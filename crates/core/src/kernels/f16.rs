//! Stage 8: f16 weights (half the bytes of f32), f32 arithmetic.
//!
//! Weights are converted to IEEE half precision once, when the session is
//! created. In the kernel, each 8 halves are widened to two f32 vectors with
//! `fcvtl` / `fcvtl2` (one instruction per 4 lanes) and fed to the same f32
//! FMAs as before. So accumulation stays f32 and the only error is the rounding
//! of each weight to 11 significant bits (relative error ≤ 2⁻¹¹ ≈ 0.05%).
//!
//! Rust's f16 NEON intrinsics (`vcvt_f32_f16`, the `f16` type) are still
//! unstable, so the conversion is two lines of inline assembly, which is stable
//! on aarch64. Elsewhere (wasm) a scalar bit-twiddling conversion is used.

use super::packed::{par_cols, par_cols_n, prefill_cols, PackedMatrix, Par, Scratch};
use super::tiled::OutPtr;
use crate::quant::{f16_to_f32, f32_to_f16};
use crate::stages::WeightFormat;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// `[d_out, d_in]` f16 weights, row-major, as raw IEEE half bits.
pub struct F16Matrix {
    pub w: Vec<u16>,
    pub d_in: usize,
    pub d_out: usize,
}

/// Widen 8 halves to 2 × 4 floats.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn widen(h: uint16x8_t) -> (float32x4_t, float32x4_t) {
    let lo: float32x4_t;
    let hi: float32x4_t;
    // Safety: pure register-to-register conversion.
    std::arch::asm!(
        "fcvtl {lo:v}.4s, {h:v}.4h",
        "fcvtl2 {hi:v}.4s, {h:v}.8h",
        h = in(vreg) h,
        lo = out(vreg) lo,
        hi = lateout(vreg) hi,
        options(pure, nomem, nostack, preserves_flags)
    );
    (lo, hi)
}

/// dot(x, w) with w in f16: 16 weights per iteration, 4 f32 accumulators.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot(x: &[f32], w: &[u16]) -> f32 {
    let d = x.len();
    let (xp, wp) = (x.as_ptr(), w.as_ptr());
    let mut a = [vdupq_n_f32(0.0); 4];
    let mut k = 0;
    while k + 16 <= d {
        let (w0, w1) = widen(vld1q_u16(wp.add(k)));
        let (w2, w3) = widen(vld1q_u16(wp.add(k + 8)));
        a[0] = vfmaq_f32(a[0], vld1q_f32(xp.add(k)), w0);
        a[1] = vfmaq_f32(a[1], vld1q_f32(xp.add(k + 4)), w1);
        a[2] = vfmaq_f32(a[2], vld1q_f32(xp.add(k + 8)), w2);
        a[3] = vfmaq_f32(a[3], vld1q_f32(xp.add(k + 12)), w3);
        k += 16;
    }
    let mut s = vaddvq_f32(vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3])));
    for kk in k..d {
        s += x[kk] * f16_to_f32(w[kk]);
    }
    s
}

#[cfg(not(target_arch = "aarch64"))]
unsafe fn dot(x: &[f32], w: &[u16]) -> f32 {
    let mut acc = [0f32; 8];
    let (xc, xr) = x.as_chunks::<8>();
    let (wc, wr) = w.as_chunks::<8>();
    for (a, b) in xc.iter().zip(wc) {
        for j in 0..8 {
            acc[j] += a[j] * f16_to_f32(b[j]);
        }
    }
    acc.iter().sum::<f32>() + xr.iter().zip(wr).map(|(a, &b)| a * f16_to_f32(b)).sum::<f32>()
}

/// f16 row → f32 (for prefill panels).
fn widen_row(src: &[u16], dst: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        let n8 = src.len() / 8 * 8;
        for k in (0..n8).step_by(8) {
            // Safety: k + 8 <= len for both slices (same length).
            unsafe {
                let (lo, hi) = widen(vld1q_u16(src.as_ptr().add(k)));
                vst1q_f32(dst.as_mut_ptr().add(k), lo);
                vst1q_f32(dst.as_mut_ptr().add(k + 4), hi);
            }
        }
        for k in n8..src.len() {
            dst[k] = f16_to_f32(src[k]);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = f16_to_f32(s);
    }
}

impl PackedMatrix for F16Matrix {
    const FORMAT: WeightFormat = WeightFormat::F16;

    fn pack(w: &[f32], d_out: usize, d_in: usize) -> Self {
        assert_eq!(w.len(), d_out * d_in);
        F16Matrix { w: w.iter().map(|&v| f32_to_f16(v)).collect(), d_in, d_out }
    }
    fn d_in(&self) -> usize {
        self.d_in
    }
    fn d_out(&self) -> usize {
        self.d_out
    }
    fn bytes(&self) -> usize {
        self.w.len() * 2
    }

    fn matmul<P: Par>(&self, out: &mut [f32], x: &[f32], bias: Option<&[f32]>, n: usize, _scratch: &mut Scratch) {
        let (d_in, d_out) = (self.d_in, self.d_out);
        assert!(x.len() >= n * d_in && out.len() >= n * d_out);
        let ptr = OutPtr(out.as_mut_ptr());
        let row = |o: usize| &self.w[o * d_in..(o + 1) * d_in];
        if n < 4 {
            par_cols::<P>(d_out, &|b, e| {
                let p = ptr;
                for i in 0..n {
                    let xr = &x[i * d_in..(i + 1) * d_in];
                    for o in b..e {
                        // Safety: row(o) has d_in = xr.len() halves; columns b..e are ours.
                        unsafe { *p.0.add(i * d_out + o) = dot(xr, row(o)) + bias.map_or(0.0, |bb| bb[o]) };
                    }
                }
            });
        } else {
            // Safety: columns b..e are disjoint per chunk.
            par_cols_n::<P>(d_out, n, &|b, e| unsafe {
                prefill_cols(ptr, x, bias, n, d_in, d_out, b, e, &|o, dst| widen_row(row(o), dst))
            });
        }
    }
}

impl F16Matrix {
    /// `dot(x, row o)`, one output (used to rescore a few LM-head rows exactly).
    pub fn dot_row(&self, x: &[f32], o: usize) -> f32 {
        assert_eq!(x.len(), self.d_in);
        // Safety: the row has exactly d_in halves.
        unsafe { dot(x, &self.w[o * self.d_in..(o + 1) * self.d_in]) }
    }
}
