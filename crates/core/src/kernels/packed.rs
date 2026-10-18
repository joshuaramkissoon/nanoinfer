//! Shared machinery for weight matrices stored in a format other than f32
//! (f16, Q8_0, Q4_0: stages 8+).
//!
//! A [`PackedMatrix`] knows how to multiply activations by itself. Every format
//! uses the same plan:
//!
//! - **Decode** (n < 4 rows): the format's own matvec reads the packed weights
//!   directly, so we stream fewer bytes (the whole point; decode is
//!   memory-bound).
//! - **Prefill**: for each 4-row weight panel, dequantise it *once* to f32 on the
//!   stack, then run a 4×4 register-blocked FMA tile (as in stage 4) over all
//!   prompt rows. Dequantisation is amortised over n rows, so prefill stays
//!   compute-bound at ~f32 speed.
//!
//! Both are split over output columns on the spin pool, like stage 6.

use super::spin;
use super::tiled::OutPtr;
use crate::quant::BlockQ8;
use crate::stages::WeightFormat;

/// Reusable per-session buffers for packed kernels (never shrink, so steady-state
/// decoding doesn't allocate).
#[derive(Debug, Default)]
pub struct Scratch {
    /// Activations quantised to Q8_0 blocks (for the integer dot products).
    pub q8: Vec<BlockQ8>,
    /// Second (residual) int8 level of two-level activations (stage 12).
    pub q8_lo: Vec<BlockQ8>,
    /// Candidate token ids (stage 12's rescored LM head).
    pub candidates: Vec<u32>,
}

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// A weight matrix `[d_out, d_in]` in some storage format.
pub trait PackedMatrix: Send + Sync + Sized {
    const FORMAT: WeightFormat;
    /// Convert row-major f32 `w[d_out, d_in]`.
    fn pack(w: &[f32], d_out: usize, d_in: usize) -> Self;
    fn d_in(&self) -> usize;
    fn d_out(&self) -> usize;
    /// Bytes of storage (what decode has to stream).
    fn bytes(&self) -> usize;
    /// `out[n, d_out] = x[n, d_in] · Wᵀ + bias`.
    fn matmul<P: Par>(&self, out: &mut [f32], x: &[f32], bias: Option<&[f32]>, n: usize, scratch: &mut Scratch);
}

/// Which parallel runtime a packed kernel runs on (a type parameter, so each
/// stage keeps the runtime it was measured with).
pub trait Par {
    /// Column chunks per thread (more = better P/E balance, more overhead).
    const CHUNKS_PER_THREAD: usize;
    /// Chunks per thread for a matmul over `n` rows (default: the constant).
    fn chunks_per_thread(_n: usize) -> usize {
        Self::CHUNKS_PER_THREAD
    }
    fn threads() -> usize;
    fn par_for(n: usize, f: &(dyn Fn(usize) + Sync));
}

/// Stage 6's spin pool (stages 8–10).
pub struct SpinPar;

impl Par for SpinPar {
    const CHUNKS_PER_THREAD: usize = 4;
    fn threads() -> usize {
        spin::num_threads()
    }
    fn par_for(n: usize, f: &(dyn Fn(usize) + Sync)) {
        spin::par_for(n, f)
    }
}

/// Split output columns into chunks on `P` and call `f(b, e)` for each.
/// Chunks are multiples of 16 columns.
pub fn par_cols<P: Par>(d_out: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    par_cols_n::<P>(d_out, 1, f)
}

/// As [`par_cols`], for a matmul over `n` rows (lets `P` chunk decode and
/// prefill differently).
pub fn par_cols_n<P: Par>(d_out: usize, n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    let width = d_out.div_ceil(P::threads() * P::chunks_per_thread(n)).next_multiple_of(16);
    let n_chunks = d_out.div_ceil(width);
    P::par_for(n_chunks, &|c| f(c * width, ((c + 1) * width).min(d_out)));
}

/// `out[i*ldo + j] = dot(x_i, w_j) (+ bias[j])` for a 4 × 4 tile, `x` rows `ldx`
/// apart and `w` rows `ldw` apart, all of length `d`.
///
/// # Safety
/// All pointers valid for the described rows; `out` tile writable.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn tile_4x4(
    out: *mut f32,
    ldo: usize,
    x: *const f32,
    ldx: usize,
    w: *const f32,
    ldw: usize,
    d: usize,
    bias: Option<&[f32]>,
) {
    let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
    let mut k = 0;
    while k + 4 <= d {
        let xv = [
            vld1q_f32(x.add(k)),
            vld1q_f32(x.add(ldx + k)),
            vld1q_f32(x.add(2 * ldx + k)),
            vld1q_f32(x.add(3 * ldx + k)),
        ];
        let wv = [
            vld1q_f32(w.add(k)),
            vld1q_f32(w.add(ldw + k)),
            vld1q_f32(w.add(2 * ldw + k)),
            vld1q_f32(w.add(3 * ldw + k)),
        ];
        for i in 0..4 {
            for j in 0..4 {
                acc[i][j] = vfmaq_f32(acc[i][j], xv[i], wv[j]);
            }
        }
        k += 4;
    }
    for i in 0..4 {
        for j in 0..4 {
            let mut s = vaddvq_f32(acc[i][j]);
            for kk in k..d {
                s += *x.add(i * ldx + kk) * *w.add(j * ldw + kk);
            }
            *out.add(i * ldo + j) = s + bias.map_or(0.0, |b| b[j]);
        }
    }
}

/// Prefill columns `b..e`: dequantise 4 weight rows at a time into an f32 panel
/// with `dequant(row, &mut panel_row)`, then multiply every prompt row by it.
///
/// # Safety
/// `out` points to `n × d_out` floats; nobody else writes columns `b..e`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_cols(
    out: OutPtr,
    x: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    d_in: usize,
    d_out: usize,
    b: usize,
    e: usize,
    dequant: &dyn Fn(usize, &mut [f32]),
) {
    let out = out.0;
    let mut panel = vec![0f32; 4 * d_in];
    let mut o = b;
    while o < e {
        let rows = (e - o).min(4);
        for r in 0..rows {
            dequant(o + r, &mut panel[r * d_in..(r + 1) * d_in]);
        }
        let bias_o = bias.map(|bb| &bb[o..o + rows]);
        #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))]
        let mut i = 0;
        #[cfg(target_arch = "aarch64")]
        if rows == 4 {
            while i + 4 <= n {
                tile_4x4(
                    out.add(i * d_out + o),
                    d_out,
                    x.as_ptr().add(i * d_in),
                    d_in,
                    panel.as_ptr(),
                    d_in,
                    d_in,
                    bias_o,
                );
                i += 4;
            }
        }
        for i in i..n {
            let xr = &x[i * d_in..(i + 1) * d_in];
            for r in 0..rows {
                let y = super::unrolled::dot(xr, &panel[r * d_in..(r + 1) * d_in]);
                *out.add(i * d_out + o + r) = y + bias_o.map_or(0.0, |bb| bb[r]);
            }
        }
        o += rows;
    }
}
