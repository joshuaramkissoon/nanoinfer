//! Stage 12: Q8_0 weights times *two-level* int8 activations.
//!
//! Stage 9 found that quantising the LM head's input to int8 is what hurts:
//! GPT-2's final LayerNorm gain has 17×-rms outlier channels, so a block of 32
//! activations containing one keeps only a few bits for the rest. Here each
//! activation block is split in two int8 parts:
//!
//! ```text
//! x ≈ d·q_hi + (d/254)·q_lo     q_hi = round(x/d),  q_lo = round((x − d·q_hi) / (d/254))
//! ```
//!
//! The residual `x − d·q_hi` is at most d/2, so `q_lo` spends its 8 bits on it,
//! for ~16-bit activation precision. The dot product is two `sdot` chains per
//! block on the *same* weight bytes: twice the arithmetic, which a
//! bandwidth-bound matvec has to spare, for none of the activation error.

use super::packed::{par_cols, par_cols_n, prefill_cols, PackedMatrix, Par, Scratch};
use super::q8::Q8Matrix;
use super::tiled::OutPtr;
use crate::quant::{BlockQ8, QK};
use crate::stages::WeightFormat;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Q8_0 weights; decode uses two-level int8 activations.
pub struct Q8HiLoMatrix(pub Q8Matrix);

/// Split `x` (n rows of d) into hi / lo Q8 blocks.
pub fn quantize_hilo(x: &[f32], n: usize, d: usize, hi: &mut Vec<BlockQ8>, lo: &mut Vec<BlockQ8>) {
    hi.clear();
    lo.clear();
    for blk in x[..n * d].as_chunks::<QK>().0 {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let dh = amax / 127.0;
        let idh = if dh != 0.0 { 1.0 / dh } else { 0.0 };
        let dl = dh / 254.0;
        let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
        let mut qh = [0i8; QK];
        let mut ql = [0i8; QK];
        for j in 0..QK {
            let h = (blk[j] * idh).round();
            qh[j] = h as i8;
            ql[j] = ((blk[j] - h * dh) * idl).round().clamp(-127.0, 127.0) as i8;
        }
        hi.push(BlockQ8 { d: dh, qs: qh });
        lo.push(BlockQ8 { d: dl, qs: ql });
    }
}

/// Σ_blocks d_w · (d_hi · q_w·q_hi + d_lo · q_w·q_lo).
///
/// # Safety
/// Requires the `dotprod` CPU feature; all slices the same length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
pub unsafe fn dot_hilo(w: &[BlockQ8], hi: &[BlockQ8], lo: &[BlockQ8]) -> f32 {
    let zero = vdupq_n_s32(0);
    let mut acc_h = vdupq_n_f32(0.0);
    let mut acc_l = vdupq_n_f32(0.0);
    for ((wb, h), l) in w.iter().zip(hi).zip(lo) {
        let w0 = vld1q_s8(wb.qs.as_ptr());
        let w1 = vld1q_s8(wb.qs.as_ptr().add(16));
        let ph = vdotq_s32(vdotq_s32(zero, w0, vld1q_s8(h.qs.as_ptr())), w1, vld1q_s8(h.qs.as_ptr().add(16)));
        let pl = vdotq_s32(vdotq_s32(zero, w0, vld1q_s8(l.qs.as_ptr())), w1, vld1q_s8(l.qs.as_ptr().add(16)));
        acc_h = vfmaq_n_f32(acc_h, vcvtq_f32_s32(ph), wb.d * h.d);
        acc_l = vfmaq_n_f32(acc_l, vcvtq_f32_s32(pl), wb.d * l.d);
    }
    vaddvq_f32(vaddq_f32(acc_h, acc_l))
}

/// # Safety
/// Always safe (scalar fallback); `unsafe` only to match the NEON signature.
#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn dot_hilo(w: &[BlockQ8], hi: &[BlockQ8], lo: &[BlockQ8]) -> f32 {
    let mut s = 0f32;
    for ((wb, h), l) in w.iter().zip(hi).zip(lo) {
        let ih: i32 = wb.qs.iter().zip(&h.qs).map(|(&a, &b)| a as i32 * b as i32).sum();
        let il: i32 = wb.qs.iter().zip(&l.qs).map(|(&a, &b)| a as i32 * b as i32).sum();
        s += wb.d * (h.d * ih as f32 + l.d * il as f32);
    }
    s
}

impl PackedMatrix for Q8HiLoMatrix {
    const FORMAT: WeightFormat = WeightFormat::Q8_0;
    fn pack(w: &[f32], d_out: usize, d_in: usize) -> Self {
        Q8HiLoMatrix(Q8Matrix::pack(w, d_out, d_in))
    }
    fn d_in(&self) -> usize {
        self.0.d_in
    }
    fn d_out(&self) -> usize {
        self.0.d_out
    }
    fn bytes(&self) -> usize {
        self.0.bytes()
    }
    fn matmul<P: Par>(&self, out: &mut [f32], x: &[f32], bias: Option<&[f32]>, n: usize, scratch: &mut Scratch) {
        let (d_in, d_out) = (self.0.d_in, self.0.d_out);
        assert!(x.len() >= n * d_in && out.len() >= n * d_out);
        let nb = d_in / QK;
        let row = |o: usize| &self.0.blocks[o * nb..(o + 1) * nb];
        let ptr = OutPtr(out.as_mut_ptr());
        if n < 4 {
            quantize_hilo(x, n, d_in, &mut scratch.q8, &mut scratch.q8_lo);
            let (hi, lo) = (&scratch.q8, &scratch.q8_lo);
            par_cols::<P>(d_out, &|b, e| {
                let p = ptr;
                for i in 0..n {
                    let (h, l) = (&hi[i * nb..(i + 1) * nb], &lo[i * nb..(i + 1) * nb]);
                    for o in b..e {
                        // Safety: equal block counts; columns b..e are ours.
                        unsafe { *p.0.add(i * d_out + o) = dot_hilo(row(o), h, l) + bias.map_or(0.0, |bb| bb[o]) };
                    }
                }
            });
        } else {
            // Prefill: same as stage 9 (dequantised panels, f32 activations).
            let dequant = |o: usize, dst: &mut [f32]| {
                for (blk, out) in row(o).iter().zip(dst.as_chunks_mut::<QK>().0) {
                    for (v, &q) in out.iter_mut().zip(&blk.qs) {
                        *v = q as f32 * blk.d;
                    }
                }
            };
            // Safety: columns b..e are disjoint per chunk.
            par_cols_n::<P>(d_out, n, &|b, e| unsafe { prefill_cols(ptr, x, bias, n, d_in, d_out, b, e, &dequant) });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hilo_is_close_to_f32_activations() {
        // With hi/lo activations, the only error left is the weights' Q8 rounding:
        // compare against dequantised-W · exact-x.
        let (d_out, d_in) = (7, 96);
        let w: Vec<f32> = (0..d_out * d_in).map(|i| ((i * 7919) % 201) as f32 / 100.0 - 1.0).collect();
        let mut x: Vec<f32> = (0..d_in).map(|i| ((i * 104729) % 97) as f32 / 48.0 - 1.0).collect();
        x[5] = 17.0; // an outlier channel, like ln_f's
        let m = Q8HiLoMatrix::pack(&w, d_out, d_in);
        let mut got = vec![0.0; d_out];
        m.matmul::<super::super::packed::SpinPar>(&mut got, &x, None, 1, &mut Scratch::default());
        for (o, &g) in got.iter().enumerate() {
            let wd: Vec<f32> = m.0.blocks[o * 3..o * 3 + 3].iter().flat_map(crate::quant::dequantize_q8).collect();
            let want: f32 = wd.iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!((want - g).abs() < 2e-3, "row {o}: {want} vs {g}");
        }
    }
}
