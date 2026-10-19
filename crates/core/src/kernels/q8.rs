//! Stage 9: Q8_0 weights (ggml-style), integer dot products.
//!
//! Each block of 32 weights is 32 × `i8` plus one f32 scale `d_w` (36 bytes:
//! 3.6× smaller than f32). For decode the activation row is quantised the same
//! way, once per matmul (`d_x` per 32 values). Then each block's dot product is
//! exact integer arithmetic:
//!
//! ```text
//! Σ_j w_j x_j  ≈  d_w · d_x · Σ_j q_w[j] · q_x[j]        (the Σ is int32, exact)
//! ```
//!
//! On aarch64 the inner sum is `sdot` (`vdotq_s32`): 16 int8 × int8 products
//! summed into 4 int32 lanes in one instruction, i.e. 4× the multiply-adds of an
//! f32 FMA per instruction, on data 4× smaller. Elsewhere [`super::i8dot`] (simd128 on wasm).
//! Prefill dequantises 4-row panels to f32 and uses the FMA tile (see
//! [`super::packed`]): x stays f32 there.

use super::packed::{par_cols, par_cols_n, prefill_cols, PackedMatrix, Par, Scratch};
use super::tiled::OutPtr;
use crate::quant::{quantize_q8, quantize_rows_q8, BlockQ8, QK};
use crate::stages::WeightFormat;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// `[d_out, d_in]` weights as Q8_0 blocks, row-major (`d_in / 32` blocks per row).
pub struct Q8Matrix {
    pub blocks: Vec<BlockQ8>,
    pub d_in: usize,
    pub d_out: usize,
}

/// Quantise `n` rows of `x[n, d]` to Q8_0 into `dst` (resized; never shrinks capacity).
pub fn quantize_activations(x: &[f32], n: usize, d: usize, dst: &mut Vec<BlockQ8>) {
    dst.clear();
    dst.extend(x[..n * d].as_chunks::<QK>().0.iter().map(quantize_q8));
}

/// Σ over blocks of d_w · d_x · (q_w · q_x), two blocks per iteration with
/// independent accumulators.
///
/// # Safety
/// Requires the `dotprod` CPU feature (every Apple M-series chip has it); `w`
/// and `x` must have the same length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
pub unsafe fn dot_q8(w: &[BlockQ8], x: &[BlockQ8]) -> f32 {
    debug_assert_eq!(w.len(), x.len());
    let nb = w.len();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let zero = vdupq_n_s32(0);
    let mut b = 0;
    while b + 2 <= nb {
        let (w0, x0, w1, x1) = (&w[b], &x[b], &w[b + 1], &x[b + 1]);
        let p0 = vdotq_s32(
            vdotq_s32(zero, vld1q_s8(w0.qs.as_ptr()), vld1q_s8(x0.qs.as_ptr())),
            vld1q_s8(w0.qs.as_ptr().add(16)),
            vld1q_s8(x0.qs.as_ptr().add(16)),
        );
        let p1 = vdotq_s32(
            vdotq_s32(zero, vld1q_s8(w1.qs.as_ptr()), vld1q_s8(x1.qs.as_ptr())),
            vld1q_s8(w1.qs.as_ptr().add(16)),
            vld1q_s8(x1.qs.as_ptr().add(16)),
        );
        acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(p0), w0.d * x0.d);
        acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(p1), w1.d * x1.d);
        b += 2;
    }
    let mut s = vaddvq_f32(vaddq_f32(acc0, acc1));
    for b in b..nb {
        s += dot_block_scalar(&w[b], &x[b]);
    }
    s
}

#[cfg(target_arch = "aarch64")]
fn dot_block_scalar(w: &BlockQ8, x: &BlockQ8) -> f32 {
    let isum: i32 = w.qs.iter().zip(&x.qs).map(|(&a, &b)| a as i32 * b as i32).sum();
    isum as f32 * w.d * x.d
}

#[cfg(not(target_arch = "aarch64"))]
/// # Safety
/// Always safe (scalar fallback); `unsafe` only to match the NEON signature.
/// Each block's integer sum comes from [`super::i8dot::isum_q8`] (simd128 / relaxed-SIMD
/// on wasm), scaled in the same order as the scalar loop: bit-identical to it.
pub unsafe fn dot_q8(w: &[BlockQ8], x: &[BlockQ8]) -> f32 {
    w.iter().zip(x).map(|(a, b)| super::i8dot::isum_q8(&a.qs, &b.qs) as f32 * a.d * b.d).sum()
}

/// One Q8_0 row → f32 (prefill panels). Plain loop: LLVM vectorises it.
fn dequant_row(src: &[BlockQ8], dst: &mut [f32]) {
    for (blk, out) in src.iter().zip(dst.as_chunks_mut::<QK>().0) {
        for (o, &q) in out.iter_mut().zip(&blk.qs) {
            *o = q as f32 * blk.d;
        }
    }
}

impl PackedMatrix for Q8Matrix {
    const FORMAT: WeightFormat = WeightFormat::Q8_0;

    fn pack(w: &[f32], d_out: usize, d_in: usize) -> Self {
        assert_eq!(w.len(), d_out * d_in);
        Q8Matrix { blocks: quantize_rows_q8(w, d_in), d_in, d_out }
    }
    fn d_in(&self) -> usize {
        self.d_in
    }
    fn d_out(&self) -> usize {
        self.d_out
    }
    fn bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<BlockQ8>()
    }

    fn matmul<P: Par>(&self, out: &mut [f32], x: &[f32], bias: Option<&[f32]>, n: usize, scratch: &mut Scratch) {
        let (d_in, d_out) = (self.d_in, self.d_out);
        assert!(x.len() >= n * d_in && out.len() >= n * d_out);
        let nb = d_in / QK;
        let row = |o: usize| &self.blocks[o * nb..(o + 1) * nb];
        let ptr = OutPtr(out.as_mut_ptr());
        if n < 4 {
            quantize_activations(x, n, d_in, &mut scratch.q8);
            let xq = &scratch.q8;
            par_cols::<P>(d_out, &|b, e| {
                let p = ptr;
                for i in 0..n {
                    let xr = &xq[i * nb..(i + 1) * nb];
                    for o in b..e {
                        // Safety: same block counts; columns b..e are ours.
                        unsafe { *p.0.add(i * d_out + o) = dot_q8(row(o), xr) + bias.map_or(0.0, |bb| bb[o]) };
                    }
                }
            });
        } else {
            // Safety: columns b..e are disjoint per chunk.
            par_cols_n::<P>(d_out, n, &|b, e| unsafe {
                prefill_cols(ptr, x, bias, n, d_in, d_out, b, e, &|o, dst| dequant_row(row(o), dst))
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_matmul_close_to_f32() {
        let (d_out, d_in) = (37, 96);
        let w: Vec<f32> = (0..d_out * d_in).map(|i| ((i * 7919) % 201) as f32 / 100.0 - 1.0).collect();
        let x: Vec<f32> = (0..5 * d_in).map(|i| ((i * 104729) % 97) as f32 / 48.0 - 1.0).collect();
        let m = Q8Matrix::pack(&w, d_out, d_in);
        for n in [1, 5] {
            let mut want = vec![0.0; n * d_out];
            super::super::reference::matmul(&mut want, &x, &w, None, n, d_in, d_out);
            let mut got = vec![0.0; n * d_out];
            m.matmul::<super::super::packed::SpinPar>(&mut got, &x, None, n, &mut Scratch::default());
            let err = want.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(err < 0.1, "n={n}: max err {err}");
        }
    }
}
