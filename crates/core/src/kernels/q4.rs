//! Stage 10: Q4_0 weights (ggml-style), 4 bits per weight.
//!
//! A block is 16 bytes of nibbles plus an f32 scale (20 bytes per 32 weights,
//! 6.4× smaller than f32). Byte `j` holds weight `j` in its low nibble and weight
//! `j + 16` in its high nibble, each stored as `q + 8` with q in -8..=7.
//!
//! Decode unpacks a block into int8 in registers (`and 0x0f`, `shr 4`, `sub 8`),
//! then runs the same `sdot` against Q8-quantised activations as stage 9. The
//! unpacking costs instructions, but decode is bound by bytes, and we now read
//! 20 per 32 weights instead of 36.

use super::packed::{par_cols, par_cols_n, prefill_cols, PackedMatrix, Par, Scratch};
use super::q8::quantize_activations;
use super::tiled::OutPtr;
use crate::quant::{quantize_rows_q4, BlockQ4, BlockQ8, QK};
use crate::stages::WeightFormat;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// `[d_out, d_in]` weights as Q4_0 blocks, row-major.
pub struct Q4Matrix {
    pub blocks: Vec<BlockQ4>,
    pub d_in: usize,
    pub d_out: usize,
}

/// Σ over blocks of d_w · d_x · (q_w · q_x) with Q4 weights.
///
/// # Safety
/// Requires the `dotprod` CPU feature; `w` and `x` must have the same length.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
pub unsafe fn dot_q4(w: &[BlockQ4], x: &[BlockQ8]) -> f32 {
    debug_assert_eq!(w.len(), x.len());
    let nb = w.len();
    let m4 = vdupq_n_u8(0x0f);
    let s8 = vdupq_n_s8(8);
    let zero = vdupq_n_s32(0);
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let unpack = |v: uint8x16_t| -> (int8x16_t, int8x16_t) {
        let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m4)), s8);
        let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)), s8);
        (lo, hi)
    };
    let mut b = 0;
    while b + 2 <= nb {
        let (w0, x0, w1, x1) = (&w[b], &x[b], &w[b + 1], &x[b + 1]);
        let (l0, h0) = unpack(vld1q_u8(w0.qs.as_ptr()));
        let (l1, h1) = unpack(vld1q_u8(w1.qs.as_ptr()));
        let p0 = vdotq_s32(vdotq_s32(zero, l0, vld1q_s8(x0.qs.as_ptr())), h0, vld1q_s8(x0.qs.as_ptr().add(16)));
        let p1 = vdotq_s32(vdotq_s32(zero, l1, vld1q_s8(x1.qs.as_ptr())), h1, vld1q_s8(x1.qs.as_ptr().add(16)));
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

fn dot_block_scalar(w: &BlockQ4, x: &BlockQ8) -> f32 {
    let mut isum = 0i32;
    for j in 0..QK / 2 {
        isum += ((w.qs[j] & 0xf) as i32 - 8) * x.qs[j] as i32;
        isum += ((w.qs[j] >> 4) as i32 - 8) * x.qs[j + QK / 2] as i32;
    }
    isum as f32 * w.d * x.d
}

#[cfg(not(target_arch = "aarch64"))]
/// # Safety
/// Always safe (scalar fallback); `unsafe` only to match the NEON signature.
pub unsafe fn dot_q4(w: &[BlockQ4], x: &[BlockQ8]) -> f32 {
    w.iter().zip(x).map(|(a, b)| dot_block_scalar(a, b)).sum()
}

/// One Q4_0 row → f32 (prefill panels).
fn dequant_row(src: &[BlockQ4], dst: &mut [f32]) {
    for (blk, out) in src.iter().zip(dst.as_chunks_mut::<QK>().0) {
        let (lo, hi) = out.split_at_mut(QK / 2);
        for j in 0..QK / 2 {
            lo[j] = ((blk.qs[j] & 0xf) as i32 - 8) as f32 * blk.d;
            hi[j] = ((blk.qs[j] >> 4) as i32 - 8) as f32 * blk.d;
        }
    }
}

impl PackedMatrix for Q4Matrix {
    const FORMAT: WeightFormat = WeightFormat::Q4_0;

    fn pack(w: &[f32], d_out: usize, d_in: usize) -> Self {
        assert_eq!(w.len(), d_out * d_in);
        Q4Matrix { blocks: quantize_rows_q4(w, d_in), d_in, d_out }
    }
    fn d_in(&self) -> usize {
        self.d_in
    }
    fn d_out(&self) -> usize {
        self.d_out
    }
    fn bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<BlockQ4>()
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
                        unsafe { *p.0.add(i * d_out + o) = dot_q4(row(o), xr) + bias.map_or(0.0, |bb| bb[o]) };
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
    fn q4_decode_matches_its_own_dequantisation() {
        // The integer path must equal "dequantise W, quantise x, multiply" exactly
        // up to float rounding; that isolates kernel bugs from quantisation error.
        let (d_out, d_in) = (9, 64);
        let w: Vec<f32> = (0..d_out * d_in).map(|i| ((i * 7919) % 201) as f32 / 100.0 - 1.0).collect();
        let x: Vec<f32> = (0..d_in).map(|i| ((i * 104729) % 97) as f32 / 48.0 - 1.0).collect();
        let m = Q4Matrix::pack(&w, d_out, d_in);
        let mut got = vec![0.0; d_out];
        m.matmul::<super::super::packed::SpinPar>(&mut got, &x, None, 1, &mut Scratch::default());
        let mut xq = Vec::new();
        quantize_activations(&x, 1, d_in, &mut xq);
        let xd: Vec<f32> = xq.iter().flat_map(crate::quant::dequantize_q8).collect();
        for (o, &g) in got.iter().enumerate() {
            let wd: Vec<f32> = m.blocks[o * 2..o * 2 + 2].iter().flat_map(crate::quant::dequantize_q4).collect();
            let want: f32 = wd.iter().zip(&xd).map(|(a, b)| a * b).sum();
            assert!((want - g).abs() < 1e-3, "row {o}: {want} vs {g}");
        }
    }
}
