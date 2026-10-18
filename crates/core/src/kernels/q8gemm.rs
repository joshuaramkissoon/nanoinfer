//! Stage 15: integer prefill for Q8_0 weights.
//!
//! Until now prefill dequantised each 4-row weight panel back to f32 and ran the
//! f32 FMA tile (stage 8's path): ~4 f32 multiply-adds per instruction. Prefill is
//! compute-bound, so the instruction mix is what matters. Here the prompt rows
//! are quantised to Q8_0 blocks too (exactly as decode already does), and a
//! 4 × 4 output tile is computed with `sdot`: per block of 32 and per (row,
//! weight-row) pair, two `sdot`s (32 int8 multiply-adds), one convert and one
//! FMA by the two scales. ~8 multiply-adds per instruction instead of ~2.7.
//!
//! Decode is unchanged (stage 9's `dot_q8`).

use super::packed::{par_cols, par_cols_n, PackedMatrix, Par, Scratch};
use super::q8::{dot_q8, quantize_activations, Q8Matrix};
use super::tiled::OutPtr;
#[cfg(target_arch = "aarch64")]
use crate::quant::BlockQ8;
use crate::quant::QK;
use crate::stages::WeightFormat;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Q8_0 weights whose prefill uses an int8 `sdot` tile.
pub struct Q8GemmMatrix(pub Q8Matrix);

/// `out[i*ldo + j] = Σ_b dw_j,b · dx_i,b · (q_w · q_x)` for a 4 × 4 tile:
/// x rows `xb[i*nb..]`, weight rows `wb[j*nb..]`, `nb` blocks each.
///
/// # Safety
/// Requires `dotprod`; `xb` has 4·nb blocks, `wb` 4·nb, `out` a writable 4×4 tile.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(clippy::needless_range_loop)]
unsafe fn tile_4x4(out: *mut f32, ldo: usize, xb: &[BlockQ8], wb: &[BlockQ8], nb: usize, bias: Option<&[f32]>) {
    let zero = vdupq_n_s32(0);
    let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
    for b in 0..nb {
        let w: [&BlockQ8; 4] = std::array::from_fn(|j| &wb[j * nb + b]);
        let wl: [int8x16_t; 4] = std::array::from_fn(|j| vld1q_s8(w[j].qs.as_ptr()));
        let wh: [int8x16_t; 4] = std::array::from_fn(|j| vld1q_s8(w[j].qs.as_ptr().add(16)));
        for i in 0..4 {
            let x = &xb[i * nb + b];
            let xl = vld1q_s8(x.qs.as_ptr());
            let xh = vld1q_s8(x.qs.as_ptr().add(16));
            for j in 0..4 {
                let p = vdotq_s32(vdotq_s32(zero, wl[j], xl), wh[j], xh);
                acc[i][j] = vfmaq_n_f32(acc[i][j], vcvtq_f32_s32(p), w[j].d * x.d);
            }
        }
    }
    for i in 0..4 {
        for j in 0..4 {
            *out.add(i * ldo + j) = vaddvq_f32(acc[i][j]) + bias.map_or(0.0, |bb| bb[j]);
        }
    }
}

impl PackedMatrix for Q8GemmMatrix {
    const FORMAT: WeightFormat = WeightFormat::Q8_0;
    fn pack(w: &[f32], d_out: usize, d_in: usize) -> Self {
        Q8GemmMatrix(Q8Matrix::pack(w, d_out, d_in))
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
        let blocks = &self.0.blocks;
        let ptr = OutPtr(out.as_mut_ptr());
        quantize_activations(x, n, d_in, &mut scratch.q8);
        let xq = &scratch.q8;
        // Rows of x (or any target without NEON) that don't fill a tile.
        let single = |b: usize, e: usize, rows: std::ops::Range<usize>| {
            let p = ptr;
            for i in rows {
                let xr = &xq[i * nb..(i + 1) * nb];
                for o in b..e {
                    // Safety: same block counts; columns b..e are this chunk's.
                    unsafe {
                        *p.0.add(i * d_out + o) =
                            dot_q8(&blocks[o * nb..(o + 1) * nb], xr) + bias.map_or(0.0, |bb| bb[o])
                    };
                }
            }
        };
        if n < 4 {
            return par_cols::<P>(d_out, &|b, e| single(b, e, 0..n));
        }
        par_cols_n::<P>(d_out, n, &|b, e| {
            #[cfg(target_arch = "aarch64")]
            {
                let p = ptr;
                let mut o = b;
                while o + 4 <= e {
                    let wb = &blocks[o * nb..(o + 4) * nb];
                    let bo = bias.map(|bb| &bb[o..o + 4]);
                    let mut i = 0;
                    while i + 4 <= n {
                        // Safety: 4 x rows and 4 weight rows of nb blocks; tile in bounds.
                        unsafe { tile_4x4(p.0.add(i * d_out + o), d_out, &xq[i * nb..(i + 4) * nb], wb, nb, bo) };
                        i += 4;
                    }
                    single(o, o + 4, i..n);
                    o += 4;
                }
                single(o, e, 0..n);
            }
            #[cfg(not(target_arch = "aarch64"))]
            single(b, e, 0..n);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_matches_per_row_integer_dot() {
        // The tile must equal stage 9's per-row integer dot exactly (same math).
        let (n, d_out, d_in) = (7, 10, 64);
        let w: Vec<f32> = (0..d_out * d_in).map(|i| ((i * 7919) % 201) as f32 / 100.0 - 1.0).collect();
        let x: Vec<f32> = (0..n * d_in).map(|i| ((i * 104729) % 97) as f32 / 48.0 - 1.0).collect();
        let b: Vec<f32> = (0..d_out).map(|i| i as f32 * 0.5).collect();
        let m = Q8GemmMatrix::pack(&w, d_out, d_in);
        let mut got = vec![0.0; n * d_out];
        m.matmul::<super::super::packed::SpinPar>(&mut got, &x, Some(&b), n, &mut Scratch::default());
        let mut xq = Vec::new();
        quantize_activations(&x, n, d_in, &mut xq);
        for i in 0..n {
            for o in 0..d_out {
                let want = unsafe { dot_q8(&m.0.blocks[o * 2..o * 2 + 2], &xq[i * 2..i * 2 + 2]) } + b[o];
                let g = got[i * d_out + o];
                assert!((want - g).abs() < 1e-4 * want.abs().max(1.0), "({i},{o}): {want} vs {g}");
            }
        }
    }
}
