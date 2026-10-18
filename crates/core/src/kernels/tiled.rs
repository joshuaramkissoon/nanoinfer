//! Stage 4 kernels: register-blocked matmul for prefill.
//!
//! During prefill all `n` prompt rows multiply the same weight matrix. Stages
//! 1–3 loop "for each prompt row → for each weight row → dot", so the whole
//! weight matrix (2–9 MB per layer matmul) is streamed once *per prompt row*.
//! It fits in the 16 MB L2, so prefill was running at L2 bandwidth and fused
//! FMAs (stage 3) didn't help at all.
//!
//! Register blocking: compute a 4 × 4 tile of the output (4 prompt rows × 4
//! weight rows) at once, in 16 NEON accumulators. Each step loads 4 chunks of x
//! and 4 chunks of w and does 16 FMAs: 0.5 loads per FMA instead of 2, i.e. every
//! weight value fetched is used 4 times from a register. And the loop order is
//! weight-block outer: a 4-row weight panel (≤ 48 KB) stays in L1 while all prompt
//! rows pass over it, so each weight is read from DRAM/L2 ~once per matmul.
//!
//! Decode (n < 4) goes through the stage 3 single-row kernel, unchanged.

#![allow(clippy::needless_range_loop)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Output tile height (prompt rows) and width (weight rows).
pub const MR: usize = 4;
pub const NR: usize = 4;

/// `out[i*ldo + o] = dot(x_i, w_o) (+ bias[o])` for a 4×4 tile.
///
/// # Safety
/// `x` points to 4 rows of `d` floats spaced `d` apart, `w` to 4 rows of `d`
/// floats spaced `d` apart, and `out` to a writable 4 × 4 tile with row stride `ldo`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn tile_4x4(out: *mut f32, ldo: usize, x: *const f32, w: *const f32, d: usize, bias: Option<&[f32]>) {
    let mut acc = [[vdupq_n_f32(0.0); NR]; MR];
    let mut k = 0;
    while k + 4 <= d {
        let xv =
            [vld1q_f32(x.add(k)), vld1q_f32(x.add(d + k)), vld1q_f32(x.add(2 * d + k)), vld1q_f32(x.add(3 * d + k))];
        let wv =
            [vld1q_f32(w.add(k)), vld1q_f32(w.add(d + k)), vld1q_f32(w.add(2 * d + k)), vld1q_f32(w.add(3 * d + k))];
        for i in 0..MR {
            for j in 0..NR {
                acc[i][j] = vfmaq_f32(acc[i][j], xv[i], wv[j]);
            }
        }
        k += 4;
    }
    for i in 0..MR {
        for j in 0..NR {
            let mut s = vaddvq_f32(acc[i][j]);
            for kk in k..d {
                s += *x.add(i * d + kk) * *w.add(j * d + kk);
            }
            *out.add(i * ldo + j) = s + bias.map_or(0.0, |b| b[j]);
        }
    }
}

/// Raw output pointer that may be shared across threads writing disjoint tiles.
#[derive(Clone, Copy)]
pub struct OutPtr(pub *mut f32);
// Safety: callers guarantee disjoint writes (each thread owns a column range).
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

/// Compute output columns `o_begin..o_end` (weight rows) for all `n` rows of x,
/// into `out` with row stride `d_out`. Columns are disjoint per caller, so
/// threads can each take a range (stage 5).
///
/// # Safety
/// `out` must point to an `n × d_out` buffer; no other thread may write the
/// same columns concurrently.
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_cols(
    out: OutPtr,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    d_in: usize,
    d_out: usize,
    o_begin: usize,
    o_end: usize,
) {
    debug_assert!(x.len() >= n * d_in && w.len() == d_in * d_out && o_end <= d_out);
    let out = out.0;
    #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))]
    let mut o = o_begin;
    #[cfg(target_arch = "aarch64")]
    while o + NR <= o_end {
        let wp = w.as_ptr().add(o * d_in);
        let b = bias.map(|b| &b[o..o + NR]);
        let mut i = 0;
        while i + MR <= n {
            tile_4x4(out.add(i * d_out + o), d_out, x.as_ptr().add(i * d_in), wp, d_in, b);
            i += MR;
        }
        // Leftover prompt rows (n % 4): one dot product each.
        for i in i..n {
            for j in 0..NR {
                let y = super::unrolled::dot(&x[i * d_in..(i + 1) * d_in], &w[(o + j) * d_in..(o + j + 1) * d_in]);
                *out.add(i * d_out + o + j) = y + b.map_or(0.0, |b| b[j]);
            }
        }
        o += NR;
    }
    // Leftover weight rows (d_out % 4, e.g. the LM head's 50257), or everything
    // on targets without NEON.
    for o in o..o_end {
        let w_row = &w[o * d_in..(o + 1) * d_in];
        for i in 0..n {
            let y = super::unrolled::dot(&x[i * d_in..(i + 1) * d_in], w_row);
            *out.add(i * d_out + o) = y + bias.map_or(0.0, |b| b[o]);
        }
    }
}

/// Same contract as [`super::reference::matmul`].
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
    assert!(x.len() >= n * d_in && out.len() >= n * d_out && w.len() == d_in * d_out);
    if n < MR {
        // Decode: nothing to reuse across rows, stage 3's kernel.
        #[cfg(target_arch = "aarch64")]
        for (x_row, out_row) in x.chunks_exact(d_in).zip(out.chunks_exact_mut(d_out)).take(n) {
            super::neon::matvec(out_row, x_row, w, bias, d_in, d_out, 0);
        }
        #[cfg(not(target_arch = "aarch64"))]
        super::unrolled::matmul(out, x, w, bias, n, d_in, d_out);
        return;
    }
    // Safety: out holds n × d_out floats (asserted) and we're the only writer.
    unsafe { matmul_cols(OutPtr(out.as_mut_ptr()), x, w, bias, n, d_in, d_out, 0, d_out) }
}

#[cfg(test)]
mod tests {
    #[test]
    fn matches_reference_with_ragged_edges() {
        // n = 7 (one full 4-row tile + 3), d_out = 10 (two tiles + 2), d_in = 13 (k tail).
        let (n, d_in, d_out) = (7, 13, 10);
        let x: Vec<f32> = (0..n * d_in).map(|i| ((i * 7) % 11) as f32 - 5.0).collect();
        let w: Vec<f32> = (0..d_out * d_in).map(|i| ((i * 3) % 5) as f32 - 2.0).collect();
        let b: Vec<f32> = (0..d_out).map(|i| i as f32).collect();
        let mut want = vec![0.0; n * d_out];
        let mut got = vec![0.0; n * d_out];
        super::super::reference::matmul(&mut want, &x, &w, Some(&b), n, d_in, d_out);
        super::matmul(&mut got, &x, &w, Some(&b), n, d_in, d_out);
        assert_eq!(want, got);
    }
}
