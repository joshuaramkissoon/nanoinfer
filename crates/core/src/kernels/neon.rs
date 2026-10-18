//! Stage 3 kernels: explicit NEON (aarch64 SIMD) matmul.
//!
//! Stage 2 let LLVM vectorise 16 accumulators, but it could only emit `fmul` +
//! `fadd` (Rust never fuses `a*b + c`, which would change rounding), and each
//! weight row was walked on its own, re-loading `x` for every output.
//!
//! Here we write the instructions ourselves: `vfmaq_f32`, one fused
//! multiply-add on 4 lanes (half the instructions of `fmul`+`fadd`, one rounding
//! instead of two), 4 accumulators × 4 lanes = 16 independent sums.
//!
//! **What didn't work:** computing 4 output rows per pass so each load of `x` is
//! shared (1.25 loads per FMA instead of 2). The inner loop was textbook (16
//! `fmla`, paired `ldp q` loads), yet decode got *slower*, 100 → 74 tok/s. A
//! microbenchmark showed why: one P-core streams ~49 GB/s from one sequential
//! stream but only ~32–38 GB/s from 4 interleaved row streams (software prefetch
//! made it worse). At batch 1 the loads were never the bottleneck, DRAM is, and
//! four streams are worse for the prefetcher than one. So the decode kernel walks
//! one weight row at a time. Row reuse comes back for prefill (stage 4), where
//! weights are served from cache.
//!
//! On anything but aarch64 (e.g. wasm32) this module is not compiled and the
//! `Neon` kernel set falls back to [`super::unrolled`], so the stage still runs.

#![allow(clippy::needless_range_loop)] // explicit lane/register indices read better in kernels

use std::arch::aarch64::*;

/// Horizontal sum of 4 lanes.
#[inline(always)]
unsafe fn hsum(v: float32x4_t) -> f32 {
    vaddvq_f32(v)
}

/// `dot(x, w)` for one row, 16 floats per iteration with 4 accumulators.
#[inline(always)]
unsafe fn dot1(x: &[f32], w: &[f32]) -> f32 {
    let d = x.len();
    let (xp, wp) = (x.as_ptr(), w.as_ptr());
    let mut a = [vdupq_n_f32(0.0); 4];
    let mut k = 0;
    while k + 16 <= d {
        for j in 0..4 {
            a[j] = vfmaq_f32(a[j], vld1q_f32(xp.add(k + 4 * j)), vld1q_f32(wp.add(k + 4 * j)));
        }
        k += 16;
    }
    let mut s = hsum(vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3])));
    while k < d {
        s += x[k] * w[k];
        k += 1;
    }
    s
}

/// Same contract as [`super::reference::matmul`], one fused-FMA dot product per output.
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
    assert!(x.len() >= n * d_in && out.len() >= n * d_out && w.len() == d_in * d_out);
    for (x_row, out_row) in x.chunks_exact(d_in).zip(out.chunks_exact_mut(d_out)).take(n) {
        matvec(out_row, x_row, w, bias, d_in, d_out, 0);
    }
}

/// `out[o - o0] = dot(x, w[o]) + bias[o]` for the output rows `o0 .. o0 + out.len()`.
/// Factored out so later stages can hand disjoint row ranges to different threads.
pub fn matvec(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, d_in: usize, d_out: usize, o0: usize) {
    assert!(x.len() == d_in && w.len() == d_in * d_out && o0 + out.len() <= d_out);
    for (i, y) in out.iter_mut().enumerate() {
        let o = o0 + i;
        // Safety: the row slice has exactly d_in = x.len() elements (checked above),
        // and dot1 only reads indices < d_in.
        *y = unsafe { dot1(x, &w[o * d_in..(o + 1) * d_in]) } + bias.map_or(0.0, |b| b[o]);
    }
}
