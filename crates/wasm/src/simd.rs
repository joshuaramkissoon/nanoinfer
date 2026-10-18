//! Browser-only kernel sets: explicit wasm SIMD.
//!
//! Core's kernels are written for the native hillclimb. Two of them don't carry
//! over to wasm: `neon` has no NEON to call, and `tiled`'s 4×4 tile becomes
//! scalar code (no `vfmaq` to lower to). These kernels are the browser rungs of
//! the ladder, written against `core::arch::wasm32` directly:
//!
//! - [`WasmSimd`]: simd128 matmul, `f32x4_mul` + `f32x4_add`. Decode walks four
//!   weight rows at once (one load of `x` feeds four rows); prefill uses a 4×4
//!   register tile (4 activation rows × 4 weight rows, 16 vector accumulators),
//!   so each weight vector loaded is used four times.
//! - [`WasmFma`]: the same loops with `f32x4_relaxed_madd`, which V8 lowers to a
//!   real fused multiply-add (`fmla` on Apple silicon). Baseline wasm has no
//!   FMA at all, so this halves the instruction count of the inner loop. Only
//!   exists in the build compiled with `+relaxed-simd`; the page feature-detects
//!   it and falls back to the simd128 build.
//!
//! They plug into core's `kv_cache::forward::<K, T>` like any other kernel set
//! (core's `Kernels` trait), so the forward pass, tracing and correctness tests
//! are shared; only the matmul changes.

use nanoinfer_core::kernels::{unrolled, Kernels};

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

/// How a kernel set multiplies and accumulates one vector.
pub trait Madd {
    const NAME: &'static str;
    #[cfg(target_arch = "wasm32")]
    fn madd(a: v128, b: v128, acc: v128) -> v128;
}

/// Baseline simd128: separate multiply and add (two roundings, two instructions).
pub struct Plain;
impl Madd for Plain {
    const NAME: &'static str = "wasm-simd";
    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    fn madd(a: v128, b: v128, acc: v128) -> v128 {
        f32x4_add(acc, f32x4_mul(a, b))
    }
}

/// Relaxed SIMD: `a*b + acc` that the engine may fuse (it does on arm64 and x86 with FMA).
#[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
pub struct Relaxed;
#[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
impl Madd for Relaxed {
    const NAME: &'static str = "wasm-fma";
    #[inline(always)]
    fn madd(a: v128, b: v128, acc: v128) -> v128 {
        f32x4_relaxed_madd(a, b, acc)
    }
}

/// A kernel set built on a [`Madd`] flavour. Everything but the matmul is
/// inherited from core's reference kernels.
pub struct Simd<M: Madd>(core::marker::PhantomData<M>);

impl<M: Madd> Kernels for Simd<M> {
    const NAME: &'static str = M::NAME;
    fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
        matmul::<M>(out, x, w, bias, n, d_in, d_out)
    }
}

pub type WasmSimd = Simd<Plain>;
#[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
pub type WasmFma = Simd<Relaxed>;

/// Same contract as core's `reference::matmul`: `out[n, d_out] = x[n, d_in] · wᵀ + b`
/// with `w` stored `[d_out, d_in]`.
pub fn matmul<M: Madd>(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
    assert!(x.len() >= n * d_in && out.len() >= n * d_out && w.len() == d_in * d_out);
    #[cfg(target_arch = "wasm32")]
    if d_in.is_multiple_of(8) {
        // Safety: bounds asserted above; we're the only writer.
        unsafe { cols::<M>(out.as_mut_ptr(), x, w, bias, n, d_in, d_out, 0, d_out) };
        return;
    }
    // Shapes the SIMD paths don't cover (never hit by GPT-2): portable fallback.
    let _ = core::marker::PhantomData::<M>;
    unrolled::matmul(out, x, w, bias, n, d_in, d_out)
}

/// Output columns `j0..j1` of the matmul (all rows).
///
/// # Safety
/// `out` points to `n × d_out` floats, shapes match, `d_in % 8 == 0`, and no
/// one else writes columns `j0..j1` concurrently.
#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn cols<M: Madd>(out: *mut f32, x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize, j0: usize, j1: usize) {
    let mut i = 0;
    while i + 4 <= n {
        tile4x4::<M>(out, x, w, bias, i, d_in, d_out, j0, j1);
        i += 4;
    }
    while i < n {
        row::<M>(out, x, w, bias, i, d_in, d_out, j0, j1);
        i += 1;
    }
}

/// `Simd<M>` with every matmul split over output columns across a rayon pool
/// of Web Workers (wasm-bindgen-rayon, shared wasm memory). Only in the
/// threads build. Core's own `threads` stage can't run in a browser: its
/// later siblings start OS threads, which wasm doesn't have, so the browser
/// gets its own threaded rung instead.
#[cfg(feature = "threads")]
pub struct Par<M: Madd>(core::marker::PhantomData<M>);

#[cfg(feature = "threads")]
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
#[cfg(feature = "threads")]
unsafe impl Send for OutPtr {}
#[cfg(feature = "threads")]
unsafe impl Sync for OutPtr {}

#[cfg(feature = "threads")]
impl<M: Madd> Kernels for Par<M> {
    const NAME: &'static str = "wasm-threads";
    fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
        use rayon::prelude::*;
        assert!(x.len() >= n * d_in && out.len() >= n * d_out && w.len() == d_in * d_out && d_in % 8 == 0);
        let threads = rayon::current_num_threads();
        // 4 chunks per thread so fast cores can steal from slow ones;
        // multiples of 16 columns.
        let width = d_out.div_ceil(threads * 4).next_multiple_of(16);
        let ptr = OutPtr(out.as_mut_ptr());
        (0..d_out.div_ceil(width)).into_par_iter().for_each(|c| {
            let p = ptr;
            let (b, e) = (c * width, ((c + 1) * width).min(d_out));
            // Safety: chunks cover disjoint column ranges.
            unsafe { cols::<M>(p.0, x, w, bias, n, d_in, d_out, b, e) }
        });
    }
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn hsum(v: v128) -> f32 {
    // (a+c, b+d, ..) then lane0 + lane1
    let s = f32x4_add(v, i32x4_shuffle::<2, 3, 0, 1>(v, v));
    f32x4_extract_lane::<0>(s) + f32x4_extract_lane::<1>(s)
}

/// Decode: one activation row against the weights, one weight row at a time
/// with four accumulators (16 floats per step). Walking a single row is one
/// sequential stream for the prefetcher; an earlier version interleaved four
/// rows to reuse each `x` load and was ~25% slower (see the LOG).
#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn row<M: Madd>(out: *mut f32, x: &[f32], w: &[f32], bias: Option<&[f32]>, i: usize, d_in: usize, d_out: usize, j0: usize, j1: usize) {
    let xp = x.as_ptr().add(i * d_in);
    let op = out.add(i * d_out);
    let main = d_in - d_in % 16;
    for j in j0..j1 {
        let wr = w.as_ptr().add(j * d_in);
        let z = f32x4_splat(0.0);
        let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
        let mut k = 0;
        while k < main {
            a0 = M::madd(v128_load(wr.add(k) as *const v128), v128_load(xp.add(k) as *const v128), a0);
            a1 = M::madd(v128_load(wr.add(k + 4) as *const v128), v128_load(xp.add(k + 4) as *const v128), a1);
            a2 = M::madd(v128_load(wr.add(k + 8) as *const v128), v128_load(xp.add(k + 8) as *const v128), a2);
            a3 = M::madd(v128_load(wr.add(k + 12) as *const v128), v128_load(xp.add(k + 12) as *const v128), a3);
            k += 16;
        }
        while k < d_in {
            a0 = M::madd(v128_load(wr.add(k) as *const v128), v128_load(xp.add(k) as *const v128), a0);
            k += 4;
        }
        *op.add(j) = hsum(f32x4_add(f32x4_add(a0, a1), f32x4_add(a2, a3))) + bias.map_or(0.0, |b| b[j]);
    }
}

/// Prefill: a 4×4 register tile. Four activation rows × four weight rows = 16
/// accumulators; per step 8 vector loads feed 16 multiply-adds, so every weight
/// vector pulled from memory is used four times.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn tile4x4<M: Madd>(out: *mut f32, x: &[f32], w: &[f32], bias: Option<&[f32]>, i: usize, d_in: usize, d_out: usize, j0: usize, j1: usize) {
    let x0 = x.as_ptr().add(i * d_in);
    let (x1, x2, x3) = (x0.add(d_in), x0.add(2 * d_in), x0.add(3 * d_in));
    let op = out;
    let mut j = j0;
    while j + 4 <= j1 {
        let w0 = w.as_ptr().add(j * d_in);
        let (w1, w2, w3) = (w0.add(d_in), w0.add(2 * d_in), w0.add(3 * d_in));
        let mut acc = [f32x4_splat(0.0); 16];
        let mut k = 0;
        while k < d_in {
            let xs = [
                v128_load(x0.add(k) as *const v128),
                v128_load(x1.add(k) as *const v128),
                v128_load(x2.add(k) as *const v128),
                v128_load(x3.add(k) as *const v128),
            ];
            let ws = [
                v128_load(w0.add(k) as *const v128),
                v128_load(w1.add(k) as *const v128),
                v128_load(w2.add(k) as *const v128),
                v128_load(w3.add(k) as *const v128),
            ];
            for r in 0..4 {
                for c in 0..4 {
                    acc[r * 4 + c] = M::madd(xs[r], ws[c], acc[r * 4 + c]);
                }
            }
            k += 4;
        }
        for r in 0..4 {
            for c in 0..4 {
                *op.add((i + r) * d_out + j + c) = hsum(acc[r * 4 + c]) + bias.map_or(0.0, |b| b[j + c]);
            }
        }
        j += 4;
    }
    for r in 0..4 {
        let xr = x0.add(r * d_in);
        let mut jj = j;
        while jj < j1 {
            let wr = w.as_ptr().add(jj * d_in);
            let mut a = f32x4_splat(0.0);
            let mut k = 0;
            while k < d_in {
                a = M::madd(v128_load(wr.add(k) as *const v128), v128_load(xr.add(k) as *const v128), a);
                k += 4;
            }
            *op.add((i + r) * d_out + jj) = hsum(a) + bias.map_or(0.0, |b| b[jj]);
            jj += 1;
        }
    }
}
