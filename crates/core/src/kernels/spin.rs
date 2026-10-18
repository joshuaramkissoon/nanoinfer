//! Stage 6 kernels: stage 5's split, dispatched on the spinning `crate::pool`
//! instead of rayon. Same chunking, same per-chunk kernels; only the cost of
//! starting and joining the parallel section changes.
//!
//! On wasm (no `std::thread::spawn`) and without the `threads` feature this is
//! exactly stage 5 (rayon if present, otherwise one thread).

#[cfg(all(feature = "threads", not(target_family = "wasm")))]
use super::tiled::OutPtr;

/// Chunks per thread (dynamic claiming balances P- vs E-cores).
pub const CHUNKS_PER_THREAD: usize = 4;

/// Threads the spin pool runs (1 without the `threads` feature).
pub fn num_threads() -> usize {
    #[cfg(all(feature = "threads", not(target_family = "wasm")))]
    {
        crate::pool::global().threads()
    }
    #[cfg(not(all(feature = "threads", not(target_family = "wasm"))))]
    {
        super::threaded::num_threads()
    }
}

/// Same contract as [`super::reference::matmul`].
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
    #[cfg(all(feature = "threads", not(target_family = "wasm")))]
    {
        assert!(x.len() >= n * d_in && out.len() >= n * d_out && w.len() == d_in * d_out);
        let pool = crate::pool::global();
        let width = d_out.div_ceil(pool.threads() * CHUNKS_PER_THREAD).next_multiple_of(16);
        let n_chunks = d_out.div_ceil(width);
        let ptr = OutPtr(out.as_mut_ptr());
        pool.run(n_chunks, &|c| {
            let (b, e) = (c * width, ((c + 1) * width).min(d_out));
            // Safety: chunks cover disjoint column ranges of `out`.
            unsafe { super::threaded::cols(ptr, x, w, bias, n, d_in, d_out, b, e) }
        });
    }
    #[cfg(not(all(feature = "threads", not(target_family = "wasm"))))]
    super::threaded::matmul(out, x, w, bias, n, d_in, d_out)
}

/// Run `f(0..n)` in parallel: on the spin pool natively, on rayon in a wasm
/// threads build, serially otherwise. For ops other than matmul (stage 7+).
pub fn par_for(n: usize, f: &(dyn Fn(usize) + Sync)) {
    #[cfg(all(feature = "threads", not(target_family = "wasm")))]
    crate::pool::global().run(n, f);
    #[cfg(all(feature = "threads", target_family = "wasm"))]
    {
        use rayon::prelude::*;
        (0..n).into_par_iter().for_each(f);
    }
    #[cfg(not(feature = "threads"))]
    (0..n).for_each(f);
}
