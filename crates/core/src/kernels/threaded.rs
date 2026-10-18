//! Stage 5 kernels: split every matmul across CPU cores.
//!
//! One core can stream ~57 GB/s; all ten together ~98 GB/s (`nanoinfer membw`).
//! Decode is memory-bound, so more cores = more bandwidth = more tok/s, up to the
//! point where the memory system, not the cores, is saturated.
//!
//! Each matmul's output columns (= weight rows) are cut into chunks, and rayon's
//! work-stealing pool hands chunks to threads. A decode chunk is a contiguous
//! slab of weight rows, i.e. one sequential stream per thread (stage 3 taught
//! us that's what the prefetcher likes). Prefill chunks run stage 4's 4×4 tile
//! kernel. We make several chunks per thread so the faster P-cores can steal work
//! from the slower E-cores instead of waiting for them at the join.
//!
//! Threads exist only with the `threads` cargo feature (the CLI enables it).
//! Without it, e.g. in the single-threaded wasm build, this is stage 4.

use super::tiled::{self, OutPtr};

/// Chunks per thread. More chunks = better balance between P and E cores, but
/// more scheduling overhead per matmul. Tuned in the LOG (stage 5).
pub const CHUNKS_PER_THREAD: usize = 4;

/// Threads the kernels will use (1 without the `threads` feature).
pub fn num_threads() -> usize {
    #[cfg(feature = "threads")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "threads"))]
    {
        1
    }
}

/// Columns `b..e` of `out[n, d_out]`: decode (n < 4) walks rows one at a time,
/// prefill uses the 4×4 tiles.
///
/// # Safety
/// `out` points to `n × d_out` floats and no one else writes columns `b..e`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn cols(
    out: OutPtr,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    n: usize,
    d_in: usize,
    d_out: usize,
    b: usize,
    e: usize,
) {
    if n < tiled::MR {
        for i in 0..n {
            let y = std::slice::from_raw_parts_mut(out.0.add(i * d_out + b), e - b);
            let x_row = &x[i * d_in..(i + 1) * d_in];
            #[cfg(target_arch = "aarch64")]
            super::neon::matvec(y, x_row, w, bias, d_in, d_out, b);
            #[cfg(not(target_arch = "aarch64"))]
            for (j, v) in y.iter_mut().enumerate() {
                let o = b + j;
                *v = super::unrolled::dot(x_row, &w[o * d_in..(o + 1) * d_in]) + bias.map_or(0.0, |bb| bb[o]);
            }
        }
    } else {
        tiled::matmul_cols(out, x, w, bias, n, d_in, d_out, b, e);
    }
}

/// Same contract as [`super::reference::matmul`].
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
    assert!(x.len() >= n * d_in && out.len() >= n * d_out && w.len() == d_in * d_out);
    let threads = num_threads();
    let ptr = OutPtr(out.as_mut_ptr());
    if threads == 1 {
        // Safety: we're the only writer of all columns.
        unsafe { cols(ptr, x, w, bias, n, d_in, d_out, 0, d_out) };
        return;
    }
    // Chunk width: a multiple of 16 columns (whole 4×4 tiles, whole cache lines of bias).
    let width = d_out.div_ceil(threads * CHUNKS_PER_THREAD).next_multiple_of(16);
    let n_chunks = d_out.div_ceil(width);
    let run = |c: usize| {
        let (b, e) = (c * width, ((c + 1) * width).min(d_out));
        // Safety: chunks cover disjoint column ranges of `out`.
        unsafe { cols(ptr, x, w, bias, n, d_in, d_out, b, e) }
    };
    #[cfg(feature = "threads")]
    {
        use rayon::prelude::*;
        (0..n_chunks).into_par_iter().for_each(run);
    }
    #[cfg(not(feature = "threads"))]
    (0..n_chunks).for_each(run);
}
