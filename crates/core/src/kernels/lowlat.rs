//! Stage 11 kernels: the same math as stages 7–9, dispatched on the
//! lower-latency [`crate::fastpool`] (one `fetch_add` per chunk claim,
//! per-thread completion counts), with fewer chunks per thread, and attention
//! parallel over heads even at short context.
//!
//! By stage 10 a decode token was ~4 ms of which the weight bytes needed ~1.5 ms.
//! The rest was fixed cost: ~49 parallel dispatches per token at up to ~8 µs
//! each before load imbalance (an empty 40-chunk dispatch on stage 6's pool), and
//! single-threaded attention (~0.5 ms/token at short context, under stage 7's
//! parallel threshold). This set attacks only that fixed cost.
//!
//! Without the `threads` feature, or on wasm, it falls back to stage 7/8 behaviour.

use super::packed::Par;
use super::tiled::OutPtr;
use super::{fastops, Rows};

/// Chunks per thread. Tried 1, 2, 3, 4, 6 for decode at 6 and 10 threads (LOG
/// stage 11): fewer, bigger chunks make the E-core stragglers worse than the
/// cheaper claims save, so it stays at stage 6's 4.
pub const CHUNKS_PER_THREAD: usize = 4;

/// Threads in the fast pool (or whatever stage 6/7 would use where it doesn't exist).
pub fn num_threads() -> usize {
    #[cfg(all(feature = "threads", not(target_family = "wasm")))]
    {
        crate::fastpool::global().threads()
    }
    #[cfg(not(all(feature = "threads", not(target_family = "wasm"))))]
    {
        super::spin::num_threads()
    }
}

pub fn par_for(n: usize, f: &(dyn Fn(usize) + Sync)) {
    #[cfg(all(feature = "threads", not(target_family = "wasm")))]
    crate::fastpool::global().run(n, f);
    #[cfg(not(all(feature = "threads", not(target_family = "wasm"))))]
    super::spin::par_for(n, f);
}

/// The fast pool as a [`Par`] runtime for packed matrices.
pub struct FastPar;

impl Par for FastPar {
    const CHUNKS_PER_THREAD: usize = CHUNKS_PER_THREAD;
    fn threads() -> usize {
        num_threads()
    }
    fn par_for(n: usize, f: &(dyn Fn(usize) + Sync)) {
        par_for(n, f)
    }
}

/// GELU, parallel above 16K elements (prefill), vectorised serial otherwise.
pub fn gelu(x: &mut [f32]) {
    let n = x.len();
    let threads = num_threads();
    if n < 1 << 14 || threads == 1 {
        return fastops::gelu_serial(x);
    }
    let chunk = n.div_ceil(threads * CHUNKS_PER_THREAD).next_multiple_of(64);
    let ptr = OutPtr(x.as_mut_ptr());
    par_for(n.div_ceil(chunk), &|c| {
        let (b, e) = (c * chunk, ((c + 1) * chunk).min(n));
        let p = ptr;
        // Safety: disjoint ranges of `x`.
        fastops::gelu_serial(unsafe { std::slice::from_raw_parts_mut(p.0.add(b), e - b) });
    });
}

/// Attention split over (query, head) units whenever there's more than one
/// thread: at decode that's 12 units of a few µs each, now worth a ~1 µs dispatch.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    out: &mut [f32],
    q: Rows,
    k: Rows,
    v: Rows,
    n_q: usize,
    q_pos0: usize,
    n_head: usize,
    head_dim: usize,
    scores: &mut [f32],
) {
    let c = n_head * head_dim;
    let units = n_q * n_head;
    let n_kv_max = q_pos0 + n_q;
    let ptr = OutPtr(out.as_mut_ptr());
    let unit = |u: usize, scratch: &mut [f32]| {
        let (i, h) = (u / n_head, u % n_head);
        let p = ptr;
        // Safety: each unit writes its own disjoint head_dim slice of `out`.
        let o = unsafe { std::slice::from_raw_parts_mut(p.0.add(i * c + h * head_dim), head_dim) };
        fastops::attend_one(o, q.head(i, h, head_dim), k, v, h, head_dim, q_pos0 + i + 1, scratch);
    };
    let threads = num_threads();
    if threads == 1 || units == 1 {
        return (0..units).for_each(|u| unit(u, scores));
    }
    let per = units.div_ceil(threads * CHUNKS_PER_THREAD).max(1);
    par_for(units.div_ceil(per), &|ch| {
        let mut stack = [0f32; 1024];
        let mut heap = Vec::new();
        let scratch: &mut [f32] = if n_kv_max <= stack.len() {
            &mut stack
        } else {
            heap.resize(n_kv_max, 0.0);
            &mut heap
        };
        for u in ch * per..((ch + 1) * per).min(units) {
            unit(u, scratch);
        }
    });
}
