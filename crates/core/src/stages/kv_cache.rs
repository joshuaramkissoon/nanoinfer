//! Stage 1: the KV cache.
//!
//! In the naive stage, appending one token recomputes every old position. But
//! look at what old positions are actually *used* for: in attention, the new
//! token's query is compared against every old **key**, and mixes every old
//! **value**. Nothing else about the past is needed. And because attention is
//! causal, an old position's K and V never change once computed.
//!
//! So: keep each layer's K and V rows in a cache. Each call now pushes only the
//! *new* tokens through the network (one row while decoding), appends their K/V
//! to the cache, and attends over the cache. Per-token work drops from
//! O(context) matmul rows to exactly 1; only attention still grows with length.
//!
//! The price is memory: 2 x n_layer x n_ctx x C floats (72 MiB for GPT-2 small at
//! full context). For big models with long contexts the KV cache, not the
//! weights, becomes the thing that limits batch size. That's a later chapter.

use super::{blocks, Activations, KvCache};
use crate::kernels::{Kernels, Rows};
use crate::trace::{traced, Op, OpKind, Trace};
use crate::weights::Model;

/// Process `new_tokens` (appended at position `kv.len`) and leave the
/// next-token logits in `acts.logits`.
pub fn forward<K: Kernels, T: Trace>(
    model: &Model,
    new_tokens: &[u32],
    kv: &mut KvCache,
    acts: &mut Activations,
    trace: &mut T,
) {
    let cfg = &model.config;
    let n = new_tokens.len();
    let c = cfg.n_embd;
    let pos0 = kv.len;
    debug_assert!(pos0 + n <= kv.capacity);
    acts.ensure(model, n);

    blocks::embed::<K, T>(model, new_tokens, pos0, acts, trace);

    for (l, layer) in model.layers.iter().enumerate() {
        blocks::qkv::<K, T>(cfg, l, layer, n, acts, trace);

        // Append this call's K and V rows to layer l's cache at positions pos0..pos0+n.
        let layer_off = l * kv.capacity * c;
        traced(
            trace,
            || Op::elementwise(OpKind::KvWrite, "kv_write", Some(l), n, 2 * c, 0, 8),
            || {
                for i in 0..n {
                    let row = &acts.qkv[i * 3 * c..][..3 * c];
                    let dst = layer_off + (pos0 + i) * c;
                    kv.k[dst..dst + c].copy_from_slice(&row[c..2 * c]);
                    kv.v[dst..dst + c].copy_from_slice(&row[2 * c..]);
                }
            },
        );

        // Attend over everything cached so far (0..pos0+n). Only Q comes from
        // this call's QKV buffer.
        let k = Rows { data: &kv.k[layer_off..layer_off + kv.capacity * c], stride: c };
        let v = Rows { data: &kv.v[layer_off..layer_off + kv.capacity * c], stride: c };
        let qkv = &acts.qkv[..n * 3 * c];
        blocks::attention::<K, T>(cfg, l, n, pos0, k, v, &mut acts.att, qkv, &mut acts.scores, trace);

        blocks::proj_and_mlp::<K, T>(cfg, l, layer, n, acts, trace);
    }
    kv.len += n;
    if T::ENABLED {
        trace.kv_cache(kv.len, kv.capacity);
    }

    blocks::lm_head::<K, T>(model, n, acts, trace);
}
