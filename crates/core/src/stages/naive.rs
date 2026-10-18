//! Stage 0: the naive forward pass.
//!
//! The simplest correct thing: every time we want one more token, run the
//! *entire* sequence so far through the network from scratch. Nothing is
//! remembered between calls.
//!
//! It's wasteful in an instructive way. A transformer's output at position `t`
//! only depends on tokens `0..=t`, so when we append token `t+1`, every
//! activation for positions `0..=t` comes out exactly the same as last time. We
//! recompute them anyway. Generating `N` tokens after a `P`-token prompt pushes
//! `(P+1) + (P+2) + ... + (P+N)` rows through every matmul: quadratic in length.
//! Stage 1 (KV cache) is the observation that only K and V of old positions are
//! ever needed again.

use super::{blocks, Activations};
use crate::kernels::{Kernels, Rows};
use crate::trace::Trace;
use crate::weights::Model;

/// Run all of `tokens` through GPT-2 and leave the next-token logits in `acts.logits`.
pub fn forward<K: Kernels, T: Trace>(model: &Model, tokens: &[u32], acts: &mut Activations, trace: &mut T) {
    let cfg = &model.config;
    let n = tokens.len();
    let c = cfg.n_embd;
    acts.ensure(model, n);

    // x = token embedding + position embedding, for every position.
    blocks::embed::<K, T>(model, tokens, 0, acts, trace);

    for (l, layer) in model.layers.iter().enumerate() {
        // Q, K, V for every position, packed per row as [q | k | v].
        blocks::qkv::<K, T>(cfg, l, layer, n, acts, trace);

        // Keys and values come straight out of this call's QKV buffer: the K of
        // row j starts at offset C, the V at 2C, and rows are 3C apart.
        let qkv = &acts.qkv[..n * 3 * c];
        let k = Rows { data: &qkv[c..], stride: 3 * c };
        let v = Rows { data: &qkv[2 * c..], stride: 3 * c };
        blocks::attention::<K, T>(cfg, l, n, 0, k, v, &mut acts.att, qkv, &mut acts.scores, trace);

        blocks::proj_and_mlp::<K, T>(cfg, l, layer, n, acts, trace);
    }

    blocks::lm_head::<K, T>(model, n, acts, trace);
}
