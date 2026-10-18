//! Building blocks shared by every stage: the pieces of a GPT-2 forward pass
//! that do not depend on *where* keys and values come from.
//!
//! A GPT-2 block is:
//! ```text
//! x = x + attn_proj(attention(qkv(ln_1(x))))
//! x = x + fc_proj(gelu(fc(ln_2(x))))
//! ```
//! Stages differ only in how many rows they push through this and how attention
//! finds its keys/values, so those two halves live here, with tracing built in.

use super::Activations;
use crate::config::Config;
use crate::kernels::{reference, Kernels, Rows};
use crate::trace::{traced, Op, OpKind, Trace};
use crate::weights::{Layer, Model};

/// `acts.x[..n] = wte[tokens] + wpe[pos0..]`.
pub fn embed<K: Kernels, T: Trace>(model: &Model, tokens: &[u32], pos0: usize, acts: &mut Activations, trace: &mut T) {
    let c = model.config.n_embd;
    let n = tokens.len();
    traced(
        trace,
        || Op::elementwise(OpKind::Embed, "embed", None, n, c, 1, 12),
        || K::embed(&mut acts.x, &model.wte, &model.wpe, tokens, pos0, c),
    );
}

/// First half of a block, up to (not including) attention:
/// `acts.qkv[..n] = ln_1(acts.x[..n]) · W_qkvᵀ + b`.
pub fn qkv<K: Kernels, T: Trace>(
    cfg: &Config,
    l: usize,
    layer: &Layer,
    n: usize,
    acts: &mut Activations,
    trace: &mut T,
) {
    let c = cfg.n_embd;
    traced(
        trace,
        || Op::elementwise(OpKind::LayerNorm, "ln_1", Some(l), n, c, 8, 8),
        || K::layernorm(&mut acts.xn, &acts.x, &layer.ln_1_w, &layer.ln_1_b, n, c, Config::LN_EPS),
    );
    traced(
        trace,
        || Op::matmul("qkv", Some(l), n, c, 3 * c),
        || K::matmul(&mut acts.qkv, &acts.xn, &layer.attn_w, Some(&layer.attn_b), n, c, 3 * c),
    );
}

/// Causal attention for the `n` query rows in `acts.qkv`, starting at absolute
/// position `pos0`, reading keys/values from `k`/`v`. Writes `acts.att`.
///
/// `k`/`v` are passed separately (not taken from `acts`) because that's the one
/// thing stages disagree about.
#[allow(clippy::too_many_arguments)]
pub fn attention<K: Kernels, T: Trace>(
    cfg: &Config,
    l: usize,
    n: usize,
    pos0: usize,
    k: Rows,
    v: Rows,
    att: &mut [f32],
    qkv: &[f32],
    scores: &mut [f32],
    trace: &mut T,
) {
    let c = cfg.n_embd;
    let hd = cfg.head_dim();
    let q = Rows { data: qkv, stride: 3 * c };
    traced(
        trace,
        || Op::attention(l, n, pos0, c, cfg.n_head),
        || K::attention(att, q, k, v, n, pos0, cfg.n_head, hd, scores),
    );
    // Optional: hand the attention map of the newest token to the trace.
    if T::ENABLED {
        if let Some((tl, th)) = trace.wants_attention() {
            if tl == l && th < cfg.n_head {
                let last = n - 1;
                let n_kv = pos0 + n;
                let q_head = &qkv[last * 3 * c + th * hd..][..hd];
                reference::attention_probs(scores, q_head, k, th, hd, n_kv);
                trace.attention(l, th, n_kv - 1, &scores[..n_kv]);
            }
        }
    }
}

/// Second half of a block: output projection, residual, MLP, residual.
pub fn proj_and_mlp<K: Kernels, T: Trace>(
    cfg: &Config,
    l: usize,
    layer: &Layer,
    n: usize,
    acts: &mut Activations,
    trace: &mut T,
) {
    let c = cfg.n_embd;
    let rc = n * c;
    traced(
        trace,
        || Op::matmul("attn_proj", Some(l), n, c, c),
        || K::matmul(&mut acts.proj, &acts.att, &layer.proj_w, Some(&layer.proj_b), n, c, c),
    );
    traced(
        trace,
        || Op::elementwise(OpKind::Residual, "residual_1", Some(l), n, c, 1, 12),
        || K::residual_add(&mut acts.x[..rc], &acts.proj[..rc]),
    );
    traced(
        trace,
        || Op::elementwise(OpKind::LayerNorm, "ln_2", Some(l), n, c, 8, 8),
        || K::layernorm(&mut acts.xn, &acts.x, &layer.ln_2_w, &layer.ln_2_b, n, c, Config::LN_EPS),
    );
    traced(
        trace,
        || Op::matmul("fc", Some(l), n, c, 4 * c),
        || K::matmul(&mut acts.fc, &acts.xn, &layer.fc_w, Some(&layer.fc_b), n, c, 4 * c),
    );
    traced(
        trace,
        || Op::elementwise(OpKind::Gelu, "gelu", Some(l), n, 4 * c, 15, 8),
        || K::gelu(&mut acts.fc[..4 * rc]),
    );
    traced(
        trace,
        || Op::matmul("fc_proj", Some(l), n, 4 * c, c),
        || K::matmul(&mut acts.proj, &acts.fc, &layer.fc_proj_w, Some(&layer.fc_proj_b), n, 4 * c, c),
    );
    traced(
        trace,
        || Op::elementwise(OpKind::Residual, "residual_2", Some(l), n, c, 1, 12),
        || K::residual_add(&mut acts.x[..rc], &acts.proj[..rc]),
    );
}

/// Final LayerNorm + LM head for the *last* row only: we only need the next
/// token's distribution, so there's no point computing logits for every
/// position (a training engine would). Writes `acts.logits`.
pub fn lm_head<K: Kernels, T: Trace>(model: &Model, n: usize, acts: &mut Activations, trace: &mut T) {
    let cfg = &model.config;
    let c = cfg.n_embd;
    let last = &acts.x[(n - 1) * c..n * c];
    traced(
        trace,
        || Op::elementwise(OpKind::LayerNorm, "ln_f", None, 1, c, 8, 8),
        || K::layernorm(&mut acts.xn[..c], last, &model.ln_f_w, &model.ln_f_b, 1, c, Config::LN_EPS),
    );
    // GPT-2 ties the LM head to the token embedding: logits = x · wteᵀ.
    traced(
        trace,
        || Op { kind: OpKind::LmHead, ..Op::matmul("lm_head", None, 1, c, cfg.n_vocab) },
        || K::matmul(&mut acts.logits, &acts.xn[..c], &model.wte, None, 1, c, cfg.n_vocab),
    );
}
