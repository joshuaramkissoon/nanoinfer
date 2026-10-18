//! Stages 8+: the KV-cache forward with weight matrices in a smaller format.
//!
//! Same computation as [`super::kv_cache`]: the same block structure, cache and
//! non-matmul kernels (`K`, normally stage 7's [`crate::kernels::FastOps`]).
//! The only difference is that the five matmuls per block and the LM head
//! multiply by a [`PackedMatrix`] (f16, Q8_0, Q4_0) instead of the f32 weights in
//! [`Model`]. The packed copy is built once, when the [`super::Session`] is created.
//!
//! LayerNorm parameters, biases and the position table stay f32 (<0.1% of the
//! bytes). The token *embedding* lookup also keeps using f32 `wte` (one row per
//! token). Only its use as the tied LM head, all 50257 rows per token, reads the
//! packed copy.

use super::{Activations, KvCache};
use crate::config::Config;
use crate::kernels::packed::{PackedMatrix, Par, Scratch};
use crate::kernels::{Kernels, Rows};
use crate::trace::{traced, Op, OpKind, Trace};
use crate::weights::Model;

/// One block's matrices in format `M`.
pub struct PackedLayer<M> {
    pub attn: M,
    pub proj: M,
    pub fc: M,
    pub fc_proj: M,
}

/// How the final LayerNorm + tied LM head are computed. The LM head is 31% of
/// GPT-2 small's bytes *and* the most quantisation-sensitive matrix, so stages
/// choose its format (and trick) separately from the blocks.
pub trait Head: Send + Sync + Sized {
    fn build(model: &Model) -> Self;
    fn bytes(&self) -> usize;
    /// `logits = head(ln_f(x))` for one residual row `x`. `xn` is `[C]` scratch.
    fn logits<K: Kernels, P: Par>(
        &self,
        model: &Model,
        x: &[f32],
        xn: &mut [f32],
        logits: &mut [f32],
        scratch: &mut Scratch,
    );
}

/// `logits = W · ln_f(x)` with `W = wte` stored as `M`.
pub struct PlainHead<M>(pub M);

impl<M: PackedMatrix> Head for PlainHead<M> {
    fn build(model: &Model) -> Self {
        PlainHead(M::pack(&model.wte, model.config.n_vocab, model.config.n_embd))
    }
    fn bytes(&self) -> usize {
        self.0.bytes()
    }
    fn logits<K: Kernels, P: Par>(
        &self,
        model: &Model,
        x: &[f32],
        xn: &mut [f32],
        logits: &mut [f32],
        scratch: &mut Scratch,
    ) {
        let c = model.config.n_embd;
        K::layernorm(xn, x, &model.ln_f_w, &model.ln_f_b, 1, c, Config::LN_EPS);
        self.0.matmul::<P>(logits, xn, None, 1, scratch);
    }
}

/// The LayerNorm gain folded into the head: with `n = (x − mean)/std`,
/// `W · (g ⊙ n + b) = (W·diag(g)) · n + W·b`. We store `W' = W·diag(g)` as `M` and
/// the per-token constant `W·b` in f32, and quantise `n`, which has no gain
/// outliers (GPT-2's ln_f gain reaches 17× its rms on a few channels). Exact in
/// f32; it only changes *what* gets rounded.
pub struct FoldedHead<M> {
    pub w: M,
    pub bias: Vec<f32>,
    /// Unit gain / zero shift: LayerNorm reduced to plain normalisation.
    ones: Vec<f32>,
    zeros: Vec<f32>,
}

impl<M: PackedMatrix> Head for FoldedHead<M> {
    fn build(model: &Model) -> Self {
        let (v, c) = (model.config.n_vocab, model.config.n_embd);
        let mut w = model.wte.clone();
        let mut bias = vec![0f32; v];
        for (row, b) in w.chunks_exact_mut(c).zip(&mut bias) {
            *b = row.iter().zip(&model.ln_f_b).map(|(a, bb)| (*a as f64) * (*bb as f64)).sum::<f64>() as f32;
            for (a, g) in row.iter_mut().zip(&model.ln_f_w) {
                *a *= g;
            }
        }
        FoldedHead { w: M::pack(&w, v, c), bias, ones: vec![1.0; c], zeros: vec![0.0; c] }
    }
    fn bytes(&self) -> usize {
        self.w.bytes() + self.bias.len() * 4
    }
    fn logits<K: Kernels, P: Par>(
        &self,
        model: &Model,
        x: &[f32],
        xn: &mut [f32],
        logits: &mut [f32],
        scratch: &mut Scratch,
    ) {
        let c = model.config.n_embd;
        K::layernorm(xn, x, &self.ones, &self.zeros, 1, c, Config::LN_EPS);
        self.w.matmul::<P>(logits, xn, Some(&self.bias), 1, scratch);
    }
}

/// Every matrix the decode loop streams: blocks in format `M`, LM head as `H`.
pub struct PackedModel<M, H = PlainHead<M>> {
    pub layers: Vec<PackedLayer<M>>,
    pub head: H,
}

impl<M: PackedMatrix, H: Head> PackedModel<M, H> {
    pub fn new(model: &Model) -> Self {
        let c = model.config.n_embd;
        let layers = model
            .layers
            .iter()
            .map(|l| PackedLayer {
                attn: M::pack(&l.attn_w, 3 * c, c),
                proj: M::pack(&l.proj_w, c, c),
                fc: M::pack(&l.fc_w, 4 * c, c),
                fc_proj: M::pack(&l.fc_proj_w, c, 4 * c),
            })
            .collect();
        PackedModel { layers, head: H::build(model) }
    }

    /// Bytes of all packed matrices.
    pub fn bytes(&self) -> usize {
        self.head.bytes()
            + self
                .layers
                .iter()
                .map(|l| l.attn.bytes() + l.proj.bytes() + l.fc.bytes() + l.fc_proj.bytes())
                .sum::<usize>()
    }
}

/// A matmul op descriptor with the packed matrix's real byte count.
fn op<M: PackedMatrix>(name: &'static str, layer: Option<usize>, n: usize, m: &M) -> Op {
    let base = Op::matmul(name, layer, n, m.d_in(), m.d_out());
    let f32_weights = 4 * (m.d_in() * m.d_out()) as u64;
    Op { bytes: base.bytes - f32_weights + m.bytes() as u64, ..base }
}

/// Process `new_tokens` (appended at `kv.len`), leaving next-token logits in
/// `acts.logits`. `scratch` is per-session space for the matmul kernels.
#[allow(clippy::too_many_arguments)]
pub fn forward<K: Kernels, M: PackedMatrix, H: Head, P: Par, T: Trace>(
    model: &Model,
    packed: &PackedModel<M, H>,
    new_tokens: &[u32],
    kv: &mut KvCache,
    acts: &mut Activations,
    scratch: &mut Scratch,
    trace: &mut T,
) {
    let cfg = &model.config;
    let n = new_tokens.len();
    let c = cfg.n_embd;
    let rc = n * c;
    let pos0 = kv.len;
    debug_assert!(pos0 + n <= kv.capacity);
    acts.ensure(model, n);

    super::blocks::embed::<K, T>(model, new_tokens, pos0, acts, trace);

    for (l, (layer, pl)) in model.layers.iter().zip(&packed.layers).enumerate() {
        traced(
            trace,
            || Op::elementwise(OpKind::LayerNorm, "ln_1", Some(l), n, c, 8, 8),
            || K::layernorm(&mut acts.xn, &acts.x, &layer.ln_1_w, &layer.ln_1_b, n, c, Config::LN_EPS),
        );
        traced(
            trace,
            || op("qkv", Some(l), n, &pl.attn),
            || pl.attn.matmul::<P>(&mut acts.qkv, &acts.xn, Some(&layer.attn_b), n, scratch),
        );

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
        let k = Rows { data: &kv.k[layer_off..layer_off + kv.capacity * c], stride: c };
        let v = Rows { data: &kv.v[layer_off..layer_off + kv.capacity * c], stride: c };
        let qkv = &acts.qkv[..n * 3 * c];
        super::blocks::attention::<K, T>(cfg, l, n, pos0, k, v, &mut acts.att, qkv, &mut acts.scores, trace);

        traced(
            trace,
            || op("attn_proj", Some(l), n, &pl.proj),
            || pl.proj.matmul::<P>(&mut acts.proj, &acts.att, Some(&layer.proj_b), n, scratch),
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
            || op("fc", Some(l), n, &pl.fc),
            || pl.fc.matmul::<P>(&mut acts.fc, &acts.xn, Some(&layer.fc_b), n, scratch),
        );
        traced(
            trace,
            || Op::elementwise(OpKind::Gelu, "gelu", Some(l), n, 4 * c, 15, 8),
            || K::gelu(&mut acts.fc[..4 * rc]),
        );
        traced(
            trace,
            || op("fc_proj", Some(l), n, &pl.fc_proj),
            || pl.fc_proj.matmul::<P>(&mut acts.proj, &acts.fc, Some(&layer.fc_proj_b), n, scratch),
        );
        traced(
            trace,
            || Op::elementwise(OpKind::Residual, "residual_2", Some(l), n, c, 1, 12),
            || K::residual_add(&mut acts.x[..rc], &acts.proj[..rc]),
        );
    }
    kv.len += n;
    if T::ENABLED {
        trace.kv_cache(kv.len, kv.capacity);
    }

    let last = &acts.x[(n - 1) * c..n * c];
    let v = cfg.n_vocab as u64;
    traced(
        trace,
        || Op {
            kind: OpKind::LmHead,
            name: "lm_head",
            layer: None,
            dims: [1, c, cfg.n_vocab],
            flops: 2 * c as u64 * v + 8 * c as u64,
            bytes: packed.head.bytes() as u64 + 4 * (c as u64 + v),
        },
        || packed.head.logits::<K, P>(model, last, &mut acts.xn[..c], &mut acts.logits, scratch),
    );
}
