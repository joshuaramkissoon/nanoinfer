//! Forward passes, one per optimisation stage, behind a single [`Session`] API.
//!
//! Each stage is a different strategy for the *same* computation, and must give
//! the same logits (tests enforce this against Hugging Face). A [`Session`] holds
//! the per-conversation state (tokens so far, KV cache, scratch buffers) and
//! dispatches to the selected stage, so the CLI and the browser can switch stages
//! at runtime and compare them.
//!
//! ## Adding a stage
//! 1. Add a variant to [`Stage`] (+ its name in `Stage::name` / `Stage::ALL`).
//! 2. If it's a new *kernel* (faster matmul, ...): add a kernel set in
//!    [`crate::kernels`] and reuse an existing forward, e.g.
//!    `kv_cache::forward::<Simd, T>`. If it's a new *strategy* (batching, a
//!    different cache layout, ...): write `stages/<name>.rs` with a `forward`
//!    built from [`blocks`]. Never edit an old stage's code path.
//! 3. Add match arms in [`Session::step`] and [`Stage::kernels`].
//! 4. The reference tests iterate `Stage::ALL`, so they cover it automatically.

pub mod blocks;
pub mod kv16;
pub mod kv_cache;
pub mod naive;
pub mod packed;

use crate::error::{Error, Result};
use crate::kernels::f16::F16Matrix;
use crate::kernels::lowlat::FastPar;
use crate::kernels::packed::SpinPar;
use crate::kernels::q4::Q4Matrix;
use crate::kernels::q8::Q8Matrix;
use crate::kernels::q8gemm::Q8GemmMatrix;
use crate::kernels::q8hilo::Q8HiLoMatrix;
use crate::kernels::{AttnNeon, FastOps, Kernels, LowLat, Neon, Reference, Spin, Threaded, Tiled, Unrolled};
use crate::trace::{ForwardInfo, Phase, Trace};
use crate::weights::Model;
use std::fmt;
use std::str::FromStr;

/// Which forward-pass implementation to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Stage 0: recompute the whole sequence for every new token. O(n²) work per token.
    Naive,
    /// Stage 1: cache each layer's keys and values; process only the new token.
    KvCache,
    /// Stage 2: KV cache + matmul with 16 independent accumulators (autovectorised).
    Unrolled,
    /// Stage 3: explicit NEON matmul (fused multiply-add, one weight row at a time).
    Neon,
    /// Stage 4: register-blocked (4×4 tile) matmul so prefill reuses each weight 4× from registers.
    Tiled,
    /// Stage 5: matmuls split across CPU cores (rayon; single-threaded without the `threads` feature).
    Threads,
    /// Stage 6: stage 5 on a persistent spin-waiting thread pool (no sleep/wake per matmul).
    SpinPool,
    /// Stage 7: stage 6 + vectorised, parallel GELU, softmax, attention and layernorm.
    FastOps,
    /// Stage 8: stage 7 with f16 weight matrices (half the bytes), f32 accumulation.
    F16,
    /// Stage 9: stage 7 with Q8_0 weights (int8 blocks of 32) and int8 sdot decode.
    Q8,
    /// Stage 10: stage 7 with Q4_0 weights (4-bit blocks of 32), unpacked to int8 for sdot.
    Q4,
    /// Stage 11: stage 9 (Q8_0, f16 head) on the low-latency pool, attention parallel at any context.
    Q8Fast,
    /// Stage 11b: stage 10 (Q4_0, f16 head) on the low-latency pool.
    Q4Fast,
    /// Stage 12: stage 11 (Q8_0) with a rescored LM head: Q8 approximate logits, f16 exact for the top candidates.
    Q8Rescore,
    /// Stage 12b: stage 11 (Q4_0) with the rescored LM head.
    Q4Rescore,
    /// Stage 13: stage 12 (Q8_0, rescored head) with register-resident NEON decode attention.
    Q8Attn,
    /// Stage 13b: stage 12b (Q4_0, rescored head) with NEON decode attention.
    Q4Attn,
    /// Stage 14: stage 13 (Q8_0, rescored head) with a head-major f16 KV cache.
    Q8Kv16,
    /// Stage 14b: stage 13b (Q4_0, rescored head) with the head-major f16 KV cache.
    Q4Kv16,
    /// Stage 15: stage 14 (Q8_0, rescored head, f16 KV) with an int8 sdot prefill tile.
    Q8Prefill,
}

impl Stage {
    pub const ALL: &'static [Stage] = &[
        Stage::Naive,
        Stage::KvCache,
        Stage::Unrolled,
        Stage::Neon,
        Stage::Tiled,
        Stage::Threads,
        Stage::SpinPool,
        Stage::FastOps,
        Stage::F16,
        Stage::Q8,
        Stage::Q4,
        Stage::Q8Fast,
        Stage::Q4Fast,
        Stage::Q8Rescore,
        Stage::Q4Rescore,
        Stage::Q8Attn,
        Stage::Q4Attn,
        Stage::Q8Kv16,
        Stage::Q4Kv16,
        Stage::Q8Prefill,
    ];

    /// Name of the kernel set this stage runs.
    pub fn kernels(self) -> &'static str {
        match self {
            Stage::Naive | Stage::KvCache => Reference::NAME,
            Stage::Unrolled => Unrolled::NAME,
            Stage::Neon => Neon::NAME,
            Stage::Tiled => Tiled::NAME,
            Stage::Threads => Threaded::NAME,
            Stage::SpinPool => Spin::NAME,
            Stage::FastOps => FastOps::NAME,
            Stage::F16 => "fast-ops + f16",
            Stage::Q8 => "fast-ops + q8_0 (f16 head)",
            Stage::Q4 => "fast-ops + q4_0 (f16 head)",
            Stage::Q8Fast => "lowlat + q8_0 (f16 head)",
            Stage::Q4Fast => "lowlat + q4_0 (f16 head)",
            Stage::Q8Rescore => "lowlat + q8_0 (rescored head)",
            Stage::Q4Rescore => "lowlat + q4_0 (rescored head)",
            Stage::Q8Attn => "attn-neon + q8_0 (rescored head)",
            Stage::Q4Attn => "attn-neon + q4_0 (rescored head)",
            Stage::Q8Kv16 => "attn-neon + q8_0 (rescored head) + f16 kv",
            Stage::Q4Kv16 => "attn-neon + q4_0 (rescored head) + f16 kv",
            Stage::Q8Prefill => "attn-neon + q8_0 sdot prefill (rescored head) + f16 kv",
        }
    }

    /// Whether this stage's kernels use more than one thread (with the
    /// `threads` feature; the thread count is the global rayon pool's).
    pub fn multithreaded(self) -> bool {
        // Stages 0–4 are single-threaded; every later stage splits work across cores.
        !matches!(self, Stage::Naive | Stage::KvCache | Stage::Unrolled | Stage::Neon | Stage::Tiled)
    }

    /// How this stage stores weight matrices (sets its decode ceiling).
    pub fn weight_format(self) -> WeightFormat {
        match self {
            Stage::F16 => WeightFormat::F16,
            Stage::Q8 => WeightFormat::Q8_0,
            Stage::Q4 => WeightFormat::Q4_0,
            Stage::Q8Fast => WeightFormat::Q8_0,
            Stage::Q4Fast => WeightFormat::Q4_0,
            Stage::Q8Rescore => WeightFormat::Q8_0,
            Stage::Q4Rescore => WeightFormat::Q4_0,
            Stage::Q8Attn => WeightFormat::Q8_0,
            Stage::Q4Attn => WeightFormat::Q4_0,
            Stage::Q8Kv16 => WeightFormat::Q8_0,
            Stage::Q4Kv16 => WeightFormat::Q4_0,
            Stage::Q8Prefill => WeightFormat::Q8_0,
            _ => WeightFormat::F32,
        }
    }

    /// Format of the tied LM head. Same as the blocks, except the integer
    /// stages, which keep it f16 (it is by far the most quantisation-sensitive matrix).
    pub fn head_format(self) -> WeightFormat {
        if matches!(
            self,
            Stage::Q8Rescore | Stage::Q4Rescore | Stage::Q8Attn | Stage::Q4Attn | Stage::Q8Kv16 | Stage::Q4Kv16
        ) {
            // Streamed bytes are the Q8 approximate head; rescoring reads ~256 f16 rows.
            return WeightFormat::Q8_0;
        }
        match self.weight_format() {
            WeightFormat::Q8_0 | WeightFormat::Q4_0 => WeightFormat::F16,
            f => f,
        }
    }

    /// Bytes this stage streams from memory per decoded token (its ceiling's
    /// denominator): block matrices at `weight_format`, LM head at `head_format`,
    /// biases / LayerNorm in f32.
    pub fn decode_bytes(self, cfg: &crate::Config) -> f64 {
        let head = (cfg.n_vocab * cfg.n_embd) as f64;
        let (b, h) = (self.weight_format(), self.head_format());
        b.decode_bytes(cfg) - head * b.bits_per_weight() / 8.0 + head * h.bits_per_weight() / 8.0
    }

    pub fn name(self) -> &'static str {
        match self {
            Stage::Naive => "naive",
            Stage::KvCache => "kv-cache",
            Stage::Unrolled => "unrolled",
            Stage::Neon => "neon",
            Stage::Tiled => "tiled",
            Stage::Threads => "threads",
            Stage::SpinPool => "spin-pool",
            Stage::FastOps => "fast-ops",
            Stage::F16 => "f16",
            Stage::Q8 => "q8",
            Stage::Q4 => "q4",
            Stage::Q8Fast => "q8-fast",
            Stage::Q4Fast => "q4-fast",
            Stage::Q8Rescore => "q8-rescore",
            Stage::Q4Rescore => "q4-rescore",
            Stage::Q8Attn => "q8-attn",
            Stage::Q4Attn => "q4-attn",
            Stage::Q8Kv16 => "q8-kv16",
            Stage::Q4Kv16 => "q4-kv16",
            Stage::Q8Prefill => "q8-prefill",
        }
    }
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Stage {
    type Err = Error;
    fn from_str(s: &str) -> Result<Stage> {
        Stage::ALL.iter().copied().find(|st| st.name() == s).ok_or_else(|| Error::UnknownStage(s.to_string()))
    }
}

/// How a stage stores its weight *matrices* (biases and LayerNorm parameters
/// always stay f32: they're <0.1% of the bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    F32,
    F16,
    /// ggml-style blocks of 32 int8 weights sharing one scale.
    Q8_0,
    /// ggml-style blocks of 32 4-bit weights sharing one scale.
    Q4_0,
}

impl WeightFormat {
    pub fn name(self) -> &'static str {
        match self {
            WeightFormat::F32 => "f32",
            WeightFormat::F16 => "f16",
            WeightFormat::Q8_0 => "q8_0",
            WeightFormat::Q4_0 => "q4_0",
        }
    }

    /// Storage cost per matrix weight, including block scales.
    pub fn bits_per_weight(self) -> f64 {
        match self {
            WeightFormat::F32 => 32.0,
            WeightFormat::F16 => 16.0,
            // 32 × int8 + one f32 scale per block
            WeightFormat::Q8_0 => (32.0 * 8.0 + 32.0) / 32.0,
            // 32 × 4 bits + one f32 scale per block
            WeightFormat::Q4_0 => (32.0 * 4.0 + 32.0) / 32.0,
        }
    }

    /// Bytes that must be streamed from memory to decode one token in this format:
    /// every matrix (incl. the tied LM head) at [`Self::bits_per_weight`], plus
    /// f32 biases / LayerNorm params. Same accounting as
    /// [`Config::decode_weight_bytes`] for f32.
    pub fn decode_bytes(self, cfg: &crate::Config) -> f64 {
        let c = cfg.n_embd;
        let matrix = cfg.n_layer * 12 * c * c + cfg.n_vocab * c;
        let other = cfg.n_params() - cfg.n_ctx * c - matrix;
        matrix as f64 * self.bits_per_weight() / 8.0 + other as f64 * 4.0
    }
}

/// Scratch buffers for activations, sized for `rows` tokens at a time. Allocated
/// once and grown on demand, so steady-state decoding never allocates.
#[derive(Debug, Default)]
pub struct Activations {
    rows: usize,
    /// Residual stream `[rows, C]`.
    pub x: Vec<f32>,
    /// LayerNorm output `[rows, C]`.
    pub xn: Vec<f32>,
    /// Fused Q|K|V projections `[rows, 3C]`.
    pub qkv: Vec<f32>,
    /// Attention output (heads concatenated) `[rows, C]`.
    pub att: Vec<f32>,
    /// Output of a projection back to width C `[rows, C]`.
    pub proj: Vec<f32>,
    /// MLP hidden layer `[rows, 4C]`.
    pub fc: Vec<f32>,
    /// Attention scores for one (query, head) `[n_ctx]`.
    pub scores: Vec<f32>,
    /// Next-token logits for the last position `[V]`.
    pub logits: Vec<f32>,
}

impl Activations {
    pub fn ensure(&mut self, model: &Model, rows: usize) {
        let cfg = &model.config;
        if self.logits.len() != cfg.n_vocab {
            self.logits = vec![0.0; cfg.n_vocab];
            self.scores = vec![0.0; cfg.n_ctx];
        }
        if rows > self.rows {
            let c = cfg.n_embd;
            self.x = vec![0.0; rows * c];
            self.xn = vec![0.0; rows * c];
            self.qkv = vec![0.0; rows * 3 * c];
            self.att = vec![0.0; rows * c];
            self.proj = vec![0.0; rows * c];
            self.fc = vec![0.0; rows * 4 * c];
            self.rows = rows;
        }
    }
}

/// Per-layer key/value cache: `[n_layer, max_seq, C]` for each of K and V.
#[derive(Debug, Default)]
pub struct KvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// Positions filled so far.
    pub len: usize,
    /// Positions allocated per layer.
    pub capacity: usize,
}

impl KvCache {
    pub fn new(model: &Model, capacity: usize) -> Self {
        let n = model.config.n_layer * capacity * model.config.n_embd;
        KvCache { k: vec![0.0; n], v: vec![0.0; n], len: 0, capacity }
    }

    /// Bytes held by the cache when full. For GPT-2 small at 1024 positions:
    /// 2 (K,V) x 12 layers x 1024 x 768 x 4 bytes = 72 MiB.
    pub fn bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * 4
    }
}

/// The state of one generation: tokens so far plus whatever the stage remembers.
/// A stage's own copy of the weight matrices in a smaller format (stages 8+),
/// built once per session from the f32 [`Model`].
enum PackedWeights {
    None,
    F16(Box<packed::PackedModel<F16Matrix>>),
    /// Q8_0 blocks, f16 LM head (see LOG stage 9: the tied head is the
    /// quantisation-sensitive matrix).
    Q8(Box<packed::PackedModel<Q8Matrix, packed::PlainHead<F16Matrix>>>),
    /// Q4_0 blocks, f16 LM head.
    Q4(Box<packed::PackedModel<Q4Matrix, packed::PlainHead<F16Matrix>>>),
    /// Q8_0 blocks, rescored head (Q8 × two-level int8 approximate, f16 exact for candidates).
    Q8R(Box<packed::PackedModel<Q8Matrix, packed::RescoredHead<Q8HiLoMatrix>>>),
    /// Q8_0 blocks with the int8 prefill tile, rescored head (stage 15).
    Q8G(Box<packed::PackedModel<Q8GemmMatrix, packed::RescoredHead<Q8HiLoMatrix>>>),
    /// Q4_0 blocks, rescored head.
    Q4R(Box<packed::PackedModel<Q4Matrix, packed::RescoredHead<Q8HiLoMatrix>>>),
}

impl PackedWeights {
    fn for_stage(model: &Model, stage: Stage) -> Self {
        match stage {
            Stage::Q8Prefill => return PackedWeights::Q8G(Box::new(packed::PackedModel::new(model))),
            Stage::Q8Rescore | Stage::Q8Attn | Stage::Q8Kv16 => {
                return PackedWeights::Q8R(Box::new(packed::PackedModel::new(model)))
            }
            Stage::Q4Rescore | Stage::Q4Attn | Stage::Q4Kv16 => {
                return PackedWeights::Q4R(Box::new(packed::PackedModel::new(model)))
            }
            _ => {}
        }
        match stage.weight_format() {
            WeightFormat::F32 => PackedWeights::None,
            WeightFormat::F16 => PackedWeights::F16(Box::new(packed::PackedModel::new(model))),
            WeightFormat::Q8_0 => PackedWeights::Q8(Box::new(packed::PackedModel::new(model))),
            WeightFormat::Q4_0 => PackedWeights::Q4(Box::new(packed::PackedModel::new(model))),
        }
    }
}

pub struct Session {
    stage: Stage,
    tokens: Vec<u32>,
    max_seq: usize,
    kv: KvCache,
    /// Stage 14+: head-major f16 cache (empty for other stages).
    kv16: kv16::KvCache16,
    acts: Activations,
    packed: PackedWeights,
    /// Scratch space for packed-weight kernels (e.g. quantised activations).
    scratch: crate::kernels::packed::Scratch,
}

impl Session {
    /// A session that can hold up to the model's full context (1024 for GPT-2).
    pub fn new(model: &Model, stage: Stage) -> Self {
        Self::with_max_seq(model, stage, model.config.n_ctx)
    }

    pub fn with_max_seq(model: &Model, stage: Stage, max_seq: usize) -> Self {
        let max_seq = max_seq.min(model.config.n_ctx);
        let uses_kv16 = matches!(stage, Stage::Q8Kv16 | Stage::Q4Kv16 | Stage::Q8Prefill);
        let kv = match stage {
            Stage::Naive => KvCache::default(),
            // f16 stages keep their cache in `kv16`; `kv` only tracks the length.
            _ if uses_kv16 => KvCache { k: Vec::new(), v: Vec::new(), len: 0, capacity: max_seq },
            _ => KvCache::new(model, max_seq),
        };
        let kv16 = if uses_kv16 { kv16::KvCache16::new(model, max_seq) } else { kv16::KvCache16::default() };
        let packed = PackedWeights::for_stage(model, stage);
        Session {
            stage,
            tokens: Vec::new(),
            max_seq,
            kv,
            kv16,
            acts: Activations::default(),
            packed,
            scratch: Default::default(),
        }
    }

    pub fn stage(&self) -> Stage {
        self.stage
    }
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }
    pub fn kv_cache(&self) -> &KvCache {
        &self.kv
    }

    /// Forget the conversation (keeps allocations).
    pub fn reset(&mut self) {
        self.tokens.clear();
        self.kv.len = 0;
    }

    /// Append `new_tokens` to the context, run the forward pass, and return the
    /// logits predicting the token after the last one.
    ///
    /// Call it once with the whole prompt (prefill), then once per generated
    /// token (decode). Same API for every stage; what differs is how much work
    /// each call does.
    pub fn step<T: Trace>(&mut self, model: &Model, new_tokens: &[u32], trace: &mut T) -> Result<&[f32]> {
        if new_tokens.is_empty() {
            return Err(Error::Format("step() needs at least one token".into()));
        }
        if let Some(&t) = new_tokens.iter().find(|&&t| t as usize >= model.config.n_vocab) {
            return Err(Error::BadToken(t));
        }
        let len = self.tokens.len() + new_tokens.len();
        if len > self.max_seq {
            return Err(Error::ContextFull { len, max: self.max_seq });
        }
        let phase = if self.tokens.is_empty() || new_tokens.len() > 1 { Phase::Prefill } else { Phase::Decode };
        self.tokens.extend_from_slice(new_tokens);

        let n_computed = match self.stage {
            Stage::Naive => len,
            _ => new_tokens.len(),
        };
        let info = ForwardInfo { stage: self.stage, phase, n_new: new_tokens.len(), n_computed, context_len: len };
        if T::ENABLED {
            trace.forward_start(&info);
        }
        match self.stage {
            Stage::Naive => naive::forward::<Reference, T>(model, &self.tokens, &mut self.acts, trace),
            Stage::KvCache => kv_cache::forward::<Reference, T>(model, new_tokens, &mut self.kv, &mut self.acts, trace),
            Stage::Unrolled => kv_cache::forward::<Unrolled, T>(model, new_tokens, &mut self.kv, &mut self.acts, trace),
            Stage::Neon => kv_cache::forward::<Neon, T>(model, new_tokens, &mut self.kv, &mut self.acts, trace),
            Stage::Tiled => kv_cache::forward::<Tiled, T>(model, new_tokens, &mut self.kv, &mut self.acts, trace),
            Stage::Threads => kv_cache::forward::<Threaded, T>(model, new_tokens, &mut self.kv, &mut self.acts, trace),
            Stage::SpinPool => kv_cache::forward::<Spin, T>(model, new_tokens, &mut self.kv, &mut self.acts, trace),
            Stage::FastOps => kv_cache::forward::<FastOps, T>(model, new_tokens, &mut self.kv, &mut self.acts, trace),
            Stage::F16 => match &self.packed {
                PackedWeights::F16(p) => packed::forward::<FastOps, F16Matrix, _, SpinPar, T>(
                    model,
                    p,
                    new_tokens,
                    &mut self.kv,
                    &mut self.acts,
                    &mut self.scratch,
                    trace,
                ),
                _ => unreachable!(),
            },
            Stage::Q8 => match &self.packed {
                PackedWeights::Q8(p) => packed::forward::<FastOps, Q8Matrix, _, SpinPar, T>(
                    model,
                    p,
                    new_tokens,
                    &mut self.kv,
                    &mut self.acts,
                    &mut self.scratch,
                    trace,
                ),
                _ => unreachable!(),
            },
            Stage::Q4 => match &self.packed {
                PackedWeights::Q4(p) => packed::forward::<FastOps, Q4Matrix, _, SpinPar, T>(
                    model,
                    p,
                    new_tokens,
                    &mut self.kv,
                    &mut self.acts,
                    &mut self.scratch,
                    trace,
                ),
                _ => unreachable!(),
            },
            Stage::Q8Fast => match &self.packed {
                PackedWeights::Q8(p) => packed::forward::<LowLat, Q8Matrix, _, FastPar, T>(
                    model,
                    p,
                    new_tokens,
                    &mut self.kv,
                    &mut self.acts,
                    &mut self.scratch,
                    trace,
                ),
                _ => unreachable!(),
            },
            Stage::Q4Fast => match &self.packed {
                PackedWeights::Q4(p) => packed::forward::<LowLat, Q4Matrix, _, FastPar, T>(
                    model,
                    p,
                    new_tokens,
                    &mut self.kv,
                    &mut self.acts,
                    &mut self.scratch,
                    trace,
                ),
                _ => unreachable!(),
            },
            Stage::Q8Rescore => match &self.packed {
                PackedWeights::Q8R(p) => packed::forward::<LowLat, Q8Matrix, _, FastPar, T>(
                    model,
                    p,
                    new_tokens,
                    &mut self.kv,
                    &mut self.acts,
                    &mut self.scratch,
                    trace,
                ),
                _ => unreachable!(),
            },
            Stage::Q4Rescore => match &self.packed {
                PackedWeights::Q4R(p) => packed::forward::<LowLat, Q4Matrix, _, FastPar, T>(
                    model,
                    p,
                    new_tokens,
                    &mut self.kv,
                    &mut self.acts,
                    &mut self.scratch,
                    trace,
                ),
                _ => unreachable!(),
            },
            Stage::Q8Attn => match &self.packed {
                PackedWeights::Q8R(p) => packed::forward::<AttnNeon, Q8Matrix, _, FastPar, T>(
                    model,
                    p,
                    new_tokens,
                    &mut self.kv,
                    &mut self.acts,
                    &mut self.scratch,
                    trace,
                ),
                _ => unreachable!(),
            },
            Stage::Q4Attn => match &self.packed {
                PackedWeights::Q4R(p) => packed::forward::<AttnNeon, Q4Matrix, _, FastPar, T>(
                    model,
                    p,
                    new_tokens,
                    &mut self.kv,
                    &mut self.acts,
                    &mut self.scratch,
                    trace,
                ),
                _ => unreachable!(),
            },
            Stage::Q8Kv16 => match &self.packed {
                PackedWeights::Q8R(p) => kv16::forward::<AttnNeon, Q8Matrix, _, FastPar, T>(
                    model,
                    p,
                    new_tokens,
                    &mut self.kv16,
                    &mut self.kv.len,
                    &mut self.acts,
                    &mut self.scratch,
                    trace,
                ),
                _ => unreachable!(),
            },
            Stage::Q4Kv16 => match &self.packed {
                PackedWeights::Q4R(p) => kv16::forward::<AttnNeon, Q4Matrix, _, FastPar, T>(
                    model,
                    p,
                    new_tokens,
                    &mut self.kv16,
                    &mut self.kv.len,
                    &mut self.acts,
                    &mut self.scratch,
                    trace,
                ),
                _ => unreachable!(),
            },
            Stage::Q8Prefill => match &self.packed {
                PackedWeights::Q8G(p) => kv16::forward::<AttnNeon, Q8GemmMatrix, _, FastPar, T>(
                    model,
                    p,
                    new_tokens,
                    &mut self.kv16,
                    &mut self.kv.len,
                    &mut self.acts,
                    &mut self.scratch,
                    trace,
                ),
                _ => unreachable!(),
            },
        }
        if T::ENABLED {
            if trace.wants_logits() {
                trace.logits(&self.acts.logits);
            }
            trace.forward_end(&info);
        }
        Ok(&self.acts.logits)
    }
}
