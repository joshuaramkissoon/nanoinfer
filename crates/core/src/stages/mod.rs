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
pub mod kv_cache;
pub mod naive;

use crate::error::{Error, Result};
use crate::kernels::{Kernels, Reference};
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
}

impl Stage {
    pub const ALL: &'static [Stage] = &[Stage::Naive, Stage::KvCache];

    /// Name of the kernel set this stage runs.
    pub fn kernels(self) -> &'static str {
        match self {
            Stage::Naive | Stage::KvCache => Reference::NAME,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Stage::Naive => "naive",
            Stage::KvCache => "kv-cache",
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
pub struct Session {
    stage: Stage,
    tokens: Vec<u32>,
    max_seq: usize,
    kv: KvCache,
    acts: Activations,
}

impl Session {
    /// A session that can hold up to the model's full context (1024 for GPT-2).
    pub fn new(model: &Model, stage: Stage) -> Self {
        Self::with_max_seq(model, stage, model.config.n_ctx)
    }

    pub fn with_max_seq(model: &Model, stage: Stage, max_seq: usize) -> Self {
        let max_seq = max_seq.min(model.config.n_ctx);
        let kv = match stage {
            Stage::Naive => KvCache::default(),
            Stage::KvCache => KvCache::new(model, max_seq),
        };
        Session { stage, tokens: Vec::new(), max_seq, kv, acts: Activations::default() }
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
            Stage::KvCache => new_tokens.len(),
        };
        let info = ForwardInfo { stage: self.stage, phase, n_new: new_tokens.len(), n_computed, context_len: len };
        if T::ENABLED {
            trace.forward_start(&info);
        }
        match self.stage {
            Stage::Naive => naive::forward::<Reference, T>(model, &self.tokens, &mut self.acts, trace),
            Stage::KvCache => kv_cache::forward::<Reference, T>(model, new_tokens, &mut self.kv, &mut self.acts, trace),
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
