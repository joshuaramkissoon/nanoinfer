//! Instrumentation: how the forward pass tells the outside world what it's doing.
//!
//! The forward pass is generic over a [`Trace`] sink and calls it around every
//! op. The default sink, [`NoTrace`], sets `ENABLED = false`; every call site is
//! guarded by `if T::ENABLED`, which is a compile-time constant, so after
//! monomorphisation the untraced engine contains no tracing code at all. Zero
//! overhead is a property of the type system, not of a runtime flag.
//!
//! Core has no clock (there is no `std::time::Instant` on wasm32-unknown-unknown),
//! so the trace events themselves carry no timestamps. A sink that wants
//! timings reads a [`Clock`] the caller supplies: `Instant` in the CLI,
//! `performance.now()` in the browser. [`Recorder`] is a ready-made sink that
//! does exactly that and keeps a flat list of events for a dashboard.

use crate::stages::Stage;

/// What kind of op ran. Coarse on purpose, so a visualisation can colour by kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    Embed,
    LayerNorm,
    MatMul,
    Attention,
    Gelu,
    Residual,
    /// Writing new K/V rows into the cache.
    KvWrite,
    /// The final projection to vocabulary logits (a matmul, but by far the
    /// widest one, so it gets its own kind).
    LmHead,
}

impl OpKind {
    /// Matches the `OpKind` strings of the web trace schema
    /// (research/03-explorable-writeup-and-wasm.md §3.4) where they overlap.
    pub fn as_str(self) -> &'static str {
        match self {
            OpKind::Embed => "embed",
            OpKind::LayerNorm => "ln",
            OpKind::MatMul => "matmul",
            OpKind::Attention => "attn",
            OpKind::Gelu => "gelu",
            OpKind::Residual => "residual",
            OpKind::KvWrite => "kv_write",
            OpKind::LmHead => "lm_head",
        }
    }
}

/// One op invocation, as reported to [`Trace::op_start`] / [`Trace::op_end`].
///
/// `flops` and `bytes` are *estimates* from the op's shape (what an ideal
/// kernel must do / move), not measurements. Dividing them by the measured
/// time gives achieved GFLOP/s and GB/s: the two axes of a roofline plot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Op {
    pub kind: OpKind,
    /// Specific name: "ln_1", "qkv", "attn", "attn_proj", "fc", "fc_proj", "lm_head", ...
    pub name: &'static str,
    /// Transformer block index, or `None` for embedding / final layers.
    pub layer: Option<usize>,
    /// Shape. For matmuls `[n, d_in, d_out]`; for attention `[n_q, n_kv, C]`;
    /// for elementwise ops `[n, d, 0]`.
    pub dims: [usize; 3],
    /// Floating point operations (a multiply-add counts as 2).
    pub flops: u64,
    /// Bytes an ideal kernel must read + write (f32 = 4 bytes).
    pub bytes: u64,
}

impl Op {
    pub fn matmul(name: &'static str, layer: Option<usize>, n: usize, d_in: usize, d_out: usize) -> Op {
        let (n64, i, o) = (n as u64, d_in as u64, d_out as u64);
        Op {
            kind: OpKind::MatMul,
            name,
            layer,
            dims: [n, d_in, d_out],
            flops: 2 * n64 * i * o + n64 * o,
            // weights + bias + input + output
            bytes: 4 * (i * o + o + n64 * i + n64 * o),
        }
    }

    pub fn elementwise(
        kind: OpKind,
        name: &'static str,
        layer: Option<usize>,
        n: usize,
        d: usize,
        flops_per: u64,
        bytes_per: u64,
    ) -> Op {
        let e = (n * d) as u64;
        Op { kind, name, layer, dims: [n, d, 0], flops: e * flops_per, bytes: e * bytes_per }
    }

    /// Causal attention for `n_q` queries starting at `q_pos0`, width `c`.
    pub fn attention(layer: usize, n_q: usize, q_pos0: usize, c: usize, n_head: usize) -> Op {
        // Query i sees L_i = q_pos0 + i + 1 keys. Per head and key: q·k (2·hd),
        // softmax (~5), and p·v (2·hd). Summed over heads: 4·C·L + 5·H·L.
        let total_keys: u64 = (0..n_q).map(|i| (q_pos0 + i + 1) as u64).sum();
        let n_kv = q_pos0 + n_q;
        let c64 = c as u64;
        Op {
            kind: OpKind::Attention,
            name: "attn",
            layer: Some(layer),
            dims: [n_q, n_kv, c],
            flops: total_keys * (4 * c64 + 5 * n_head as u64),
            // read q, K, V once; write out
            bytes: 4 * (2 * n_q as u64 * c64 + 2 * n_kv as u64 * c64),
        }
    }
}

/// Prefill = processing the prompt (many tokens per forward). Decode = one new
/// token per forward. They have very different performance characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Prefill,
    Decode,
}

/// Reported once per forward call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardInfo {
    pub stage: Stage,
    pub phase: Phase,
    /// New tokens fed in this call.
    pub n_new: usize,
    /// Tokens actually pushed through the network this call. For the naive stage
    /// this is the *whole* context every time; for the KV cache stage it equals `n_new`.
    pub n_computed: usize,
    /// Context length after this call.
    pub context_len: usize,
}

/// The instrumentation sink. All methods have empty defaults, so a sink only
/// implements what it cares about.
pub trait Trace {
    /// When `false`, the forward pass skips every tracing call at compile time.
    const ENABLED: bool = true;

    fn forward_start(&mut self, _info: &ForwardInfo) {}
    fn forward_end(&mut self, _info: &ForwardInfo) {}
    fn op_start(&mut self, _op: &Op) {}
    fn op_end(&mut self, _op: &Op) {}

    /// The KV cache now holds `len` positions out of `capacity` (stage 1+).
    fn kv_cache(&mut self, _len: usize, _capacity: usize) {}

    /// Return `true` to receive the final-position logits via [`Trace::logits`]
    /// (e.g. to draw top-k next-token probabilities).
    fn wants_logits(&self) -> bool {
        false
    }
    fn logits(&mut self, _logits: &[f32]) {}

    /// Return `Some((layer, head))` to receive the attention probabilities of the
    /// last query position for that head via [`Trace::attention`]. Costs one
    /// extra q·K pass for that head only, and only when asked.
    fn wants_attention(&self) -> Option<(usize, usize)> {
        None
    }
    /// `probs[j]` = how much the newest token attends to position `j`.
    fn attention(&mut self, _layer: usize, _head: usize, _query_pos: usize, _probs: &[f32]) {}
}

/// The default sink: compiled out entirely.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoTrace;

impl Trace for NoTrace {
    const ENABLED: bool = false;
}

/// Run `f` between `op_start`/`op_end`. When tracing is disabled this is just `f()`.
#[inline(always)]
pub(crate) fn traced<T: Trace, R>(trace: &mut T, op: impl FnOnce() -> Op, f: impl FnOnce() -> R) -> R {
    if T::ENABLED {
        let op = op();
        trace.op_start(&op);
        let r = f();
        trace.op_end(&op);
        r
    } else {
        f()
    }
}

/// A monotonic clock supplied by the host. Seconds as `f64`.
pub trait Clock {
    fn now(&self) -> f64;
}

impl<F: Fn() -> f64> Clock for F {
    fn now(&self) -> f64 {
        self()
    }
}

/// One timed op, as recorded by [`Recorder`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpEvent {
    pub op: Op,
    /// Index of the forward call this op belongs to.
    pub forward: usize,
    /// Seconds since the recorder was created.
    pub start: f64,
    pub duration: f64,
}

/// One forward call, as recorded by [`Recorder`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardEvent {
    pub info: ForwardInfo,
    pub start: f64,
    pub duration: f64,
    pub kv_len: Option<usize>,
}

/// A ready-made sink for dashboards: records every op with timings from an
/// injected clock, plus (optionally) top-k probabilities and an attention map.
pub struct Recorder<C: Clock> {
    clock: C,
    t0: f64,
    op_t: f64,
    fwd_t: f64,
    pub forwards: Vec<ForwardEvent>,
    pub ops: Vec<OpEvent>,
    /// How many top tokens to keep per forward (0 = don't look at logits).
    pub top_k: usize,
    /// `(token, probability)` for the last forward, highest first.
    pub top: Vec<(u32, f32)>,
    /// Which `(layer, head)` attention map to capture, if any.
    pub attention_head: Option<(usize, usize)>,
    /// Attention probabilities of the newest token, for the last forward.
    pub attention_probs: Vec<f32>,
    kv_len: Option<usize>,
}

impl<C: Clock> Recorder<C> {
    pub fn new(clock: C) -> Self {
        let t0 = clock.now();
        Recorder {
            clock,
            t0,
            op_t: 0.0,
            fwd_t: 0.0,
            forwards: Vec::new(),
            ops: Vec::new(),
            top_k: 0,
            top: Vec::new(),
            attention_head: None,
            attention_probs: Vec::new(),
            kv_len: None,
        }
    }

    fn now(&self) -> f64 {
        self.clock.now() - self.t0
    }

    /// Drop recorded events (keeps settings).
    pub fn clear(&mut self) {
        self.forwards.clear();
        self.ops.clear();
    }
}

impl<C: Clock> Trace for Recorder<C> {
    fn forward_start(&mut self, _info: &ForwardInfo) {
        self.fwd_t = self.now();
        self.kv_len = None;
    }
    fn forward_end(&mut self, info: &ForwardInfo) {
        let now = self.now();
        self.forwards.push(ForwardEvent {
            info: *info,
            start: self.fwd_t,
            duration: now - self.fwd_t,
            kv_len: self.kv_len,
        });
    }
    fn op_start(&mut self, _op: &Op) {
        self.op_t = self.now();
    }
    fn op_end(&mut self, op: &Op) {
        let now = self.now();
        self.ops.push(OpEvent { op: *op, forward: self.forwards.len(), start: self.op_t, duration: now - self.op_t });
    }
    fn kv_cache(&mut self, len: usize, _capacity: usize) {
        self.kv_len = Some(len);
    }
    fn wants_logits(&self) -> bool {
        self.top_k > 0
    }
    fn logits(&mut self, logits: &[f32]) {
        self.top = crate::sampler::top_k_probs(logits, self.top_k);
    }
    fn wants_attention(&self) -> Option<(usize, usize)> {
        self.attention_head
    }
    fn attention(&mut self, _layer: usize, _head: usize, _query_pos: usize, probs: &[f32]) {
        self.attention_probs.clear();
        self.attention_probs.extend_from_slice(probs);
    }
}
