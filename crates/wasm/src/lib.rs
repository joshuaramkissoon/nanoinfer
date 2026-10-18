//! # nanoinfer-wasm
//!
//! A thin wasm-bindgen facade over `nanoinfer-core`, built for one job: driving
//! the live visualisations in `web/`. Everything numeric the page shows comes
//! out of here.
//!
//! - [`Engine`]: weights + tokenizer + a [`Session`] on a selectable [`Stage`].
//!   `forward()` runs one real forward pass and returns a compact trace (per-op
//!   timings, FLOPs, bytes, KV length, top-k next-token probabilities and,
//!   optionally, one head's attention row).
//! - [`stages()`]: the stage list, read from `Stage::ALL`, so stages the
//!   hillclimb adds to core appear in the browser without touching this crate.
//! - [`Engine::attention_maps`]: an analysis pass that captures *every* head's
//!   full causal attention matrix, built from core's own `blocks`.
//! - [`bench_bandwidth`] / [`bench_peak_gflops`]: the two roofs of the roofline,
//!   measured in this tab.
//!
//! Timing comes from `performance.now()` (core has no clock). It's ~5 µs
//! resolution when the page is cross-origin isolated, ~100 µs otherwise; the
//! page reports which.

use nanoinfer_core::kernels::{reference, Reference, Rows};
use nanoinfer_core::stages::{blocks, Activations};
use nanoinfer_core::trace::{ForwardInfo, Op, Phase, Trace};
use nanoinfer_core::{sampler, Model, NoTrace, Sampler, Session, Stage, Tokenizer};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    /// Milliseconds, from the global `performance` (works in a Worker too).
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn perf_now() -> f64;
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(s: &str);
}

#[wasm_bindgen(start)]
pub fn init() {
    std::panic::set_hook(Box::new(|info| console_error(&format!("nanoinfer-wasm panic: {info}"))));
}

/// Version of the packed trace layout returned by [`Engine::forward`].
pub const TRACE_VERSION: f64 = 1.0;
/// Header length (f64s) of a packed trace.
pub const HEADER_LEN: usize = 16;
/// f64s per op record.
pub const OP_STRIDE: usize = 10;

fn err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

/// Every stage core knows about, as
/// `[{ name, kernels, index, weightFormat, bitsPerWeight }]`.
#[wasm_bindgen]
pub fn stages() -> js_sys::Array {
    let out = js_sys::Array::new();
    for (i, s) in Stage::ALL.iter().enumerate() {
        let o = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&o, &"name".into(), &s.name().into());
        let _ = js_sys::Reflect::set(&o, &"kernels".into(), &s.kernels().into());
        let _ = js_sys::Reflect::set(&o, &"index".into(), &(i as u32).into());
        let wf = s.weight_format();
        let _ = js_sys::Reflect::set(&o, &"weightFormat".into(), &wf.name().into());
        let _ = js_sys::Reflect::set(&o, &"bitsPerWeight".into(), &wf.bits_per_weight().into());
        out.push(&o);
    }
    out
}

/// Layout constants of the packed trace, so JS never hard-codes them.
#[wasm_bindgen]
pub fn trace_layout() -> Vec<f64> {
    vec![TRACE_VERSION, HEADER_LEN as f64, OP_STRIDE as f64]
}

/// A growable byte buffer living in wasm memory. JS streams the checkpoint into
/// it chunk by chunk, so the 500 MB file is never held twice on the JS side.
#[wasm_bindgen]
pub struct Bytes {
    buf: Vec<u8>,
}

#[wasm_bindgen]
impl Bytes {
    #[wasm_bindgen(constructor)]
    pub fn new(capacity: usize) -> Bytes {
        Bytes { buf: Vec::with_capacity(capacity) }
    }
    pub fn push(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }
    pub fn len(&self) -> usize {
        self.buf.len()
    }
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

/// The trace sink: like core's `Recorder`, but writes straight into a flat
/// `f64` buffer that crosses to JS in one copy.
struct Sink {
    names: Vec<&'static str>,
    kinds: Vec<&'static str>,
    ops: Vec<f64>,
    t0: f64,
    t_fwd: f64,
    t_op: f64,
    fwd_ms: f64,
    info: Option<ForwardInfo>,
    kv: Option<(usize, usize)>,
    top_k: usize,
    top: Vec<(u32, f32)>,
    entropy: f32,
    attn_head: Option<(usize, usize)>,
    attn: Vec<f32>,
}

fn intern(table: &mut Vec<&'static str>, s: &'static str) -> usize {
    match table.iter().position(|&t| t == s) {
        Some(i) => i,
        None => {
            table.push(s);
            table.len() - 1
        }
    }
}

impl Sink {
    fn new() -> Self {
        Sink {
            names: Vec::new(),
            kinds: Vec::new(),
            ops: Vec::with_capacity(256 * OP_STRIDE),
            t0: perf_now(),
            t_fwd: 0.0,
            t_op: 0.0,
            fwd_ms: 0.0,
            info: None,
            kv: None,
            top_k: 10,
            top: Vec::new(),
            entropy: 0.0,
            attn_head: None,
            attn: Vec::new(),
        }
    }
    fn now(&self) -> f64 {
        perf_now() - self.t0
    }
    fn begin(&mut self) {
        self.ops.clear();
        self.top.clear();
        self.attn.clear();
        self.kv = None;
        self.info = None;
    }
}

impl Trace for Sink {
    fn forward_start(&mut self, _info: &ForwardInfo) {
        self.t_fwd = self.now();
    }
    fn forward_end(&mut self, info: &ForwardInfo) {
        self.fwd_ms = self.now() - self.t_fwd;
        self.info = Some(*info);
    }
    fn op_start(&mut self, _op: &Op) {
        self.t_op = self.now();
    }
    fn op_end(&mut self, op: &Op) {
        let end = self.now();
        let k = intern(&mut self.kinds, op.kind.as_str());
        let n = intern(&mut self.names, op.name);
        self.ops.extend_from_slice(&[
            k as f64,
            n as f64,
            op.layer.map_or(-1.0, |l| l as f64),
            op.dims[0] as f64,
            op.dims[1] as f64,
            op.dims[2] as f64,
            op.flops as f64,
            op.bytes as f64,
            self.t_op - self.t_fwd,
            end - self.t_op,
        ]);
    }
    fn kv_cache(&mut self, len: usize, capacity: usize) {
        self.kv = Some((len, capacity));
    }
    fn wants_logits(&self) -> bool {
        // We always copy logits ourselves (see Engine::forward); this is only
        // for the top-k/entropy summary.
        false
    }
    fn wants_attention(&self) -> Option<(usize, usize)> {
        self.attn_head
    }
    fn attention(&mut self, _layer: usize, _head: usize, _query_pos: usize, probs: &[f32]) {
        self.attn.clear();
        self.attn.extend_from_slice(probs);
    }
}

/// Shannon entropy (nats) of softmax(logits).
fn entropy(logits: &[f32]) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0.0f64;
    for &l in logits {
        z += ((l - max) as f64).exp();
    }
    let lz = z.ln();
    let mut h = 0.0f64;
    for &l in logits {
        let lp = (l - max) as f64 - lz;
        h -= lp.exp() * lp;
    }
    h as f32
}

#[wasm_bindgen]
pub struct Engine {
    model: Model,
    tok: Tokenizer,
    /// Independent conversations, addressed by slot id. Slot 0 always exists;
    /// more let the page race two stages against each other token by token.
    slots: Vec<Option<Slot>>,
    sink: Sink,
    sampler: Sampler,
}

struct Slot {
    session: Session,
    logits: Vec<f32>,
}

#[wasm_bindgen]
impl Engine {
    /// Build an engine from the bytes of HF `model.safetensors`, `vocab.json`
    /// and `merges.txt`. Consumes `weights` (its memory is reused afterwards).
    #[wasm_bindgen(constructor)]
    pub fn new(weights: Bytes, vocab: &[u8], merges: &[u8], stage: &str) -> Result<Engine, JsError> {
        let model = Model::from_safetensors(&weights.buf).map_err(err)?;
        drop(weights);
        let tok = Tokenizer::from_vocab_and_merges(vocab, merges).map_err(err)?;
        let stage: Stage = stage.parse().map_err(err)?;
        let session = Session::new(&model, stage);
        Ok(Engine {
            model,
            tok,
            slots: vec![Some(Slot { session, logits: Vec::new() })],
            sink: Sink::new(),
            sampler: Sampler::greedy(),
        })
    }

    // ---- model / tokenizer --------------------------------------------------

    /// `[n_vocab, n_ctx, n_embd, n_head, n_layer, n_params, decode_weight_bytes_f32]`.
    pub fn config(&self) -> Vec<f64> {
        let c = &self.model.config;
        vec![
            c.n_vocab as f64,
            c.n_ctx as f64,
            c.n_embd as f64,
            c.n_head as f64,
            c.n_layer as f64,
            c.n_params() as f64,
            c.decode_weight_bytes(4) as f64,
        ]
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tok.encode(text)
    }

    /// All token byte strings concatenated; see [`Engine::vocab_offsets`].
    /// Lets JS decode any token (including partial UTF-8) without a round trip.
    pub fn vocab_blob(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for id in 0..self.tok.vocab_size() as u32 {
            out.extend_from_slice(self.tok.token_bytes(id));
        }
        out
    }
    /// `n_vocab + 1` offsets into [`Engine::vocab_blob`].
    pub fn vocab_offsets(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.tok.vocab_size() + 1);
        let mut o = 0u32;
        out.push(0);
        for id in 0..self.tok.vocab_size() as u32 {
            o += self.tok.token_bytes(id).len() as u32;
            out.push(o);
        }
        out
    }

    // ---- sessions -------------------------------------------------------------

    fn slot(&mut self, slot: u32) -> Result<&mut Slot, JsError> {
        self.slots.get_mut(slot as usize).and_then(|s| s.as_mut()).ok_or_else(|| err(format!("no session {slot}")))
    }

    /// Open a new session on `stage` holding up to `max_seq` tokens (0 = the
    /// model's full context). Returns its slot id.
    pub fn open(&mut self, stage: &str, max_seq: usize) -> Result<u32, JsError> {
        let stage: Stage = stage.parse().map_err(err)?;
        let max_seq = if max_seq == 0 { self.model.config.n_ctx } else { max_seq };
        let slot = Slot { session: Session::with_max_seq(&self.model, stage, max_seq), logits: Vec::new() };
        let id = match self.slots.iter().position(|s| s.is_none()) {
            Some(i) => {
                self.slots[i] = Some(slot);
                i
            }
            None => {
                self.slots.push(Some(slot));
                self.slots.len() - 1
            }
        };
        Ok(id as u32)
    }

    /// Free a session's memory (its KV cache can be tens of MB).
    pub fn close(&mut self, slot: u32) {
        if slot != 0 {
            if let Some(s) = self.slots.get_mut(slot as usize) {
                *s = None;
            }
        }
    }

    pub fn stage(&mut self, slot: u32) -> Result<String, JsError> {
        Ok(self.slot(slot)?.session.stage().name().to_string())
    }

    /// Restart session `slot` on `stage` (keeps its max_seq).
    pub fn set_stage(&mut self, slot: u32, stage: &str) -> Result<(), JsError> {
        let stage: Stage = stage.parse().map_err(err)?;
        let model = &self.model;
        let s = self.slots.get_mut(slot as usize).and_then(|s| s.as_mut()).ok_or_else(|| err("no session"))?;
        if stage != s.session.stage() {
            s.session = Session::with_max_seq(model, stage, s.session.max_seq());
        } else {
            s.session.reset();
        }
        s.logits.clear();
        Ok(())
    }

    pub fn reset(&mut self, slot: u32) -> Result<(), JsError> {
        let s = self.slot(slot)?;
        s.session.reset();
        s.logits.clear();
        Ok(())
    }

    pub fn tokens(&mut self, slot: u32) -> Result<Vec<u32>, JsError> {
        Ok(self.slot(slot)?.session.tokens().to_vec())
    }

    pub fn max_seq(&mut self, slot: u32) -> Result<usize, JsError> {
        Ok(self.slot(slot)?.session.max_seq())
    }

    /// How many top tokens to report per forward (default 10), and which
    /// `(layer, head)` attention row to capture (`layer < 0` = none).
    pub fn set_capture(&mut self, top_k: usize, layer: i32, head: i32) {
        self.sink.top_k = top_k;
        self.sink.attn_head = if layer >= 0 && head >= 0 { Some((layer as usize, head as usize)) } else { None };
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.sampler = Sampler::new(self.sampler.temperature, self.sampler.top_k, seed);
    }

    /// Op-name and op-kind string tables (indices used in the packed trace).
    /// Grow-only; refetch when the header's table sizes change.
    pub fn op_names(&self) -> js_sys::Array {
        self.sink.names.iter().map(|&s| JsValue::from_str(s)).collect()
    }
    pub fn op_kinds(&self) -> js_sys::Array {
        self.sink.kinds.iter().map(|&s| JsValue::from_str(s)).collect()
    }

    /// Append `tokens` to the context and run one real forward pass.
    ///
    /// Returns a packed trace (all f64):
    /// ```text
    /// header[16] = version, n_ops, n_top, n_attn, phase(0 prefill/1 decode),
    ///              n_new, n_computed, context_len, kv_len(-1), kv_capacity,
    ///              forward_ms, entropy_nats, n_names, n_kinds, attn_layer, attn_head
    /// top[n_top*2]   = (token id, probability), most likely first
    /// attn[n_attn]   = newest token's attention over positions 0..n_attn
    /// ops[n_ops*10]  = kind, name, layer, d0, d1, d2, flops, bytes, start_ms, dur_ms
    /// ```
    pub fn forward(&mut self, slot: u32, tokens: &[u32]) -> Result<Vec<f64>, JsError> {
        self.sink.begin();
        let model = &self.model;
        let s = self.slots.get_mut(slot as usize).and_then(|s| s.as_mut()).ok_or_else(|| err("no session"))?;
        let logits = s.session.step(model, tokens, &mut self.sink).map_err(err)?;
        s.logits.clear();
        s.logits.extend_from_slice(logits);
        if self.sink.top_k > 0 {
            self.sink.top = sampler::top_k_probs(&s.logits, self.sink.top_k);
            self.sink.entropy = entropy(&s.logits);
        }
        let s = &self.sink;
        let info = s.info.ok_or_else(|| err("forward produced no trace"))?;
        let n_ops = s.ops.len() / OP_STRIDE;
        let mut out = Vec::with_capacity(HEADER_LEN + s.top.len() * 2 + s.attn.len() + s.ops.len());
        let (al, ah) = s.attn_head.map_or((-1.0, -1.0), |(l, h)| (l as f64, h as f64));
        out.extend_from_slice(&[
            TRACE_VERSION,
            n_ops as f64,
            s.top.len() as f64,
            s.attn.len() as f64,
            if info.phase == Phase::Prefill { 0.0 } else { 1.0 },
            info.n_new as f64,
            info.n_computed as f64,
            info.context_len as f64,
            s.kv.map_or(-1.0, |k| k.0 as f64),
            s.kv.map_or(-1.0, |k| k.1 as f64),
            s.fwd_ms,
            s.entropy as f64,
            s.names.len() as f64,
            s.kinds.len() as f64,
            al,
            ah,
        ]);
        for &(id, p) in &s.top {
            out.push(id as f64);
            out.push(p as f64);
        }
        out.extend(s.attn.iter().map(|&p| p as f64));
        out.extend_from_slice(&s.ops);
        Ok(out)
    }

    /// Sample the next token from the last forward's logits. Re-callable with
    /// different settings without re-running the model.
    pub fn sample(&mut self, slot: u32, temperature: f32, top_k: usize) -> Result<u32, JsError> {
        self.sampler.temperature = temperature;
        self.sampler.top_k = top_k;
        let logits = &self.slots.get(slot as usize).and_then(|s| s.as_ref()).ok_or_else(|| err("no session"))?.logits;
        if logits.is_empty() {
            return Err(err("sample() before forward()"));
        }
        Ok(self.sampler.sample(logits))
    }

    /// Probability of `id` under session `slot`'s last softmax.
    pub fn prob_of(&mut self, slot: u32, id: u32) -> Result<f32, JsError> {
        let logits = &self.slot(slot)?.logits;
        if logits.is_empty() || id as usize >= logits.len() {
            return Ok(0.0);
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
        Ok((logits[id as usize] - max).exp() / z)
    }

    /// Analysis pass: run `tokens` through the network (core's own blocks and
    /// reference kernels) and capture every head's full causal attention
    /// matrix. Returns `[n_layer, n_head, T, T]` row-major, where row `i` is
    /// query `i` and only `j <= i` is non-zero. Independent of the session.
    pub fn attention_maps(&self, tokens: &[u32]) -> Result<Vec<f32>, JsError> {
        let model = &self.model;
        let cfg = model.config;
        let n = tokens.len();
        if n == 0 || n > cfg.n_ctx {
            return Err(err(format!("attention_maps: need 1..={} tokens", cfg.n_ctx)));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= cfg.n_vocab) {
            return Err(err(format!("bad token {t}")));
        }
        let (c, nh, hd) = (cfg.n_embd, cfg.n_head, cfg.head_dim());
        let mut acts = Activations::default();
        acts.ensure(model, n);
        let mut out = vec![0.0f32; cfg.n_layer * nh * n * n];
        let mut row = vec![0.0f32; n];
        let mut tr = NoTrace;
        blocks::embed::<Reference, NoTrace>(model, tokens, 0, &mut acts, &mut tr);
        for (l, layer) in model.layers.iter().enumerate() {
            blocks::qkv::<Reference, NoTrace>(&cfg, l, layer, n, &mut acts, &mut tr);
            {
                let qkv = &acts.qkv[..n * 3 * c];
                let k = Rows { data: &qkv[c..], stride: 3 * c };
                for h in 0..nh {
                    let base = (l * nh + h) * n * n;
                    for i in 0..n {
                        let q = &qkv[i * 3 * c + h * hd..][..hd];
                        reference::attention_probs(&mut row, q, k, h, hd, i + 1);
                        out[base + i * n..base + i * n + i + 1].copy_from_slice(&row[..i + 1]);
                    }
                }
            }
            let qkv = acts.qkv[..n * 3 * c].to_vec();
            let k = Rows { data: &qkv[c..], stride: 3 * c };
            let v = Rows { data: &qkv[2 * c..], stride: 3 * c };
            blocks::attention::<Reference, NoTrace>(&cfg, l, n, 0, k, v, &mut acts.att, &qkv, &mut acts.scores, &mut tr);
            blocks::proj_and_mlp::<Reference, NoTrace>(&cfg, l, layer, n, &mut acts, &mut tr);
        }
        Ok(out)
    }
}

// ---- microbenchmarks: the two roofs ---------------------------------------------

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

/// Streaming-read bandwidth in GB/s: sum a `mb`-megabyte f32 buffer, best of
/// `iters`. This is the "speed of light" denominator for decode in this tab.
#[wasm_bindgen]
pub fn bench_bandwidth(mb: usize, iters: usize) -> f64 {
    let n = mb * 1024 * 1024 / 4;
    let buf: Vec<f32> = (0..n).map(|i| (i & 7) as f32).collect();
    let mut best = f64::INFINITY;
    let mut sink = 0.0f32;
    for _ in 0..iters.max(1) {
        let t = perf_now();
        sink += stream_sum(&buf);
        let dt = perf_now() - t;
        best = best.min(dt);
    }
    std::hint::black_box(sink);
    (n * 4) as f64 / (best / 1e3) / 1e9
}

#[cfg(target_arch = "wasm32")]
fn stream_sum(x: &[f32]) -> f32 {
    // 8 independent accumulators so add latency never limits the read rate.
    let mut acc = [f32x4_splat(0.0); 8];
    let chunks = x.chunks_exact(32);
    let rem = chunks.remainder();
    for c in chunks {
        // Safety: c has 32 f32s; v128_load tolerates unaligned addresses.
        unsafe {
            let p = c.as_ptr() as *const v128;
            for (k, a) in acc.iter_mut().enumerate() {
                *a = f32x4_add(*a, v128_load(p.add(k)));
            }
        }
    }
    let mut a = acc[0];
    for v in &acc[1..] {
        a = f32x4_add(a, *v);
    }
    let mut s = f32x4_extract_lane::<0>(a) + f32x4_extract_lane::<1>(a) + f32x4_extract_lane::<2>(a) + f32x4_extract_lane::<3>(a);
    for &v in rem {
        s += v;
    }
    s
}

#[cfg(not(target_arch = "wasm32"))]
fn stream_sum(x: &[f32]) -> f32 {
    x.iter().sum()
}

/// Peak single-thread compute in GFLOP/s: 8 independent SIMD multiply-add
/// chains held in registers (no memory traffic). The "factory" roof.
#[wasm_bindgen]
pub fn bench_peak_gflops(millions: usize) -> f64 {
    let iters = millions * 1_000_000 / 32;
    let t = perf_now();
    let r = peak_loop(iters);
    let dt = perf_now() - t;
    std::hint::black_box(r);
    // per iter: 8 chains x 4 lanes x (mul + add) = 64 flops
    (iters as f64 * 64.0) / (dt / 1e3) / 1e9
}

#[cfg(target_arch = "wasm32")]
fn peak_loop(iters: usize) -> f32 {
    let m = f32x4_splat(0.999_999);
    let b = f32x4_splat(1e-7);
    let mut a = [f32x4_splat(1.0); 8];
    for _ in 0..iters {
        for v in a.iter_mut() {
            *v = f32x4_add(f32x4_mul(*v, m), b);
        }
        std::hint::black_box(&a);
    }
    let mut s = 0.0;
    for v in a {
        s += f32x4_extract_lane::<0>(v);
    }
    s
}

#[cfg(not(target_arch = "wasm32"))]
fn peak_loop(iters: usize) -> f32 {
    let mut a = [1.0f32; 32];
    for _ in 0..iters {
        for v in a.iter_mut() {
            *v = *v * 0.999_999 + 1e-7;
        }
    }
    a.iter().sum()
}
