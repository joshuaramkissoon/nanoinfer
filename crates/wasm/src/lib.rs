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

pub mod simd;
#[cfg(feature = "webgpu")]
pub mod webgpu;

use nanoinfer_core::kernels::{reference, Kernels, Rows};
use nanoinfer_core::stages::{blocks, kv_cache, Activations, KvCache};
use nanoinfer_core::trace::{ForwardInfo, Op, Phase, Trace};
use nanoinfer_core::{sampler, Config, Model, NoTrace, Sampler, Session, Stage, Tokenizer};
use wasm_bindgen::prelude::*;

#[cfg(feature = "threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

/// The fastest kernel set this build has, used by the analysis passes
/// (attention maps, logit lens), which are prefill-shaped.
#[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
type Fast = simd::WasmFma;
#[cfg(not(all(target_arch = "wasm32", target_feature = "relaxed-simd")))]
type Fast = simd::WasmSimd;

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

/// Stages that exist only in the browser build: kernel sets from [`simd`] that
/// plug into core's KV-cache forward pass. Never part of core's `Stage::ALL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BrowserStage {
    Simd,
    #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
    Fma,
    #[cfg(feature = "threads")]
    Threads,
}

impl BrowserStage {
    const ALL: &'static [BrowserStage] = &[
        BrowserStage::Simd,
        #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
        BrowserStage::Fma,
        #[cfg(feature = "threads")]
        BrowserStage::Threads,
    ];
    fn name(self) -> &'static str {
        match self {
            BrowserStage::Simd => simd::Plain::NAME,
            #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
            BrowserStage::Fma => simd::Relaxed::NAME,
            #[cfg(feature = "threads")]
            BrowserStage::Threads => "wasm-threads",
        }
    }
    fn note(self) -> &'static str {
        match self {
            BrowserStage::Simd => "browser rung: explicit simd128 matmul, 4 weight rows per pass, 4x4 register tile for prefill",
            #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
            BrowserStage::Fma => "browser rung: same kernel with relaxed-SIMD fused multiply-add",
            #[cfg(feature = "threads")]
            BrowserStage::Threads => "browser rung: wasm-fma's matmuls split across a pool of Web Workers (wasm-bindgen-rayon)",
        }
    }
}

use simd::Madd as _;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnyStage {
    Core(Stage),
    Browser(BrowserStage),
}

fn parse_stage(s: &str) -> Result<AnyStage, JsError> {
    if let Some(b) = BrowserStage::ALL.iter().find(|b| b.name() == s) {
        return Ok(AnyStage::Browser(*b));
    }
    let st: Stage = s.parse().map_err(err)?;
    if !runnable(st) {
        return Err(err(format!("stage {s} can't run in this wasm build")));
    }
    Ok(AnyStage::Core(st))
}

/// Threads core's multithreaded stages get in this build (always 1: core's
/// `threads` feature is never enabled for wasm, see Cargo.toml).
fn n_threads() -> usize {
    nanoinfer_core::kernels::threaded::num_threads()
}

/// What a core stage actually runs when compiled to wasm, if that differs from
/// native (empty when it's the same code).
fn wasm_note(s: Stage) -> String {
    // How the weight format's inner loop runs without NEON (kernels/i8dot.rs, kernels/f16.rs).
    let fmt = |s: Stage| match s.weight_format().name() {
        "f16" => "f16 weights are widened with simd128 integer ops (no fcvtl)".to_string(),
        f => format!("{f} decode uses simd128 int8 dots (no sdot)"),
    };
    match s {
        Stage::Neon => "no NEON in wasm: runs the unrolled kernels".into(),
        Stage::Tiled => "no NEON in wasm: the 4x4 tile is scalar code".into(),
        s if s.multithreaded() && n_threads() <= 1 && s.weight_format().name() != "f32" => {
            format!("no OS threads in wasm: runs single-threaded; {}", fmt(s))
        }
        s if s.multithreaded() && n_threads() <= 1 => "no OS threads in wasm: runs single-threaded".into(),
        s if s.weight_format().name() != "f32" => fmt(s),
        _ => String::new(),
    }
}

/// Every core stage runs in wasm: core's `threads` feature (OS threads) is
/// never enabled here, so its multithreaded stages degrade to one thread.
fn runnable(_s: Stage) -> bool {
    true
}

/// Which build this is: "simd128", "relaxed" (relaxed-SIMD) or "threads".
#[wasm_bindgen]
pub fn build_name() -> String {
    if cfg!(feature = "threads") {
        "threads".into()
    } else if cfg!(all(target_arch = "wasm32", target_feature = "relaxed-simd")) {
        "relaxed".into()
    } else {
        "simd128".into()
    }
}

/// Workers in the `wasm-threads` pool (1 unless this is the threads build and
/// `initThreadPool` has run).
#[wasm_bindgen]
pub fn thread_count() -> usize {
    #[cfg(feature = "threads")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "threads"))]
    {
        1
    }
}

/// Every stage this build can run: core's `Stage::ALL` (so new hillclimb stages
/// appear automatically) followed by the browser-only rungs, as
/// `[{ name, kernels, index, weightFormat, bitsPerWeight, browserOnly, wasmNote }]`.
#[wasm_bindgen]
pub fn stages() -> js_sys::Array {
    let out = js_sys::Array::new();
    let set = |o: &js_sys::Object, k: &str, v: JsValue| {
        let _ = js_sys::Reflect::set(o, &k.into(), &v);
    };
    for (i, s) in Stage::ALL.iter().enumerate().filter(|(_, s)| runnable(**s)) {
        let o = js_sys::Object::new();
        set(&o, "name", s.name().into());
        set(&o, "kernels", s.kernels().into());
        set(&o, "index", (i as u32).into());
        let wf = s.weight_format();
        set(&o, "weightFormat", wf.name().into());
        set(&o, "bitsPerWeight", wf.bits_per_weight().into());
        set(&o, "browserOnly", false.into());
        set(&o, "wasmNote", wasm_note(*s).into());
        out.push(&o);
    }
    for (i, b) in BrowserStage::ALL.iter().enumerate() {
        let o = js_sys::Object::new();
        set(&o, "name", b.name().into());
        set(&o, "kernels", b.name().into());
        set(&o, "index", ((Stage::ALL.len() + i) as u32).into());
        set(&o, "weightFormat", "f32".into());
        set(&o, "bitsPerWeight", 32.0.into());
        set(&o, "browserOnly", true.into());
        set(&o, "wasmNote", b.note().into());
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
///
/// [`Bytes::from_f16`] makes a buffer that accepts an **f16** safetensors file
/// (half the download) and widens it to the f32 file core expects while it
/// streams in: the header is rewritten (dtype F32, offsets doubled) and every
/// half is converted as it arrives, so the heap never holds both copies.
#[wasm_bindgen]
pub struct Bytes {
    buf: Vec<u8>,
    f16: Option<F16Stream>,
}

enum F16Stream {
    /// collecting the 8-byte header length, then the JSON header
    Header(Vec<u8>),
    /// converting data; holds an odd trailing byte between chunks
    Data(Option<u8>),
}

/// IEEE half -> single (handles subnormals, inf, NaN).
#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // subnormal: renormalise
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// Rewrite an all-F16 safetensors header as the equivalent F32 header.
fn widen_header(json: &[u8]) -> Result<Vec<u8>, String> {
    let mut v: serde_json::Value = serde_json::from_slice(json).map_err(|e| e.to_string())?;
    let obj = v.as_object_mut().ok_or("header is not an object")?;
    for (name, t) in obj.iter_mut() {
        if name == "__metadata__" {
            continue;
        }
        let dtype = t["dtype"].as_str().unwrap_or("");
        if dtype != "F16" {
            return Err(format!("{name}: expected F16, got {dtype}"));
        }
        t["dtype"] = "F32".into();
        let offs = t["data_offsets"].as_array().ok_or("missing data_offsets")?;
        let (a, b) = (offs[0].as_u64().unwrap_or(0) * 2, offs[1].as_u64().unwrap_or(0) * 2);
        t["data_offsets"] = serde_json::json!([a, b]);
    }
    let mut out = serde_json::to_vec(&v).map_err(|e| e.to_string())?;
    while out.len() % 8 != 0 {
        out.push(b' ');
    }
    Ok(out)
}

#[wasm_bindgen]
impl Bytes {
    #[wasm_bindgen(constructor)]
    pub fn new(capacity: usize) -> Bytes {
        Bytes { buf: Vec::with_capacity(capacity), f16: None }
    }
    /// A buffer for an f16 safetensors file of `f16_size` bytes; reserves the
    /// size of the widened f32 file.
    pub fn from_f16(f16_size: usize) -> Bytes {
        Bytes { buf: Vec::with_capacity(f16_size * 2 + 4096), f16: Some(F16Stream::Header(Vec::new())) }
    }
    pub fn push(&mut self, chunk: &[u8]) -> Result<(), JsError> {
        let Some(st) = self.f16.as_mut() else {
            self.buf.extend_from_slice(chunk);
            return Ok(());
        };
        let mut chunk = chunk;
        if let F16Stream::Header(h) = st {
            h.extend_from_slice(chunk);
            if h.len() < 8 {
                return Ok(());
            }
            let n = u64::from_le_bytes(h[..8].try_into().unwrap()) as usize;
            if h.len() < 8 + n {
                return Ok(());
            }
            let header = widen_header(&h[8..8 + n]).map_err(err)?;
            let rest = h[8 + n..].to_vec();
            self.buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
            self.buf.extend_from_slice(&header);
            *st = F16Stream::Data(None);
            self.push_halves(&rest);
            return Ok(());
        }
        if let Some(F16Stream::Data(Some(b0))) = self.f16 {
            if let Some((&b1, tail)) = chunk.split_first() {
                self.buf.extend_from_slice(&f16_to_f32(u16::from_le_bytes([b0, b1])).to_le_bytes());
                self.f16 = Some(F16Stream::Data(None));
                chunk = tail;
            }
        }
        self.push_halves(chunk);
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.buf.len()
    }
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

impl Bytes {
    fn push_halves(&mut self, data: &[u8]) {
        let pairs = data.chunks_exact(2);
        let odd = pairs.remainder().first().copied();
        self.buf.reserve(data.len() * 2);
        for p in pairs {
            self.buf.extend_from_slice(&f16_to_f32(u16::from_le_bytes([p[0], p[1]])).to_le_bytes());
        }
        self.f16 = Some(F16Stream::Data(odd));
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
    /// more let the page race stages against each other token by token.
    slots: Vec<Option<Slot>>,
    sink: Sink,
    sampler: Sampler,
}

struct Slot {
    runner: Runner,
    logits: Vec<f32>,
}

/// A conversation on either a core stage (core's own `Session`) or a
/// browser-only stage (core's KV-cache forward pass with a wasm kernel set).
enum Runner {
    Core(Session),
    Browser(BrowserSession),
}

struct BrowserSession {
    stage: BrowserStage,
    tokens: Vec<u32>,
    max_seq: usize,
    kv: KvCache,
    acts: Activations,
}

impl BrowserSession {
    fn new(model: &Model, stage: BrowserStage, max_seq: usize) -> Self {
        let max_seq = max_seq.min(model.config.n_ctx);
        BrowserSession { stage, tokens: Vec::new(), max_seq, kv: KvCache::new(model, max_seq), acts: Activations::default() }
    }
    /// Mirrors `Session::step` for the KV-cache stages.
    fn step<T: Trace>(&mut self, model: &Model, new: &[u32], trace: &mut T) -> Result<&[f32], JsError> {
        if new.is_empty() {
            return Err(err("step() needs at least one token"));
        }
        if let Some(&t) = new.iter().find(|&&t| t as usize >= model.config.n_vocab) {
            return Err(err(format!("bad token {t}")));
        }
        let len = self.tokens.len() + new.len();
        if len > self.max_seq {
            return Err(err(format!("context full: {len} > {}", self.max_seq)));
        }
        let phase = if self.tokens.is_empty() || new.len() > 1 { Phase::Prefill } else { Phase::Decode };
        self.tokens.extend_from_slice(new);
        // ForwardInfo wants a core stage; the sink ignores it. Tiled is the
        // closest relative (same forward pass, a different matmul).
        let info = ForwardInfo { stage: Stage::Tiled, phase, n_new: new.len(), n_computed: new.len(), context_len: len };
        trace.forward_start(&info);
        match self.stage {
            BrowserStage::Simd => kv_cache::forward::<simd::WasmSimd, T>(model, new, &mut self.kv, &mut self.acts, trace),
            #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
            BrowserStage::Fma => kv_cache::forward::<simd::WasmFma, T>(model, new, &mut self.kv, &mut self.acts, trace),
            #[cfg(feature = "threads")]
            BrowserStage::Threads => kv_cache::forward::<simd::Par<simd::Relaxed>, T>(model, new, &mut self.kv, &mut self.acts, trace),
        }
        trace.forward_end(&info);
        Ok(&self.acts.logits)
    }
}

impl Runner {
    fn new(model: &Model, stage: AnyStage, max_seq: usize) -> Runner {
        match stage {
            AnyStage::Core(s) => Runner::Core(Session::with_max_seq(model, s, max_seq)),
            AnyStage::Browser(b) => Runner::Browser(BrowserSession::new(model, b, max_seq)),
        }
    }
    fn stage(&self) -> AnyStage {
        match self {
            Runner::Core(s) => AnyStage::Core(s.stage()),
            Runner::Browser(b) => AnyStage::Browser(b.stage),
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Runner::Core(s) => s.stage().name(),
            Runner::Browser(b) => b.stage.name(),
        }
    }
    fn max_seq(&self) -> usize {
        match self {
            Runner::Core(s) => s.max_seq(),
            Runner::Browser(b) => b.max_seq,
        }
    }
    fn reset(&mut self) {
        match self {
            Runner::Core(s) => s.reset(),
            Runner::Browser(b) => {
                b.tokens.clear();
                b.kv.len = 0;
            }
        }
    }
    fn tokens(&self) -> &[u32] {
        match self {
            Runner::Core(s) => s.tokens(),
            Runner::Browser(b) => &b.tokens,
        }
    }
    fn step<T: Trace>(&mut self, model: &Model, new: &[u32], trace: &mut T) -> Result<&[f32], JsError> {
        match self {
            Runner::Core(s) => s.step(model, new, trace).map_err(err),
            Runner::Browser(b) => b.step(model, new, trace),
        }
    }
}

/// Softmax summary of one logits row: (max, log Σ exp(l - max), entropy in nats).
fn softmax_stats(row: &[f32]) -> (f32, f64, f64) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0.0f64;
    let mut zl = 0.0f64;
    for &l in row {
        let d = (l - max) as f64;
        let e = d.exp();
        z += e;
        zl += e * d;
    }
    let lz = z.ln();
    // H = -Σ p log p = lz - Σ p·d
    (max, lz, lz - zl / z)
}

#[wasm_bindgen]
impl Engine {
    /// Build an engine from the bytes of HF `model.safetensors`, `vocab.json`
    /// and `merges.txt`. Consumes `weights` (its memory is reused afterwards).
    #[wasm_bindgen(constructor)]
    pub fn new(weights: Bytes, vocab: &[u8], merges: &[u8], stage: &str) -> Result<Engine, JsError> {
        if matches!(weights.f16, Some(F16Stream::Header(_))) {
            return Err(err("f16 checkpoint ended inside its header"));
        }
        let model = Model::from_safetensors(&weights.buf).map_err(err)?;
        drop(weights);
        let tok = Tokenizer::from_vocab_and_merges(vocab, merges).map_err(err)?;
        let stage = parse_stage(stage)?;
        let runner = Runner::new(&model, stage, model.config.n_ctx);
        Ok(Engine {
            model,
            tok,
            slots: vec![Some(Slot { runner, logits: Vec::new() })],
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
        let stage = parse_stage(stage)?;
        let max_seq = if max_seq == 0 { self.model.config.n_ctx } else { max_seq };
        let slot = Slot { runner: Runner::new(&self.model, stage, max_seq), logits: Vec::new() };
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
        Ok(self.slot(slot)?.runner.name().to_string())
    }

    /// Restart session `slot` on `stage` (keeps its max_seq).
    pub fn set_stage(&mut self, slot: u32, stage: &str) -> Result<(), JsError> {
        let stage = parse_stage(stage)?;
        let model = &self.model;
        let s = self.slots.get_mut(slot as usize).and_then(|s| s.as_mut()).ok_or_else(|| err("no session"))?;
        if stage != s.runner.stage() {
            s.runner = Runner::new(model, stage, s.runner.max_seq());
        } else {
            s.runner.reset();
        }
        s.logits.clear();
        Ok(())
    }

    pub fn reset(&mut self, slot: u32) -> Result<(), JsError> {
        let s = self.slot(slot)?;
        s.runner.reset();
        s.logits.clear();
        Ok(())
    }

    pub fn tokens(&mut self, slot: u32) -> Result<Vec<u32>, JsError> {
        Ok(self.slot(slot)?.runner.tokens().to_vec())
    }

    pub fn max_seq(&mut self, slot: u32) -> Result<usize, JsError> {
        Ok(self.slot(slot)?.runner.max_seq())
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
        let logits = s.runner.step(model, tokens, &mut self.sink)?;
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

    /// The raw logits of session `slot`'s last forward (all `n_vocab` of them),
    /// so the page can re-sample them at any temperature / top-k itself.
    pub fn logits(&mut self, slot: u32) -> Result<Vec<f32>, JsError> {
        Ok(self.slot(slot)?.logits.clone())
    }

    /// Analysis pass: run `tokens` through the network (core's own blocks and
    /// reference kernels) and capture every head's full causal attention
    /// matrix. Returns `[n_layer, n_head, T, T]` row-major, where row `i` is
    /// query `i` and only `j <= i` is non-zero. Independent of the session.
    pub fn attention_maps(&self, tokens: &[u32]) -> Result<Vec<f32>, JsError> {
        let cfg = self.model.config;
        let n = self.check_tokens(tokens)?;
        let (c, nh, hd) = (cfg.n_embd, cfg.n_head, cfg.head_dim());
        let mut out = vec![0.0f32; cfg.n_layer * nh * n * n];
        let mut row = vec![0.0f32; n];
        self.analysis_pass(tokens, |l, qkv, _x| {
            if let Some(qkv) = qkv {
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
        });
        Ok(out)
    }

    /// The residual stream at position `pos`, after the embedding and after
    /// every block: `[n_layer + 1, n_embd]`. This is the vector that "is" the
    /// token as it flows up the network.
    pub fn residual_stream(&self, tokens: &[u32], pos: usize) -> Result<Vec<f32>, JsError> {
        let cfg = self.model.config;
        let n = self.check_tokens(tokens)?;
        if pos >= n {
            return Err(err("residual_stream: pos out of range"));
        }
        let c = cfg.n_embd;
        let mut out = Vec::with_capacity((cfg.n_layer + 1) * c);
        self.analysis_pass(tokens, |_l, qkv, x| {
            if qkv.is_none() {
                out.extend_from_slice(&x[pos * c..(pos + 1) * c]);
            }
        });
        Ok(out)
    }

    /// The logit lens: decode the residual stream *after every layer* with the
    /// model's own final LayerNorm + LM head, as if the network stopped there.
    /// Shows what the model "would say" at each depth.
    ///
    /// For each depth `s` in `0..=n_layer` (0 = straight after the embedding)
    /// and each requested position, returns `2*top_k + 4` floats:
    /// ```text
    /// ids[top_k], probs[top_k], entropy_nats,
    /// p(final layer's top-1 token), p(the token that actually came next, -1 if last),
    /// ‖residual‖₂
    /// ```
    /// laid out `[depth][position][...]`.
    pub fn logit_lens(&self, tokens: &[u32], positions: &[u32], top_k: usize) -> Result<Vec<f32>, JsError> {
        let cfg = self.model.config;
        let n = self.check_tokens(tokens)?;
        if positions.is_empty() || positions.iter().any(|&p| p as usize >= n) {
            return Err(err("logit_lens: positions must be in 0..tokens.len()"));
        }
        let (c, v, depth) = (cfg.n_embd, cfg.n_vocab, cfg.n_layer + 1);
        let np = positions.len();
        let k = top_k.clamp(1, 32);
        // residual rows for the requested positions, after embed and every block
        let mut saved = vec![0.0f32; depth * np * c];
        let mut s_i = 0;
        self.analysis_pass(tokens, |_l, qkv, x| {
            if qkv.is_none() {
                for (pi, &p) in positions.iter().enumerate() {
                    let p = p as usize;
                    saved[(s_i * np + pi) * c..][..c].copy_from_slice(&x[p * c..(p + 1) * c]);
                }
                s_i += 1;
            }
        });
        let stride = 2 * k + 4;
        let mut out = vec![0.0f32; depth * np * stride];
        let mut xn = vec![0.0f32; np * c];
        let mut logits = vec![0.0f32; np * v];
        let mut final_top = vec![0u32; np];
        for s in (0..depth).rev() {
            let xs = &saved[s * np * c..(s + 1) * np * c];
            Fast::layernorm(&mut xn, xs, &self.model.ln_f_w, &self.model.ln_f_b, np, c, Config::LN_EPS);
            Fast::matmul(&mut logits, &xn, &self.model.wte, None, np, c, v);
            for pi in 0..np {
                let row = &logits[pi * v..(pi + 1) * v];
                let (max, lz, h) = softmax_stats(row);
                let p_of = |id: u32| ((row[id as usize] - max) as f64 - lz).exp() as f32;
                let top = sampler::top_k_probs(row, k);
                if s == depth - 1 {
                    final_top[pi] = top[0].0;
                }
                let o = &mut out[(s * np + pi) * stride..][..stride];
                for (j, &(id, p)) in top.iter().enumerate() {
                    o[j] = id as f32;
                    o[k + j] = p;
                }
                o[2 * k] = h as f32;
                o[2 * k + 1] = p_of(final_top[pi]);
                let pos = positions[pi] as usize;
                o[2 * k + 2] = if pos + 1 < n { p_of(tokens[pos + 1]) } else { -1.0 };
                let xr = &xs[pi * c..(pi + 1) * c];
                o[2 * k + 3] = xr.iter().map(|a| a * a).sum::<f32>().sqrt();
            }
        }
        Ok(out)
    }
}

impl Engine {
    fn check_tokens(&self, tokens: &[u32]) -> Result<usize, JsError> {
        let cfg = &self.model.config;
        let n = tokens.len();
        if n == 0 || n > cfg.n_ctx {
            return Err(err(format!("analysis: need 1..={} tokens", cfg.n_ctx)));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= cfg.n_vocab) {
            return Err(err(format!("bad token {t}")));
        }
        Ok(n)
    }

    /// One prefill-shaped pass over `tokens`, built from core's blocks with the
    /// fastest kernels this build has. Calls `visit(layer, Some(qkv), x)` after
    /// each layer's QKV projection and `visit(layer, None, x)` with the residual
    /// stream after the embedding (layer 0) and after each block (layer l+1).
    fn analysis_pass(&self, tokens: &[u32], mut visit: impl FnMut(usize, Option<&[f32]>, &[f32])) {
        let model = &self.model;
        let cfg = model.config;
        let n = tokens.len();
        let c = cfg.n_embd;
        let mut acts = Activations::default();
        acts.ensure(model, n);
        let mut tr = NoTrace;
        blocks::embed::<Fast, NoTrace>(model, tokens, 0, &mut acts, &mut tr);
        visit(0, None, &acts.x[..n * c]);
        for (l, layer) in model.layers.iter().enumerate() {
            blocks::qkv::<Fast, NoTrace>(&cfg, l, layer, n, &mut acts, &mut tr);
            let qkv = acts.qkv[..n * 3 * c].to_vec();
            visit(l, Some(&qkv), &acts.x[..n * c]);
            let k = Rows { data: &qkv[c..], stride: 3 * c };
            let v = Rows { data: &qkv[2 * c..], stride: 3 * c };
            blocks::attention::<Fast, NoTrace>(&cfg, l, n, 0, k, v, &mut acts.att, &qkv, &mut acts.scores, &mut tr);
            blocks::proj_and_mlp::<Fast, NoTrace>(&cfg, l, layer, n, &mut acts, &mut tr);
            visit(l + 1, None, &acts.x[..n * c]);
        }
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

/// Peak single-thread compute in GFLOP/s: 12 independent SIMD multiply-add
/// chains held in registers (no memory traffic). The "factory" roof. Uses the
/// best multiply-add this build has: separate `mul` + `add` in the simd128
/// build, fused `relaxed_madd` in the relaxed/threads builds.
#[wasm_bindgen]
pub fn bench_peak_gflops(millions: usize) -> f64 {
    let iters = millions * 1_000_000 / 96;
    let t = perf_now();
    let r = peak_loop(iters);
    let dt = perf_now() - t;
    std::hint::black_box(r);
    // per iter: 12 chains x 4 lanes x (mul + add) = 96 flops
    (iters as f64 * 96.0) / (dt / 1e3) / 1e9
}

/// Which multiply-add [`bench_peak_gflops`] measured: "mul+add" or "fma".
#[wasm_bindgen]
pub fn peak_kind() -> String {
    if cfg!(all(target_arch = "wasm32", target_feature = "relaxed-simd")) { "fma".into() } else { "mul+add".into() }
}

#[cfg(target_arch = "wasm32")]
fn peak_loop(iters: usize) -> f32 {
    #[cfg(target_feature = "relaxed-simd")]
    type M = simd::Relaxed;
    #[cfg(not(target_feature = "relaxed-simd"))]
    type M = simd::Plain;
    // An earlier version called black_box(&a) every iteration, which forced
    // the accumulators through memory and read ~25% low (decode measured
    // *above* that "peak"). The chains are loop-carried, so nothing can fold.
    let m = std::hint::black_box(f32x4_splat(0.999_999));
    let b = std::hint::black_box(f32x4_splat(1e-7));
    let mut a = [f32x4_splat(1.0); 12];
    for _ in 0..iters {
        for v in a.iter_mut() {
            *v = <M as simd::Madd>::madd(*v, m, b);
        }
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
