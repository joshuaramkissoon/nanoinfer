//! The WebGPU rungs: `nanoinfer-webgpu` (wgpu on the browser's WebGPU) behind
//! free functions, because everything GPU is async and wasm-bindgen can't
//! hold `&mut Engine` across an await. Sessions live in a registry keyed by
//! slot id; weights are uploaded once per dtype from the CPU engine's
//! already-loaded `Model` and shared by every stage that uses that dtype.
//!
//! Traces use the same packed layout as [`crate::Engine::forward`], with
//! their own name/kind tables ([`webgpu_op_names`], [`webgpu_op_kinds`]).
//! Stages that wait per op report real per-op wall times; one-submit stages
//! report one `gpu` op for the whole forward (WebGPU has no cheap way to time
//! a single dispatch inside a pass).

use crate::{err, perf_now, Engine, HEADER_LEN, TRACE_VERSION};
use nanoinfer_core::sampler;
use nanoinfer_webgpu::weights::Weights;
use nanoinfer_webgpu::{probe, DType, Engine as GpuEngine, Gpu, Session, Sync, WgpuStage};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use nanoinfer_webgpu::wgpu;
use wasm_bindgen::prelude::*;

struct Slot {
    s: Session,
    logits: Vec<f32>,
}

thread_local! {
    static GPU: RefCell<Option<Rc<Gpu>>> = const { RefCell::new(None) };
    static WEIGHTS: RefCell<HashMap<(DType, DType, bool), Rc<Weights>>> = RefCell::new(HashMap::new());
    static ENGINES: RefCell<HashMap<WgpuStage, Rc<GpuEngine>>> = RefCell::new(HashMap::new());
    static SLOTS: RefCell<HashMap<u32, Slot>> = RefCell::new(HashMap::new());
    static NEXT: Cell<u32> = const { Cell::new(1) };
    static NAMES: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
    static KINDS: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

fn gpu() -> Result<Rc<Gpu>, JsError> {
    GPU.with(|g| g.borrow().clone()).ok_or_else(|| err("WebGPU not initialised (call webgpu_init)"))
}

fn set(o: &js_sys::Object, k: &str, v: JsValue) {
    let _ = js_sys::Reflect::set(o, &k.into(), &v);
}

/// Request an adapter + device. Resolves to `{adapter, backend, features[],
/// maxStorageBufferBindingSize, maxBufferSize, maxStorageBuffersPerShaderStage}`
/// or rejects when this browser has no usable WebGPU.
#[wasm_bindgen]
pub async fn webgpu_init() -> Result<JsValue, JsError> {
    if GPU.with(|g| g.borrow().is_none()) {
        let g = Gpu::new().await.map_err(err)?;
        GPU.with(|s| *s.borrow_mut() = Some(Rc::new(g)));
    }
    let g = gpu()?;
    let o = js_sys::Object::new();
    set(&o, "adapter", g.info.name.clone().into());
    set(&o, "backend", format!("{:?}", g.info.backend).into());
    let feats = js_sys::Array::new();
    for (f, n) in [
        (wgpu::Features::SHADER_F16, "shader-f16"),
        (wgpu::Features::TIMESTAMP_QUERY, "timestamp-query"),
        (wgpu::Features::SUBGROUP, "subgroups"),
    ] {
        if g.has(f) {
            feats.push(&n.into());
        }
    }
    set(&o, "features", feats.into());
    set(&o, "maxStorageBufferBindingSize", (g.limits.max_storage_buffer_binding_size as f64).into());
    set(&o, "maxBufferSize", (g.limits.max_buffer_size as f64).into());
    set(&o, "maxStorageBuffersPerShaderStage", g.limits.max_storage_buffers_per_shader_stage.into());
    Ok(o.into())
}

/// The WebGPU stages, as StageInfo objects (same shape as `stages()`).
#[wasm_bindgen]
pub fn webgpu_stages() -> js_sys::Array {
    let out = js_sys::Array::new();
    let g = GPU.with(|g| g.borrow().clone());
    for (i, s) in WgpuStage::ALL.iter().enumerate() {
        let supported = g.as_ref().is_some_and(|g| s.supported(g));
        let o = js_sys::Object::new();
        let r = s.recipe();
        set(&o, "name", s.name().into());
        set(&o, "kernels", s.kernels().into());
        set(&o, "index", (1000 + i as u32).into());
        set(&o, "weightFormat", r.dtype.name().into());
        set(&o, "bitsPerWeight", (r.dtype.bytes_per_weight() * 8.0).into());
        set(&o, "headFormat", r.head_dtype.name().into());
        set(&o, "browserOnly", true.into());
        set(&o, "backend", "webgpu".into());
        set(&o, "supported", supported.into());
        let note = if supported { String::new() } else { "needs WebGPU subgroups, which wgpu doesn't expose in browsers yet".to_string() };
        set(&o, "wasmNote", note.into());
        out.push(&o);
    }
    out
}

fn engine_for(engine: &Engine, stage: WgpuStage) -> Result<Rc<GpuEngine>, JsError> {
    if let Some(e) = ENGINES.with(|m| m.borrow().get(&stage).cloned()) {
        return Ok(e);
    }
    let g = gpu()?;
    if !stage.supported(&g) {
        return Err(err(format!("{stage} needs a WebGPU feature this browser doesn't expose")));
    }
    let r = stage.recipe();
    let key = (r.dtype, r.head_dtype, r.ln_fold);
    let w = match WEIGHTS.with(|m| m.borrow().get(&key).cloned()) {
        Some(w) => w,
        None => {
            let w = Rc::new(Weights::upload(&g, &engine.model, r.dtype, r.head_dtype, r.ln_fold));
            WEIGHTS.with(|m| m.borrow_mut().insert(key, w.clone()));
            w
        }
    };
    let e = Rc::new(GpuEngine::with_weights(g, engine.model.config, stage, w));
    ENGINES.with(|m| m.borrow_mut().insert(stage, e.clone()));
    Ok(e)
}

/// Open a session on a WebGPU stage (uploads that dtype's weights on first
/// use, from the CPU engine's model). Returns its slot id.
#[wasm_bindgen]
pub fn webgpu_open(engine: &Engine, stage: &str, max_seq: usize) -> Result<u32, JsError> {
    let stage: WgpuStage = stage.parse().map_err(err)?;
    let e = engine_for(engine, stage)?;
    let s = e.session(max_seq);
    let id = NEXT.with(|n| {
        let id = n.get();
        n.set(id + 1);
        id
    });
    SLOTS.with(|m| m.borrow_mut().insert(id, Slot { s, logits: vec![] }));
    Ok(id)
}

#[wasm_bindgen]
pub fn webgpu_close(slot: u32) {
    SLOTS.with(|m| m.borrow_mut().remove(&slot));
}

/// Drop uploaded weights no open session uses (frees GPU memory).
#[wasm_bindgen]
pub fn webgpu_trim() {
    let used: Vec<WgpuStage> = SLOTS.with(|m| m.borrow().values().map(|s| s.s.e.stage).collect());
    ENGINES.with(|m| m.borrow_mut().retain(|k, _| used.contains(k)));
    WEIGHTS.with(|w| w.borrow_mut().retain(|_, v| Rc::strong_count(v) > 1));
}

#[wasm_bindgen]
pub fn webgpu_reset(slot: u32) -> Result<(), JsError> {
    SLOTS.with(|m| {
        let mut m = m.borrow_mut();
        let s = m.get_mut(&slot).ok_or_else(|| err(format!("no webgpu session {slot}")))?;
        s.s.reset();
        s.logits.clear();
        Ok(())
    })
}

#[wasm_bindgen]
pub fn webgpu_stage(slot: u32) -> Result<String, JsError> {
    SLOTS.with(|m| m.borrow().get(&slot).map(|s| s.s.e.stage.name().to_string()).ok_or_else(|| err("no webgpu session")))
}

fn take(slot: u32) -> Result<Slot, JsError> {
    SLOTS.with(|m| m.borrow_mut().remove(&slot)).ok_or_else(|| err(format!("no webgpu session {slot}")))
}
fn put(slot: u32, s: Slot) {
    SLOTS.with(|m| m.borrow_mut().insert(slot, s));
}

fn intern(table: &'static std::thread::LocalKey<RefCell<Vec<&'static str>>>, s: &'static str) -> f64 {
    table.with(|t| {
        let mut t = t.borrow_mut();
        match t.iter().position(|&x| x == s) {
            Some(i) => i as f64,
            None => {
                t.push(s);
                (t.len() - 1) as f64
            }
        }
    })
}

fn kind_of(name: &str) -> &'static str {
    match name {
        "embed" => "embed",
        "ln_1" | "ln_2" | "ln_f" => "ln",
        "qkv" | "attn_proj" | "fc" | "fc_proj" => "matmul",
        "attn" => "attn",
        "gelu" => "gelu",
        "residual" | "residual2" => "residual",
        "kv_write" => "kv_write",
        "lm_head" => "lm_head",
        _ => "gpu",
    }
}

#[wasm_bindgen]
pub fn webgpu_op_names() -> js_sys::Array {
    NAMES.with(|t| t.borrow().iter().map(|&s| JsValue::from_str(s)).collect())
}
#[wasm_bindgen]
pub fn webgpu_op_kinds() -> js_sys::Array {
    KINDS.with(|t| t.borrow().iter().map(|&s| JsValue::from_str(s)).collect())
}

/// Append `tokens` and run one forward on the GPU; resolves to a packed
/// trace (same layout as `Engine.forward`). The logits are read back (for
/// top-k, sampling and the page's figures).
#[wasm_bindgen]
pub async fn webgpu_forward(slot: u32, tokens: Vec<u32>, top_k: usize) -> Result<Vec<f64>, JsError> {
    let mut sl = take(slot)?;
    let per_op = sl.s.e.recipe.sync == Sync::PerOp;
    sl.s.profile = per_op;
    let pos0 = sl.s.len();
    let t0 = perf_now();
    let res = sl.s.step(&tokens, &perf_now).await;
    let ms = perf_now() - t0;
    let logits = match res {
        Ok(l) => l,
        Err(e) => {
            put(slot, sl);
            return Err(err(e));
        }
    };
    sl.logits = logits;
    let top = if top_k > 0 { sampler::top_k_probs(&sl.logits, top_k) } else { vec![] };
    let ent = if top_k > 0 { crate::entropy(&sl.logits) } else { 0.0 };
    let cfg = sl.s.e.config;
    let n = tokens.len();
    let mut ops: Vec<f64> = vec![];
    let mut op = |name: &'static str, layer: i32, dims: [f64; 3], flops: f64, bytes: f64, start: f64, dur: f64| {
        ops.extend_from_slice(&[intern(&KINDS, kind_of(name)), intern(&NAMES, name), layer as f64, dims[0], dims[1], dims[2], flops, bytes, start, dur]);
    };
    if per_op {
        let c = cfg.n_embd as f64;
        let bpw = sl.s.e.recipe.dtype.bytes_per_weight();
        let mut t = 0.0;
        for &(name, layer, d) in &sl.s.last.op_ms {
            let (rows, cols) = match name {
                "qkv" => (3.0 * c, c),
                "attn_proj" => (c, c),
                "fc" => (4.0 * c, c),
                "fc_proj" => (c, 4.0 * c),
                "lm_head" => (cfg.n_vocab as f64 / 2.0, c),
                _ => (0.0, 0.0),
            };
            let rows_n = if name == "lm_head" { 1.0 } else { n as f64 };
            op(name, layer, [rows_n, rows, cols], 2.0 * rows_n * rows * cols, rows * cols * bpw, t, d);
            t += d;
        }
    } else {
        let w = sl.s.e.stage.decode_weight_bytes(&cfg);
        let flops = 2.0 * (cfg.n_layer * 12 * cfg.n_embd * cfg.n_embd) as f64 * n as f64 + 2.0 * (cfg.n_vocab * cfg.n_embd) as f64;
        op("gpu", -1, [n as f64, sl.s.last.dispatches as f64, sl.s.last.submits as f64], flops, w, 0.0, ms);
    }
    let n_ops = ops.len() / crate::OP_STRIDE;
    let ctx = pos0 + n;
    let mut out = Vec::with_capacity(HEADER_LEN + top.len() * 2 + ops.len());
    out.extend_from_slice(&[
        TRACE_VERSION,
        n_ops as f64,
        top.len() as f64,
        0.0,
        if pos0 == 0 { 0.0 } else { 1.0 },
        n as f64,
        n as f64,
        ctx as f64,
        ctx as f64,
        sl.s.max_seq() as f64,
        ms,
        ent as f64,
        NAMES.with(|t| t.borrow().len()) as f64,
        KINDS.with(|t| t.borrow().len()) as f64,
        -1.0,
        -1.0,
    ]);
    for &(id, p) in &top {
        out.push(id as f64);
        out.push(p as f64);
    }
    out.extend_from_slice(&ops);
    put(slot, sl);
    Ok(out)
}

/// Sample from a WebGPU session's last logits with the CPU engine's sampler
/// (so seeds and settings behave exactly like the wasm stages).
#[wasm_bindgen]
pub fn webgpu_sample(engine: &mut Engine, slot: u32, temperature: f32, top_k: usize) -> Result<u32, JsError> {
    engine.sampler.temperature = temperature;
    engine.sampler.top_k = top_k;
    SLOTS.with(|m| {
        let m = m.borrow();
        let s = m.get(&slot).ok_or_else(|| err("no webgpu session"))?;
        if s.logits.is_empty() {
            return Err(err("sample() before forward()"));
        }
        Ok(engine.sampler.sample(&s.logits))
    })
}

#[wasm_bindgen]
pub fn webgpu_prob_of(slot: u32, id: u32) -> f32 {
    SLOTS.with(|m| {
        let m = m.borrow();
        let Some(s) = m.get(&slot) else { return 0.0 };
        let l = &s.logits;
        if l.is_empty() || id as usize >= l.len() {
            return 0.0;
        }
        let max = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = l.iter().map(|&v| (v - max).exp()).sum();
        (l[id as usize] - max).exp() / z
    })
}

#[wasm_bindgen]
pub fn webgpu_logits(slot: u32) -> Vec<f32> {
    SLOTS.with(|m| m.borrow().get(&slot).map(|s| s.logits.clone()).unwrap_or_default())
}

/// Greedy-decode `n` tokens after `first` with the stage's fastest path
/// (the GPU-resident loop where the recipe has one). Resolves to
/// `[total_ms, dispatches, submits, gpu_ms|NaN, ...tokens]`.
#[wasm_bindgen]
pub async fn webgpu_greedy(slot: u32, first: u32, n: usize) -> Result<Vec<f64>, JsError> {
    let mut sl = take(slot)?;
    sl.s.profile = false;
    let t0 = perf_now();
    let r = sl.s.decode_greedy(first, n, &perf_now).await;
    let ms = perf_now() - t0;
    let st = sl.s.last.clone();
    put(slot, sl);
    let toks = r.map_err(err)?;
    let mut out = vec![ms, st.dispatches as f64, st.submits as f64, st.gpu_ms.unwrap_or(f64::NAN)];
    out.extend(toks.iter().map(|&t| t as f64));
    Ok(out)
}

/// The Ladder's fixed-shape benchmark, run entirely here: prefill `prompt`
/// (greedy pick), then `gen` greedy decode steps on the stage's fastest path.
/// Resolves to `[prefill_ms, decode_ms, dispatches/token, submits/token,
/// gpu_ms/token|NaN]`. Resets the session first and after.
#[wasm_bindgen]
pub async fn webgpu_bench(slot: u32, prompt: Vec<u32>, gen: usize) -> Result<Vec<f64>, JsError> {
    let mut sl = take(slot)?;
    sl.s.profile = false;
    sl.s.reset();
    let t0 = perf_now();
    let first = sl.s.step_greedy(&prompt, &perf_now).await;
    let prefill = perf_now() - t0;
    let first = match first {
        Ok(f) => f,
        Err(e) => {
            put(slot, sl);
            return Err(err(e));
        }
    };
    let t0 = perf_now();
    let r = sl.s.decode_greedy(first, gen, &perf_now).await;
    let decode = perf_now() - t0;
    let st = sl.s.last.clone();
    sl.s.reset();
    put(slot, sl);
    r.map_err(err)?;
    let f = st.forwards.max(1) as f64;
    Ok(vec![prefill, decode, st.dispatches as f64 / f, st.submits as f64 / f, st.gpu_ms.map_or(f64::NAN, |g| g / f)])
}

/// Per-op steady-state times for one decode step: `[name, layer, ms]`
/// triples flattened into an array of objects. Resets the session.
#[wasm_bindgen]
pub async fn webgpu_profile(slot: u32, tokens: Vec<u32>, reps: usize) -> Result<js_sys::Array, JsError> {
    let mut sl = take(slot)?;
    let r = sl.s.profile_ops(&tokens, reps.max(1), &perf_now).await;
    sl.s.reset();
    sl.logits.clear();
    put(slot, sl);
    let out = js_sys::Array::new();
    for (name, layer, ms) in r.map_err(err)? {
        let o = js_sys::Object::new();
        set(&o, "name", name.into());
        set(&o, "layer", layer.into());
        set(&o, "ms", ms.into());
        out.push(&o);
    }
    Ok(out)
}

/// The rules of the game in this tab: dispatch, pass, submit and round-trip
/// costs, and GPU read bandwidth. Resolves to `[{label, value, unit}]`.
#[wasm_bindgen]
pub async fn webgpu_probe(reps: usize) -> Result<js_sys::Array, JsError> {
    let g = gpu()?;
    let r = probe::run(&g, &perf_now, reps.max(3)).await;
    let out = js_sys::Array::new();
    for (l, v, u) in r.rows {
        let o = js_sys::Object::new();
        set(&o, "label", l.into());
        set(&o, "value", v.into());
        set(&o, "unit", u.into());
        out.push(&o);
    }
    Ok(out)
}
