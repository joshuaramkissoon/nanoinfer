//! The GPT-2 forward pass as WebGPU dispatches.
//!
//! [`Session::plan`] walks the network exactly like core's `kv_cache` stage
//! and asks the [`Recipe`] which kernel each op uses, producing a list of
//! dispatches. [`Session::execute`] then either submits them one by one
//! (waiting for each: `webgpu-naive`) or encodes them all into one compute
//! pass and one submit. Bind groups and per-op parameters are created once
//! per (phase, op) and reused, so a steady-state token only records
//! `setPipeline` / `setBindGroup` / `dispatchWorkgroups`.

use crate::gpu::{Gpu, Kernel, Step, STEP_STRIDE};
use crate::weights::{Mat, Weights};
use crate::{DType, Matmul, Matvec, Recipe, Sync, WgpuStage};
use nanoinfer_core::{Config, Model};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

const COMMON: &str = include_str!("shaders/common.wgsl");
const MAT_COMMON: &str = include_str!("shaders/mat_common.wgsl");

fn w_src(d: DType) -> &'static str {
    match d {
        DType::F32 => include_str!("shaders/w_f32.wgsl"),
        DType::F16 => include_str!("shaders/w_f16.wgsl"),
        DType::Q8_0 => include_str!("shaders/w_q8.wgsl"),
    }
}
fn wdot_src(d: DType) -> &'static str {
    match d {
        DType::F32 => include_str!("shaders/wdot_f32.wgsl"),
        DType::F16 => include_str!("shaders/wdot_f16.wgsl"),
        DType::Q8_0 => include_str!("shaders/wdot_q8.wgsl"),
    }
}
/// Threads per row in the matvec: ~3 vector loads in flight per thread on a
/// 768-wide row (the Metal lesson), whatever the load width.
fn tpr(d: DType) -> u32 {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(t) = std::env::var("NANOINFER_WGPU_TPR").ok().and_then(|v| v.parse().ok()) {
        return t;
    }
    match d {
        DType::F32 => 64,
        DType::F16 | DType::Q8_0 => 32,
    }
}

/// Max forwards one submit can carry (one step-uniform slot each).
pub const MAX_STEPS: usize = 64;
/// Tokens per submit in the GPU-resident greedy loop.
pub const RESIDENT_CHUNK: usize = 16;
const ARGMAX_GROUPS: u32 = 64;
/// Keys per flash-decoding chunk (attention_fd.wgsl's CH).
const FD_CHUNK: usize = 64;
const PARAM_SLOTS: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MatKind {
    Matvec,
    MatvecSg,
    MatvecLn,
    Naive,
    Tiled,
}

/// GPU-resident model for one stage. Weights are shared (`Rc`) between
/// engines of the same dtype.
pub struct Engine {
    pub gpu: Rc<Gpu>,
    pub config: Config,
    pub stage: WgpuStage,
    pub recipe: Recipe,
    pub w: Rc<Weights>,
    kernels: RefCell<HashMap<String, Rc<Kernel>>>,
}

impl Engine {
    pub fn new(gpu: Rc<Gpu>, model: &Model, stage: WgpuStage) -> Engine {
        let r = stage.recipe();
        let w = Rc::new(Weights::upload(&gpu, model, r.dtype, r.head_dtype, r.ln_fold));
        Self::with_weights(gpu, model.config, stage, w)
    }

    /// Reuse already-uploaded weights (must match the stage's dtypes).
    pub fn with_weights(gpu: Rc<Gpu>, config: Config, stage: WgpuStage, w: Rc<Weights>) -> Engine {
        let recipe = stage.recipe();
        assert_eq!((w.dtype, w.head_dtype, w.ln_folded), (recipe.dtype, recipe.head_dtype, recipe.ln_fold), "weights dtype / folding mismatch");
        Engine { gpu, config, stage, recipe, w, kernels: RefCell::new(HashMap::new()) }
    }

    pub fn session(self: &Rc<Self>, max_seq: usize) -> Session {
        Session::new(self.clone(), max_seq)
    }

    fn cached(&self, key: String, make: impl FnOnce() -> Kernel) -> Rc<Kernel> {
        if let Some(k) = self.kernels.borrow().get(&key) {
            return k.clone();
        }
        let k = Rc::new(make());
        self.kernels.borrow_mut().insert(key, k.clone());
        k
    }

    fn mat_kernel(&self, kind: MatKind, d: DType, epi: u32) -> Rc<Kernel> {
        let key = format!("mat/{kind:?}/{d:?}/{epi}");
        self.cached(key, || {
            let (body, entry, wdot, enable) = match kind {
                MatKind::Matvec => (include_str!("shaders/matvec.wgsl"), "matvec", true, ""),
                // naga takes subgroup builtins without `enable subgroups;` (and rejects
                // the directive); a browser would need it.
                MatKind::MatvecSg => (include_str!("shaders/matvec_sg.wgsl"), "matvec_sg", true, if cfg!(target_arch = "wasm32") { "enable subgroups;\n" } else { "" }),
                MatKind::MatvecLn => (include_str!("shaders/matvec_ln.wgsl"), "matvec_ln", true, ""),
                MatKind::Naive => (include_str!("shaders/matmul_naive.wgsl"), "matmul_naive", false, ""),
                MatKind::Tiled => (include_str!("shaders/matmul_tiled.wgsl"), "matmul_tiled", false, ""),
            };
            let src = format!(
                "{enable}{COMMON}\n{MAT_COMMON}\n{}\n{}\n{body}",
                w_src(d),
                if wdot { wdot_src(d) } else { "" }
            );
            let module = format!("{entry}/{d:?}");
            let mut consts = vec![("EPI", epi as f64)];
            if wdot {
                consts.push(("TPR", tpr(d) as f64));
            }
            self.gpu.kernel(&module, &src, entry, "rrrrwwwu", &consts)
        })
    }

    fn embed_kernel(&self, d: DType) -> Rc<Kernel> {
        self.cached(format!("embed/{d:?}"), || {
            let src = format!("{COMMON}\n{}\n{}", w_src(d), include_str!("shaders/embed.wgsl"));
            self.gpu.kernel(&format!("embed/{d:?}"), &src, "embed", "rrrrwwwu", &[])
        })
    }

    fn attn4_kernel(&self, kv16: bool) -> Rc<Kernel> {
        self.cached(format!("attention4/{kv16}"), || {
            let kv = if kv16 { include_str!("shaders/kv_f16.wgsl") } else { include_str!("shaders/kv_f32.wgsl") };
            let s = format!("{COMMON}\n{kv}\n{}", include_str!("shaders/attention4.wgsl"));
            self.gpu.kernel(&format!("attention4/{kv16}"), &s, "attn_par4", "rrrwwu", &[])
        })
    }

    fn op_kernel(&self, file: &'static str, src: &'static str, entry: &'static str, sig: &str) -> Rc<Kernel> {
        self.cached(format!("{file}/{entry}"), || {
            let s = format!("{COMMON}\n{src}");
            self.gpu.kernel(file, &s, entry, sig, &[])
        })
    }
}

/// Counters for the most recent call.
#[derive(Debug, Clone, Default)]
pub struct RunStats {
    pub dispatches: usize,
    pub submits: usize,
    pub forwards: usize,
    /// GPU time from pass timestamps (PerForward only, if timestamp-query).
    pub gpu_ms: Option<f64>,
    /// Per-op wall times (PerOp only): (name, layer, ms).
    pub op_ms: Vec<(&'static str, i32, f64)>,
}

/// One planned dispatch.
pub struct Disp {
    pub k: Rc<Kernel>,
    bg: usize,
    pub dims: [u32; 3],
    step: u32,
    pub name: &'static str,
    pub layer: i32,
    /// (rows, cols) for matrix ops, for traces
    pub shape: (usize, usize),
    pub n: usize,
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
struct Key {
    name: &'static str,
    layer: i32,
    part: u32,
    prefill: bool,
}

/// Every buffer a session owns.
pub struct Bufs {
    gen: wgpu::Buffer,
    x: wgpu::Buffer,
    xn: wgpu::Buffer,
    qkv: wgpu::Buffer,
    att: wgpu::Buffer,
    proj: wgpu::Buffer,
    fc: wgpu::Buffer,
    scores: wgpu::Buffer,
    pub logits: wgpu::Buffer,
    arg_v: wgpu::Buffer,
    arg_i: wgpu::Buffer,
    kc: wgpu::Buffer,
    vc: wgpu::Buffer,
    dummy_r: wgpu::Buffer,
    dummy_w: [wgpu::Buffer; 2],
    steps: wgpu::Buffer,
    step_bg: wgpu::BindGroup,
    rb_logits: wgpu::Buffer,
    rb_gen: wgpu::Buffer,
    ts: Option<(wgpu::QuerySet, wgpu::Buffer, wgpu::Buffer)>,
}

/// Bind groups + per-op parameter slots, created on first use and reused.
struct Binder {
    params: wgpu::Buffer,
    n_params: usize,
    bgs: Vec<wgpu::BindGroup>,
    index: HashMap<Key, usize>,
}

impl Binder {
    fn bind(&mut self, gpu: &Gpu, key: Key, k: &Kernel, bufs: &[&wgpu::Buffer], params: [u32; 8]) -> usize {
        if let Some(&i) = self.index.get(&key) {
            return i;
        }
        assert!(self.n_params < PARAM_SLOTS, "out of param slots");
        let off = self.n_params as u64 * 256;
        self.n_params += 1;
        gpu.queue.write_buffer(&self.params, off, bytemuck::cast_slice(&params));
        let mut all: Vec<(&wgpu::Buffer, u64, Option<u64>)> = bufs.iter().map(|b| (*b, 0, None)).collect();
        all.push((&self.params, off, Some(32)));
        self.bgs.push(gpu.bind(k, &all));
        let i = self.bgs.len() - 1;
        self.index.insert(key, i);
        i
    }
}

/// Activations + KV cache for one conversation.
pub struct Session {
    pub e: Rc<Engine>,
    pos: usize,
    max_seq: usize,
    pub b: Bufs,
    binder: Binder,
    pub last: RunStats,
    /// Record per-op wall times in PerOp mode.
    pub profile: bool,
}

const F: u64 = 4;

/// Shared state for planning one forward.
struct Planner<'a> {
    e: &'a Engine,
    b: &'a Bufs,
    binder: &'a mut Binder,
    plan: &'a mut Vec<Disp>,
    n: usize,
    step: u32,
    max_seq: usize,
}

impl Planner<'_> {
    fn push(&mut self, k: Rc<Kernel>, key: Key, bufs: &[&wgpu::Buffer], params: [u32; 8], dims: [u32; 3], shape: (usize, usize), n: usize) {
        let bg = self.binder.bind(&self.e.gpu, key, &k, bufs, params);
        self.plan.push(Disp { k, bg, dims, step: self.step, name: key.name, layer: key.layer, shape, n });
    }

    fn small(&mut self, k: &Rc<Kernel>, name: &'static str, layer: i32, bufs: &[&wgpu::Buffer], params: [u32; 8], dims: [u32; 3]) {
        let key = Key { name, layer, part: 0, prefill: self.n > 1 };
        self.push(k.clone(), key, bufs, params, dims, (0, 0), self.n);
    }

    #[allow(clippy::too_many_arguments)]
    fn mat(&mut self, name: &'static str, layer: i32, part: u32, mat: &Mat, x: &wgpu::Buffer, y: &wgpu::Buffer, bias: Option<&wgpu::Buffer>, epi: u32, ldy: u32, y_off: u32, x_last: bool) {
        self.mat_ln(name, layer, part, mat, x, y, bias, epi, ldy, y_off, x_last, false)
    }

    /// `mat`; with `ln` (a matrix with a folded LayerNorm, whose bias buffer
    /// also holds its row sums) a decode matvec takes the raw residual as `x`
    /// and normalises on the fly.
    #[allow(clippy::too_many_arguments)]
    fn mat_ln(&mut self, name: &'static str, layer: i32, part: u32, mat: &Mat, x: &wgpu::Buffer, y: &wgpu::Buffer, bias: Option<&wgpu::Buffer>, epi: u32, ldy: u32, y_off: u32, x_last: bool, ln: bool) {
        let e = self.e;
        let r = e.recipe;
        let n = self.n;
        let prefill = n > 1 && !x_last;
        let kind = if prefill {
            match r.matmul {
                Matmul::Naive => MatKind::Naive,
                Matmul::Tiled => MatKind::Tiled,
            }
        } else {
            match (r.matvec, ln) {
                (Matvec::Workgroup, true) => MatKind::MatvecLn,
                (Matvec::Naive, _) => MatKind::Naive,
                (Matvec::Workgroup, _) => MatKind::Matvec,
                (Matvec::Subgroup, _) => MatKind::MatvecSg,
            }
        };
        assert!(!ln || prefill || kind == MatKind::MatvecLn, "folded LayerNorm needs the workgroup matvec");
        let k = e.mat_kernel(kind, mat.dtype, epi);
        let rows_n = if x_last { 1 } else { n };
        let dims = match kind {
            MatKind::Naive => [(mat.rows as u32).div_ceil(64), rows_n as u32, 1],
            MatKind::Matvec | MatKind::MatvecSg | MatKind::MatvecLn => [(mat.rows as u32).div_ceil(256 / tpr(mat.dtype)), rows_n as u32, 1],
            MatKind::Tiled => [(mat.rows as u32).div_ceil(64), (rows_n as u32).div_ceil(64), 1],
        };
        let c = e.config.n_embd as u32;
        let layer_off = (layer.max(0) as u32) * self.max_seq as u32 * c;
        let params = [mat.rows as u32, mat.cols as u32, ldy, y_off, x_last as u32, bias.is_some() as u32, layer_off, c];
        let b = self.b;
        let sc = mat.scales.as_ref().unwrap_or(&b.dummy_r);
        let (kc, vc) = if epi == 3 { (&b.kc, &b.vc) } else { (&b.dummy_w[0], &b.dummy_w[1]) };
        let bufs = [x, &mat.w, sc, bias.unwrap_or(&b.dummy_r), y, kc, vc];
        let key = Key { name, layer, part, prefill };
        self.push(k, key, &bufs, params, dims, (mat.rows, mat.cols), rows_n);
    }

    /// One forward of `n` tokens (already in `gen`).
    fn forward(&mut self, argmax: bool) {
        let e = self.e;
        let b = self.b;
        let w = &*e.w;
        let r = e.recipe;
        let cfg = e.config;
        let c = cfg.n_embd as u32;
        let nu = self.n as u32;
        let par = r.par;
        let fused = r.fused;
        // embed (one dispatch per wte part)
        let ek = e.embed_kernel(r.head_dtype);
        for (pi, (row0, part)) in w.head.iter().enumerate() {
            let key = Key { name: "embed", layer: -1, part: pi as u32, prefill: self.n > 1 };
            let sc = part.scales.as_ref().unwrap_or(&b.dummy_r);
            let bufs = [&b.gen, &part.w, sc, &w.wpe, &b.x, &b.dummy_w[0], &b.dummy_w[1]];
            let params = [part.rows as u32, c, 0, *row0 as u32, 0, (pi == 0) as u32, 0, c];
            self.push(ek.clone(), key, &bufs, params, [(nu * c).div_ceil(256), 1, 1], (0, 0), self.n);
        }
        let ln_src = include_str!("shaders/layernorm.wgsl");
        let at_src = include_str!("shaders/attention.wgsl");
        let ln64 = r.ln64 && cfg.n_embd == 768;
        let ln_entry = if ln64 { "layernorm_wg64" } else if par { "layernorm_par" } else { "layernorm_naive" };
        let ln = e.op_kernel("layernorm", ln_src, ln_entry, "rrrwu");
        let ln_dims = if par || ln64 { [nu, 1, 1] } else { [nu.div_ceil(64), 1, 1] };
        // attn4 is a decode kernel: prefill (many token rows, plenty of
        // workgroups) is ~5% faster on attn_par.
        let attn = if r.attn4 && (self.n == 1 || r.kv16) {
            e.attn4_kernel(r.kv16)
        } else {
            e.op_kernel("attention", at_src, if par { "attn_par" } else { "attn_naive" }, "rrrwwu")
        };
        let h = cfg.n_head as u32;
        let fd_chunks = self.max_seq.div_ceil(FD_CHUNK) as u32;
        let fd = (r.flashdec && self.n == 1).then(|| {
            let src = include_str!("shaders/attention_fd.wgsl");
            (e.op_kernel("attention_fd", src, "attn_fd_split", "rrrwwu"), e.op_kernel("attention_fd", src, "attn_fd_merge", "rrrwwu"))
        });
        let attn_dims = if par { [nu, h, 1] } else { [nu.div_ceil(64), h, 1] };
        let kvw = e.op_kernel("kv_write", include_str!("shaders/kv_write.wgsl"), "kv_write", "rwwu");
        let kvw16 = e.op_kernel("kv_write16", include_str!("shaders/kv_write16.wgsl"), "kv_write16", "rwwu");
        assert!(!r.kv16 || (r.attn4 && fused && !r.flashdec), "kv16 needs attn4 + fused epilogues");
        let gelu = e.op_kernel("gelu", include_str!("shaders/gelu.wgsl"), "gelu_inplace", "wu");
        let add = e.op_kernel("add", include_str!("shaders/add.wgsl"), "add_inplace", "wru");
        let elem = |len: u32| [(nu * len).div_ceil(256), 1, 1];
        let p1 = |a: u32| [a, 0, 0, 0, 0, 0, 0, 0];

        // Folded LayerNorms: decode skips ln_1/ln_2 (the matvec normalises);
        // prefill keeps them, with unit gain and zero bias (weights.rs).
        let fold_dec = r.ln_fold && self.n == 1;
        assert!(!r.ln_fold || fused, "ln_fold needs fused epilogues");
        for (l, lw) in w.layers.iter().enumerate() {
            let li = l as i32;
            let layer_off = l as u32 * self.max_seq as u32 * c;
            // f16 KV: qkv writes plain f32 rows and kv_write16 packs K/V.
            let qkv_epi = if r.kv16 { 0 } else { 3 };
            if fold_dec {
                self.mat_ln("qkv", li, 0, &lw.attn, &b.x, &b.qkv, Some(&lw.attn_b), qkv_epi, 3 * c, 0, false, true);
            } else if fused {
                self.small(&ln, "ln_1", li, &[&b.x, &lw.ln_1_w, &lw.ln_1_b, &b.xn], p1(c), ln_dims);
                self.mat("qkv", li, 0, &lw.attn, &b.xn, &b.qkv, Some(&lw.attn_b), qkv_epi, 3 * c, 0, false);
            } else {
                self.small(&ln, "ln_1", li, &[&b.x, &lw.ln_1_w, &lw.ln_1_b, &b.xn], p1(c), ln_dims);
                self.mat("qkv", li, 0, &lw.attn, &b.xn, &b.qkv, Some(&lw.attn_b), 0, 3 * c, 0, false);
                self.small(&kvw, "kv_write", li, &[&b.qkv, &b.kc, &b.vc], [c, layer_off, 0, 0, 0, 0, 0, 0], elem(c));
            }
            if r.kv16 {
                self.small(&kvw16, "kv_write", li, &[&b.qkv, &b.kc, &b.vc], [c, layer_off, 0, 0, 0, 0, 0, 0], [(nu * c / 2).div_ceil(256), 1, 1]);
            }
            let att_bufs = [&b.qkv, &b.kc, &b.vc, &b.att, &b.scores];
            if let Some((split, merge)) = &fd {
                let p = [c, h, layer_off, fd_chunks, 0, 0, 0, 0];
                self.small(split, "attn", li, &att_bufs, p, [fd_chunks, h, 1]);
                self.small(merge, "attn_merge", li, &att_bufs, p, [1, h, 1]);
            } else {
                self.small(&attn, "attn", li, &att_bufs, [c, h, layer_off, self.max_seq as u32, 0, 0, 0, 0], attn_dims);
            }
            if fused {
                self.mat("attn_proj", li, 0, &lw.proj, &b.att, &b.x, Some(&lw.proj_b), 2, c, 0, false);
            } else {
                self.mat("attn_proj", li, 0, &lw.proj, &b.att, &b.proj, Some(&lw.proj_b), 0, c, 0, false);
                self.small(&add, "residual", li, &[&b.x, &b.proj], p1(c), elem(c));
            }
            if fold_dec {
                self.mat_ln("fc", li, 0, &lw.fc, &b.x, &b.fc, Some(&lw.fc_b), 1, 4 * c, 0, false, true);
                self.mat("fc_proj", li, 0, &lw.fc_proj, &b.fc, &b.x, Some(&lw.fc_proj_b), 2, c, 0, false);
            } else if fused {
                self.small(&ln, "ln_2", li, &[&b.x, &lw.ln_2_w, &lw.ln_2_b, &b.xn], p1(c), ln_dims);
                self.mat("fc", li, 0, &lw.fc, &b.xn, &b.fc, Some(&lw.fc_b), 1, 4 * c, 0, false);
                self.mat("fc_proj", li, 0, &lw.fc_proj, &b.fc, &b.x, Some(&lw.fc_proj_b), 2, c, 0, false);
            } else {
                self.small(&ln, "ln_2", li, &[&b.x, &lw.ln_2_w, &lw.ln_2_b, &b.xn], p1(c), ln_dims);
                self.mat("fc", li, 0, &lw.fc, &b.xn, &b.fc, Some(&lw.fc_b), 0, 4 * c, 0, false);
                self.small(&gelu, "gelu", li, &[&b.fc], p1(4 * c), elem(4 * c));
                self.mat("fc_proj", li, 0, &lw.fc_proj, &b.fc, &b.proj, Some(&lw.fc_proj_b), 0, c, 0, false);
                self.small(&add, "residual2", li, &[&b.x, &b.proj], p1(c), elem(c));
            }
        }
        self.small(&ln, "ln_f", -1, &[&b.x, &w.ln_f_w, &w.ln_f_b, &b.xn], p1(c), ln_dims);
        for (pi, (row0, part)) in w.head.iter().enumerate() {
            self.mat("lm_head", -1, pi as u32, part, &b.xn, &b.logits, None, 0, cfg.n_vocab as u32, *row0 as u32, true);
        }
        if argmax {
            let a_src = include_str!("shaders/argmax.wgsl");
            let a1 = e.op_kernel("argmax", a_src, "argmax_partial", "rwwwu");
            let a2 = e.op_kernel("argmax", a_src, "argmax_final", "rwwwu");
            let v = cfg.n_vocab as u32;
            let bufs = [&b.logits, &b.arg_v, &b.arg_i, &b.gen];
            self.small(&a1, "argmax", -1, &bufs, [v, ARGMAX_GROUPS, 0, 0, 0, 0, 0, 0], [ARGMAX_GROUPS, 1, 1]);
            self.small(&a2, "argmax_final", -1, &bufs, [v, ARGMAX_GROUPS, 0, 0, 0, 0, 0, 0], [1, 1, 1]);
        }
    }
}

impl Session {
    fn new(e: Rc<Engine>, max_seq: usize) -> Self {
        let cfg = e.config;
        let max_seq = if max_seq == 0 { cfg.n_ctx } else { max_seq.min(cfg.n_ctx) };
        let (c, rows, v) = (cfg.n_embd as u64, max_seq as u64, cfg.n_vocab as u64);
        let g = &e.gpu;
        let kv = (cfg.n_layer as u64) * rows * c * if e.recipe.kv16 { 2 } else { F };
        let scores = if e.recipe.flashdec {
            // (o[64], m, s) per (head, chunk), padded to 17 vec4s
            cfg.n_head as u64 * (max_seq.div_ceil(FD_CHUNK) as u64) * 17 * 16
        } else if e.recipe.par {
            16
        } else {
            rows * cfg.n_head as u64 * rows * F
        };
        let steps = g.uniform("steps", STEP_STRIDE * MAX_STEPS as u64);
        let step_bg = g.step_bind(&steps);
        let ts = g.has(wgpu::Features::TIMESTAMP_QUERY).then(|| {
            let qs = g.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("ts"), ty: wgpu::QueryType::Timestamp, count: 2 });
            let rs = g.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ts-resolve"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            (qs, rs, g.readback("ts-rb", 16))
        });
        let b = Bufs {
            gen: g.buffer("gen", (rows + 1) * 4),
            x: g.buffer("x", rows * c * F),
            xn: g.buffer("xn", rows * c * F),
            qkv: g.buffer("qkv", rows * 3 * c * F),
            att: g.buffer("att", rows * c * F),
            proj: g.buffer("proj", rows * c * F),
            fc: g.buffer("fc", rows * 4 * c * F),
            scores: g.buffer("scores", scores),
            logits: g.buffer("logits", v * F),
            arg_v: g.buffer("arg_v", ARGMAX_GROUPS as u64 * 4),
            arg_i: g.buffer("arg_i", ARGMAX_GROUPS as u64 * 4),
            kc: g.buffer("k_cache", kv),
            vc: g.buffer("v_cache", kv),
            dummy_r: g.buffer("dummy_r", 16),
            dummy_w: [g.buffer("dummy_w0", 16), g.buffer("dummy_w1", 16)],
            steps,
            step_bg,
            rb_logits: g.readback("rb_logits", v * F),
            rb_gen: g.readback("rb_gen", (rows + 1) * 4),
            ts,
        };
        let binder = Binder { params: g.uniform("params", 256 * PARAM_SLOTS as u64), n_params: 0, bgs: vec![], index: HashMap::new() };
        Session { e, pos: 0, max_seq, b, binder, last: RunStats::default(), profile: false }
    }

    pub fn len(&self) -> usize {
        self.pos
    }
    pub fn is_empty(&self) -> bool {
        self.pos == 0
    }
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    fn check(&self, n: usize) -> Result<(), String> {
        if n == 0 {
            return Err("step() needs at least one token".into());
        }
        if self.pos + n > self.max_seq {
            return Err(format!("context full: {} > {}", self.pos + n, self.max_seq));
        }
        Ok(())
    }

    /// Plan one forward of `n` tokens using step slot `step`.
    pub fn plan(&mut self, plan: &mut Vec<Disp>, n: usize, step: u32, argmax: bool) {
        Planner { e: &self.e, b: &self.b, binder: &mut self.binder, plan, n, step, max_seq: self.max_seq }.forward(argmax);
    }

    fn write_steps(&self, steps: &[Step]) {
        let mut bytes = vec![0u8; steps.len() * STEP_STRIDE as usize];
        for (i, s) in steps.iter().enumerate() {
            bytes[i * STEP_STRIDE as usize..][..16].copy_from_slice(bytemuck::bytes_of(s));
        }
        self.e.gpu.queue.write_buffer(&self.b.steps, 0, &bytes);
    }

    fn record(&self, pass: &mut wgpu::ComputePass<'_>, d: &Disp) {
        pass.set_pipeline(&d.k.pipeline);
        pass.set_bind_group(0, &self.binder.bgs[d.bg], &[]);
        pass.set_bind_group(1, &self.b.step_bg, &[d.step * STEP_STRIDE as u32]);
        pass.dispatch_workgroups(d.dims[0], d.dims[1], d.dims[2]);
    }

    /// Run a plan; `copy` = (src, src offset, dst, bytes) appended after the last op.
    async fn execute(&mut self, plan: &[Disp], copy: Option<(&wgpu::Buffer, u64, &wgpu::Buffer, u64)>, now: &dyn Fn() -> f64) {
        let g = &self.e.gpu;
        let mut stats = RunStats { dispatches: plan.len(), forwards: 1, ..Default::default() };
        match self.e.recipe.sync {
            Sync::PerOp => {
                for (i, d) in plan.iter().enumerate() {
                    let t0 = now();
                    let mut enc = g.device.create_command_encoder(&Default::default());
                    {
                        let mut pass = enc.begin_compute_pass(&Default::default());
                        self.record(&mut pass, d);
                    }
                    if i + 1 == plan.len() {
                        if let Some((src, so, dst, n)) = copy {
                            enc.copy_buffer_to_buffer(src, so, dst, 0, n);
                        }
                    }
                    g.queue.submit([enc.finish()]);
                    g.wait_idle().await;
                    if self.profile {
                        stats.op_ms.push((d.name, d.layer, now() - t0));
                    }
                    stats.submits += 1;
                }
            }
            Sync::PerForward => {
                let mut enc = g.device.create_command_encoder(&Default::default());
                {
                    let tw = self.b.ts.as_ref().map(|(qs, _, _)| wgpu::ComputePassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    });
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("forward"), timestamp_writes: tw });
                    for d in plan {
                        self.record(&mut pass, d);
                    }
                }
                if let Some((qs, rs, rb)) = &self.b.ts {
                    enc.resolve_query_set(qs, 0..2, rs, 0);
                    enc.copy_buffer_to_buffer(rs, 0, rb, 0, 16);
                }
                if let Some((src, so, dst, n)) = copy {
                    enc.copy_buffer_to_buffer(src, so, dst, 0, n);
                }
                g.queue.submit([enc.finish()]);
                stats.submits = 1;
                if let Some((_, _, rb)) = &self.b.ts {
                    let bts = g.read(rb, 16).await;
                    let t: &[u64] = bytemuck::cast_slice(&bts);
                    if t[1] > t[0] {
                        stats.gpu_ms = Some((t[1] - t[0]) as f64 * g.queue.get_timestamp_period() as f64 / 1e6);
                    }
                } else if copy.is_none() {
                    g.wait_idle().await;
                }
            }
        }
        self.last = stats;
    }

    fn put_tokens(&self, tokens: &[u32]) {
        self.e.gpu.queue.write_buffer(&self.b.gen, self.pos as u64 * 4, bytemuck::cast_slice(tokens));
    }

    /// Forward over `tokens` (appended to the context); returns the next-token logits.
    pub async fn step(&mut self, tokens: &[u32], now: &dyn Fn() -> f64) -> Result<Vec<f32>, String> {
        self.check(tokens.len())?;
        self.put_tokens(tokens);
        let n = tokens.len();
        self.write_steps(&[Step { n: n as u32, pos0: self.pos as u32, ..Default::default() }]);
        let mut plan = vec![];
        self.plan(&mut plan, n, 0, false);
        let v = self.e.config.n_vocab as u64 * 4;
        let (lg, rb) = (self.b.logits.clone(), self.b.rb_logits.clone());
        self.execute(&plan, Some((&lg, 0, &rb, v)), now).await;
        let bytes = self.e.gpu.read(&rb, v).await;
        self.pos += n;
        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    /// Like [`step`](Self::step) but returns the greedy token. With
    /// `gpu_argmax` only 4 bytes come back.
    pub async fn step_greedy(&mut self, tokens: &[u32], now: &dyn Fn() -> f64) -> Result<u32, String> {
        if !self.e.recipe.gpu_argmax {
            let l = self.step(tokens, now).await?;
            return Ok(nanoinfer_core::sampler::argmax(&l));
        }
        self.check(tokens.len())?;
        self.put_tokens(tokens);
        let n = tokens.len();
        let pos0 = self.pos;
        self.write_steps(&[Step { n: n as u32, pos0: pos0 as u32, ..Default::default() }]);
        let mut plan = vec![];
        self.plan(&mut plan, n, 0, true);
        let (gen, rb) = (self.b.gen.clone(), self.b.rb_gen.clone());
        let off = (pos0 + n) as u64 * 4;
        self.execute(&plan, Some((&gen, off, &rb, 4)), now).await;
        let bts = self.e.gpu.read(&rb, 4).await;
        self.pos += n;
        Ok(u32::from_le_bytes([bts[0], bts[1], bts[2], bts[3]]))
    }

    /// Greedy-decode `n` tokens, the first one fed `first`; returns them.
    /// With `resident`, up to [`RESIDENT_CHUNK`] forwards go in one submit:
    /// each forward's embed reads the token the previous one's argmax wrote
    /// into `gen`, and the CPU reads the chunk's tokens back once.
    pub async fn decode_greedy(&mut self, first: u32, n: usize, now: &dyn Fn() -> f64) -> Result<Vec<u32>, String> {
        let r = self.e.recipe;
        let mut total = RunStats::default();
        let mut out = Vec::with_capacity(n);
        let mut gpu_ms = Some(0.0);
        if !(r.resident && r.gpu_argmax && r.sync == Sync::PerForward) {
            let mut next = first;
            for _ in 0..n {
                next = self.step_greedy(&[next], now).await?;
                out.push(next);
                total.dispatches += self.last.dispatches;
                total.submits += self.last.submits;
                total.forwards += 1;
                gpu_ms = gpu_ms.zip(self.last.gpu_ms).map(|(a, b)| a + b);
            }
        } else {
            self.check(n)?;
            self.put_tokens(&[first]);
            let mut done = 0;
            while done < n {
                let k = (n - done).min(RESIDENT_CHUNK).min(MAX_STEPS);
                let pos0 = self.pos;
                let steps: Vec<Step> = (0..k).map(|i| Step { n: 1, pos0: (pos0 + i) as u32, ..Default::default() }).collect();
                self.write_steps(&steps);
                let mut plan = vec![];
                for i in 0..k {
                    self.plan(&mut plan, 1, i as u32, true);
                }
                let (gen, rb) = (self.b.gen.clone(), self.b.rb_gen.clone());
                let off = (pos0 + 1) as u64 * 4;
                self.execute(&plan, Some((&gen, off, &rb, k as u64 * 4)), now).await;
                let bts = self.e.gpu.read(&rb, k as u64 * 4).await;
                out.extend_from_slice(bytemuck::cast_slice::<u8, u32>(&bts));
                total.dispatches += self.last.dispatches;
                total.submits += self.last.submits;
                total.forwards += k;
                gpu_ms = gpu_ms.zip(self.last.gpu_ms).map(|(a, b)| a + b);
                self.pos += k;
                done += k;
            }
        }
        total.gpu_ms = gpu_ms;
        self.last = total;
        Ok(out)
    }

    /// Steady-state cost of every op of one forward: each dispatch alone,
    /// repeated `reps` times back to back in one pass, one submit + wait,
    /// wall time / reps, minus the cost of an empty submit + wait. (Pass
    /// timestamps on wgpu-Metal proved unreliable for small passes; this
    /// works the same natively and in a browser.) Small matrices hit in
    /// cache on repeats, so trust GB/s only for the large ones. Leaves the
    /// session's activations and cache dirty: `reset()` afterwards.
    pub async fn profile_ops(&mut self, tokens: &[u32], reps: usize, now: &dyn Fn() -> f64) -> Result<Vec<(&'static str, i32, f64)>, String> {
        self.check(tokens.len())?;
        let g = self.e.gpu.clone();
        self.put_tokens(tokens);
        let n = tokens.len();
        self.write_steps(&[Step { n: n as u32, pos0: self.pos as u32, ..Default::default() }]);
        let mut plan = vec![];
        self.plan(&mut plan, n, 0, self.e.recipe.gpu_argmax);
        let mut base = vec![];
        for _ in 0..5 {
            let t0 = now();
            let enc = g.device.create_command_encoder(&Default::default());
            g.queue.submit([enc.finish()]);
            g.wait_idle().await;
            base.push(now() - t0);
        }
        base.sort_by(f64::total_cmp);
        let base = base[2];
        let mut out = vec![];
        for d in &plan {
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let t0 = now();
                let mut enc = g.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    for _ in 0..reps {
                        self.record(&mut pass, d);
                    }
                }
                g.queue.submit([enc.finish()]);
                g.wait_idle().await;
                best = best.min(now() - t0);
            }
            out.push((d.name, d.layer, ((best - base) / reps as f64).max(0.0)));
        }
        Ok(out)
    }

    /// Copy of the last forward's logits (still on the GPU after any call).
    pub async fn read_logits(&mut self) -> Vec<f32> {
        let g = &self.e.gpu;
        let v = self.e.config.n_vocab as u64 * 4;
        let mut enc = g.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.b.logits, 0, &self.b.rb_logits, 0, v);
        g.queue.submit([enc.finish()]);
        bytemuck::cast_slice(&g.read(&self.b.rb_logits, v).await).to_vec()
    }
}

