//! The GPT-2 forward pass, encoded as Metal dispatches.
//!
//! One function, [`MetalSession::encode_forward`], walks the network exactly
//! like `nanoinfer_core::stages::kv_cache` and asks the [`Recipe`] which kernel
//! to use for each op. Adding a stage means adding kernels and a recipe, not
//! editing what an earlier stage runs.

use crate::gpu::{Arg, Buf, Gpu, RunStats, Runner};
use crate::weights::{Mat, Weights};
use crate::{DType, Matmul, Matvec, MetalStage, Par, Recipe};
use nanoinfer_core::{Config, Model};

/// GPU-resident model for one stage (weights in that stage's dtype).
pub struct MetalEngine {
    pub gpu: Gpu,
    pub config: Config,
    pub stage: MetalStage,
    recipe: Recipe,
    w: Weights,
}

impl MetalEngine {
    pub fn new(model: &Model, stage: MetalStage) -> Result<MetalEngine, String> {
        let gpu = Gpu::new()?;
        let recipe = stage.recipe();
        let w = Weights::upload(&gpu, model, recipe.dtype);
        Ok(MetalEngine { gpu, config: model.config, stage, recipe, w })
    }

    pub fn session(&self) -> MetalSession<'_> {
        MetalSession::new(self)
    }
}

/// Activations + KV cache for one conversation.
pub struct MetalSession<'e> {
    e: &'e MetalEngine,
    pos: usize,
    max_seq: usize,
    tokens: Buf,
    x: Buf,
    xn: Buf,
    qkv: Buf,
    att: Buf,
    proj: Buf,
    fc: Buf,
    scores: Buf,
    logits: Buf,
    argmax: Buf,
    k_cache: Buf,
    v_cache: Buf,
    /// Counters for the most recent `step`.
    pub last: RunStats,
    /// Set to `Some(vec![])` to collect per-op GPU times (PerOp sync only).
    pub profile: Option<Vec<(&'static str, f64)>>,
}

const F: usize = 4; // bytes per f32

/// SIMD matvec shape: rows per SIMD-group, SIMD-groups per threadgroup
/// (chosen by `examples/matvec_sweep.rs`).
const MV_ROWS: usize = 4;
const MV_SIMDGROUPS: usize = 4;
const MV_KERNEL: &str = "matvec_simd_r4";

impl<'e> MetalSession<'e> {
    fn new(e: &'e MetalEngine) -> Self {
        let cfg = e.config;
        let (c, rows, v) = (cfg.n_embd, cfg.n_ctx, cfg.n_vocab);
        let g = &e.gpu;
        let kv = cfg.n_layer * cfg.n_ctx * c * F;
        let scores = if e.recipe.attn == Par::Naive { rows * cfg.n_head * cfg.n_ctx * F } else { 16 };
        MetalSession {
            e,
            pos: 0,
            max_seq: cfg.n_ctx,
            tokens: g.buffer(rows * 4),
            x: g.buffer(rows * c * F),
            xn: g.buffer(rows * c * F),
            qkv: g.buffer(rows * 3 * c * F),
            att: g.buffer(rows * c * F),
            proj: g.buffer(rows * c * F),
            fc: g.buffer(rows * 4 * c * F),
            scores: g.buffer(scores),
            logits: g.buffer(v * F),
            argmax: g.buffer(1024 * 8),
            k_cache: g.buffer(kv),
            v_cache: g.buffer(kv),
            last: RunStats::default(),
            profile: None,
        }
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

    fn check(&self, tokens: &[u32]) -> Result<(), String> {
        if tokens.is_empty() {
            return Err("step() needs at least one token".into());
        }
        if let Some(t) = tokens.iter().find(|&&t| t as usize >= self.e.config.n_vocab) {
            return Err(format!("bad token {t}"));
        }
        if self.pos + tokens.len() > self.max_seq {
            return Err(format!("context full: {} > {}", self.pos + tokens.len(), self.max_seq));
        }
        Ok(())
    }

    /// Run the forward pass over `tokens` (appended to the context) and return the
    /// next-token logits.
    pub fn step(&mut self, tokens: &[u32]) -> Result<&[f32], String> {
        self.check(tokens)?;
        self.run(tokens, false);
        Ok(self.logits.slice(self.e.config.n_vocab))
    }

    /// Like [`step`](Self::step) but returns the argmax token. Stages with
    /// `gpu_argmax` never copy the logits to the CPU side at all.
    pub fn step_greedy(&mut self, tokens: &[u32]) -> Result<u32, String> {
        self.check(tokens)?;
        if self.e.recipe.gpu_argmax {
            self.run(tokens, true);
            Ok(self.argmax.slice::<u32>(1)[0])
        } else {
            self.run(tokens, false);
            Ok(nanoinfer_core::sampler::argmax(self.logits.slice(self.e.config.n_vocab)))
        }
    }

    fn run(&mut self, tokens: &[u32], argmax: bool) {
        self.tokens.write(0, tokens);
        let mut r = self.e.gpu.runner(self.e.recipe.sync);
        r.profile = self.profile.take();
        self.encode_forward(&mut r, tokens.len(), argmax);
        r.flush();
        self.profile = r.profile.take();
        self.last = r.stats;
        self.pos += tokens.len();
    }

    /// `out[n, rows] = x[n, cols] · Wᵀ (+ bias)`.
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        r: &mut Runner,
        label: &'static str,
        out: &Buf,
        x: &Buf,
        x_off: usize,
        w: &Mat,
        bias: Option<&Buf>,
        n: usize,
    ) {
        let rec = self.e.recipe;
        let (d_in, d_out) = (w.cols as u32, w.rows as u32);
        let has_bias = bias.is_some() as u32;
        let bias = bias.unwrap_or(&self.x); // any buffer; unused when has_bias == 0
        let args = [
            Arg::Buf(out),
            Arg::BufAt(x, x_off),
            Arg::Buf(&w.buf),
            Arg::Buf(bias),
            Arg::U32(n as u32),
            Arg::U32(d_in),
            Arg::U32(d_out),
            Arg::U32(has_bias),
        ];
        let rows = w.rows;
        // Decode (and prefill stages without a real matmul) use the matvec.
        let matmul = if n == 1 { None } else { Some(rec.matmul) };
        let mv = match matmul {
            None => Some(rec.matvec),
            Some(Matmul::Naive) => Some(Matvec::Naive),
            Some(Matmul::Matvec) => Some(Matvec::Simd),
            Some(_) => None,
        };
        if let Some(mv) = mv {
            match (mv, rec.dtype) {
                (Matvec::Naive, DType::F32) => r.dispatch(label, "matmul_naive", &args, n * rows, 256),
                (Matvec::Simd, dtype) => {
                    let (kernel, rows_per_sg, nsg) = matvec_kernel(dtype);
                    let groups = (rows.div_ceil(rows_per_sg * nsg), n, 1);
                    r.dispatch_tg(label, kernel, &args, groups, (32 * nsg, 1, 1), 0)
                }
                other => unimplemented!("matvec {other:?}"),
            }
            return;
        }
        match (rec.matmul, rec.dtype) {
            (Matmul::Tiled, DType::F32) => {
                r.dispatch_tg(label, "matmul_tiled", &args, (rows.div_ceil(32), n.div_ceil(32), 1), (256, 1, 1), 0)
            }
            (Matmul::SimdMatrix, DType::F32) => {
                r.dispatch_tg(label, "matmul_mma", &args, (rows.div_ceil(64), n.div_ceil(32), 1), (128, 1, 1), 0)
            }
            other => unimplemented!("matmul {other:?}"),
        }
    }

    fn layernorm(&self, r: &mut Runner, label: &'static str, x_off: usize, w: &Buf, b: &Buf, n: usize) {
        let c = self.e.config.n_embd as u32;
        match self.e.recipe.norm {
            Par::Naive => r.dispatch(
                label,
                "layernorm_naive",
                &[
                    Arg::Buf(&self.xn),
                    Arg::BufAt(&self.x, x_off),
                    Arg::Buf(w),
                    Arg::Buf(b),
                    Arg::U32(n as u32),
                    Arg::U32(c),
                ],
                n,
                32,
            ),
            Par::Simd => r.dispatch_tg(
                label,
                "layernorm_simd",
                &[
                    Arg::Buf(&self.xn),
                    Arg::BufAt(&self.x, x_off),
                    Arg::Buf(w),
                    Arg::Buf(b),
                    Arg::U32(n as u32),
                    Arg::U32(c),
                ],
                (n, 1, 1),
                (256, 1, 1),
                32 * F,
            ),
        }
    }

    fn attention(&self, r: &mut Runner, l: usize, n: usize) {
        let cfg = &self.e.config;
        let layer_off = l * cfg.n_ctx * cfg.n_embd * F;
        match self.e.recipe.attn {
            Par::Naive => r.dispatch(
                "attn",
                "attention_naive",
                &[
                    Arg::Buf(&self.att),
                    Arg::Buf(&self.qkv),
                    Arg::BufAt(&self.k_cache, layer_off),
                    Arg::BufAt(&self.v_cache, layer_off),
                    Arg::Buf(&self.scores),
                    Arg::U32(n as u32),
                    Arg::U32(self.pos as u32),
                    Arg::U32(cfg.n_head as u32),
                    Arg::U32(cfg.head_dim() as u32),
                    Arg::U32(cfg.n_ctx as u32),
                ],
                n * cfg.n_head,
                32,
            ),
            Par::Simd => r.dispatch_tg(
                "attn",
                "attention_simd",
                &[
                    Arg::Buf(&self.att),
                    Arg::Buf(&self.qkv),
                    Arg::BufAt(&self.k_cache, layer_off),
                    Arg::BufAt(&self.v_cache, layer_off),
                    Arg::U32(n as u32),
                    Arg::U32(self.pos as u32),
                    Arg::U32(cfg.n_head as u32),
                ],
                (n * cfg.n_head, 1, 1),
                (256, 1, 1),
                (1024 + 32 + 256) * F,
            ),
        }
    }

    fn encode_forward(&self, r: &mut Runner, n: usize, _argmax: bool) {
        let cfg = self.e.config;
        let w = &self.e.w;
        let c = cfg.n_embd;
        let (cu, nu, pos) = (c as u32, n as u32, self.pos as u32);
        r.dispatch(
            "embed",
            "embed",
            &[
                Arg::Buf(&self.x),
                Arg::Buf(&w.wte),
                Arg::Buf(&w.wpe),
                Arg::Buf(&self.tokens),
                Arg::U32(pos),
                Arg::U32(cu),
                Arg::U32(nu),
            ],
            n * c,
            256,
        );
        for (l, lw) in w.layers.iter().enumerate() {
            let layer_off = l * cfg.n_ctx * c * F;
            self.layernorm(r, "ln_1", 0, &lw.ln1_w, &lw.ln1_b, n);
            self.linear(r, "qkv", &self.qkv, &self.xn, 0, &lw.attn_w, Some(&lw.attn_b), n);
            r.dispatch(
                "kv_write",
                "kv_write",
                &[
                    Arg::BufAt(&self.k_cache, layer_off),
                    Arg::BufAt(&self.v_cache, layer_off),
                    Arg::Buf(&self.qkv),
                    Arg::U32(pos),
                    Arg::U32(cu),
                    Arg::U32(nu),
                ],
                n * c,
                256,
            );
            self.attention(r, l, n);
            self.linear(r, "attn_proj", &self.proj, &self.att, 0, &lw.proj_w, Some(&lw.proj_b), n);
            self.residual(r, "residual_1", n * c);
            self.layernorm(r, "ln_2", 0, &lw.ln2_w, &lw.ln2_b, n);
            self.linear(r, "fc", &self.fc, &self.xn, 0, &lw.fc_w, Some(&lw.fc_b), n);
            r.dispatch("gelu", "gelu_naive", &[Arg::Buf(&self.fc), Arg::U32((n * 4 * c) as u32)], n * 4 * c, 256);
            self.linear(r, "fc_proj", &self.proj, &self.fc, 0, &lw.fcp_w, Some(&lw.fcp_b), n);
            self.residual(r, "residual_2", n * c);
        }
        // Only the last position's logits are needed.
        self.layernorm(r, "ln_f", (n - 1) * c * F, &w.ln_f_w, &w.ln_f_b, 1);
        self.linear(r, "lm_head", &self.logits, &self.xn, 0, &w.lm_head, None, 1);
    }

    fn residual(&self, r: &mut Runner, label: &'static str, len: usize) {
        r.dispatch(label, "residual_add", &[Arg::Buf(&self.x), Arg::Buf(&self.proj), Arg::U32(len as u32)], len, 256);
    }
}

/// The SIMD-group matvec kernel for a weight dtype: (name, rows per
/// SIMD-group, SIMD-groups per threadgroup).
fn matvec_kernel(dtype: DType) -> (&'static str, usize, usize) {
    match dtype {
        DType::F32 => (MV_KERNEL, MV_ROWS, MV_SIMDGROUPS),
        other => unimplemented!("matvec for {other:?}"),
    }
}
