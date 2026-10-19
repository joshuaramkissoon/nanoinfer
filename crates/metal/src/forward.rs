//! The GPT-2 forward pass, encoded as Metal dispatches.
//!
//! One function, [`MetalSession::encode_forward`], walks the network exactly
//! like `nanoinfer_core::stages::kv_cache` and asks the [`Recipe`] which kernel
//! to use for each op. Adding a stage means adding kernels and a recipe, not
//! editing what an earlier stage runs.

use crate::gpu::{Arg, Buf, Gpu, RunStats, Runner};
use crate::weights::{Mat, Weights};
use crate::{Attn, DType, Fusion, Matmul, Matvec, MetalStage, Par, Recipe};
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
        Self::with_recipe(model, stage, stage.recipe())
    }

    /// An engine with a custom recipe (for experiments; `stage` is only a label).
    pub fn with_recipe(model: &Model, stage: MetalStage, recipe: Recipe) -> Result<MetalEngine, String> {
        let gpu = Gpu::new()?;
        let w = Weights::upload(&gpu, model, recipe.dtype, recipe.head_dtype);
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
    /// Argmax partials (value, index) per threadgroup.
    arg_v: Buf,
    arg_i: Buf,
    /// Token ids by absolute position, written by the CPU or by `argmax_final`
    /// (gen[pos + 1] = argmax of the forward at pos). Lets decode run without
    /// the CPU in the loop.
    gen: Buf,
    /// Flash-decoding partials [n_head, max_chunks, 2 + head_dim].
    att_part: Buf,
    k_cache: Buf,
    v_cache: Buf,
    /// Q8 KV cache scales, f32 [n_layer][n_ctx][n_head] (16 bytes unless `kv_q8`).
    k_scale: Buf,
    v_scale: Buf,
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

/// Keys per flash-decoding chunk (must match ATT_CHUNK in attn.metal).
const ATT_CHUNK: usize = 64;
/// Threadgroups (of 1024 threads) in the first argmax pass.
const ARGMAX_GROUPS: usize = 32;
/// Fused matvec shape: 4 rows per SIMD-group, 8 SIMD-groups = 32 rows per threadgroup.
const FUSED_ROWS: usize = 4;
const FUSED_SIMDGROUPS: usize = 8;

impl<'e> MetalSession<'e> {
    fn new(e: &'e MetalEngine) -> Self {
        let cfg = e.config;
        let (c, rows, v) = (cfg.n_embd, cfg.n_ctx, cfg.n_vocab);
        let g = &e.gpu;
        let kv = cfg.n_layer
            * cfg.n_ctx
            * c
            * if e.recipe.kv_q8 {
                1
            } else if e.recipe.kv_f16 {
                2
            } else {
                F
            };
        let kv_scales = if e.recipe.kv_q8 { cfg.n_layer * cfg.n_ctx * cfg.n_head * F } else { 16 };
        let scores = if e.recipe.attn == Attn::Naive { rows * cfg.n_head * cfg.n_ctx * F } else { 16 };
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
            argmax: g.buffer(16),
            arg_v: g.buffer(ARGMAX_GROUPS * 4),
            arg_i: g.buffer(ARGMAX_GROUPS * 4),
            gen: g.buffer((rows + 1) * 4),
            att_part: g.buffer(cfg.n_head * cfg.n_ctx.div_ceil(ATT_CHUNK) * (cfg.head_dim() + 2) * F),
            k_cache: g.buffer(kv),
            v_cache: g.buffer(kv),
            k_scale: g.buffer(kv_scales),
            v_scale: g.buffer(kv_scales),
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
        self.gen.write(self.pos, &tokens[tokens.len() - 1..]);
        let mut r = self.e.gpu.runner(self.e.recipe.sync);
        r.profile = self.profile.take();
        self.encode_forward(&mut r, tokens.len(), argmax);
        r.flush();
        self.profile = r.profile.take();
        self.last = r.stats;
        self.pos += tokens.len();
    }

    /// Greedy-decode `n` tokens, the first one fed `first`; returns the `n`
    /// generated tokens. Same result as calling [`step_greedy`](Self::step_greedy)
    /// in a loop. With `Recipe::pipelined`, the CPU never waits between
    /// tokens: every forward reads its input token from `gen[pos]`, which the
    /// previous forward's argmax wrote, so all `n` command buffers are
    /// committed up front and the GPU runs them back to back.
    pub fn decode_greedy(&mut self, first: u32, n: usize) -> Result<Vec<u32>, String> {
        self.check(&[first])?;
        let rec = self.e.recipe;
        if !(rec.pipelined && rec.gpu_argmax && rec.fused != Fusion::None) {
            let mut out = Vec::with_capacity(n);
            let mut next = first;
            for _ in 0..n {
                next = self.step_greedy(&[next])?;
                out.push(next);
            }
            return Ok(out);
        }
        if self.pos + n > self.max_seq {
            return Err(format!("context full: {} > {}", self.pos + n, self.max_seq));
        }
        let pos0 = self.pos;
        self.gen.write(pos0, &[first]);
        let mut stats = RunStats::default();
        let mut pending = Vec::with_capacity(n);
        for _ in 0..n {
            let mut r = self.e.gpu.runner(rec.sync);
            self.encode_decode_fused(&mut r, true);
            stats.dispatches += r.stats.dispatches;
            pending.push(r.commit());
            self.pos += 1;
        }
        for p in pending.into_iter().flatten() {
            stats.gpu_seconds += p.wait();
            stats.command_buffers += 1;
        }
        self.last = stats;
        Ok(self.gen.slice::<u32>(pos0 + n + 1)[pos0 + 1..].to_vec())
    }

    /// Greedy decode that hands each token to `on_token` as soon as its
    /// forward completes, the way a server streams to a client. At most
    /// `depth` forwards are in flight: `depth = 1` commits a token, waits,
    /// reads its argmax back and only then encodes the next (per-token sync,
    /// like llama.cpp); `depth = 2` commits token t+1 before reading token t
    /// (one token of lookahead, like MLX's async eval), so the GPU never idles
    /// while the CPU reads back and encodes. `on_token` returning `false`
    /// stops; forwards already in flight finish and are rolled back, so the
    /// session ends just after the last delivered token. Returns the number of
    /// tokens delivered. Stages without the pipelined recipe fall back to
    /// [`step_greedy`](Self::step_greedy) per token (depth 1).
    pub fn decode_stream(
        &mut self,
        first: u32,
        n: usize,
        depth: usize,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<usize, String> {
        self.check(&[first])?;
        let rec = self.e.recipe;
        if !(rec.pipelined && rec.gpu_argmax && rec.fused != Fusion::None) {
            let mut next = first;
            for i in 0..n {
                next = self.step_greedy(&[next])?;
                if !on_token(next) {
                    return Ok(i + 1);
                }
            }
            return Ok(n);
        }
        if self.pos + n > self.max_seq {
            return Err(format!("context full: {} > {}", self.pos + n, self.max_seq));
        }
        let depth = depth.max(1);
        let pos0 = self.pos;
        self.gen.write(pos0, &[first]);
        let mut stats = RunStats::default();
        let mut inflight = std::collections::VecDeque::with_capacity(depth);
        let (mut committed, mut delivered, mut stopped) = (0, 0, false);
        while delivered < n {
            // Keep `depth` forwards queued; forward i reads gen[pos0 + i] (written
            // by forward i-1's argmax on the GPU) and writes gen[pos0 + i + 1].
            while !stopped && committed < n && inflight.len() < depth {
                let mut r = self.e.gpu.runner(rec.sync);
                self.encode_decode_fused(&mut r, true);
                stats.dispatches += r.stats.dispatches;
                inflight.extend(r.commit());
                self.pos += 1;
                committed += 1;
            }
            let Some(p) = inflight.pop_front() else { break };
            stats.gpu_seconds += p.wait();
            stats.command_buffers += 1;
            if stopped {
                continue; // draining forwards past the stop
            }
            // Forward `delivered` is complete; later ones only write later slots.
            let slot = pos0 + delivered + 1;
            let tok = self.gen.slice::<u32>(slot + 1)[slot];
            delivered += 1;
            if !on_token(tok) {
                stopped = true;
            }
        }
        for p in inflight {
            stats.gpu_seconds += p.wait();
            stats.command_buffers += 1;
        }
        self.pos = pos0 + delivered;
        self.last = stats;
        Ok(delivered)
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
            match (mv, w.dtype) {
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
        match (rec.matmul, w.dtype) {
            (Matmul::Tiled, DType::F32) => {
                r.dispatch_tg(label, "matmul_tiled", &args, (rows.div_ceil(32), n.div_ceil(32), 1), (256, 1, 1), 0)
            }
            (Matmul::SimdMatrix, DType::F32) => {
                r.dispatch_tg(label, "matmul_mma", &args, (rows.div_ceil(64), n.div_ceil(32), 1), (128, 1, 1), 0)
            }
            (Matmul::SimdMatrix, dtype) => {
                let kernel = match dtype {
                    DType::F16 => "matmul_mma_f16",
                    DType::Q8_0 => "matmul_mma_q8_0",
                    _ => "matmul_mma_q4_0",
                };
                r.dispatch_tg(label, kernel, &args, (rows.div_ceil(64), n.div_ceil(32), 1), (128, 1, 1), 0)
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
        let layer_off = self.kv_layer_off(l);
        let attn = match self.e.recipe.attn {
            Attn::SplitK | Attn::Flash if n == 1 => return self.attention_split(r, l),
            a => a,
        };
        match attn {
            Attn::Naive => r.dispatch(
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
            Attn::Flash => r.dispatch_tg(
                "attn",
                self.kv_kernel("attn_prefill_flash"),
                &[
                    Arg::Buf(&self.att),
                    Arg::Buf(&self.qkv),
                    Arg::BufAt(&self.k_cache, layer_off),
                    Arg::BufAt(&self.v_cache, layer_off),
                    Arg::U32(n as u32),
                    Arg::U32(self.pos as u32),
                    Arg::U32(cfg.n_head as u32),
                    Arg::BufAt(&self.k_scale, self.kv_scale_off(l)),
                    Arg::BufAt(&self.v_scale, self.kv_scale_off(l)),
                ],
                (n.div_ceil(32), cfg.n_head, 1),
                (128, 1, 1),
                0,
            ),
            Attn::Simd | Attn::SplitK => r.dispatch_tg(
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

    /// Byte offset of layer `l` in the K/V cache buffers.
    fn kv_layer_off(&self, l: usize) -> usize {
        let cfg = &self.e.config;
        l * cfg.n_ctx
            * cfg.n_embd
            * if self.e.recipe.kv_q8 {
                1
            } else if self.e.recipe.kv_f16 {
                2
            } else {
                F
            }
    }

    /// The f16-KV variant of a kernel that reads or writes the cache.
    /// Byte offset of layer `l` in the Q8 scale buffers.
    fn kv_scale_off(&self, l: usize) -> usize {
        let cfg = &self.e.config;
        l * cfg.n_ctx * cfg.n_head * F
    }

    /// Q8 KV: quantise the K/V rows of `qkv` (n tokens from pos0) into the cache.
    fn kv_quant8(&self, r: &mut Runner, l: usize, pos0: usize, n: usize) {
        let cfg = &self.e.config;
        r.dispatch_tg(
            "kv_write",
            "kv_quant8",
            &[
                Arg::BufAt(&self.k_cache, self.kv_layer_off(l)),
                Arg::BufAt(&self.v_cache, self.kv_layer_off(l)),
                Arg::BufAt(&self.k_scale, self.kv_scale_off(l)),
                Arg::BufAt(&self.v_scale, self.kv_scale_off(l)),
                Arg::Buf(&self.qkv),
                Arg::U32(pos0 as u32),
                Arg::U32(cfg.n_head as u32),
            ],
            (n * cfg.n_head, 2, 1),
            (64, 1, 1),
            4 * F,
        );
    }

    fn kv_kernel(&self, name: &'static str) -> &'static str {
        if self.e.recipe.kv_q8 {
            return match name {
                "attn_prefill_flash" => "attn_prefill_flash_q8",
                "attn_decode_split" => "attn_decode_split_q8",
                other => unimplemented!("{other} has no Q8-KV variant"),
            };
        }
        if !self.e.recipe.kv_f16 {
            return name;
        }
        match name {
            "attn_prefill_flash" => "attn_prefill_flash_h",
            "attn_decode_split" => "attn_decode_split_h",
            "kv_write" => "kv_write_h",
            other => unimplemented!("{other} has no f16-KV variant"),
        }
    }

    /// Flash-decoding: (head, 64-key chunk) threadgroups, then a merge per head.
    fn attention_split(&self, r: &mut Runner, l: usize) {
        let cfg = &self.e.config;
        let layer_off = self.kv_layer_off(l);
        let n_kv = self.pos + 1;
        let chunks = n_kv.div_ceil(ATT_CHUNK);
        let max_chunks = cfg.n_ctx.div_ceil(ATT_CHUNK) as u32;
        r.dispatch_tg(
            "attn",
            self.kv_kernel("attn_decode_split"),
            &[
                Arg::Buf(&self.qkv),
                Arg::BufAt(&self.k_cache, layer_off),
                Arg::BufAt(&self.v_cache, layer_off),
                Arg::Buf(&self.att_part),
                Arg::U32(n_kv as u32),
                Arg::U32(cfg.n_head as u32),
                Arg::U32(max_chunks),
                Arg::BufAt(&self.k_scale, self.kv_scale_off(l)),
                Arg::BufAt(&self.v_scale, self.kv_scale_off(l)),
            ],
            (cfg.n_head, chunks, 1),
            (256, 1, 1),
            (64 + 32 + 256) * F,
        );
        r.dispatch_tg(
            "attn",
            "attn_decode_merge",
            &[
                Arg::Buf(&self.att_part),
                Arg::Buf(&self.att),
                Arg::U32(chunks as u32),
                Arg::U32(cfg.n_head as u32),
                Arg::U32(max_chunks),
            ],
            (cfg.n_head, 1, 1),
            (64, 1, 1),
            0,
        );
    }

    fn encode_forward(&self, r: &mut Runner, n: usize, argmax: bool) {
        if self.e.recipe.fused != Fusion::None && n == 1 {
            return self.encode_decode_fused(r, argmax);
        }
        self.encode_forward_unfused(r, n);
        if argmax {
            self.encode_argmax(r);
        }
    }

    /// Argmax over the logits on the GPU; writes `argmax[0]` and `gen[pos + n]`.
    fn encode_argmax(&self, r: &mut Runner) {
        let v = self.e.config.n_vocab as u32;
        r.dispatch_tg(
            "argmax",
            "argmax_partial",
            &[Arg::Buf(&self.logits), Arg::Buf(&self.arg_v), Arg::Buf(&self.arg_i), Arg::U32(v)],
            (ARGMAX_GROUPS, 1, 1),
            (1024, 1, 1),
            64 * F,
        );
        // The slot this forward's prediction belongs to (pos is not yet advanced).
        let slot = (self.pos + r.pending_tokens()) as u32;
        r.dispatch_tg(
            "argmax",
            "argmax_final",
            &[
                Arg::Buf(&self.arg_v),
                Arg::Buf(&self.arg_i),
                Arg::U32(ARGMAX_GROUPS as u32),
                Arg::Buf(&self.argmax),
                Arg::Buf(&self.gen),
                Arg::U32(slot),
            ],
            (1, 1, 1),
            (32, 1, 1),
            0,
        );
    }

    /// One decode step (n = 1) with fused kernels: 5 dispatches per layer.
    /// Input token is `gen[pos]`.
    fn encode_decode_fused(&self, r: &mut Runner, argmax: bool) {
        let cfg = self.e.config;
        let w = &self.e.w;
        let c = cfg.n_embd;
        let pos = self.pos as u32;
        r.set_pending_tokens(1);
        r.dispatch(
            "embed",
            "embed_gen",
            &[
                Arg::Buf(&self.x),
                Arg::Buf(&w.wte),
                Arg::Buf(&w.wpe),
                Arg::Buf(&self.gen),
                Arg::U32(pos),
                Arg::U32(pos),
                Arg::U32(c as u32),
            ],
            c,
            256,
        );
        if self.e.recipe.fused == Fusion::Epilogue {
            return self.encode_decode_epilogue(r, argmax);
        }
        for (l, lw) in w.layers.iter().enumerate() {
            let layer_off = self.kv_layer_off(l);
            let kc = Arg::BufAt(&self.k_cache, layer_off);
            let vc = Arg::BufAt(&self.v_cache, layer_off);
            self.fused(
                r,
                "qkv",
                "mvf_ln_qkv",
                &self.qkv,
                &self.x,
                &lw.attn_w,
                &lw.attn_b,
                Some((&lw.ln1_w, &lw.ln1_b)),
                Some((kc, vc)),
            );
            self.attention(r, l, 1);
            self.fused(r, "attn_proj", "mvf_resid", &self.x, &self.att, &lw.proj_w, &lw.proj_b, None, None);
            self.fused(
                r,
                "fc",
                "mvf_ln_gelu",
                &self.fc,
                &self.x,
                &lw.fc_w,
                &lw.fc_b,
                Some((&lw.ln2_w, &lw.ln2_b)),
                None,
            );
            self.fused(r, "fc_proj", "mvf_resid", &self.x, &self.fc, &lw.fcp_w, &lw.fcp_b, None, None);
        }
        // ln_f + LM head; `bias` is unused (has_bias = 0).
        let head = &w.lm_head;
        let args = [
            Arg::Buf(&self.logits),
            Arg::Buf(&self.x),
            Arg::Buf(&head.buf),
            Arg::Buf(&self.x),
            Arg::Buf(&w.ln_f_w),
            Arg::Buf(&w.ln_f_b),
            Arg::Buf(&self.x),
            Arg::Buf(&self.x),
            Arg::U32(c as u32),
            Arg::U32(head.rows as u32),
            Arg::U32(0),
            Arg::U32(pos),
        ];
        let kernel = format!("mvf_ln_{}", head.dtype.name());
        self.dispatch_fused(r, "lm_head", &kernel, &args, head.rows, c);
        if argmax {
            self.encode_argmax(r);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn fused(
        &self,
        r: &mut Runner,
        label: &'static str,
        kernel: &str,
        out: &Buf,
        x: &Buf,
        w: &Mat,
        bias: &Buf,
        ln: Option<(&Buf, &Buf)>,
        kv: Option<(Arg, Arg)>,
    ) {
        let (lw, lb) = ln.unwrap_or((bias, bias));
        let (kc, vc) = kv.unwrap_or((Arg::Buf(out), Arg::Buf(out)));
        let args = [
            Arg::Buf(out),
            Arg::Buf(x),
            Arg::Buf(&w.buf),
            Arg::Buf(bias),
            Arg::Buf(lw),
            Arg::Buf(lb),
            kc,
            vc,
            Arg::U32(w.cols as u32),
            Arg::U32(w.rows as u32),
            Arg::U32(1),
            Arg::U32(self.pos as u32),
        ];
        let kernel = format!("{kernel}_{}", w.dtype.name());
        self.dispatch_fused(r, label, &kernel, &args, w.rows, w.cols);
    }

    fn dispatch_fused(
        &self,
        r: &mut Runner,
        label: &'static str,
        kernel: &str,
        args: &[Arg],
        rows: usize,
        d_in: usize,
    ) {
        let rows_per_tg = FUSED_ROWS * FUSED_SIMDGROUPS;
        r.dispatch_tg(
            label,
            kernel,
            args,
            (rows.div_ceil(rows_per_tg), 1, 1),
            (32 * FUSED_SIMDGROUPS, 1, 1),
            (32 + d_in) * F,
        );
    }

    /// Fusion::Epilogue decode body (after the embedding): LayerNorms stay
    /// separate; bias+GELU, residuals and KV writes ride in matvec epilogues.
    fn encode_decode_epilogue(&self, r: &mut Runner, argmax: bool) {
        let w = &self.e.w;
        for (l, lw) in w.layers.iter().enumerate() {
            let layer_off = self.kv_layer_off(l);
            let kv = Some((Arg::BufAt(&self.k_cache, layer_off), Arg::BufAt(&self.v_cache, layer_off)));
            self.layernorm(r, "ln_1", 0, &lw.ln1_w, &lw.ln1_b, 1);
            if self.e.recipe.kv_q8 {
                self.epi(r, "qkv", "mve", &self.qkv, &self.xn, &lw.attn_w, Some(&lw.attn_b), None);
                self.kv_quant8(r, l, self.pos, 1);
                self.attention(r, l, 1);
                self.epi(r, "attn_proj", "mve_resid", &self.x, &self.att, &lw.proj_w, Some(&lw.proj_b), None);
                self.layernorm(r, "ln_2", 0, &lw.ln2_w, &lw.ln2_b, 1);
                self.epi(r, "fc", "mve_gelu", &self.fc, &self.xn, &lw.fc_w, Some(&lw.fc_b), None);
                self.epi(r, "fc_proj", "mve_resid", &self.x, &self.fc, &lw.fcp_w, Some(&lw.fcp_b), None);
                continue;
            }
            self.epi(
                r,
                "qkv",
                if self.e.recipe.kv_f16 { "mve_qkv_h" } else { "mve_qkv" },
                &self.qkv,
                &self.xn,
                &lw.attn_w,
                Some(&lw.attn_b),
                kv,
            );
            self.attention(r, l, 1);
            self.epi(r, "attn_proj", "mve_resid", &self.x, &self.att, &lw.proj_w, Some(&lw.proj_b), None);
            self.layernorm(r, "ln_2", 0, &lw.ln2_w, &lw.ln2_b, 1);
            self.epi(r, "fc", "mve_gelu", &self.fc, &self.xn, &lw.fc_w, Some(&lw.fc_b), None);
            self.epi(r, "fc_proj", "mve_resid", &self.x, &self.fc, &lw.fcp_w, Some(&lw.fcp_b), None);
        }
        self.layernorm(r, "ln_f", 0, &w.ln_f_w, &w.ln_f_b, 1);
        self.epi(r, "lm_head", "mve", &self.logits, &self.xn, &w.lm_head, None, None);
        if argmax {
            self.encode_argmax(r);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn epi(
        &self,
        r: &mut Runner,
        label: &'static str,
        kernel: &str,
        out: &Buf,
        x: &Buf,
        w: &Mat,
        bias: Option<&Buf>,
        kv: Option<(Arg, Arg)>,
    ) {
        let (kc, vc) = kv.unwrap_or((Arg::Buf(out), Arg::Buf(out)));
        let has_bias = bias.is_some() as u32;
        let bias = bias.unwrap_or(x);
        let args = [
            Arg::Buf(out),
            Arg::Buf(x),
            Arg::Buf(&w.buf),
            Arg::Buf(bias),
            Arg::Buf(bias), // 4, 5: unused (LayerNorm params in matvec_fused)
            Arg::Buf(bias),
            kc,
            vc,
            Arg::U32(w.cols as u32),
            Arg::U32(w.rows as u32),
            Arg::U32(has_bias),
            Arg::U32(self.pos as u32),
        ];
        if self.e.recipe.lanes && w.dtype != DType::F32 {
            // matvec_lanes: 8 lanes per row, a SIMD-group owns R * 4 rows.
            let (rr, nsg) = lanes_shape();
            let kernel = format!("{}_{}_r{rr}", kernel.replace("mve", "mvl"), w.dtype.name());
            let rows_per_tg = rr * 4 * nsg;
            r.dispatch_tg(label, &kernel, &args, (w.rows.div_ceil(rows_per_tg), 1, 1), (32 * nsg, 1, 1), 0);
            return;
        }
        let rows_per_sg = match w.dtype {
            DType::F32 | DType::F16 => 4,
            DType::Q8_0 | DType::Q4_0 => 2,
        };
        let kernel = format!("{kernel}_{}", w.dtype.name());
        r.dispatch_tg(label, &kernel, &args, (w.rows.div_ceil(rows_per_sg * 4), 1, 1), (128, 1, 1), 0);
    }

    fn encode_forward_unfused(&self, r: &mut Runner, n: usize) {
        r.set_pending_tokens(n);
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
            let layer_off = self.kv_layer_off(l);
            self.layernorm(r, "ln_1", 0, &lw.ln1_w, &lw.ln1_b, n);
            self.linear(r, "qkv", &self.qkv, &self.xn, 0, &lw.attn_w, Some(&lw.attn_b), n);
            if self.e.recipe.kv_q8 {
                self.kv_quant8(r, l, self.pos, n);
            } else {
                r.dispatch(
                    "kv_write",
                    self.kv_kernel("kv_write"),
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
            }
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

/// matvec_lanes shape: (row-sets per SIMD-group, SIMD-groups per threadgroup).
/// `NANOINFER_LANES=R,NSG` overrides it for sweeps.
fn lanes_shape() -> (usize, usize) {
    static S: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        std::env::var("NANOINFER_LANES")
            .ok()
            .and_then(|v| {
                let (a, b) = v.split_once(',')?;
                Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
            })
            .unwrap_or(LANES_DEFAULT)
    })
}
const LANES_DEFAULT: (usize, usize) = (1, 4);

/// The SIMD-group matvec kernel for a weight dtype: (name, rows per
/// SIMD-group, SIMD-groups per threadgroup).
fn matvec_kernel(dtype: DType) -> (&'static str, usize, usize) {
    match dtype {
        DType::F32 => (MV_KERNEL, MV_ROWS, MV_SIMDGROUPS),
        DType::F16 => ("matvec_f16_r4", 4, 4),
        DType::Q8_0 => ("matvec_q8_0_r2", 2, 4),
        DType::Q4_0 => ("matvec_q4_0_r2", 2, 4),
    }
}
