//! Round 2: batched, ragged, paged forward passes.
//!
//! A [`Batch`] owns the KV cache for many sequences at once and runs forward
//! passes over a ragged list of rows (one row = one token of one sequence).
//! The same forward serves every serving pattern in round 2:
//!
//! - **batched decode**: one row per sequence, B sequences side by side; the
//!   matvecs become skinny matrix products, so each weight byte read from
//!   DRAM is used B times ([`Batch::decode_resident`] keeps the loop on the GPU);
//! - **prefill / chunked prefill**: many consecutive rows of one sequence;
//! - **continuous batching**: prefill chunks of new requests mixed with decode
//!   rows of running ones, in the same forward;
//! - **speculative verification**: k + 1 rows of one sequence, greedy
//!   predictions at every row ([`Batch::step`] with `all_logits`).
//!
//! The KV cache is paged: per layer a pool of [`KV_BLOCK`]-position blocks,
//! and a per-sequence block table (vLLM's PagedAttention). Blocks are handed
//! out on demand and returned when a sequence finishes, so memory is used by
//! tokens that exist rather than by `max_len` reservations.
//!
//! Every kernel used here for ≤ [`MAX_NB`] rows computes each token's dot
//! products in the same order whatever the batch size, so a sequence's tokens
//! are bit-identical alone or batched (tested). With more rows than
//! `gemm_min_rows` the forward switches to the simdgroup-matrix GEMMs, which
//! are faster for big batches but round differently.

use crate::gpu::{Arg, Buf, Gpu, Pending, Runner, Sync};
use crate::weights::{Mat, Weights};
use crate::DType;
use nanoinfer_core::{Config, Model};

/// Positions per KV-cache page (must match KV_BLOCK in batch.metal).
pub const KV_BLOCK: usize = 16;
/// Largest token chunk the batched matvec multiplies per weight load.
pub const MAX_NB: usize = 16;
const ATT_CHUNK: usize = 64;
const F: usize = 4;
/// SIMD-groups per threadgroup in the batched matvec (4 rows each).
const MVB_NSG: usize = 4;
/// Rows from which `gemm` uses the 64 x 64-tile `matmul_mma2` kernels.
pub const GEMM2_MIN_ROWS: usize = 128;

/// Sizes and policies of a [`Batch`].
#[derive(Debug, Clone, Copy)]
pub struct BatchOpts {
    pub dtype: DType,
    pub head_dtype: DType,
    /// Most rows in one forward (prefill chunks included).
    pub max_rows: usize,
    /// Most sequences alive at once.
    pub max_seqs: usize,
    /// KV-cache pages in the pool (each holds KV_BLOCK positions of every layer).
    pub kv_blocks: usize,
    /// Forwards with at least this many rows use the GEMM kernels instead of
    /// the batched matvec (`usize::MAX`: never).
    pub gemm_min_rows: usize,
    /// Weight rows per lane in the batched matvec, by chunk size NB = 1, 2,
    /// 4, 8, 16 (0 = the R = 1 `matvec_batch` kernel). Results are
    /// bit-identical for every choice; only speed changes.
    pub rows_per_lane: [usize; 5],
}

impl BatchOpts {
    pub fn new(dtype: DType) -> BatchOpts {
        let head_dtype = if dtype == DType::Q4_0 { DType::Q8_0 } else { dtype };
        BatchOpts {
            dtype,
            head_dtype,
            max_rows: 512,
            max_seqs: 64,
            kv_blocks: 1024,
            gemm_min_rows: 32,
            rows_per_lane: rows_per_lane_default(),
        }
    }
}

/// `NANOINFER_MVR=a,b,c,d,e` overrides [`BatchOpts::rows_per_lane`] (sweeps).
fn rows_per_lane_default() -> [usize; 5] {
    let parsed = std::env::var("NANOINFER_MVR").ok().and_then(|v| {
        let xs: Vec<usize> = v.split(',').map(|x| x.trim().parse().ok()).collect::<Option<_>>()?;
        <[usize; 5]>::try_from(xs).ok()
    });
    parsed.unwrap_or([0, 2, 2, 2, 2])
}

/// GPU-resident weights for batched forwards.
pub struct BatchEngine {
    pub gpu: Gpu,
    pub config: Config,
    pub opts: BatchOpts,
    w: Weights,
}

impl BatchEngine {
    pub fn new(model: &Model, opts: BatchOpts) -> Result<BatchEngine, String> {
        let gpu = Gpu::new()?;
        let w = Weights::upload(&gpu, model, opts.dtype, opts.head_dtype);
        Ok(BatchEngine { gpu, config: model.config, opts, w })
    }

    pub fn batch(&self) -> Batch<'_> {
        Batch::new(self)
    }

    /// Bytes of one KV page (all layers, K and V, f16).
    pub fn kv_block_bytes(&self) -> usize {
        2 * self.config.n_layer * KV_BLOCK * self.config.n_embd * 2
    }
}

/// One row of a forward: token `pos` of sequence `seq`.
#[derive(Debug, Clone, Copy)]
struct Row {
    seq: u32,
    pos: u32,
}

/// Counters for the last forward(s).
#[derive(Debug, Default, Clone, Copy)]
pub struct BatchStats {
    pub rows: usize,
    pub dispatches: usize,
    pub gpu_seconds: f64,
    pub command_buffers: usize,
}

/// KV cache + activations for a set of sequences.
pub struct Batch<'e> {
    e: &'e BatchEngine,
    max_blocks: usize,
    // CPU-side bookkeeping
    seq_len: Vec<usize>,
    seq_live: Vec<bool>,
    seq_blocks: Vec<Vec<u32>>,
    free_blocks: Vec<u32>,
    // GPU buffers
    rows: Buf,
    lrows: Buf,
    toks: Buf,
    table: Buf,
    k_pool: Buf,
    v_pool: Buf,
    x: Buf,
    xn: Buf,
    qkv: Buf,
    att: Buf,
    proj: Buf,
    fc: Buf,
    att_part: Buf,
    hx: Buf,
    logits: Buf,
    out: Buf,
    max_logit_rows: usize,
    pub last: BatchStats,
}

impl<'e> Batch<'e> {
    fn new(e: &'e BatchEngine) -> Self {
        let cfg = e.config;
        let o = e.opts;
        let g = &e.gpu;
        let c = cfg.n_embd;
        let max_blocks = cfg.n_ctx.div_ceil(KV_BLOCK);
        let max_logit_rows = o.max_rows.min(o.max_seqs.max(MAX_NB + 1));
        let pool = cfg.n_layer * o.kv_blocks * KV_BLOCK * c * 2;
        let max_chunks = cfg.n_ctx.div_ceil(ATT_CHUNK);
        Batch {
            e,
            max_blocks,
            seq_len: vec![0; o.max_seqs],
            seq_live: vec![false; o.max_seqs],
            seq_blocks: vec![Vec::new(); o.max_seqs],
            free_blocks: (0..o.kv_blocks as u32).rev().collect(),
            rows: g.buffer(o.max_rows * 16),
            lrows: g.buffer(o.max_rows * 4),
            toks: g.buffer(o.max_seqs * cfg.n_ctx * 4),
            table: g.buffer(o.max_seqs * max_blocks * 4),
            k_pool: g.buffer(pool),
            v_pool: g.buffer(pool),
            x: g.buffer(o.max_rows * c * F),
            xn: g.buffer(o.max_rows * c * F),
            qkv: g.buffer(o.max_rows * 3 * c * F),
            att: g.buffer(o.max_rows * c * F),
            proj: g.buffer(o.max_rows * c * F),
            fc: g.buffer(o.max_rows * 4 * c * F),
            att_part: g.buffer(o.max_rows * cfg.n_head * max_chunks * (cfg.head_dim() + 2) * F),
            hx: g.buffer(max_logit_rows * c * F),
            logits: g.buffer(max_logit_rows * cfg.n_vocab * F),
            out: g.buffer(max_logit_rows * 4),
            max_logit_rows,
            last: BatchStats::default(),
        }
    }

    pub fn config(&self) -> Config {
        self.e.config
    }

    /// Start a new (empty) sequence; `None` if all sequence slots are taken.
    pub fn add_seq(&mut self) -> Option<usize> {
        let s = self.seq_live.iter().position(|l| !l)?;
        self.seq_live[s] = true;
        self.seq_len[s] = 0;
        Some(s)
    }

    /// Finish a sequence and return its KV pages to the pool.
    pub fn free_seq(&mut self, s: usize) {
        self.truncate(s, 0);
        self.seq_live[s] = false;
    }

    pub fn seq_len(&self, s: usize) -> usize {
        self.seq_len[s]
    }

    /// Forget a sequence's positions from `len` on (speculative rollback).
    /// Pages wholly past `len` go back to the pool.
    pub fn truncate(&mut self, s: usize, len: usize) {
        assert!(len <= self.seq_len[s]);
        self.seq_len[s] = len;
        let keep = len.div_ceil(KV_BLOCK);
        while self.seq_blocks[s].len() > keep {
            self.free_blocks.push(self.seq_blocks[s].pop().unwrap());
        }
    }

    /// The KV pages (block table) of sequence `s`, in position order.
    pub fn block_table(&self, s: usize) -> &[u32] {
        &self.seq_blocks[s]
    }

    /// KV pages currently allocated / in the pool.
    pub fn blocks_used(&self) -> usize {
        self.e.opts.kv_blocks - self.free_blocks.len()
    }
    pub fn free_block_count(&self) -> usize {
        self.free_blocks.len()
    }

    /// Make sure sequence `s` has pages for positions `< len`.
    fn ensure_blocks(&mut self, s: usize, len: usize) -> Result<(), String> {
        if len > self.e.config.n_ctx {
            return Err(format!("sequence {s}: context full ({len} > {})", self.e.config.n_ctx));
        }
        while self.seq_blocks[s].len() * KV_BLOCK < len {
            let b = self.free_blocks.pop().ok_or("KV cache pool exhausted")?;
            let i = self.seq_blocks[s].len();
            self.table.write(s * self.max_blocks + i, &[b]);
            self.seq_blocks[s].push(b);
        }
        Ok(())
    }

    /// One forward over `work`: each `(seq, tokens)` appends `tokens` to that
    /// sequence. Returns, per work item, the greedy next token after each of
    /// its tokens (`all_logits`) or after its last token only.
    pub fn step(&mut self, work: &[(usize, &[u32])], all_logits: bool) -> Result<Vec<Vec<u32>>, String> {
        let (p, n_logit, rows) = self.encode_step(work, all_logits)?;
        let gpu = p.wait();
        self.last = BatchStats { rows, dispatches: self.last.dispatches, gpu_seconds: gpu, command_buffers: 1 };
        let out = self.out.slice::<u32>(n_logit);
        let mut res = Vec::with_capacity(work.len());
        let mut k = 0;
        for (s, toks) in work {
            let m = if all_logits { toks.len() } else { 1 };
            res.push(out[k..k + m].to_vec());
            k += m;
            self.seq_len[*s] += toks.len();
        }
        Ok(res)
    }

    /// Encode and commit one forward without waiting; returns (pending, logit rows, rows).
    fn encode_step(&mut self, work: &[(usize, &[u32])], all: bool) -> Result<(Pending, usize, usize), String> {
        let cfg = self.e.config;
        let mut rows = Vec::new();
        let mut lrows = Vec::new();
        for &(s, toks) in work {
            if !self.seq_live[s] {
                return Err(format!("sequence {s} is not live"));
            }
            if toks.is_empty() {
                return Err("empty work item".into());
            }
            if let Some(t) = toks.iter().find(|&&t| t as usize >= cfg.n_vocab) {
                return Err(format!("bad token {t}"));
            }
            let len = self.seq_len[s];
            self.ensure_blocks(s, len + toks.len())?;
            self.toks.write(s * cfg.n_ctx + len, toks);
            for i in 0..toks.len() {
                if all || i + 1 == toks.len() {
                    lrows.push(rows.len() as u32);
                }
                rows.push(Row { seq: s as u32, pos: (len + i) as u32 });
            }
        }
        if rows.len() > self.e.opts.max_rows {
            return Err(format!("{} rows > max_rows {}", rows.len(), self.e.opts.max_rows));
        }
        if lrows.len() > self.max_logit_rows {
            return Err(format!("{} logit rows > {}", lrows.len(), self.max_logit_rows));
        }
        let packed: Vec<u32> = rows.iter().flat_map(|r| [r.seq, r.pos, 0, 0]).collect();
        self.rows.write(0, &packed);
        self.lrows.write(0, &lrows);
        let max_pos = rows.iter().map(|r| r.pos as usize).max().unwrap();
        let mut r = self.e.gpu.runner(Sync::PerForward);
        self.encode_forward(&mut r, rows.len(), lrows.len(), 0, max_pos);
        self.last.dispatches = r.stats.dispatches;
        let p = r.commit().expect("nothing encoded");
        Ok((p, lrows.len(), rows.len()))
    }

    /// Greedy-decode `steps` tokens for each of `seqs` (one row per sequence
    /// per step), feeding `first[i]` to `seqs[i]` first. The loop stays on the
    /// GPU: step k's embedding reads the token step k − 1's argmax wrote, all
    /// command buffers are committed up front, and the CPU waits once.
    /// Returns the generated tokens per sequence.
    pub fn decode_resident(&mut self, seqs: &[usize], first: &[u32], steps: usize) -> Result<Vec<Vec<u32>>, String> {
        let cfg = self.e.config;
        assert_eq!(seqs.len(), first.len());
        let n = seqs.len();
        if n == 0 || steps == 0 {
            return Ok(vec![vec![]; n]);
        }
        if n > self.e.opts.max_rows || n > self.max_logit_rows {
            return Err(format!("batch {n} too large"));
        }
        let mut packed = Vec::with_capacity(4 * n);
        let mut max_pos = 0;
        for (&s, &t) in seqs.iter().zip(first) {
            let len = self.seq_len[s];
            self.ensure_blocks(s, len + steps)?;
            self.toks.write(s * cfg.n_ctx + len, &[t]);
            packed.extend([s as u32, len as u32, 0, 0]);
            max_pos = max_pos.max(len);
        }
        self.rows.write(0, &packed);
        self.lrows.write(0, &(0..n as u32).collect::<Vec<_>>());
        let mut pending = Vec::with_capacity(steps);
        let mut dispatches = 0;
        for k in 0..steps {
            let mut r = self.e.gpu.runner(Sync::PerForward);
            self.encode_forward(&mut r, n, n, k, max_pos + k);
            dispatches += r.stats.dispatches;
            pending.push(r.commit().expect("encoded"));
        }
        let mut gpu = 0.0;
        for p in pending {
            gpu += p.wait();
        }
        self.last = BatchStats { rows: n, dispatches: dispatches / steps, gpu_seconds: gpu, command_buffers: steps };
        let mut res = Vec::with_capacity(n);
        for &s in seqs {
            let len = self.seq_len[s];
            res.push(self.toks.slice::<u32>(s * cfg.n_ctx + len + steps + 1)[s * cfg.n_ctx + len + 1..].to_vec());
            self.seq_len[s] += steps;
        }
        Ok(res)
    }

    /// Encode a forward over the `n` rows in `self.rows` (positions + `step`),
    /// with greedy argmax for the `m` rows listed in `self.lrows`.
    fn encode_forward(&self, r: &mut Runner, n: usize, m: usize, step: usize, max_pos: usize) {
        let cfg = self.e.config;
        let w = &self.e.w;
        let c = cfg.n_embd;
        let (nu, cu, stepu) = (n as u32, c as u32, step as u32);
        r.dispatch(
            "embed",
            "b_embed",
            &[
                Arg::Buf(&self.x),
                Arg::Buf(&w.wte),
                Arg::Buf(&w.wpe),
                Arg::Buf(&self.toks),
                Arg::Buf(&self.rows),
                Arg::U32(nu),
                Arg::U32(cu),
                Arg::U32(stepu),
                Arg::U32(cfg.n_ctx as u32),
            ],
            n * c,
            256,
        );
        let gemm = n >= self.e.opts.gemm_min_rows;
        let layer_bytes = self.e.opts.kv_blocks * KV_BLOCK * c * 2;
        for (l, lw) in w.layers.iter().enumerate() {
            let kv = (l * layer_bytes, step);
            self.layernorm(r, "ln_1", &lw.ln1_w, &lw.ln1_b, n);
            if gemm {
                self.gemm(r, "qkv", &self.qkv, &self.xn, &lw.attn_w, Some(&lw.attn_b), n);
                r.dispatch(
                    "kv_write",
                    "b_kv_write",
                    &[
                        Arg::BufAt(&self.k_pool, kv.0),
                        Arg::BufAt(&self.v_pool, kv.0),
                        Arg::Buf(&self.qkv),
                        Arg::Buf(&self.rows),
                        Arg::Buf(&self.table),
                        Arg::U32(nu),
                        Arg::U32(cu),
                        Arg::U32(stepu),
                        Arg::U32(self.max_blocks as u32),
                    ],
                    n * c,
                    256,
                );
            } else {
                self.mvb(r, "qkv", "mvb_qkv", &self.qkv, &self.xn, &lw.attn_w, Some(&lw.attn_b), n, kv);
            }
            self.attention(r, l, n, step, max_pos);
            if gemm {
                self.gemm(r, "attn_proj", &self.proj, &self.att, &lw.proj_w, Some(&lw.proj_b), n);
                self.residual(r, "residual_1", n * c);
            } else {
                self.mvb(r, "attn_proj", "mvb_resid", &self.x, &self.att, &lw.proj_w, Some(&lw.proj_b), n, kv);
            }
            self.layernorm(r, "ln_2", &lw.ln2_w, &lw.ln2_b, n);
            if gemm {
                self.gemm(r, "fc", &self.fc, &self.xn, &lw.fc_w, Some(&lw.fc_b), n);
                r.dispatch("gelu", "gelu_naive", &[Arg::Buf(&self.fc), Arg::U32((n * 4 * c) as u32)], n * 4 * c, 256);
                self.gemm(r, "fc_proj", &self.proj, &self.fc, &lw.fcp_w, Some(&lw.fcp_b), n);
                self.residual(r, "residual_2", n * c);
            } else {
                self.mvb(r, "fc", "mvb_gelu", &self.fc, &self.xn, &lw.fc_w, Some(&lw.fc_b), n, kv);
                self.mvb(r, "fc_proj", "mvb_resid", &self.x, &self.fc, &lw.fcp_w, Some(&lw.fcp_b), n, kv);
            }
        }
        // LM head over the m logit rows only.
        r.dispatch_tg(
            "ln_f",
            "b_ln_gather",
            &[
                Arg::Buf(&self.hx),
                Arg::Buf(&self.x),
                Arg::Buf(&w.ln_f_w),
                Arg::Buf(&w.ln_f_b),
                Arg::Buf(&self.lrows),
                Arg::U32(cu),
            ],
            (m, 1, 1),
            (256, 1, 1),
            32 * F,
        );
        if m >= self.e.opts.gemm_min_rows {
            self.gemm(r, "lm_head", &self.logits, &self.hx, &w.lm_head, None, m);
        } else {
            self.mvb(r, "lm_head", "mvb", &self.logits, &self.hx, &w.lm_head, None, m, (0, step));
        }
        r.dispatch_tg(
            "argmax",
            "b_argmax_rows",
            &[
                Arg::Buf(&self.logits),
                Arg::Buf(&self.out),
                Arg::Buf(&self.toks),
                Arg::Buf(&self.rows),
                Arg::Buf(&self.lrows),
                Arg::U32(cfg.n_vocab as u32),
                Arg::U32(stepu),
                Arg::U32(cfg.n_ctx as u32),
            ],
            (m, 1, 1),
            (1024, 1, 1),
            64 * F,
        );
    }

    fn layernorm(&self, r: &mut Runner, label: &'static str, w: &Buf, b: &Buf, n: usize) {
        r.dispatch_tg(
            label,
            "layernorm_simd",
            &[
                Arg::Buf(&self.xn),
                Arg::Buf(&self.x),
                Arg::Buf(w),
                Arg::Buf(b),
                Arg::U32(n as u32),
                Arg::U32(self.e.config.n_embd as u32),
            ],
            (n, 1, 1),
            (256, 1, 1),
            32 * F,
        );
    }

    fn residual(&self, r: &mut Runner, label: &'static str, len: usize) {
        r.dispatch(label, "residual_add", &[Arg::Buf(&self.x), Arg::Buf(&self.proj), Arg::U32(len as u32)], len, 256);
    }

    /// Batched matvec over n rows: chunks of NB rows share each weight load.
    #[allow(clippy::too_many_arguments)]
    fn mvb(
        &self,
        r: &mut Runner,
        label: &'static str,
        kernel: &str,
        out: &Buf,
        x: &Buf,
        w: &Mat,
        bias: Option<&Buf>,
        n: usize,
        (kv_off, step): (usize, usize),
    ) {
        // Chunk size: a power of two up to 8, then 12 or 16 (9 rows on the
        // 16-row kernel cost ~25% more than 8: speculative verify at k = 8).
        let nb = match n {
            0..=8 => n.next_power_of_two(),
            9..=12 => 12,
            _ => MAX_NB,
        };
        let has_bias = bias.is_some() as u32;
        let bias = bias.unwrap_or(x);
        let args = [
            Arg::Buf(out),
            Arg::Buf(x),
            Arg::Buf(&w.buf),
            Arg::Buf(bias),
            Arg::BufAt(&self.k_pool, kv_off),
            Arg::BufAt(&self.v_pool, kv_off),
            Arg::Buf(&self.rows),
            Arg::Buf(&self.table),
            Arg::U32(w.cols as u32),
            Arg::U32(w.rows as u32),
            Arg::U32(has_bias),
            Arg::U32(n as u32),
            Arg::U32(step as u32),
            Arg::U32(self.max_blocks as u32),
        ];
        // rows per lane by chunk size; 12 shares the 16-row setting
        let rpl = self.e.opts.rows_per_lane[if nb == 12 { 4 } else { nb.trailing_zeros() as usize }];
        let (kernel, rows_per_sg) = if rpl == 0 {
            (format!("{kernel}_{}_b{nb}", w.dtype.name()), 4)
        } else {
            (format!("{}_{}_b{nb}_r{rpl}", kernel.replacen("mvb", "mvr", 1), w.dtype.name()), 4 * rpl)
        };
        let groups = (w.rows.div_ceil(rows_per_sg * MVB_NSG), n.div_ceil(nb), 1);
        r.dispatch_tg(label, &kernel, &args, groups, (32 * MVB_NSG, 1, 1), 0);
    }

    /// simdgroup-matrix GEMM (the prefill kernels), out = x · Wᵀ (+ bias).
    #[allow(clippy::too_many_arguments)]
    fn gemm(&self, r: &mut Runner, label: &'static str, out: &Buf, x: &Buf, w: &Mat, bias: Option<&Buf>, n: usize) {
        let has_bias = bias.is_some() as u32;
        let bias = bias.unwrap_or(x);
        let args = [
            Arg::Buf(out),
            Arg::Buf(x),
            Arg::Buf(&w.buf),
            Arg::Buf(bias),
            Arg::U32(n as u32),
            Arg::U32(w.cols as u32),
            Arg::U32(w.rows as u32),
            Arg::U32(has_bias),
        ];
        // From 128 rows the 64 x 64-tile GEMM with a half weight tile is 5-15%
        // faster (examples/gemm_sweep.rs); below it, half its tile is padding.
        let big = n >= GEMM2_MIN_ROWS && w.dtype != DType::F32;
        let kernel = match (w.dtype, big) {
            (DType::F32, _) => "matmul_mma",
            (DType::F16, false) => "matmul_mma_f16",
            (DType::Q8_0, false) => "matmul_mma_q8_0",
            (DType::Q4_0, false) => "matmul_mma_q4_0",
            (DType::F16, true) => "matmul_mma2_f16",
            (DType::Q8_0, true) => "matmul_mma2_q8_0",
            (DType::Q4_0, true) => "matmul_mma2_q4_0",
        };
        let bm = if big { 64 } else { 32 };
        r.dispatch_tg(label, kernel, &args, (w.rows.div_ceil(64), n.div_ceil(bm), 1), (128, 1, 1), 0);
    }

    fn attention(&self, r: &mut Runner, l: usize, n: usize, step: usize, max_pos: usize) {
        let cfg = self.e.config;
        let layer_bytes = l * self.e.opts.kv_blocks * KV_BLOCK * cfg.n_embd * 2;
        let max_chunks = cfg.n_ctx.div_ceil(ATT_CHUNK) as u32;
        let chunks = (max_pos + 1).div_ceil(ATT_CHUNK);
        r.dispatch_tg(
            "attn",
            "b_attn_split",
            &[
                Arg::Buf(&self.qkv),
                Arg::BufAt(&self.k_pool, layer_bytes),
                Arg::BufAt(&self.v_pool, layer_bytes),
                Arg::Buf(&self.att_part),
                Arg::Buf(&self.rows),
                Arg::Buf(&self.table),
                Arg::U32(cfg.n_head as u32),
                Arg::U32(max_chunks),
                Arg::U32(step as u32),
                Arg::U32(self.max_blocks as u32),
            ],
            (cfg.n_head, chunks, n),
            (256, 1, 1),
            (64 + 32 + 256) * F,
        );
        r.dispatch_tg(
            "attn",
            "b_attn_merge",
            &[
                Arg::Buf(&self.att_part),
                Arg::Buf(&self.att),
                Arg::Buf(&self.rows),
                Arg::U32(cfg.n_head as u32),
                Arg::U32(max_chunks),
                Arg::U32(step as u32),
            ],
            (cfg.n_head, n, 1),
            (64, 1, 1),
            0,
        );
    }
}
