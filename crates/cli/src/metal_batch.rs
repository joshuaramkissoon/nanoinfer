//! `nanoinfer metal-batch`: batched decode on the GPU (round 2).
//!
//! B independent sequences decode together: each step is one forward with
//! B rows, so every weight byte read from DRAM serves B tokens. Prints
//! aggregate tok/s and per-stream latency for each B, the bytes and FLOPs
//! per step, and appends to `bench/results.jsonl` (stage `metal-batch`) and
//! a small summary for the web page (`bench/batching.json`).

use crate::bench::{begin, git_sha, machine, median, prefer_performance_cores, stamp, stddev, timestamp, BENCH_TEXT};
use clap::Parser;
use nanoinfer_metal::batch::KV_BLOCK;
use nanoinfer_metal::{BatchEngine, BatchOpts, DType};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "nanoinfer metal-batch")]
pub struct BatchArgs {
    #[arg(long, default_value = "weights/gpt2")]
    weights: PathBuf,
    /// Weight dtypes to sweep (comma-separated: f32, f16, q8_0, q4_0).
    #[arg(long, default_value = "q4_0")]
    dtype: String,
    /// Batch sizes to sweep.
    #[arg(long, default_value = "1,2,4,8,16,32,64")]
    batch: String,
    #[arg(long, default_value_t = 64)]
    prompt_len: usize,
    /// Decode steps per run (each step produces B tokens).
    #[arg(long, default_value_t = 64)]
    tokens: usize,
    #[arg(long, default_value_t = 3)]
    runs: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    /// Rows from which the forward switches from the batched matvec to the
    /// simdgroup-matrix GEMMs: a number, or "never".
    #[arg(long, default_value = "32")]
    gemm_min_rows: String,
    /// Label for this series in the summary JSON (default: dtype + kernel policy).
    #[arg(long)]
    label: Option<String>,
    #[arg(long, default_value_t = 120.0)]
    bandwidth_gbs: f64,
    #[arg(long, default_value = "bench/results.jsonl")]
    out: PathBuf,
    #[arg(long, default_value = "bench/batching.json")]
    summary: PathBuf,
    #[arg(long)]
    no_record: bool,
    /// Record even if the engine sources have uncommitted changes (marked dirty).
    #[arg(long)]
    allow_dirty: bool,
    #[arg(long, default_value = "")]
    note: String,
}

pub fn parse_dtype(s: &str) -> Result<DType, String> {
    match s {
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "q8_0" | "q8" => Ok(DType::Q8_0),
        "q4_0" | "q4" => Ok(DType::Q4_0),
        _ => Err(format!("unknown dtype {s}")),
    }
}

/// Bytes of weights one forward reads (each matrix once, whatever the batch).
pub fn weight_bytes(cfg: &nanoinfer_core::Config, dtype: DType, head: DType) -> f64 {
    let c = cfg.n_embd;
    let blocks = cfg.n_layer * 12 * c * c;
    let headp = cfg.n_vocab * c;
    let small = cfg.n_params() - cfg.n_ctx * c - blocks - headp;
    blocks as f64 * dtype.bytes_per_weight() + headp as f64 * head.bytes_per_weight() + small as f64 * 4.0
}

/// `n` prompts of `len` tokens, each starting at a different offset of the bench text.
pub fn prompts(tok: &nanoinfer_core::Tokenizer, n: usize, len: usize) -> Vec<Vec<u32>> {
    let mut text = Vec::new();
    while text.len() < len + 8 * n + 1 {
        text.extend(tok.encode(BENCH_TEXT));
    }
    (0..n).map(|i| text[8 * i..8 * i + len].to_vec()).collect()
}

pub fn run(a: BatchArgs) -> Result<(), Box<dyn std::error::Error>> {
    begin(!a.no_record, a.allow_dirty)?;
    prefer_performance_cores();
    let tok = crate::load_tokenizer(&a.weights)?;
    let model = crate::load_model(&a.weights)?;
    let cfg = model.config;
    let batches: Vec<usize> = a.batch.split(',').map(|s| s.trim().parse()).collect::<Result<_, _>>()?;
    let max_b = *batches.iter().max().unwrap();
    let gemm_min_rows = if a.gemm_min_rows == "never" { usize::MAX } else { a.gemm_min_rows.parse()? };
    let ctx = a.prompt_len + a.tokens;
    if ctx > cfg.n_ctx {
        return Err("prompt_len + tokens > n_ctx".into());
    }
    let model_name = model_label(&a.weights);
    for dt in a.dtype.split(',') {
        let dtype = parse_dtype(dt.trim())?;
        let opts = BatchOpts {
            max_seqs: max_b,
            max_rows: 512.max(max_b),
            kv_blocks: max_b * ctx.div_ceil(KV_BLOCK) + 4,
            gemm_min_rows,
            ..BatchOpts::new(dtype)
        };
        let engine = BatchEngine::new(&model, opts)?;
        let wbytes = weight_bytes(&cfg, opts.dtype, opts.head_dtype);
        let kv_per_pos = (2 * cfg.n_layer * cfg.n_embd * 2) as f64;
        let flops_per_tok = 2.0 * (cfg.n_layer * 12 * cfg.n_embd * cfg.n_embd + cfg.n_vocab * cfg.n_embd) as f64;
        let label = a
            .label
            .clone()
            .unwrap_or_else(|| format!("{}{}", dtype.name(), if gemm_min_rows == usize::MAX { " matvec" } else { "" }));
        eprintln!(
            "metal-batch: {} {} ({label}), {:.1} MB weights/step, KV pool {} pages ({:.0} MB), gemm from {} rows",
            engine.gpu.name(),
            dtype.name(),
            wbytes / 1e6,
            opts.kv_blocks,
            opts.kv_blocks as f64 * engine.kv_block_bytes() as f64 / 1e6,
            a.gemm_min_rows
        );
        println!(
            "{:>4} {:>10} {:>9} {:>9} {:>8} {:>8} {:>8} {:>9} {:>7}",
            "B", "agg tok/s", "±", "stream", "step ms", "GPU ms", "GB/s", "GFLOP/s", "vs B=1"
        );
        let mut points = Vec::new();
        let mut base = None;
        for &b in &batches {
            let ps = prompts(&tok, b, a.prompt_len);
            let (mut agg, mut gpu_ms) = (vec![], vec![]);
            for run in 0..a.warmup + a.runs {
                let mut batch = engine.batch();
                let seqs: Vec<usize> = (0..b).map(|_| batch.add_seq().unwrap()).collect();
                // Prefill in chunks of whole prompts that fit in max_rows.
                let per = (opts.max_rows / a.prompt_len).max(1);
                let mut first = Vec::with_capacity(b);
                for (ss, pp) in seqs.chunks(per).zip(ps.chunks(per)) {
                    let work: Vec<(usize, &[u32])> = ss.iter().zip(pp).map(|(&s, p)| (s, p.as_slice())).collect();
                    first.extend(batch.step(&work, false)?.into_iter().map(|v| v[0]));
                }
                let t = Instant::now();
                batch.decode_resident(&seqs, &first, a.tokens)?;
                let wall = t.elapsed().as_secs_f64();
                if run >= a.warmup {
                    agg.push((b * a.tokens) as f64 / wall);
                    gpu_ms.push(1e3 * batch.last.gpu_seconds / a.tokens as f64);
                }
            }
            let (am, asd, gm) = (median(&agg), stddev(&agg), median(&gpu_ms));
            let step_ms = 1e3 * b as f64 / am;
            let avg_ctx = a.prompt_len as f64 + a.tokens as f64 / 2.0;
            let bytes = wbytes + b as f64 * avg_ctx * kv_per_pos;
            let gbs = bytes / (step_ms * 1e-3) / 1e9;
            let gflops = b as f64 * flops_per_tok / (step_ms * 1e-3) / 1e9;
            let base_v = *base.get_or_insert(am);
            println!(
                "{b:>4} {am:>10.1} {asd:>9.1} {:>9.1} {step_ms:>8.3} {gm:>8.3} {gbs:>8.1} {gflops:>9.1} {:>6.2}x",
                am / b as f64,
                am / base_v
            );
            points.push(json!({
                "batch": b,
                "agg_tok_s": am,
                "agg_tok_s_std": asd,
                "per_stream_tok_s": am / b as f64,
                "step_ms": step_ms,
                "gpu_ms_per_step": gm,
                "gbs": gbs,
                "gflops": gflops,
                "bytes_per_step": bytes,
                "flops_per_step": b as f64 * flops_per_tok,
            }));
            if !a.no_record {
                let line = json!({
                    "timestamp": timestamp(),
                    "stage": "metal-batch",
                    "kernels": format!("metal/batch/{}/gemm>={}", dtype.name(), a.gemm_min_rows),
                    "backend": "metal",
                    "model": model_name,
                    "dtype": dtype.name(),
                    "threads": 1,
                    "batch": b,
                    "prompt_len": a.prompt_len,
                    "gen_tokens": a.tokens,
                    "runs": a.runs,
                    "decode_tok_s": am,
                    "decode_tok_s_std": asd,
                    "per_stream_tok_s": am / b as f64,
                    "step_ms": step_ms,
                    "gpu_ms_per_step": gm,
                    "decode_gbs": gbs,
                    "decode_gflops": gflops,
                    "bandwidth_gbs": a.bandwidth_gbs,
                    "machine": machine(),
                    "traced": false,
                    "note": a.note,
                });
                append_jsonl(&a.out, &line)?;
            }
        }
        if !a.no_record {
            let series = json!({
                "label": label,
                "dtype": dtype.name(),
                "gemm_min_rows": a.gemm_min_rows,
                "weight_bytes": wbytes,
                "kv_bytes_per_position": kv_per_pos,
                "points": points,
                "note": a.note,
            });
            update_summary(
                &a.summary,
                &label,
                series,
                json!({
                    "what": "Batched greedy decode on the M4 GPU: B sequences per forward. agg_tok_s = B * steps / wall; per_stream_tok_s = one sequence's speed; step_ms = per-token latency every stream sees.",
                    "model": model_name,
                    "machine": machine(),
                    "prompt_len": a.prompt_len,
                    "decode_steps": a.tokens,
                    "git_sha": git_sha(),
                    "timestamp": timestamp(),
                    "note": a.note,
                }),
            )?;
        }
    }
    Ok(())
}

pub fn model_label(weights: &Path) -> String {
    match weights.file_name().and_then(|s| s.to_str()) {
        Some("gpt2") | None => "gpt2-124m".into(),
        Some(other) => other.to_string(),
    }
}

/// Append one bench record, stamped with its provenance ([`stamp`]; the GPU
/// benches drive the GPU from one user-interactive thread).
pub fn append_jsonl(path: &Path, line: &Value) -> Result<(), Box<dyn std::error::Error>> {
    let mut line = line.clone();
    stamp(&mut line, "interactive");
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{line}")?;
    Ok(())
}

/// Merge `series` (keyed by its label) into the summary file, refreshing `meta`.
pub fn update_summary(path: &Path, label: &str, series: Value, meta: Value) -> Result<(), Box<dyn std::error::Error>> {
    let mut doc: Value = std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(json!({}));
    let obj = doc.as_object_mut().ok_or("summary is not an object")?;
    for (k, v) in meta.as_object().unwrap() {
        obj.insert(k.clone(), v.clone());
    }
    let list = obj.entry("series").or_insert(json!([]));
    let arr = list.as_array_mut().ok_or("series is not an array")?;
    arr.retain(|s| s["label"] != label);
    arr.push(series);
    std::fs::write(path, serde_json::to_string_pretty(&doc)? + "\n")?;
    eprintln!("updated {}", path.display());
    Ok(())
}
