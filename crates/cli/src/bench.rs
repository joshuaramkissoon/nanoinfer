//! `nanoinfer bench`: prefill and decode throughput, measured separately.
//!
//! Prefill (feeding the prompt) and decode (one token at a time) stress the
//! hardware in completely different ways, so averaging them into one "tok/s"
//! hides everything interesting. Each run:
//!
//! 1. fresh session, prompt of exactly `--prompt-len` tokens
//! 2. time ONE forward over the whole prompt          -> prefill tok/s
//! 3. time `--tokens` single-token forwards (greedy)  -> decode tok/s
//!
//! We do `--warmup` untimed runs first (page in the mmap, warm caches, let the
//! CPU clock up), then `--runs` timed runs, and report median ± stddev. A JSON
//! line is appended to `bench/results.jsonl` so the hillclimb has a history.

use clap::Args;
use nanoinfer_core::trace::Recorder;
use nanoinfer_core::{sampler, Model, NoTrace, Session, Stage, Trace};
use serde_json::json;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Args)]
pub struct BenchArgs {
    #[arg(long, default_value = "kv-cache")]
    stage: Stage,
    /// Decode steps to time per run.
    #[arg(long, default_value_t = 64)]
    tokens: usize,
    /// Prompt length in tokens.
    #[arg(long, default_value_t = 64)]
    prompt_len: usize,
    #[arg(long, default_value_t = 5)]
    runs: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    /// Peak memory bandwidth in GB/s used for the theoretical decode ceiling
    /// (Apple M4 base: ~120 GB/s).
    #[arg(long, default_value_t = 120.0)]
    bandwidth_gbs: f64,
    /// Where to append the result line.
    #[arg(long, default_value = "bench/results.jsonl")]
    out: PathBuf,
    /// Don't append to the results file.
    #[arg(long)]
    no_record: bool,
    /// Run with a timing Recorder attached instead of NoTrace, to measure the
    /// cost of tracing itself.
    #[arg(long)]
    with_trace: bool,
    /// Free-form note stored with the result.
    #[arg(long, default_value = "")]
    note: String,
}

/// Prompt text for benchmarking: ordinary English, repeated to the requested length.
pub(crate) const BENCH_TEXT: &str = "The history of computing is a story of making machines do arithmetic faster. \
Early computers filled rooms and performed a few thousand operations per second. ";

pub(crate) fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

pub(crate) fn stddev(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt()
}

fn cmd_output(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub(crate) fn git_sha() -> String {
    let sha = cmd_output("git", &["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = cmd_output("git", &["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    if dirty {
        format!("{sha}-dirty")
    } else {
        sha
    }
}

pub(crate) fn machine() -> String {
    cmd_output("sysctl", &["-n", "machdep.cpu.brand_string"])
        .unwrap_or_else(|| format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH))
}

/// RFC 3339 UTC timestamp without pulling in a date crate.
pub(crate) fn timestamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // Howard Hinnant's civil_from_days.
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

/// Ask macOS to schedule this thread on a performance core. Without this, a
/// long single-threaded benchmark can be migrated to an efficiency core
/// mid-run (we saw decode drop from 19 to 6.5 tok/s between runs).
#[cfg(target_os = "macos")]
pub fn prefer_performance_cores() {
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    // Safety: plain libSystem call affecting only the current thread.
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

#[cfg(not(target_os = "macos"))]
pub fn prefer_performance_cores() {}

/// One timed run: (prefill seconds, decode seconds), pushing each decode
/// step's duration onto `steps`.
fn time_run_steps<T: Trace>(
    model: &Model,
    stage: Stage,
    prompt: &[u32],
    tokens: usize,
    trace: &mut T,
    steps: &mut Vec<f64>,
) -> nanoinfer_core::Result<(f64, f64)> {
    let mut s = Session::new(model, stage);
    let t = Instant::now();
    let mut next = sampler::argmax(s.step(model, prompt, trace)?);
    let t_prefill = t.elapsed().as_secs_f64();
    let t = Instant::now();
    for _ in 0..tokens {
        let ts = Instant::now();
        next = sampler::argmax(s.step(model, &[next], trace)?);
        steps.push(ts.elapsed().as_secs_f64());
    }
    Ok((t_prefill, t.elapsed().as_secs_f64()))
}

pub fn run(
    weights: &Path,
    a: BenchArgs,
    pool_threads: usize,
    qos: crate::pool::Qos,
) -> Result<(), Box<dyn std::error::Error>> {
    // Only the threaded stages use the pool; everything else is one thread.
    let threads = if a.stage.multithreaded() { pool_threads } else { 1 };
    let tok = crate::load_tokenizer(weights)?;
    let model = crate::load_model(weights)?;
    let cfg = model.config;
    if a.prompt_len + a.tokens > cfg.n_ctx {
        return Err(format!("prompt_len + tokens must be <= {}", cfg.n_ctx).into());
    }
    let mut prompt = Vec::new();
    while prompt.len() < a.prompt_len {
        prompt.extend(tok.encode(BENCH_TEXT));
    }
    prompt.truncate(a.prompt_len);

    let fmt = a.stage.weight_format();
    let weight_bytes = a.stage.decode_bytes(&cfg);
    let ceiling = a.bandwidth_gbs * 1e9 / weight_bytes;
    // FLOPs per token (matmuls only; attention is <1% here). Decode runs every
    // matrix once per token; prefill runs the blocks per token but the LM head
    // only once per prompt (we only need the last position's logits).
    let c = cfg.n_embd;
    let block_flops = 2.0 * (cfg.n_layer * 12 * c * c) as f64;
    let head_flops = 2.0 * (cfg.n_vocab * c) as f64;
    let decode_flops = block_flops + head_flops;
    let prefill_flops = block_flops + head_flops / a.prompt_len as f64;
    eprintln!(
        "bench: stage={} threads={threads} qos={} prompt_len={} tokens={} runs={} (+{} warmup)",
        a.stage,
        qos.name(),
        a.prompt_len,
        a.tokens,
        a.runs,
        a.warmup
    );
    eprintln!(
        "decode ceiling: {:.0} GB/s / {:.1} MB of weights per token = {:.0} tok/s",
        a.bandwidth_gbs,
        weight_bytes / 1e6,
        ceiling
    );

    let mut prefill = Vec::new();
    let mut decode = Vec::new();
    let mut steps = Vec::new();
    for run in 0..a.warmup + a.runs {
        let mut run_steps = Vec::new();
        let (t_prefill, t_decode) = if a.with_trace {
            let start = Instant::now();
            let mut rec = Recorder::new(move || start.elapsed().as_secs_f64());
            time_run_steps(&model, a.stage, &prompt, a.tokens, &mut rec, &mut run_steps)?
        } else {
            time_run_steps(&model, a.stage, &prompt, a.tokens, &mut NoTrace, &mut run_steps)?
        };
        if run >= a.warmup {
            steps.extend(run_steps);
        }
        let (p, d) = (a.prompt_len as f64 / t_prefill, a.tokens as f64 / t_decode);
        let label = if run < a.warmup { "warmup" } else { "run" };
        eprintln!("  {label:<6} prefill {p:>9.2} tok/s   decode {d:>8.3} tok/s");
        if run >= a.warmup {
            prefill.push(p);
            decode.push(d);
        }
    }

    let (pm, ps, dm, ds) = (median(&prefill), stddev(&prefill), median(&decode), stddev(&decode));
    // Median single-step time over every timed token: robust to the preemption
    // spikes a busy machine adds to whole-run totals.
    let step_tok_s = 1.0 / median(&steps);
    let decode_gbs = dm * weight_bytes / 1e9;
    let prefill_gflops = pm * prefill_flops / 1e9;
    let decode_gflops = dm * decode_flops / 1e9;
    println!(
        "{} ({threads}t): prefill {:.2} ± {:.2} tok/s ({:.1} GFLOP/s) | decode {:.3} ± {:.3} tok/s ({:.1} GB/s, {:.1} GFLOP/s) | {:.2}% of {:.0} tok/s ceiling",
        a.stage,
        pm,
        ps,
        prefill_gflops,
        dm,
        ds,
        decode_gbs,
        decode_gflops,
        100.0 * dm / ceiling,
        ceiling
    );
    println!(
        "  per-token median: {step_tok_s:.1} tok/s ({:.1} GB/s, {:.1}% of ceiling)",
        step_tok_s * weight_bytes / 1e9,
        100.0 * step_tok_s / ceiling
    );

    if !a.no_record {
        let line = json!({
            "timestamp": timestamp(),
            "git_sha": git_sha(),
            "stage": a.stage.name(),
            "kernels": a.stage.kernels(),
            "model": "gpt2-124m",
            "dtype": fmt.name(),
            "threads": threads,
            "qos": qos.name(),
            "prompt_len": a.prompt_len,
            "gen_tokens": a.tokens,
            "runs": a.runs,
            "prefill_tok_s": pm,
            "prefill_tok_s_std": ps,
            "decode_tok_s": dm,
            "decode_tok_s_std": ds,
            "decode_ceiling_tok_s": ceiling,
            "decode_gbs": decode_gbs,
            "decode_step_median_tok_s": step_tok_s,
            "decode_gflops": decode_gflops,
            "prefill_gflops": prefill_gflops,
            "pct_of_ceiling": 100.0 * dm / ceiling,
            "bandwidth_gbs": a.bandwidth_gbs,
            "machine": machine(),
            "traced": a.with_trace,
            "note": a.note,
        });
        if let Some(dir) = a.out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&a.out)?;
        writeln!(f, "{line}")?;
        eprintln!("appended to {}", a.out.display());
    }
    Ok(())
}
