//! `nanoinfer bench --stage metal-*`: the same prefill/decode benchmark as
//! [`crate::bench`], run on the GPU through `nanoinfer-metal`.
//!
//! `main` routes here before clap parses the CPU `bench` command (whose
//! `--stage` only knows CPU stages), so `bench.rs` stays untouched. Besides
//! wall-clock tok/s it records what the GPU itself reports: summed
//! `GPUEndTime - GPUStartTime` per token, dispatches and command buffers per
//! token, and achieved GB/s against the bandwidth ceiling.

use crate::bench::{git_sha, machine, median, prefer_performance_cores, stddev, timestamp, BENCH_TEXT};
use clap::Parser;
use nanoinfer_metal::{MetalEngine, MetalStage};
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "nanoinfer bench")]
struct MetalBenchArgs {
    #[arg(long, global = true, default_value = "weights/gpt2")]
    weights: PathBuf,
    #[arg(long)]
    stage: MetalStage,
    #[arg(long, default_value_t = 64)]
    tokens: usize,
    #[arg(long, default_value_t = 64)]
    prompt_len: usize,
    #[arg(long, default_value_t = 5)]
    runs: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 120.0)]
    bandwidth_gbs: f64,
    #[arg(long, default_value = "bench/results.jsonl")]
    out: PathBuf,
    #[arg(long)]
    no_record: bool,
    #[arg(long, default_value = "")]
    note: String,
    /// Print steady-state GPU time per op for one decode step (each op timed
    /// alone, repeated back to back in its own command buffer).
    #[arg(long)]
    profile: bool,
}

/// If argv is `bench ... --stage metal-*`, run the Metal bench and return its result.
pub fn intercept() -> Option<Result<(), Box<dyn std::error::Error>>> {
    let argv: Vec<String> = std::env::args().collect();
    let is_bench = argv.iter().skip(1).any(|a| a == "bench");
    let metal = argv.windows(2).any(|w| w[0] == "--stage" && w[1].starts_with("metal"))
        || argv.iter().any(|a| a.starts_with("--stage=metal"));
    if !(is_bench && metal) {
        return None;
    }
    let args = MetalBenchArgs::parse_from(argv.iter().filter(|a| *a != "bench"));
    Some(run(args))
}

struct RunResult {
    prefill_s: f64,
    decode_s: f64,
    prefill_gpu_s: f64,
    decode_gpu_s: f64,
    dispatches: usize,
    command_buffers: usize,
}

fn time_run(engine: &MetalEngine, prompt: &[u32], tokens: usize) -> Result<RunResult, String> {
    let mut s = engine.session();
    let t = Instant::now();
    let mut next = s.step_greedy(prompt)?;
    let prefill_s = t.elapsed().as_secs_f64();
    let prefill_gpu_s = s.last.gpu_seconds;
    let (mut decode_gpu_s, mut dispatches, mut command_buffers) = (0.0, 0, 0);
    let t = Instant::now();
    for _ in 0..tokens {
        next = s.step_greedy(&[next])?;
        decode_gpu_s += s.last.gpu_seconds;
        dispatches = s.last.dispatches;
        command_buffers = s.last.command_buffers;
    }
    Ok(RunResult {
        prefill_s,
        decode_s: t.elapsed().as_secs_f64(),
        prefill_gpu_s,
        decode_gpu_s,
        dispatches,
        command_buffers,
    })
}

fn run(a: MetalBenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    prefer_performance_cores();
    let tok = crate::load_tokenizer(&a.weights)?;
    let model = crate::load_model(&a.weights)?;
    let cfg = model.config;
    if a.prompt_len + a.tokens > cfg.n_ctx {
        return Err(format!("prompt_len + tokens must be <= {}", cfg.n_ctx).into());
    }
    let mut prompt = Vec::new();
    while prompt.len() < a.prompt_len {
        prompt.extend(tok.encode(BENCH_TEXT));
    }
    prompt.truncate(a.prompt_len);

    let t = Instant::now();
    let engine = MetalEngine::new(&model, a.stage)?;
    eprintln!("metal: {} ({}), weights uploaded in {:.2}s", engine.gpu.name(), a.stage, t.elapsed().as_secs_f64());
    drop(model);

    let weight_bytes = a.stage.decode_weight_bytes(&cfg);
    let ceiling = a.bandwidth_gbs * 1e9 / weight_bytes;
    let c = cfg.n_embd;
    let block_flops = 2.0 * (cfg.n_layer * 12 * c * c) as f64;
    let head_flops = 2.0 * (cfg.n_vocab * c) as f64;
    let prefill_flops = block_flops + head_flops / a.prompt_len as f64;
    eprintln!(
        "bench: stage={} prompt_len={} tokens={} runs={} (+{} warmup)",
        a.stage, a.prompt_len, a.tokens, a.runs, a.warmup
    );
    eprintln!(
        "decode ceiling: {:.0} GB/s / {:.1} MB of weights per token = {:.0} tok/s",
        a.bandwidth_gbs,
        weight_bytes / 1e6,
        ceiling
    );

    let (mut prefill, mut decode, mut gpu_ms, mut prefill_gpu_ms) = (vec![], vec![], vec![], vec![]);
    let mut last = None;
    for run in 0..a.warmup + a.runs {
        let r = time_run(&engine, &prompt, a.tokens)?;
        let (p, d) = (a.prompt_len as f64 / r.prefill_s, a.tokens as f64 / r.decode_s);
        let g = 1e3 * r.decode_gpu_s / a.tokens as f64;
        let label = if run < a.warmup { "warmup" } else { "run" };
        eprintln!(
            "  {label:<6} prefill {p:>9.2} tok/s   decode {d:>8.3} tok/s   (wall {:.3} ms/tok, GPU {g:.3} ms/tok)",
            1e3 / d
        );
        if run >= a.warmup {
            prefill.push(p);
            decode.push(d);
            gpu_ms.push(g);
            prefill_gpu_ms.push(1e3 * r.prefill_gpu_s);
        }
        last = Some(r);
    }
    let last = last.unwrap();

    let (pm, ps, dm, ds) = (median(&prefill), stddev(&prefill), median(&decode), stddev(&decode));
    let gm = median(&gpu_ms);
    let wall_ms = 1e3 / dm;
    let decode_gbs = dm * weight_bytes / 1e9;
    let gpu_gbs = weight_bytes / (gm * 1e-3) / 1e9;
    let prefill_gflops = pm * prefill_flops / 1e9;
    println!(
        "{}: prefill {:.2} ± {:.2} tok/s ({:.1} GFLOP/s) | decode {:.3} ± {:.3} tok/s ({:.1} GB/s) | {:.2}% of {:.0} tok/s ceiling",
        a.stage,
        pm,
        ps,
        prefill_gflops,
        dm,
        ds,
        decode_gbs,
        100.0 * dm / ceiling,
        ceiling
    );
    println!(
        "  per decode token: wall {wall_ms:.3} ms, GPU busy {gm:.3} ms ({:.0}% of wall, {gpu_gbs:.1} GB/s while busy), {} dispatches, {} command buffers",
        100.0 * gm / wall_ms,
        last.dispatches,
        last.command_buffers
    );

    if a.profile {
        let bpw = a.stage.recipe().dtype.bytes_per_weight();
        let mut s = engine.session();
        s.profile = Some(Vec::new());
        s.step(&prompt)?;
        let prefill_prof = s.profile.replace(Vec::new()).unwrap_or_default();
        s.step(&[prompt[0]])?;
        let decode_prof = s.profile.take().unwrap_or_default();
        print_profile(&format!("one prefill of {} tokens", prompt.len()), &prefill_prof, &cfg, bpw, prompt.len());
        print_profile(&format!("one decode step, context {}", prompt.len() + 1), &decode_prof, &cfg, bpw, 1);
    }

    if !a.no_record {
        let line = json!({
            "timestamp": timestamp(),
            "git_sha": git_sha(),
            "stage": a.stage.name(),
            "kernels": a.stage.kernels(),
            "backend": "metal",
            "model": "gpt2-124m",
            "dtype": a.stage.recipe().dtype.name(),
            "threads": 1,
            "prompt_len": a.prompt_len,
            "gen_tokens": a.tokens,
            "runs": a.runs,
            "prefill_tok_s": pm,
            "prefill_tok_s_std": ps,
            "decode_tok_s": dm,
            "decode_tok_s_std": ds,
            "decode_ceiling_tok_s": ceiling,
            "decode_gbs": decode_gbs,
            "prefill_gflops": prefill_gflops,
            "pct_of_ceiling": 100.0 * dm / ceiling,
            "bandwidth_gbs": a.bandwidth_gbs,
            "gpu_ms_per_token": gm,
            "wall_ms_per_token": wall_ms,
            "gpu_gbs_while_busy": gpu_gbs,
            "prefill_gpu_ms": median(&prefill_gpu_ms),
            "dispatches_per_token": last.dispatches,
            "command_buffers_per_token": last.command_buffers,
            "machine": machine(),
            "traced": false,
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

/// Aggregate per-op GPU times by label and print them, with GB/s of weights
/// for the matrix ops and GFLOP/s when `n` tokens went through them.
fn print_profile(title: &str, prof: &[(&'static str, f64)], cfg: &nanoinfer_core::Config, bpw: f64, n: usize) {
    let c = cfg.n_embd;
    let mut by: Vec<(&str, f64, usize)> = Vec::new();
    for (name, t) in prof {
        match by.iter_mut().find(|e| e.0 == *name) {
            Some(e) => {
                e.1 += t;
                e.2 += 1;
            }
            None => by.push((name, *t, 1)),
        }
    }
    by.sort_by(|x, y| y.1.total_cmp(&x.1));
    let total: f64 = by.iter().map(|e| e.1).sum();
    println!("  GPU time per op ({title}; sum {:.3} ms):", total * 1e3);
    // Weights per op (all layers), and how many token rows went through it.
    let weights = |name: &str| -> Option<(f64, usize)> {
        let per = match name {
            "qkv" => 3 * c * c,
            "attn_proj" => c * c,
            "fc" | "fc_proj" => 4 * c * c,
            "lm_head" => return Some(((cfg.n_vocab * c) as f64, 1)),
            _ => return None,
        };
        Some(((cfg.n_layer * per) as f64, n))
    };
    for (name, t, k) in by {
        let rate = weights(name)
            .map(|(params, rows)| {
                format!("{:>6.1} GB/s {:>7.1} GFLOP/s", params * bpw / t / 1e9, 2.0 * params * rows as f64 / t / 1e9)
            })
            .unwrap_or_default();
        println!("    {name:<11} {:>9.1} us  {:>5.1}%  ({k:>2} dispatches)  {rate}", t * 1e6, 100.0 * t / total);
    }
}
