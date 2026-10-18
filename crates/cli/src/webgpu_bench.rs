//! `nanoinfer bench --stage webgpu-*`: the prefill/decode benchmark on wgpu
//! (natively: wgpu's Metal backend), the same code the browser runs.
//! `nanoinfer webgpu-probe` prints dispatch / submit / round-trip costs.
//!
//! `main` routes here before clap parses the CPU `bench` command, like
//! `metal_bench.rs`.

use crate::bench::{git_sha, machine, median, prefer_performance_cores, stddev, timestamp, BENCH_TEXT};
use clap::Parser;
use nanoinfer_webgpu::gpu::block_on;
use nanoinfer_webgpu::{probe, Engine, Gpu, WgpuStage};
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "nanoinfer bench")]
struct Args {
    #[arg(long, global = true, default_value = "weights/gpt2")]
    weights: PathBuf,
    #[arg(long)]
    stage: WgpuStage,
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
    /// PerOp stages: print wall time per op for one decode step.
    #[arg(long)]
    profile: bool,
}

pub fn intercept() -> Option<Result<(), Box<dyn std::error::Error>>> {
    let argv: Vec<String> = std::env::args().collect();
    if argv.get(1).map(String::as_str) == Some("webgpu-probe") {
        return Some(run_probe());
    }
    let is_bench = argv.iter().skip(1).any(|a| a == "bench");
    let webgpu = argv.windows(2).any(|w| w[0] == "--stage" && w[1].starts_with("webgpu"))
        || argv.iter().any(|a| a.starts_with("--stage=webgpu"));
    if !(is_bench && webgpu) {
        return None;
    }
    Some(run(Args::parse_from(argv.iter().filter(|a| *a != "bench"))))
}

fn clock() -> impl Fn() -> f64 {
    let t = Instant::now();
    move || t.elapsed().as_secs_f64() * 1e3
}

fn run_probe() -> Result<(), Box<dyn std::error::Error>> {
    prefer_performance_cores();
    let gpu = block_on(Gpu::new())?;
    println!("adapter: {}", gpu.name());
    let now = clock();
    for (l, v, u) in block_on(probe::run(&gpu, &now, 7)).rows {
        println!("{l:<75} {v:>9.2} {u}");
    }
    Ok(())
}

fn run(a: Args) -> Result<(), Box<dyn std::error::Error>> {
    prefer_performance_cores();
    let tok = crate::load_tokenizer(&a.weights)?;
    let model = crate::load_model(&a.weights)?;
    let cfg = model.config;
    let mut prompt = Vec::new();
    while prompt.len() < a.prompt_len {
        prompt.extend(tok.encode(BENCH_TEXT));
    }
    prompt.truncate(a.prompt_len);
    let gpu = Rc::new(block_on(Gpu::new())?);
    if !a.stage.supported(&gpu) {
        return Err(format!("{} is not supported on {}", a.stage, gpu.name()).into());
    }
    let t = Instant::now();
    let engine = Rc::new(Engine::new(gpu.clone(), &model, a.stage));
    eprintln!("wgpu: {} ({}), weights uploaded in {:.2}s", gpu.name(), a.stage, t.elapsed().as_secs_f64());
    drop(model);
    let now = clock();
    let weight_bytes = a.stage.decode_weight_bytes(&cfg);
    let ceiling = a.bandwidth_gbs * 1e9 / weight_bytes;
    let c = cfg.n_embd;
    let prefill_flops = 2.0 * (cfg.n_layer * 12 * c * c) as f64 + 2.0 * (cfg.n_vocab * c) as f64 / a.prompt_len as f64;
    eprintln!(
        "bench: stage={} prompt_len={} tokens={} runs={} (+{} warmup)",
        a.stage, a.prompt_len, a.tokens, a.runs, a.warmup
    );

    let (mut prefill, mut decode, mut gpu_ms) = (vec![], vec![], vec![]);
    let mut last = Default::default();
    for run in 0..a.warmup + a.runs {
        let mut s = engine.session(a.prompt_len + a.tokens + 1);
        let t0 = Instant::now();
        let first = block_on(s.step_greedy(&prompt, &now))?;
        let ps = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        block_on(s.decode_greedy(first, a.tokens, &now))?;
        let ds = t0.elapsed().as_secs_f64();
        let (p, d) = (a.prompt_len as f64 / ps, a.tokens as f64 / ds);
        let g = s.last.gpu_ms.map(|g| g / a.tokens as f64);
        let label = if run < a.warmup { "warmup" } else { "run" };
        eprintln!(
            "  {label:<6} prefill {p:>9.2} tok/s   decode {d:>8.3} tok/s   (wall {:.3} ms/tok, GPU {} ms/tok)",
            1e3 / d,
            g.map(|g| format!("{g:.3}")).unwrap_or("?".into())
        );
        if run >= a.warmup {
            prefill.push(p);
            decode.push(d);
            if let Some(g) = g {
                gpu_ms.push(g);
            }
        }
        last = s.last.clone();
    }
    let (pm, ps, dm, ds) = (median(&prefill), stddev(&prefill), median(&decode), stddev(&decode));
    let gm = if gpu_ms.is_empty() { f64::NAN } else { median(&gpu_ms) };
    let wall_ms = 1e3 / dm;
    let per_tok = |x: usize| x as f64 / last.forwards.max(1) as f64;
    println!(
        "{}: prefill {pm:.2} ± {ps:.2} tok/s ({:.1} GFLOP/s) | decode {dm:.3} ± {ds:.3} tok/s ({:.1} GB/s) | {:.2}% of {ceiling:.0} tok/s ceiling",
        a.stage,
        pm * prefill_flops / 1e9,
        dm * weight_bytes / 1e9,
        100.0 * dm / ceiling
    );
    println!(
        "  per decode token: wall {wall_ms:.3} ms, GPU busy {gm:.3} ms ({:.0}% of wall), {:.0} dispatches, {:.2} submits",
        100.0 * gm / wall_ms,
        per_tok(last.dispatches),
        per_tok(last.submits)
    );
    if a.profile {
        let mut s = engine.session(a.prompt_len + 2);
        block_on(s.step(&prompt, &now))?;
        let ops = if engine.recipe.sync == nanoinfer_webgpu::Sync::PerOp {
            s.profile = true;
            block_on(s.step(&[prompt[0]], &now))?;
            println!("  wall time per op, one decode step (submit + wait each):");
            s.last.op_ms.clone()
        } else {
            println!("  steady-state time per op, one decode step (each op x20 in its own submit):");
            block_on(s.profile_ops(&[prompt[0]], 20, &now))?
        };
        let mut by: Vec<(&str, f64, usize)> = vec![];
        for (name, _, ms) in &ops {
            match by.iter_mut().find(|e| e.0 == *name) {
                Some(e) => {
                    e.1 += ms;
                    e.2 += 1
                }
                None => by.push((name, *ms, 1)),
            }
        }
        by.sort_by(|x, y| y.1.total_cmp(&x.1));
        let tot: f64 = by.iter().map(|e| e.1).sum();
        println!("    (sum {tot:.3} ms)");
        for (n, ms, k) in by {
            println!("    {n:<12} {:>9.1} us  {:>5.1}%  ({k} dispatches, {:.1} us each)", ms * 1e3, 100.0 * ms / tot, ms * 1e3 / k as f64);
        }
    }
    if !a.no_record {
        let line = json!({
            "timestamp": timestamp(),
            "git_sha": git_sha(),
            "stage": a.stage.name(),
            "kernels": a.stage.kernels(),
            "backend": "wgpu-metal",
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
            "decode_gbs": dm * weight_bytes / 1e9,
            "prefill_gflops": pm * prefill_flops / 1e9,
            "pct_of_ceiling": 100.0 * dm / ceiling,
            "bandwidth_gbs": a.bandwidth_gbs,
            "gpu_ms_per_token": if gm.is_nan() { serde_json::Value::Null } else { json!(gm) },
            "wall_ms_per_token": wall_ms,
            "dispatches_per_token": per_tok(last.dispatches),
            "submits_per_token": per_tok(last.submits),
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
