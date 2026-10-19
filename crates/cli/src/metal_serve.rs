//! `nanoinfer metal-serve`: static vs continuous batching on a synthetic
//! workload (round 2).
//!
//! Requests with varying prompt and output lengths arrive (all at once, or as
//! a Poisson process). Two schedulers drive the same paged, ragged forward
//! ([`nanoinfer_metal::Batch::step`], CPU in the loop every iteration):
//!
//! - **static**: take up to B waiting requests, prefill them, decode until
//!   the *longest* one finishes (finished slots ride along as wasted rows),
//!   then take the next B. KV is reserved for prompt + max output per slot.
//! - **continuous** (Orca-style iteration-level scheduling): every iteration
//!   retires finished sequences and admits waiting ones while a slot and KV
//!   pages are free; new prompts are prefilled in chunks under a per-iteration
//!   token budget, mixed into the same forward as the running decodes.
//!
//! The clock is the measured wall time of each iteration; idle gaps waiting
//! for the next arrival are skipped (added to the clock without running).
//! Reports throughput, TTFT, TPOT, KV usage, and a Gantt trace for the web
//! page (`bench/serving.json`).

use crate::bench::{git_sha, machine, timestamp, BENCH_TEXT};
use crate::metal_bench::metal_batch::{append_jsonl, model_label, parse_dtype};
use clap::Parser;
use nanoinfer_metal::batch::KV_BLOCK;
use nanoinfer_metal::{BatchEngine, BatchOpts};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "nanoinfer metal-serve")]
pub struct ServeArgs {
    #[arg(long, default_value = "weights/gpt2")]
    weights: PathBuf,
    #[arg(long, default_value = "q4_0")]
    dtype: String,
    /// Number of requests.
    #[arg(long, default_value_t = 128)]
    requests: usize,
    /// Most sequences in flight (static batch size / continuous slot count).
    #[arg(long, default_value_t = 32)]
    max_batch: usize,
    /// Prompt lengths are uniform in [min, max].
    #[arg(long, default_value_t = 16)]
    prompt_min: usize,
    #[arg(long, default_value_t = 256)]
    prompt_max: usize,
    /// Output lengths are uniform in [min, max] (no EOS: fixed per request).
    #[arg(long, default_value_t = 16)]
    out_min: usize,
    #[arg(long, default_value_t = 256)]
    out_max: usize,
    /// Mean arrival rate in requests/s (0 = everything arrives at t = 0).
    #[arg(long, default_value_t = 0.0)]
    rate: f64,
    /// Continuous batching: most prompt tokens prefilled per iteration.
    #[arg(long, default_value_t = 256)]
    prefill_budget: usize,
    /// KV pool pages (16 positions each); default: enough for max_batch full slots.
    #[arg(long)]
    kv_blocks: Option<usize>,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    /// Requests to include in the Gantt trace of the summary.
    #[arg(long, default_value_t = 32)]
    gantt: usize,
    #[arg(long, default_value = "bench/results.jsonl")]
    out: PathBuf,
    #[arg(long, default_value = "bench/serving.json")]
    summary: PathBuf,
    #[arg(long)]
    no_record: bool,
    /// Record even if the engine sources have uncommitted changes (marked dirty).
    #[arg(long)]
    allow_dirty: bool,
    #[arg(long, default_value = "")]
    note: String,
}

#[derive(Clone)]
struct Req {
    id: usize,
    arrival: f64,
    prompt: Vec<u32>,
    out_len: usize,
}

#[derive(Default, Clone)]
struct ReqLog {
    admitted: f64,
    first_token: f64,
    finished: f64,
    generated: usize,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unif(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() % (hi - lo + 1) as u64) as usize
    }
}

fn workload(a: &ServeArgs, text: &[u32]) -> Vec<Req> {
    let mut rng = Rng(a.seed.wrapping_mul(0x9e3779b97f4a7c15) | 1);
    let mut t = 0.0;
    (0..a.requests)
        .map(|id| {
            if a.rate > 0.0 {
                t += -rng.unif().max(1e-12).ln() / a.rate;
            }
            let pl = rng.range(a.prompt_min, a.prompt_max);
            let off = rng.range(0, text.len() - pl - 1);
            Req { id, arrival: t, prompt: text[off..off + pl].to_vec(), out_len: rng.range(a.out_min, a.out_max) }
        })
        .collect()
}

/// One iteration's record for the trace.
struct Iter {
    t0: f64,
    dt: f64,
    decode_rows: usize,
    prefill_rows: usize,
    /// Slots holding a sequence (including finished ones waiting in a static batch).
    occupied: usize,
    kv_pages: usize,
}

/// The paged KV cache at one iteration (continuous policy): for the viz.
struct KvSnap {
    t: f64,
    iter: usize,
    /// (request id, sequence length, block table)
    seqs: Vec<(usize, usize, Vec<u32>)>,
}

struct Outcome {
    logs: Vec<ReqLog>,
    iters: Vec<Iter>,
    kv_snaps: Vec<KvSnap>,
    makespan: f64,
    /// Peak KV pages reserved by the policy (static: prompt + max output per slot).
    reserved_peak: usize,
}

/// Snapshot the continuous scheduler's block tables every this many iterations.
const KV_SNAP_EVERY: usize = 10;

fn pages(n: usize) -> usize {
    n.div_ceil(KV_BLOCK)
}

fn run_static(engine: &BatchEngine, reqs: &[Req], max_batch: usize) -> Result<Outcome, String> {
    let mut b = engine.batch();
    let mut logs = vec![ReqLog::default(); reqs.len()];
    let mut iters = Vec::new();
    let mut clock = 0.0f64;
    let mut next = 0;
    let mut reserved_peak = 0;
    let max_rows = engine.opts.max_rows;
    while next < reqs.len() {
        clock = clock.max(reqs[next].arrival);
        // Everything that has arrived, up to max_batch.
        let mut group = vec![];
        while next < reqs.len() && group.len() < max_batch && reqs[next].arrival <= clock {
            group.push(next);
            next += 1;
        }
        let max_out = group.iter().map(|&i| reqs[i].out_len).max().unwrap();
        reserved_peak = reserved_peak.max(group.iter().map(|&i| pages(reqs[i].prompt.len() + max_out)).sum());
        let seqs: Vec<usize> = group.iter().map(|_| b.add_seq().unwrap()).collect();
        for &i in &group {
            logs[i].admitted = clock;
        }
        // Prefill the whole group (chunked by max_rows), padding-free.
        let mut last = vec![0u32; group.len()];
        let mut k = 0;
        while k < group.len() {
            let mut work: Vec<(usize, &[u32])> = vec![];
            let mut rows = 0;
            let start = k;
            while k < group.len() && (work.is_empty() || rows + reqs[group[k]].prompt.len() <= max_rows) {
                let p = &reqs[group[k]].prompt;
                let p = &p[..p.len().min(max_rows)];
                work.push((seqs[k], p));
                rows += p.len();
                k += 1;
            }
            let t = Instant::now();
            let out = b.step(&work, false)?;
            let dt = t.elapsed().as_secs_f64();
            iters.push(Iter {
                t0: clock,
                dt,
                decode_rows: 0,
                prefill_rows: rows,
                occupied: group.len(),
                kv_pages: b.blocks_used(),
            });
            clock += dt;
            for (j, o) in out.into_iter().enumerate() {
                last[start + j] = o[0];
            }
        }
        for &i in &group {
            logs[i].first_token = clock;
            logs[i].generated = 1;
        }
        // Decode until the longest request is done; finished ones keep their slot.
        for _step in 1..max_out {
            let toks: Vec<[u32; 1]> = last.iter().map(|&t| [t]).collect();
            let work: Vec<(usize, &[u32])> = seqs.iter().zip(&toks).map(|(&s, t)| (s, t.as_slice())).collect();
            let t = Instant::now();
            let out = b.step(&work, false)?;
            let dt = t.elapsed().as_secs_f64();
            iters.push(Iter {
                t0: clock,
                dt,
                decode_rows: group.len(),
                prefill_rows: 0,
                occupied: group.len(),
                kv_pages: b.blocks_used(),
            });
            clock += dt;
            for (j, &i) in group.iter().enumerate() {
                last[j] = out[j][0];
                if logs[i].generated < reqs[i].out_len {
                    logs[i].generated += 1;
                    if logs[i].generated == reqs[i].out_len {
                        logs[i].finished = clock;
                    }
                }
            }
        }
        for (j, &i) in group.iter().enumerate() {
            if reqs[i].out_len == 1 {
                logs[i].finished = logs[i].first_token;
            }
            b.free_seq(seqs[j]);
        }
    }
    Ok(Outcome { logs, iters, kv_snaps: vec![], makespan: clock, reserved_peak })
}

fn run_continuous(engine: &BatchEngine, reqs: &[Req], max_batch: usize, budget: usize) -> Result<Outcome, String> {
    struct Live {
        req: usize,
        seq: usize,
        prefilled: usize,
        last: u32,
    }
    let mut b = engine.batch();
    let mut logs = vec![ReqLog::default(); reqs.len()];
    let mut iters = Vec::new();
    let mut kv_snaps: Vec<KvSnap> = Vec::new();
    let mut clock = 0.0f64;
    let mut waiting: VecDeque<usize> = VecDeque::new();
    let mut next = 0;
    let mut live: Vec<Live> = Vec::new();
    let mut done = 0;
    let max_rows = engine.opts.max_rows;
    while done < reqs.len() {
        while next < reqs.len() && reqs[next].arrival <= clock {
            waiting.push_back(next);
            next += 1;
        }
        if live.is_empty() && waiting.is_empty() {
            clock = reqs[next].arrival;
            continue;
        }
        // Admit while a slot is free and the pool can hold the prompt plus a
        // little headroom (pages are otherwise allocated as sequences grow).
        while live.len() < max_batch {
            let Some(&i) = waiting.front() else { break };
            let need = pages(reqs[i].prompt.len() + 1) + live.len();
            if b.free_block_count() < need {
                break;
            }
            waiting.pop_front();
            let seq = b.add_seq().ok_or("no sequence slot")?;
            logs[i].admitted = clock;
            live.push(Live { req: i, seq, prefilled: 0, last: 0 });
        }
        // Build the iteration: one decode row per running sequence, plus
        // prefill chunks (oldest first) under the token budget.
        let mut work: Vec<(usize, Vec<u32>)> = vec![];
        let mut kinds = vec![]; // (live index, is_prefill, chunk len)
        let mut decode_rows = 0;
        for (j, l) in live.iter().enumerate() {
            if l.prefilled == reqs[l.req].prompt.len() {
                work.push((l.seq, vec![l.last]));
                kinds.push((j, false, 1));
                decode_rows += 1;
            }
        }
        let mut budget_left = budget.min(max_rows - decode_rows);
        for (j, l) in live.iter().enumerate() {
            let p = &reqs[l.req].prompt;
            if l.prefilled < p.len() && budget_left > 0 {
                let n = (p.len() - l.prefilled).min(budget_left);
                work.push((l.seq, p[l.prefilled..l.prefilled + n].to_vec()));
                kinds.push((j, true, n));
                budget_left -= n;
            }
        }
        let refs: Vec<(usize, &[u32])> = work.iter().map(|(s, t)| (*s, t.as_slice())).collect();
        let t = Instant::now();
        let out = b.step(&refs, false)?;
        let dt = t.elapsed().as_secs_f64();
        iters.push(Iter {
            t0: clock,
            dt,
            decode_rows,
            prefill_rows: kinds.iter().filter(|k| k.1).map(|k| k.2).sum(),
            occupied: live.len(),
            kv_pages: b.blocks_used(),
        });
        if (iters.len() - 1) % KV_SNAP_EVERY == 0 {
            let seqs = live.iter().map(|l| (reqs[l.req].id, b.seq_len(l.seq), b.block_table(l.seq).to_vec())).collect();
            kv_snaps.push(KvSnap { t: clock, iter: iters.len() - 1, seqs });
        }
        clock += dt;
        for ((j, is_prefill, n), o) in kinds.into_iter().zip(out) {
            let l = &mut live[j];
            let r = &reqs[l.req];
            if is_prefill {
                l.prefilled += n;
                if l.prefilled < r.prompt.len() {
                    continue; // mid-prompt chunk: its prediction is not a token yet
                }
                logs[l.req].first_token = clock;
            }
            logs[l.req].generated += 1;
            l.last = o[0];
            if logs[l.req].generated == r.out_len {
                logs[l.req].finished = clock;
            }
        }
        // Retire finished sequences.
        let mut k = 0;
        while k < live.len() {
            if logs[live[k].req].generated == reqs[live[k].req].out_len && logs[live[k].req].finished > 0.0 {
                b.free_seq(live[k].seq);
                live.swap_remove(k);
                done += 1;
            } else {
                k += 1;
            }
        }
    }
    let reserved_peak = iters.iter().map(|i| i.kv_pages).max().unwrap_or(0);
    Ok(Outcome { logs, iters, kv_snaps, makespan: clock, reserved_peak })
}

fn pct(xs: &mut [f64], p: f64) -> f64 {
    xs.sort_by(f64::total_cmp);
    if xs.is_empty() {
        return 0.0;
    }
    xs[((xs.len() - 1) as f64 * p).round() as usize]
}

fn summarise(name: &str, reqs: &[Req], o: &Outcome, gantt: usize) -> Value {
    let gen: usize = o.logs.iter().map(|l| l.generated).sum();
    let mut ttft: Vec<f64> = reqs.iter().zip(&o.logs).map(|(r, l)| l.first_token - r.arrival).collect();
    let mut tpot: Vec<f64> = reqs
        .iter()
        .zip(&o.logs)
        .filter(|(r, _)| r.out_len > 1)
        .map(|(r, l)| (l.finished - l.first_token) / (r.out_len - 1) as f64)
        .collect();
    let mut e2e: Vec<f64> = reqs.iter().zip(&o.logs).map(|(r, l)| l.finished - r.arrival).collect();
    let rows: usize = o.iters.iter().map(|i| i.decode_rows + i.prefill_rows).sum();
    let wasted: usize =
        o.iters.iter().map(|i| i.occupied.saturating_sub(i.decode_rows)).filter(|_| name == "static").sum();
    let busy: f64 = o.iters.iter().map(|i| i.dt).sum();
    let kv_avg = o.iters.iter().map(|i| i.kv_pages as f64 * i.dt).sum::<f64>() / busy.max(1e-12);
    let (t50, t99) = (pct(&mut ttft, 0.5), pct(&mut ttft, 0.99));
    let (p50, p99) = (pct(&mut tpot, 0.5), pct(&mut tpot, 0.99));
    let (e50, e99) = (pct(&mut e2e, 0.5), pct(&mut e2e, 0.99));
    println!(
        "{name:<11} {:>7.0} tok/s  makespan {:>6.2} s  iters {:>5}  TTFT p50 {:>7.1} ms p99 {:>7.1} ms  TPOT p50 {:>6.2} ms p99 {:>6.2} ms  e2e p50 {:>6.2} s  KV peak {} pages",
        gen as f64 / o.makespan,
        o.makespan,
        o.iters.len(),
        1e3 * t50,
        1e3 * t99,
        1e3 * p50,
        1e3 * p99,
        e50,
        o.reserved_peak
    );
    let g: Vec<Value> = reqs
        .iter()
        .zip(&o.logs)
        .take(gantt)
        .map(|(r, l)| {
            json!({"id": r.id, "arrival": r.arrival, "admitted": l.admitted, "first_token": l.first_token,
                   "finished": l.finished, "prompt_len": r.prompt.len(), "out_len": r.out_len})
        })
        .collect();
    // Downsample the iteration trace to <= 120 points for the page.
    let stride = o.iters.len().div_ceil(120).max(1);
    let it: Vec<Value> = o
        .iters
        .iter()
        .step_by(stride)
        .map(|i| {
            json!({"t": i.t0, "dt": i.dt, "decode_rows": i.decode_rows, "prefill_rows": i.prefill_rows,
                   "occupied": i.occupied, "kv_pages": i.kv_pages})
        })
        .collect();
    // Paged-KV snapshots (continuous only), <= 20 for the page, and the
    // internal fragmentation of the pages in use (unused tail positions).
    let snap_stride = o.kv_snaps.len().div_ceil(20).max(1);
    let snaps: Vec<Value> = o
        .kv_snaps
        .iter()
        .step_by(snap_stride)
        .map(|k| {
            let seqs: Vec<Value> =
                k.seqs.iter().map(|(id, len, pages)| json!({"id": id, "len": len, "pages": pages})).collect();
            json!({"t": k.t, "iter": k.iter, "seqs": seqs})
        })
        .collect();
    let (mut slots, mut used, mut n_seq, mut n_snap) = (0usize, 0usize, 0usize, 0usize);
    for k in &o.kv_snaps {
        for (_, len, pages) in &k.seqs {
            slots += pages.len() * KV_BLOCK;
            used += len;
            n_seq += 1;
        }
        n_snap += 1;
    }
    let kv_stats = (n_snap > 0).then(|| {
        json!({"snapshots": n_snap, "avg_live_seqs": n_seq as f64 / n_snap as f64,
               "avg_pages_per_seq": slots as f64 / KV_BLOCK as f64 / n_seq.max(1) as f64,
               "internal_fragmentation": 1.0 - used as f64 / slots.max(1) as f64})
    });
    json!({
        "policy": name,
        "kv_snapshots": snaps,
        "kv_stats": kv_stats,
        "throughput_tok_s": gen as f64 / o.makespan,
        "generated_tokens": gen,
        "makespan_s": o.makespan,
        "gpu_busy_s": busy,
        "iterations": o.iters.len(),
        "rows_computed": rows,
        "wasted_decode_rows": wasted,
        "ttft_ms": {"p50": 1e3 * t50, "p99": 1e3 * t99},
        "tpot_ms": {"p50": 1e3 * p50, "p99": 1e3 * p99},
        "e2e_s": {"p50": e50, "p99": e99},
        "kv_pages_peak": o.reserved_peak,
        "kv_pages_avg_used": kv_avg,
        "gantt": g,
        "iters": it,
    })
}

pub fn run(a: ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    crate::bench::begin(!a.no_record, a.allow_dirty)?;
    crate::bench::prefer_performance_cores();
    let tok = crate::load_tokenizer(&a.weights)?;
    let model = crate::load_model(&a.weights)?;
    let cfg = model.config;
    let mut text = Vec::new();
    while text.len() < 4 * a.prompt_max + 64 {
        text.extend(tok.encode(BENCH_TEXT));
    }
    if a.prompt_max + a.out_max > cfg.n_ctx {
        return Err("prompt_max + out_max > n_ctx".into());
    }
    let reqs = workload(&a, &text);
    let dtype = parse_dtype(&a.dtype)?;
    let kv_blocks = a.kv_blocks.unwrap_or(a.max_batch * pages(a.prompt_max + a.out_max) + 8);
    let opts = BatchOpts { max_seqs: a.max_batch, max_rows: 512, kv_blocks, ..BatchOpts::new(dtype) };
    let engine = BatchEngine::new(&model, opts)?;
    let total_out: usize = reqs.iter().map(|r| r.out_len).sum();
    let total_in: usize = reqs.iter().map(|r| r.prompt.len()).sum();
    eprintln!(
        "metal-serve: {} requests (prompt {}-{}, output {}-{}; {total_in} prompt + {total_out} output tokens), rate {} req/s, max batch {}, {} KV pages ({:.0} MB), {}",
        a.requests,
        a.prompt_min,
        a.prompt_max,
        a.out_min,
        a.out_max,
        a.rate,
        a.max_batch,
        kv_blocks,
        kv_blocks as f64 * engine.kv_block_bytes() as f64 / 1e6,
        dtype.name()
    );
    // Warm up the pipelines once.
    run_continuous(&engine, &reqs[..reqs.len().min(4)], a.max_batch, a.prefill_budget)?;
    let st = run_static(&engine, &reqs, a.max_batch)?;
    let s_static = summarise("static", &reqs, &st, a.gantt);
    let ct = run_continuous(&engine, &reqs, a.max_batch, a.prefill_budget)?;
    let s_cont = summarise("continuous", &reqs, &ct, a.gantt);
    let speedup = s_cont["throughput_tok_s"].as_f64().unwrap() / s_static["throughput_tok_s"].as_f64().unwrap();
    println!("continuous / static throughput: {speedup:.2}x");
    if !a.no_record {
        let model_name = model_label(&a.weights);
        for s in [&s_static, &s_cont] {
            let line = json!({
                "timestamp": timestamp(),
                "stage": format!("metal-serve-{}", s["policy"].as_str().unwrap()),
                "kernels": format!("metal/batch/{}", dtype.name()),
                "backend": "metal",
                "model": model_name,
                "dtype": dtype.name(),
                "threads": 1,
                "requests": a.requests,
                "max_batch": a.max_batch,
                "rate": a.rate,
                "decode_tok_s": s["throughput_tok_s"],
                "ttft_p50_ms": s["ttft_ms"]["p50"],
                "ttft_p99_ms": s["ttft_ms"]["p99"],
                "tpot_p50_ms": s["tpot_ms"]["p50"],
                "tpot_p99_ms": s["tpot_ms"]["p99"],
                "makespan_s": s["makespan_s"],
                "kv_pages_peak": s["kv_pages_peak"],
                "machine": machine(),
                "traced": false,
                "note": a.note,
            });
            append_jsonl(&a.out, &line)?;
        }
        let mut key = format!("rate={}", a.rate);
        if a.prefill_budget != 256 {
            key += &format!(",budget={}", a.prefill_budget);
        }
        let mut doc: Value =
            std::fs::read(&a.summary).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(json!({}));
        let obj = doc.as_object_mut().ok_or("summary is not an object")?;
        obj.insert("what".into(), json!("Static vs continuous batching on the M4 GPU, synthetic workload, same paged ragged forward. Times in seconds of measured GPU wall time (idle gaps waiting for arrivals skipped)."));
        obj.insert("model".into(), json!(model_name));
        obj.insert("machine".into(), json!(machine()));
        obj.insert("git_sha".into(), json!(git_sha()));
        obj.insert("timestamp".into(), json!(timestamp()));
        let runs = obj.entry("runs").or_insert(json!({}));
        runs.as_object_mut().unwrap().insert(
            key,
            json!({
                "workload": {"requests": a.requests, "prompt": [a.prompt_min, a.prompt_max], "output": [a.out_min, a.out_max],
                             "rate_req_s": a.rate, "max_batch": a.max_batch, "prefill_budget": a.prefill_budget,
                             "kv_pages": kv_blocks, "kv_page_bytes": engine.kv_block_bytes(), "dtype": dtype.name(),
                             "prompt_tokens": total_in, "output_tokens": total_out},
                "static": s_static,
                "continuous": s_cont,
                "speedup": speedup,
                "note": a.note,
            }),
        );
        round_floats(&mut doc);
        std::fs::write(&a.summary, serde_json::to_string(&doc)? + "\n")?;
        eprintln!("updated {}", a.summary.display());
    }
    Ok(())
}

/// Keep the summary small: 4 significant digits is plenty for a chart.
pub fn round_floats(v: &mut Value) {
    match v {
        Value::Number(n) if n.is_f64() => {
            let x = n.as_f64().unwrap();
            if x != 0.0 && x.is_finite() {
                let scale = 10f64.powi(3 - x.abs().log10().floor() as i32);
                *v = json!((x * scale).round() / scale);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(round_floats),
        Value::Object(o) => o.values_mut().for_each(round_floats),
        _ => {}
    }
}
