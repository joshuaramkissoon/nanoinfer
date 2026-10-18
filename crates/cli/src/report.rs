//! Turning a [`Recorder`]'s events into a terminal profile or a JSONL trace.
//!
//! The JSONL format follows the event schema proposed for the web
//! visualisations (research/03-explorable-writeup-and-wasm.md §3.4): one JSON
//! object per line with a `t` field: `run_start`, `op`, `kv`, `capture`,
//! `token`, `run_end`. Times are microseconds since the run started.

use nanoinfer_core::trace::{Clock, Phase, Recorder};
use nanoinfer_core::{Config, Stage, Tokenizer};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

fn phase_str(p: Phase) -> &'static str {
    match p {
        Phase::Prefill => "prefill",
        Phase::Decode => "decode",
    }
}

/// Per-op table: where does the time go, and how close to the roofline is each op?
pub fn print_profile<C: Clock>(rec: &Recorder<C>) {
    for phase in [Phase::Prefill, Phase::Decode] {
        // name -> (count, seconds, flops, bytes)
        let mut agg: BTreeMap<&str, (usize, f64, u64, u64)> = BTreeMap::new();
        let mut total = 0.0;
        let mut n_fwd = 0;
        for f in rec.forwards.iter().filter(|f| f.info.phase == phase) {
            total += f.duration;
            n_fwd += 1;
        }
        if n_fwd == 0 {
            continue;
        }
        for e in rec.ops.iter().filter(|e| rec.forwards.get(e.forward).is_some_and(|f| f.info.phase == phase)) {
            let a = agg.entry(e.op.name).or_default();
            a.0 += 1;
            a.1 += e.duration;
            a.2 += e.op.flops;
            a.3 += e.op.bytes;
        }
        let mut rows: Vec<_> = agg.into_iter().collect();
        rows.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
        eprintln!("\n{} ({} forward calls, {:.1} ms total)", phase_str(phase), n_fwd, total * 1e3);
        eprintln!("  {:<11} {:>6} {:>10} {:>7} {:>9} {:>8}", "op", "calls", "ms", "share", "GFLOP/s", "GB/s");
        for (name, (count, secs, flops, bytes)) in rows {
            eprintln!(
                "  {:<11} {:>6} {:>10.2} {:>6.1}% {:>9.2} {:>8.2}",
                name,
                count,
                secs * 1e3,
                100.0 * secs / total,
                flops as f64 / secs / 1e9,
                bytes as f64 / secs / 1e9
            );
        }
    }
}

/// Accumulates JSONL trace events as generation proceeds.
pub struct TraceWriter {
    lines: Vec<Value>,
    ops_done: usize,
    cfg: Config,
}

impl TraceWriter {
    pub fn new(cfg: Config, stage: Stage, prompt: &[u32]) -> Self {
        let start = json!({
            "t": "run_start", "run": 0, "ts_us": 0,
            "config": {
                "model": "gpt2-124m", "dtype": "f32", "backend": "native", "threads": 1,
                "step": stage.name(), "knobs": { "kernels": stage.kernels() },
                "isolated": true, "timer_res_us": 0.042
            },
            "prompt_tokens": prompt,
        });
        TraceWriter { lines: vec![start], ops_done: 0, cfg }
    }

    /// Emit events for the most recent forward call, which produced token `next`.
    pub fn forward<C: Clock>(&mut self, rec: &Recorder<C>, tok: &Tokenizer, next: u32) {
        let Some(f) = rec.forwards.last() else { return };
        let idx = rec.forwards.len() - 1;
        let us = |s: f64| (s * 1e6 * 1000.0).round() / 1000.0;
        for e in &rec.ops[self.ops_done..] {
            self.lines.push(json!({
                "t": "op", "run": 0, "tok": idx,
                "layer": e.op.layer.map_or(-1, |l| l as i64),
                "op": e.op.name, "kind": e.op.kind.as_str(),
                "start_us": us(e.start), "dur_us": us(e.duration),
                "flops": e.op.flops, "bytes": e.op.bytes, "shape": e.op.dims,
            }));
        }
        self.ops_done = rec.ops.len();
        if let Some((len, capacity)) = f.kv {
            // K and V, every layer, n_embd floats each.
            let bytes = len * 2 * self.cfg.n_layer * self.cfg.n_embd * 4;
            self.lines.push(json!({ "t": "kv", "run": 0, "len": len, "capacity": capacity, "bytes": bytes }));
        }
        if let Some((layer, head)) = rec.attention_head {
            self.lines.push(json!({
                "t": "capture", "run": 0, "tok": idx, "layer": layer, "head": head,
                "name": "attn_probs", "shape": [rec.attention_probs.len()], "dtype": "f32",
                "data": rec.attention_probs,
            }));
        }
        let p = rec.top.iter().find(|t| t.0 == next).map_or(f32::NAN, |t| t.1);
        let entropy: f32 = rec.top.iter().map(|&(_, p)| if p > 0.0 { -p * p.ln() } else { 0.0 }).sum();
        self.lines.push(json!({
            "t": "token", "run": 0, "idx": idx, "phase": phase_str(f.info.phase),
            "id": next, "text": String::from_utf8_lossy(tok.token_bytes(next)),
            "logprob": p.ln(), "entropy_topk": entropy,
            "topk": rec.top.iter().map(|&(id, p)| json!([id, p])).collect::<Vec<_>>(),
            "dur_us": us(f.duration),
            "n_computed": f.info.n_computed, "context_len": f.info.context_len,
        }));
    }

    pub fn finish<C: Clock>(mut self, rec: &Recorder<C>, path: &Path) -> std::io::Result<()> {
        let prefill: f64 = rec.forwards.iter().filter(|f| f.info.phase == Phase::Prefill).map(|f| f.duration).sum();
        let dec: Vec<f64> = rec.forwards.iter().filter(|f| f.info.phase == Phase::Decode).map(|f| f.duration).collect();
        let n_prompt = rec.forwards.first().map_or(0, |f| f.info.n_new);
        self.lines.push(json!({
            "t": "run_end", "run": 0, "n_prompt": n_prompt, "n_gen": rec.forwards.len(),
            "prefill_ms": prefill * 1e3,
            "decode_tok_s": if dec.is_empty() { 0.0 } else { dec.len() as f64 / dec.iter().sum::<f64>() },
        }));
        let mut out = String::new();
        for l in &self.lines {
            out.push_str(&l.to_string());
            out.push('\n');
        }
        std::fs::write(path, out)
    }
}
