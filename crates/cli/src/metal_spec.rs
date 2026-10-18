//! `nanoinfer metal-spec`: greedy speculative decoding on the GPU (round 2).
//!
//! A small draft model (GPT-2 124M) proposes k tokens one at a time; the big
//! target model (gpt2-medium / gpt2-large, same tokenizer) checks all of them
//! in ONE forward over k + 1 positions (`Batch::step` with all logits). The
//! longest prefix where draft == target's greedy choice is kept, plus the
//! target's own token at the first mismatch (or a bonus token if all k
//! match), so every round yields 1..=k+1 tokens for one target forward.
//! Rejected positions are rolled back with `Batch::truncate`.
//!
//! Because the target's batched kernels are batch-invariant (each row's dot
//! products are accumulated in the same order for 1 row or k + 1), the output
//! is bit-for-bit the target's own greedy decode; the bench asserts it.

use crate::bench::{git_sha, machine, median, prefer_performance_cores, timestamp};
use crate::metal_bench::metal_batch::{append_jsonl, model_label, parse_dtype};
use clap::Parser;
use nanoinfer_core::Tokenizer;
use nanoinfer_metal::{Batch, BatchEngine, BatchOpts};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "nanoinfer metal-spec")]
pub struct SpecArgs {
    #[arg(long, default_value = "weights/gpt2-medium")]
    target: PathBuf,
    #[arg(long, default_value = "weights/gpt2")]
    draft: PathBuf,
    #[arg(long, default_value = "f16")]
    target_dtype: String,
    #[arg(long, default_value = "q4_0")]
    draft_dtype: String,
    /// Draft lengths to sweep.
    #[arg(long, default_value = "1,2,3,4,5,6,8")]
    k: String,
    /// Tokens to generate per prompt.
    #[arg(long, default_value_t = 128)]
    tokens: usize,
    /// Repeats per (prompt, k); the median wall time is reported.
    #[arg(long, default_value_t = 3)]
    runs: usize,
    /// Only prompts whose name contains this.
    #[arg(long, default_value = "")]
    filter: String,
    #[arg(long, default_value = "bench/results.jsonl")]
    out: PathBuf,
    #[arg(long, default_value = "bench/specdec.json")]
    summary: PathBuf,
    #[arg(long)]
    no_record: bool,
    #[arg(long, default_value = "")]
    note: String,
}

/// Prompt categories: code and lists should be easy for a small draft,
/// open-ended prose hard.
fn prompt_set(wiki: Option<String>) -> Vec<(&'static str, String)> {
    let mut v = vec![
        (
            "code",
            "import numpy as np\n\ndef softmax(x):\n    \"\"\"Compute softmax values for each row of x.\"\"\"\n    e = np.exp(x - np.max(x, axis=1, keepdims=True))\n    return e / e.sum(axis=1, keepdims=True)\n\n\ndef cross_entropy(logits, labels):\n".to_string(),
        ),
        (
            "list",
            "January, February, March, April, May, June, July, August, September, October, November, December. Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday. 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,".to_string(),
        ),
        (
            "license",
            "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation".to_string(),
        ),
        (
            "prose",
            "The history of computing is a story of making machines do arithmetic faster. Early mechanical calculators".to_string(),
        ),
        ("story", "Once upon a time, in a small village at the edge of a dark forest, there lived a girl named".to_string()),
        ("qa", "Q: What is the capital of France?\nA: Paris.\nQ: What is the largest planet in the solar system?\nA:".to_string()),
    ];
    if let Some(w) = wiki {
        v.push(("wikitext", w));
    }
    v
}

struct SpecRun {
    out: Vec<u32>,
    wall: f64,
    rounds: usize,
    /// Accepted draft tokens per round.
    accepted: Vec<usize>,
    draft_s: f64,
    verify_s: f64,
}

/// Target alone: prefill then GPU-resident greedy decode (the best batch-1 path).
fn target_alone(tb: &mut Batch, prompt: &[u32], n: usize, resident: bool) -> Result<(Vec<u32>, f64), String> {
    let s = tb.add_seq().unwrap();
    let first = tb.step(&[(s, prompt)], false)?[0][0];
    let t = Instant::now();
    let mut out = vec![first];
    if resident {
        out.extend(tb.decode_resident(&[s], &[first], n - 1)?.remove(0));
    } else {
        for _ in 1..n {
            let next = tb.step(&[(s, &[*out.last().unwrap()])], false)?[0][0];
            out.push(next);
        }
    }
    let wall = t.elapsed().as_secs_f64();
    tb.free_seq(s);
    Ok((out, wall))
}

fn speculative(tb: &mut Batch, db: &mut Batch, prompt: &[u32], n: usize, k: usize) -> Result<SpecRun, String> {
    let ts = tb.add_seq().unwrap();
    let ds = db.add_seq().unwrap();
    let mut g = tb.step(&[(ts, prompt)], false)?[0][0];
    db.step(&[(ds, prompt)], false)?;
    let t0 = Instant::now();
    let mut out = vec![g];
    let mut dfeed = vec![g];
    let (mut rounds, mut accepted, mut draft_s, mut verify_s) = (0, vec![], 0.0, 0.0);
    while out.len() < n {
        // Draft k tokens: one forward over the not-yet-seen tokens, then k - 1 on the GPU.
        let t = Instant::now();
        let d1 = db.step(&[(ds, &dfeed)], false)?[0][0];
        let dbase = db.seq_len(ds);
        let mut d = vec![d1];
        if k > 1 {
            d.extend(db.decode_resident(&[ds], &[d1], k - 1)?.remove(0));
        }
        draft_s += t.elapsed().as_secs_f64();
        // Verify: target over [g, d1..dk], greedy prediction after every position.
        let t = Instant::now();
        let tlen = tb.seq_len(ts);
        let mut v = Vec::with_capacity(k + 1);
        v.push(g);
        v.extend(&d);
        let p = tb.step(&[(ts, &v)], true)?.remove(0);
        verify_s += t.elapsed().as_secs_f64();
        let acc = (0..k).take_while(|&i| d[i] == p[i]).count();
        out.extend(&d[..acc]);
        out.push(p[acc]);
        accepted.push(acc);
        rounds += 1;
        // Roll back: the target keeps g, d1..d_acc; the draft has processed
        // dfeed + d1..d_{k-1}.
        tb.truncate(ts, tlen + 1 + acc);
        if acc < k {
            db.truncate(ds, dbase + acc);
            dfeed = vec![p[acc]];
        } else {
            dfeed = vec![d[k - 1], p[k]];
        }
        g = p[acc];
    }
    let wall = t0.elapsed().as_secs_f64();
    out.truncate(n);
    tb.free_seq(ts);
    db.free_seq(ds);
    Ok(SpecRun { out, wall, rounds, accepted, draft_s, verify_s })
}

/// Expected tokens per target forward for acceptance rate a and draft length k.
fn expected_tokens(a: f64, k: usize) -> f64 {
    if (1.0 - a).abs() < 1e-9 {
        return (k + 1) as f64;
    }
    (1.0 - a.powi(k as i32 + 1)) / (1.0 - a)
}

fn piece(tok: &Tokenizer, id: u32) -> String {
    String::from_utf8_lossy(tok.token_bytes(id)).into_owned()
}

pub fn run(a: SpecArgs) -> Result<(), Box<dyn std::error::Error>> {
    prefer_performance_cores();
    let tok = crate::load_tokenizer(&a.draft)?;
    let tmodel = crate::load_model(&a.target)?;
    let dmodel = crate::load_model(&a.draft)?;
    let (tdt, ddt) = (parse_dtype(&a.target_dtype)?, parse_dtype(&a.draft_dtype)?);
    let ks: Vec<usize> = a.k.split(',').map(|s| s.trim().parse()).collect::<Result<_, _>>()?;
    let opts = |dtype| BatchOpts { max_seqs: 2, max_rows: 512, kv_blocks: 160, ..BatchOpts::new(dtype) };
    let te = BatchEngine::new(&tmodel, opts(tdt))?;
    let de = BatchEngine::new(&dmodel, opts(ddt))?;
    drop((tmodel, dmodel));
    let (tname, dname) = (model_label(&a.target), model_label(&a.draft));
    let mut tb = te.batch();
    let mut db = de.batch();
    let wiki = std::fs::read("weights/wikitext-2-raw-test.txt").ok().map(|b| {
        let t = String::from_utf8_lossy(&b);
        let ids = tok.encode(&t.chars().skip(2000).take(4000).collect::<String>());
        tok.decode(&ids[..96])
    });
    eprintln!("metal-spec: target {tname} ({}), draft {dname} ({}), {} tokens", tdt.name(), ddt.name(), a.tokens);
    let mut cats = Vec::new();
    for (name, text) in prompt_set(wiki) {
        if !name.contains(&a.filter) {
            continue;
        }
        let prompt = tok.encode(&text);
        // Baselines (median of runs): target alone resident and CPU-loop, draft alone.
        let (mut t_res, mut t_loop, mut d_res) = (vec![], vec![], vec![]);
        let mut reference = vec![];
        target_alone(&mut tb, &prompt, a.tokens, true)?; // warm-up
        for _ in 0..a.runs {
            let (o, w) = target_alone(&mut tb, &prompt, a.tokens, true)?;
            reference = o;
            t_res.push(w);
            t_loop.push(target_alone(&mut tb, &prompt, a.tokens, false)?.1);
            d_res.push(target_alone(&mut db, &prompt, a.tokens, true)?.1);
        }
        let (tr, tl, dr) = (median(&t_res), median(&t_loop), median(&d_res));
        let c = dr / tr;
        println!(
            "[{name}] {} prompt tokens | target alone {:.1} tok/s (GPU loop), {:.1} (CPU loop) | draft alone {:.1} tok/s | c = {c:.3}",
            prompt.len(),
            (a.tokens - 1) as f64 / tr,
            (a.tokens - 1) as f64 / tl,
            (a.tokens - 1) as f64 / dr
        );
        println!("   k  rounds  accept  alpha  tok/fwd  formula  draft ms  verify ms  tok/s  speedup  (vs CPU loop)");
        let mut per_k = Vec::new();
        for &k in &ks {
            let mut walls = vec![];
            let mut last = None;
            speculative(&mut tb, &mut db, &prompt, a.tokens, k)?; // warm-up
            for _ in 0..a.runs {
                let r = speculative(&mut tb, &mut db, &prompt, a.tokens, k)?;
                if r.out != reference {
                    let same = r.out.iter().zip(&reference).take_while(|(x, y)| x == y).count();
                    return Err(format!(
                        "[{name}] k={k}: speculative output differs from target greedy at token {same}"
                    )
                    .into());
                }
                walls.push(r.wall);
                last = Some(r);
            }
            let r = last.unwrap();
            let w = median(&walls);
            let acc: usize = r.accepted.iter().sum();
            // Trials: each round tests drafts until the first miss (or all k).
            let trials: usize = r.accepted.iter().map(|&x| if x < k { x + 1 } else { k }).sum();
            let alpha = acc as f64 / trials as f64;
            let tpf = (a.tokens - 1) as f64 / r.rounds as f64;
            let toks = (a.tokens - 1) as f64 / w;
            println!(
                "{k:>4} {:>7} {:>7.2} {alpha:>6.3} {tpf:>8.2} {:>8.2} {:>9.2} {:>10.2} {toks:>6.1} {:>7.2}x {:>7.2}x",
                r.rounds,
                acc as f64 / r.rounds as f64,
                expected_tokens(alpha, k),
                1e3 * r.draft_s / r.rounds as f64,
                1e3 * r.verify_s / r.rounds as f64,
                tr / w,
                tl / w
            );
            // Token-level playback for the page: (piece, accepted?) per output token.
            let mut play = vec![];
            let mut i = 1; // out[0] comes from the prefill
            for &acc in &r.accepted {
                for j in 0..=acc {
                    if i < r.out.len() {
                        play.push(json!([piece(&tok, r.out[i]), if j < acc { 1 } else { 0 }]));
                    }
                    i += 1;
                }
            }
            per_k.push(json!({
                "k": k,
                "rounds": r.rounds,
                "accepted_per_round": acc as f64 / r.rounds as f64,
                "alpha": alpha,
                "tokens_per_target_forward": tpf,
                "formula_tokens_per_forward": expected_tokens(alpha, k),
                "draft_ms_per_round": 1e3 * r.draft_s / r.rounds as f64,
                "verify_ms_per_round": 1e3 * r.verify_s / r.rounds as f64,
                "tok_s": toks,
                "speedup_vs_target": tr / w,
                "speedup_vs_target_cpu_loop": tl / w,
                "accept_hist": (0..=k).map(|x| r.accepted.iter().filter(|&&y| y == x).count()).collect::<Vec<_>>(),
                "playback": if k == 4 { json!(play) } else { Value::Null },
            }));
            if !a.no_record {
                append_jsonl(
                    &a.out,
                    &json!({
                        "timestamp": timestamp(),
                        "git_sha": git_sha(),
                        "stage": "metal-spec",
                        "kernels": format!("metal/batch/spec/{}+{}", tdt.name(), ddt.name()),
                        "backend": "metal",
                        "model": tname,
                        "draft_model": dname,
                        "dtype": tdt.name(),
                        "draft_dtype": ddt.name(),
                        "threads": 1,
                        "prompt": name,
                        "prompt_len": prompt.len(),
                        "gen_tokens": a.tokens,
                        "runs": a.runs,
                        "k": k,
                        "alpha": alpha,
                        "tokens_per_target_forward": tpf,
                        "decode_tok_s": toks,
                        "target_alone_tok_s": (a.tokens - 1) as f64 / tr,
                        "speedup": tr / w,
                        "machine": machine(),
                        "traced": false,
                        "note": a.note,
                    }),
                )?;
            }
        }
        cats.push(json!({
            "prompt": name,
            "text": text,
            "prompt_tokens": prompt.len(),
            "target_tok_s": (a.tokens - 1) as f64 / tr,
            "target_cpu_loop_tok_s": (a.tokens - 1) as f64 / tl,
            "draft_tok_s": (a.tokens - 1) as f64 / dr,
            "c": c,
            "continuation": tok.decode(&reference),
            "per_k": per_k,
        }));
        if !a.no_record {
            write_summary(&a, &tname, &dname, tdt, ddt, &cats)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_summary(a: &SpecArgs, tname: &str, dname: &str, tdt: nanoinfer_metal::DType, ddt: nanoinfer_metal::DType, cats: &[Value]) -> Result<(), Box<dyn std::error::Error>> {
    {
        let key = format!("{tname}:{}+{dname}:{}", tdt.name(), ddt.name());
        let mut doc: Value =
            std::fs::read(&a.summary).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(json!({}));
        let obj = doc.as_object_mut().ok_or("summary is not an object")?;
        obj.insert("what".into(), json!("Greedy speculative decoding on the M4 GPU: draft proposes k tokens, target verifies k+1 positions in one forward. Output asserted identical to target greedy. alpha = accepted / tested draft tokens; formula = (1 - alpha^(k+1)) / (1 - alpha); c = draft time / target time per token."));
        obj.insert("machine".into(), json!(machine()));
        obj.insert("git_sha".into(), json!(git_sha()));
        obj.insert("timestamp".into(), json!(timestamp()));
        obj.entry("pairs").or_insert(json!({})).as_object_mut().unwrap().insert(
            key,
            json!({"target": tname, "target_dtype": tdt.name(), "draft": dname, "draft_dtype": ddt.name(),
                   "tokens": a.tokens, "note": a.note, "prompts": cats}),
        );
        crate::metal_bench::metal_serve::round_floats(&mut doc);
        std::fs::write(&a.summary, serde_json::to_string(&doc)? + "\n")?;
        eprintln!("updated {}", a.summary.display());
    }
    Ok(())
}
