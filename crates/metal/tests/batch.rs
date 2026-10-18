//! The batched / ragged / paged forward (`nanoinfer_metal::batch`):
//! f32 must reproduce HF's greedy continuations; any dtype must give a
//! sequence bit-identical tokens alone or batched (batch invariance), on
//! the CPU-driven and the GPU-resident paths, and after KV pages are
//! recycled between sequences.
#![cfg(target_os = "macos")]

use nanoinfer_core::Model;
use nanoinfer_metal::{BatchEngine, BatchOpts, DType};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::OnceLock;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn reference() -> &'static Value {
    static R: OnceLock<Value> = OnceLock::new();
    R.get_or_init(|| {
        serde_json::from_slice(&std::fs::read(root().join("tests/fixtures/reference.json")).unwrap()).unwrap()
    })
}

fn model() -> Option<&'static Model> {
    static M: OnceLock<Model> = OnceLock::new();
    let path = root().join("weights/gpt2/model.safetensors");
    if !path.exists() {
        assert!(std::env::var("NANOINFER_REQUIRE_WEIGHTS").is_err(), "weights missing");
        eprintln!("SKIP: weights/gpt2 missing (run scripts/fetch_weights.sh)");
        return None;
    }
    Some(M.get_or_init(|| Model::from_safetensors(&std::fs::read(path).unwrap()).unwrap()))
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn prompts() -> Vec<Vec<u32>> {
    reference()["prompts"].as_array().unwrap().iter().map(|p| ids(&p["ids"])).collect()
}

#[test]
fn batch_f32_matches_hf_greedy() {
    let Some(model) = model() else { return };
    let r = reference();
    for gemm_min_rows in [usize::MAX, 4] {
        let opts = BatchOpts { gemm_min_rows, ..BatchOpts::new(DType::F32) };
        let engine = BatchEngine::new(model, opts).unwrap();
        let mut b = engine.batch();
        for g in r["greedy"].as_array().unwrap() {
            let prompt = ids(&r["prompts"][g["prompt"].as_u64().unwrap() as usize]["ids"]);
            let expected = ids(&g["new_ids"]);
            let s = b.add_seq().unwrap();
            let mut next = b.step(&[(s, &prompt)], false).unwrap()[0][0];
            let mut out = vec![next];
            while out.len() < expected.len() {
                next = b.step(&[(s, &[next])], false).unwrap()[0][0];
                out.push(next);
            }
            assert_eq!(out, expected, "gemm_min_rows {gemm_min_rows}");
            b.free_seq(s);
        }
    }
}

/// Greedy tokens for each prompt, decoded alone (batch of 1, CPU loop).
fn alone(engine: &BatchEngine, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut b = engine.batch();
    let s = b.add_seq().unwrap();
    let mut next = b.step(&[(s, prompt)], false).unwrap()[0][0];
    let mut out = vec![next];
    for _ in 1..n {
        next = b.step(&[(s, &[next])], false).unwrap()[0][0];
        out.push(next);
    }
    out
}

#[test]
fn batched_decode_is_batch_invariant() {
    let Some(model) = model() else { return };
    for dtype in [DType::Q4_0, DType::F16] {
        let opts = BatchOpts { gemm_min_rows: usize::MAX, kv_blocks: 64, ..BatchOpts::new(dtype) };
        let engine = BatchEngine::new(model, opts).unwrap();
        let ps = prompts();
        let n = 20;
        let expect: Vec<Vec<u32>> = ps.iter().map(|p| alone(&engine, p, n)).collect();

        // All prompts in one ragged prefill, then batched decode on the CPU loop.
        let mut b = engine.batch();
        let seqs: Vec<usize> = ps.iter().map(|_| b.add_seq().unwrap()).collect();
        let work: Vec<(usize, &[u32])> = seqs.iter().zip(&ps).map(|(&s, p)| (s, p.as_slice())).collect();
        let first: Vec<u32> = b.step(&work, false).unwrap().into_iter().map(|v| v[0]).collect();
        let mut got: Vec<Vec<u32>> = first.iter().map(|&t| vec![t]).collect();
        for _ in 1..n / 2 {
            let toks: Vec<[u32; 1]> = got.iter().map(|g| [*g.last().unwrap()]).collect();
            let work: Vec<(usize, &[u32])> = seqs.iter().zip(&toks).map(|(&s, t)| (s, t.as_slice())).collect();
            for (g, t) in got.iter_mut().zip(b.step(&work, false).unwrap()) {
                g.push(t[0]);
            }
        }
        // ...then the rest GPU-resident.
        let last: Vec<u32> = got.iter().map(|g| *g.last().unwrap()).collect();
        let rest = b.decode_resident(&seqs, &last, n - n / 2).unwrap();
        for (g, r) in got.iter_mut().zip(rest) {
            g.extend(r);
        }
        assert_eq!(got, expect, "{dtype:?}: batched vs alone");

        // Recycle pages: free two sequences, start a new one that reuses their
        // blocks while the others keep decoding.
        b.free_seq(seqs[0]);
        b.free_seq(seqs[2]);
        let s = b.add_seq().unwrap();
        let t = b.step(&[(s, ps[1].as_slice())], false).unwrap()[0][0];
        let mut fresh = vec![t];
        fresh.extend(b.decode_resident(&[s], &[t], n - 1).unwrap().remove(0));
        assert_eq!(fresh, expect[1], "{dtype:?}: after page reuse");
    }
}

/// Verification rows (all logits) must predict, at every position, what
/// one-token-at-a-time decoding predicts there; truncate rolls back.
#[test]
fn all_logits_and_truncate_match_sequential() {
    let Some(model) = model() else { return };
    let opts = BatchOpts { gemm_min_rows: usize::MAX, ..BatchOpts::new(DType::Q4_0) };
    let engine = BatchEngine::new(model, opts).unwrap();
    let p = &prompts()[2];
    let seq = alone(&engine, p, 12);
    let mut b = engine.batch();
    let s = b.add_seq().unwrap();
    b.step(&[(s, p.as_slice())], false).unwrap();
    // Feed seq[0..9] in one forward: prediction after seq[i] must be seq[i + 1].
    let preds = b.step(&[(s, &seq[..9])], true).unwrap().remove(0);
    assert_eq!(preds, seq[1..10]);
    // Roll back 5 positions and continue: same tokens.
    b.truncate(s, p.len() + 4);
    let next = b.step(&[(s, &seq[4..5])], false).unwrap()[0][0];
    assert_eq!(next, seq[5]);
}
