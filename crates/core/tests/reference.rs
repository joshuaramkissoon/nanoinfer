//! Correctness against Hugging Face transformers.
//!
//! Fixtures come from `scripts/gen_reference.py` (committed in tests/fixtures/).
//! Weights are NOT committed: run `scripts/fetch_weights.sh` first. Without
//! weights these tests print a loud SKIP and pass, unless
//! `NANOINFER_REQUIRE_WEIGHTS=1` is set, in which case they fail.

use nanoinfer_core::trace::{Clock, Recorder};
use nanoinfer_core::{sampler, Model, NoTrace, Session, Stage, Tokenizer};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::OnceLock;

const TOL: f32 = 1e-3;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn reference() -> &'static Value {
    static R: OnceLock<Value> = OnceLock::new();
    R.get_or_init(|| {
        let s = std::fs::read(root().join("tests/fixtures/reference.json")).expect("fixtures missing");
        serde_json::from_slice(&s).unwrap()
    })
}

fn weights_dir() -> Option<PathBuf> {
    let d = root().join("weights/gpt2");
    if d.join("model.safetensors").exists() {
        Some(d)
    } else if std::env::var("NANOINFER_REQUIRE_WEIGHTS").is_ok() {
        panic!("weights missing: run scripts/fetch_weights.sh");
    } else {
        eprintln!("SKIP: weights/gpt2 missing (run scripts/fetch_weights.sh)");
        None
    }
}

fn model() -> Option<&'static Model> {
    static M: OnceLock<Model> = OnceLock::new();
    let dir = weights_dir()?;
    Some(M.get_or_init(|| Model::from_safetensors(&std::fs::read(dir.join("model.safetensors")).unwrap()).unwrap()))
}

fn tokenizer() -> Option<&'static Tokenizer> {
    static T: OnceLock<Tokenizer> = OnceLock::new();
    let dir = weights_dir()?;
    Some(T.get_or_init(|| {
        let v = std::fs::read(dir.join("vocab.json")).unwrap();
        let m = std::fs::read(dir.join("merges.txt")).unwrap();
        Tokenizer::from_vocab_and_merges(&v, &m).unwrap()
    }))
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn ref_logits(file: &str) -> Vec<f32> {
    let b = std::fs::read(root().join("tests/fixtures").join(file)).unwrap();
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn tokenizer_matches_hf() {
    let Some(tok) = tokenizer() else { return };
    for case in reference()["tokenizer"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expected = ids(&case["ids"]);
        assert_eq!(tok.encode(text), expected, "encode mismatch for {text:?}");
        assert_eq!(tok.decode(&expected), text, "decode round-trip failed for {text:?}");
    }
    for p in reference()["prompts"].as_array().unwrap() {
        assert_eq!(tok.encode(p["text"].as_str().unwrap()), ids(&p["ids"]));
    }
}

#[test]
fn last_logits_match_hf_for_every_stage() {
    let Some(model) = model() else { return };
    for &stage in Stage::ALL {
        for p in reference()["prompts"].as_array().unwrap() {
            let prompt = ids(&p["ids"]);
            let expected = ref_logits(p["logits_file"].as_str().unwrap());
            let mut s = Session::new(model, stage);
            let logits = s.step(model, &prompt, &mut NoTrace).unwrap();
            let d = max_abs_diff(logits, &expected);
            println!("{stage} {:?}: max |diff| = {d:.2e}", p["text"].as_str().unwrap());
            assert!(d < TOL, "{stage}: max abs diff {d} >= {TOL}");
            assert_eq!(sampler::argmax(logits) as u64, p["argmax"].as_u64().unwrap());
        }
    }
}

#[test]
fn kv_cache_token_by_token_matches_hf() {
    // Feed the prompt one token at a time: exercises the decode path (n = 1,
    // growing cache) rather than a single prefill.
    let Some(model) = model() else { return };
    for p in reference()["prompts"].as_array().unwrap() {
        let prompt = ids(&p["ids"]);
        let expected = ref_logits(p["logits_file"].as_str().unwrap());
        let mut s = Session::new(model, Stage::KvCache);
        for t in &prompt[..prompt.len() - 1] {
            s.step(model, std::slice::from_ref(t), &mut NoTrace).unwrap();
        }
        let logits = s.step(model, &prompt[prompt.len() - 1..], &mut NoTrace).unwrap();
        let d = max_abs_diff(logits, &expected);
        assert!(d < TOL, "token-by-token max abs diff {d}");
    }
}

fn greedy(model: &Model, stage: Stage, prompt: &[u32], n: usize) -> Vec<u32> {
    let mut s = Session::new(model, stage);
    let mut next = sampler::argmax(s.step(model, prompt, &mut NoTrace).unwrap());
    let mut out = vec![next];
    while out.len() < n {
        next = sampler::argmax(s.step(model, &[next], &mut NoTrace).unwrap());
        out.push(next);
    }
    out
}

#[test]
fn greedy_generation_matches_hf() {
    let Some(model) = model() else { return };
    let r = reference();
    for g in r["greedy"].as_array().unwrap() {
        let prompt = ids(&r["prompts"][g["prompt"].as_u64().unwrap() as usize]["ids"]);
        let expected = ids(&g["new_ids"]);
        assert_eq!(greedy(model, Stage::KvCache, &prompt, expected.len()), expected, "kv-cache greedy");
        // Naive is O(n²), so check a shorter prefix.
        assert_eq!(greedy(model, Stage::Naive, &prompt, 8), expected[..8], "naive greedy");
    }
}

#[test]
fn naive_and_kv_cache_agree_while_decoding() {
    let Some(model) = model() else { return };
    let prompt = ids(&reference()["prompts"][1]["ids"]);
    let mut a = Session::new(model, Stage::Naive);
    let mut b = Session::new(model, Stage::KvCache);
    let mut tok = prompt.clone();
    for _ in 0..6 {
        let la = a.step(model, &tok, &mut NoTrace).unwrap().to_vec();
        let lb = b.step(model, &tok, &mut NoTrace).unwrap();
        assert!(max_abs_diff(&la, lb) < TOL);
        tok = vec![sampler::argmax(&la)];
    }
}

#[test]
fn trace_reports_ops_kv_topk_and_attention() {
    let Some(model) = model() else { return };
    let prompt = ids(&reference()["prompts"][0]["ids"]);
    // A fake clock that ticks 1µs per call: deterministic, no std::time needed.
    let ticks = std::cell::Cell::new(0.0);
    let clock = || {
        ticks.set(ticks.get() + 1e-6);
        ticks.get()
    };
    let mut rec = Recorder::new(&clock as &dyn Fn() -> f64);
    rec.top_k = 5;
    rec.attention_head = Some((3, 7));

    let mut s = Session::new(model, Stage::KvCache);
    s.step(model, &prompt, &mut rec).unwrap();
    s.step(model, &[11], &mut rec).unwrap();

    assert_eq!(rec.forwards.len(), 2);
    assert_eq!(rec.forwards[1].kv.map(|k| k.0), Some(prompt.len() + 1));
    // 12 layers x 10 ops (ln, qkv, kv_write, attn, proj, res, ln, fc, gelu, fc_proj, res = 11) + embed + ln_f + lm_head
    let per_forward = rec.ops.iter().filter(|o| o.forward == 1).count();
    assert_eq!(per_forward, 12 * 11 + 3);
    assert!(rec.ops.iter().all(|o| o.duration > 0.0 && o.op.flops > 0 || o.op.name == "kv_write"));
    assert_eq!(rec.top.len(), 5);
    assert!(rec.top.windows(2).all(|w| w[0].1 >= w[1].1));
    assert_eq!(rec.attention_probs.len(), prompt.len() + 1);
    assert!((rec.attention_probs.iter().sum::<f32>() - 1.0).abs() < 1e-4);
}

// Make sure `Clock` stays implementable for plain closures (the wasm host will do this).
#[allow(dead_code)]
fn clock_is_object_safe(c: &dyn Clock) -> f64 {
    c.now()
}
