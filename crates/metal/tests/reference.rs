//! Every Metal stage against Hugging Face fixtures (tests/fixtures).
//!
//! f32 stages must match to 1e-3 like the CPU engine and reproduce HF's greedy
//! continuation token for token. Lower-precision stages get a looser logit
//! tolerance; f16 must still match greedy, quantised stages must keep top-1 on
//! the fixture prompts. Without weights the tests print SKIP and pass.
#![cfg(target_os = "macos")]

use nanoinfer_core::Model;
use nanoinfer_metal::{DType, MetalEngine, MetalStage};
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

fn ref_logits(file: &str) -> Vec<f32> {
    let b = std::fs::read(root().join("tests/fixtures").join(file)).unwrap();
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

/// Max |Δlogit| allowed. Block-quantised stages are judged by KL instead.
fn tolerance(d: DType) -> f32 {
    match d {
        DType::F32 => 1e-3,
        DType::F16 => 0.25,
        DType::Q8_0 | DType::Q4_0 => f32::INFINITY,
    }
}

/// Max KL(HF || ours) in nats over the next-token distribution.
fn kl_tolerance(d: DType) -> f64 {
    match d {
        DType::F32 | DType::F16 => 1e-3,
        DType::Q8_0 => 0.01,
        DType::Q4_0 => 0.25,
    }
}

fn kl(p_logits: &[f32], q_logits: &[f32]) -> f64 {
    let lse = |l: &[f32]| {
        let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + l.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
    };
    let (lp, lq) = (lse(p_logits), lse(q_logits));
    p_logits.iter().zip(q_logits).map(|(&a, &b)| (a as f64 - lp).exp() * ((a as f64 - lp) - (b as f64 - lq))).sum()
}

#[test]
fn every_metal_stage_matches_hf_logits() {
    let Some(model) = model() else { return };
    for &stage in MetalStage::ALL {
        let engine = MetalEngine::new(model, stage).unwrap();
        let dtype = stage.recipe().dtype;
        for p in reference()["prompts"].as_array().unwrap() {
            let prompt = ids(&p["ids"]);
            let expected = ref_logits(p["logits_file"].as_str().unwrap());
            let mut s = engine.session();
            let logits = s.step(&prompt).unwrap();
            let d = logits.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            let top = nanoinfer_core::sampler::argmax(logits);
            let k = kl(&expected, logits);
            println!("{stage} {:?}: max |diff| = {d:.2e}, KL = {k:.2e}, top-1 {top}", p["text"].as_str().unwrap());
            assert!(d < tolerance(dtype), "{stage}: max abs diff {d}");
            assert!(k < kl_tolerance(dtype), "{stage}: KL {k}");
            assert_eq!(top as u64, p["argmax"].as_u64().unwrap(), "{stage}: top-1 differs");
            // Decode path (n = 1) too: feed the prompt one token at a time.
            let mut s = engine.session();
            for t in &prompt[..prompt.len() - 1] {
                s.step(std::slice::from_ref(t)).unwrap();
            }
            let logits = s.step(&prompt[prompt.len() - 1..]).unwrap();
            let d = logits.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(d < tolerance(dtype), "{stage} token-by-token: max abs diff {d}");
        }
    }
}

#[test]
fn metal_greedy_matches_hf() {
    let Some(model) = model() else { return };
    let r = reference();
    for &stage in MetalStage::ALL {
        if !matches!(stage.recipe().dtype, DType::F32 | DType::F16) {
            continue;
        }
        let engine = MetalEngine::new(model, stage).unwrap();
        for g in r["greedy"].as_array().unwrap() {
            let prompt = ids(&r["prompts"][g["prompt"].as_u64().unwrap() as usize]["ids"]);
            let expected = ids(&g["new_ids"]);
            let mut s = engine.session();
            let mut next = s.step_greedy(&prompt).unwrap();
            let mut out = vec![next];
            while out.len() < expected.len() {
                next = s.step_greedy(&[next]).unwrap();
                out.push(next);
            }
            assert_eq!(out, expected, "{stage} greedy");
        }
    }
}

/// GPU argmax and the pipelined greedy loop must pick exactly the tokens a
/// CPU argmax over the same stage's logits picks.
#[test]
fn gpu_greedy_paths_agree_with_cpu_argmax() {
    let Some(model) = model() else { return };
    let prompt = ids(&reference()["prompts"][2]["ids"]);
    for &stage in MetalStage::ALL {
        let engine = MetalEngine::new(model, stage).unwrap();
        let mut a = engine.session();
        let mut next = nanoinfer_core::sampler::argmax(a.step(&prompt).unwrap());
        let first = next;
        let mut cpu = vec![];
        for _ in 0..16 {
            next = nanoinfer_core::sampler::argmax(a.step(&[next]).unwrap());
            cpu.push(next);
        }
        let mut b = engine.session();
        assert_eq!(b.step_greedy(&prompt).unwrap(), first, "{stage}");
        assert_eq!(b.decode_greedy(first, 16).unwrap(), cpu, "{stage} decode_greedy");
        // Streaming delivery, with and without lookahead, gives the same tokens;
        // stopping early rolls the in-flight forwards back, so decoding resumes
        // exactly where the delivered tokens end.
        for depth in [1, 2, 3] {
            let mut c = engine.session();
            c.step_greedy(&prompt).unwrap();
            let mut got = vec![];
            assert_eq!(
                c.decode_stream(first, 16, depth, |t| {
                    got.push(t);
                    true
                })
                .unwrap(),
                16
            );
            assert_eq!(got, cpu, "{stage} decode_stream depth {depth}");
            let mut d = engine.session();
            d.step_greedy(&prompt).unwrap();
            let mut got = vec![];
            let k = d
                .decode_stream(first, 16, depth, |t| {
                    got.push(t);
                    got.len() < 5
                })
                .unwrap();
            assert_eq!((k, d.len()), (5, prompt.len() + 5), "{stage} stop at depth {depth}");
            got.extend(d.decode_greedy(got[4], 11).unwrap());
            assert_eq!(got, cpu, "{stage} resume after stop, depth {depth}");
        }
    }
}

/// The fused decode kernels exist for every dtype (only q4 has a named
/// stage): f32 and f16 must still reproduce HF's greedy continuation exactly.
#[test]
fn fused_pipelined_matches_hf_greedy_in_every_dtype() {
    let Some(model) = model() else { return };
    let r = reference();
    for (dtype, fused) in [DType::F32, DType::F16, DType::Q8_0, DType::Q4_0]
        .into_iter()
        .flat_map(|d| [(d, nanoinfer_metal::Fusion::Full), (d, nanoinfer_metal::Fusion::Epilogue)])
    {
        let head_dtype = if dtype == DType::Q4_0 { DType::Q8_0 } else { dtype };
        let recipe = nanoinfer_metal::Recipe { dtype, head_dtype, fused, ..MetalStage::Pipelined.recipe() };
        let engine = MetalEngine::with_recipe(model, MetalStage::Pipelined, recipe).unwrap();
        for g in r["greedy"].as_array().unwrap() {
            let prompt = ids(&r["prompts"][g["prompt"].as_u64().unwrap() as usize]["ids"]);
            let expected = ids(&g["new_ids"]);
            let mut s = engine.session();
            let first = s.step_greedy(&prompt).unwrap();
            let mut out = vec![first];
            out.extend(s.decode_greedy(first, expected.len() - 1).unwrap());
            let agree = out.iter().zip(&expected).take_while(|(a, b)| a == b).count();
            println!(
                "fused ({fused:?}) + pipelined {dtype:?}: first {agree}/{} greedy tokens match HF",
                expected.len()
            );
            if matches!(dtype, DType::F32 | DType::F16) {
                assert_eq!(out, expected, "{dtype:?}");
            }
        }
    }
}

/// Fixture prompts are shorter than one 64-key chunk, so check the split-K /
/// flash attention paths against attention_simd at a context of 300 tokens
/// (5 chunks), in f32 so only the attention arithmetic differs.
#[test]
fn split_and_flash_attention_match_simd_at_long_context() {
    let Some(model) = model() else { return };
    let mut seq = vec![];
    while seq.len() < 300 {
        seq.extend(ids(&reference()["prompts"][3]["ids"]));
    }
    seq.truncate(300);
    let base = nanoinfer_metal::Recipe { dtype: DType::F32, head_dtype: DType::F32, ..MetalStage::Mma.recipe() };
    let run = |attn: nanoinfer_metal::Attn, kv_f16: bool| {
        let recipe = nanoinfer_metal::Recipe { attn, kv_f16, ..base };
        let engine = MetalEngine::with_recipe(model, MetalStage::Mma, recipe).unwrap();
        let mut s = engine.session();
        // Two prefill chunks, so the second one starts at pos0 = 100.
        s.step(&seq[..100]).unwrap();
        let prefill = s.step(&seq[100..250]).unwrap().to_vec();
        for &t in &seq[250..299] {
            s.step(&[t]).unwrap();
        }
        (prefill, s.step(&seq[299..]).unwrap().to_vec())
    };
    let (p0, d0) = run(nanoinfer_metal::Attn::Simd, false);
    for (attn, kv_f16, tol) in [
        (nanoinfer_metal::Attn::SplitK, false, 1e-3),
        (nanoinfer_metal::Attn::Flash, false, 1e-3),
        (nanoinfer_metal::Attn::Flash, true, 0.1),
    ] {
        let (p, d) = run(attn, kv_f16);
        let dp = p.iter().zip(&p0).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        let dd = d.iter().zip(&d0).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        println!("{attn:?} (f16 KV: {kv_f16}) vs Simd at context 300: prefill max diff {dp:.2e}, decode {dd:.2e}");
        assert!(dp < tol && dd < tol, "{attn:?} kv_f16={kv_f16}: {dp} {dd}");
    }
}

/// Q8 KV cache (metal-kv8) at a context of 300 tokens, f32 weights so only the
/// KV quantisation differs: small KL vs the f32 cache, same top-1.
#[test]
fn q8_kv_cache_close_to_f32_at_long_context() {
    let Some(model) = model() else { return };
    let mut seq = vec![];
    while seq.len() < 300 {
        seq.extend(ids(&reference()["prompts"][3]["ids"]));
    }
    seq.truncate(300);
    let run = |kv_q8: bool| {
        let recipe = nanoinfer_metal::Recipe {
            dtype: DType::F32,
            head_dtype: DType::F32,
            kv_q8,
            kv_f16: false,
            ..MetalStage::Kv8.recipe()
        };
        let engine = MetalEngine::with_recipe(model, MetalStage::Kv8, recipe).unwrap();
        let mut s = engine.session();
        s.step(&seq[..100]).unwrap();
        let prefill = s.step(&seq[100..250]).unwrap().to_vec();
        for &t in &seq[250..299] {
            s.step(&[t]).unwrap();
        }
        (prefill, s.step(&seq[299..]).unwrap().to_vec())
    };
    let (p0, d0) = run(false);
    let (p, d) = run(true);
    for (name, a, b) in [("prefill", &p, &p0), ("decode", &d, &d0)] {
        let diff = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        let k = kl(b, a);
        println!("Q8 KV vs f32 KV at context 300, {name}: max |diff| {diff:.3e}, KL {k:.2e}");
        assert!(k < 0.01, "{name}: KL {k}");
        assert_eq!(nanoinfer_core::sampler::argmax(a), nanoinfer_core::sampler::argmax(b), "{name} top-1");
    }
}
