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

fn tolerance(d: DType) -> f32 {
    match d {
        DType::F32 => 1e-3,
        DType::F16 => 0.1,
        DType::Q8_0 => 1.0,
        DType::Q4_0 => 4.0,
    }
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
            println!("{stage} {:?}: max |diff| = {d:.2e}, top-1 {top}", p["text"].as_str().unwrap());
            assert!(d < tolerance(dtype), "{stage}: max abs diff {d}");
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
