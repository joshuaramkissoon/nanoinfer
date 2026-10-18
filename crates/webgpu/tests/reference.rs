//! Every WebGPU stage against the Hugging Face fixtures (tests/fixtures),
//! natively through wgpu's Metal backend.
//!
//! f32 stages must match HF logits to 1e-3 and reproduce its greedy
//! continuation; f16 must keep greedy; Q8 is judged by KL and top-1. The GPU
//! argmax and the resident loop must pick exactly what a CPU argmax over the
//! same stage's logits picks. Without weights the tests print SKIP and pass.

use nanoinfer_core::Model;
use nanoinfer_webgpu::gpu::block_on;
use nanoinfer_webgpu::{DType, Engine, Gpu, WgpuStage};
use serde_json::Value;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::OnceLock;
use std::time::Instant;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn reference() -> &'static Value {
    static R: OnceLock<Value> = OnceLock::new();
    R.get_or_init(|| serde_json::from_slice(&std::fs::read(root().join("tests/fixtures/reference.json")).unwrap()).unwrap())
}
fn model() -> Option<&'static Model> {
    static M: OnceLock<Model> = OnceLock::new();
    let path = root().join("weights/gpt2/model.safetensors");
    if !path.exists() {
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
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let lse = |l: &[f32]| {
        let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + l.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
    };
    let (lp, lq) = (lse(p), lse(q));
    p.iter().zip(q).map(|(&a, &b)| (a as f64 - lp).exp() * ((a as f64 - lp) - (b as f64 - lq))).sum()
}
fn tolerance(d: DType) -> (f32, f64) {
    match d {
        DType::F32 => (1e-3, 1e-4),
        DType::F16 => (0.25, 1e-3),
        DType::Q8_0 => (f32::INFINITY, 0.02),
    }
}
fn now() -> impl Fn() -> f64 {
    let t = Instant::now();
    move || t.elapsed().as_secs_f64() * 1e3
}

#[test]
fn every_webgpu_stage_matches_hf() {
    let Some(model) = model() else { return };
    let gpu = Rc::new(block_on(Gpu::new()).unwrap());
    let now = now();
    let r = reference();
    for &stage in WgpuStage::ALL {
        if !stage.supported(&gpu) {
            eprintln!("skip {stage}: unsupported here");
            continue;
        }
        let engine = Rc::new(Engine::new(gpu.clone(), model, stage));
        let rec = stage.recipe();
        let (tol, kl_tol) = tolerance(rec.dtype.max_lossy(rec.head_dtype));
        for p in r["prompts"].as_array().unwrap() {
            let prompt = ids(&p["ids"]);
            let expected = ref_logits(p["logits_file"].as_str().unwrap());
            let mut s = engine.session(64);
            let logits = block_on(s.step(&prompt, &now)).unwrap();
            let d = logits.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            let k = kl(&expected, &logits);
            let top = nanoinfer_core::sampler::argmax(&logits);
            println!("{stage} {:?}: max |diff| {d:.2e}, KL {k:.2e}, top-1 {top}", p["text"].as_str().unwrap());
            assert!(d < tol, "{stage}: max abs diff {d}");
            assert!(k < kl_tol, "{stage}: KL {k}");
            assert_eq!(top as u64, p["argmax"].as_u64().unwrap(), "{stage}: top-1");
            // decode path: one token at a time
            let mut s = engine.session(64);
            for t in &prompt[..prompt.len() - 1] {
                block_on(s.step(std::slice::from_ref(t), &now)).unwrap();
            }
            let logits = block_on(s.step(&prompt[prompt.len() - 1..], &now)).unwrap();
            let d = logits.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(d < tol, "{stage} token-by-token: max abs diff {d}");
        }
        // greedy vs HF (f32/f16), and the GPU greedy paths vs CPU argmax (all)
        for g in r["greedy"].as_array().unwrap() {
            let prompt = ids(&r["prompts"][g["prompt"].as_u64().unwrap() as usize]["ids"]);
            let expected = ids(&g["new_ids"]);
            let mut s = engine.session(prompt.len() + expected.len() + 1);
            let first = block_on(s.step_greedy(&prompt, &now)).unwrap();
            let mut out = vec![first];
            out.extend(block_on(s.decode_greedy(first, expected.len() - 1, &now)).unwrap());
            let agree = out.iter().zip(&expected).take_while(|(a, b)| a == b).count();
            println!("{stage} greedy: first {agree}/{} tokens match HF", expected.len());
            if rec.dtype != DType::Q8_0 {
                assert_eq!(out, expected, "{stage} greedy");
            }
            // CPU argmax over the same stage's logits, token by token
            let mut c = engine.session(prompt.len() + expected.len() + 1);
            let mut next = nanoinfer_core::sampler::argmax(&block_on(c.step(&prompt, &now)).unwrap());
            let mut cpu = vec![next];
            while cpu.len() < out.len() {
                next = nanoinfer_core::sampler::argmax(&block_on(c.step(&[next], &now)).unwrap());
                cpu.push(next);
            }
            assert_eq!(out, cpu, "{stage}: GPU greedy path != CPU argmax");
        }
    }
}
