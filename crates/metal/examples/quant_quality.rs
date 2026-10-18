//! Which matrices can be quantised? KL(HF f32 || ours) and top-1 agreement on
//! the fixture prompts for mixed recipes (block dtype x LM-head dtype).
//!
//! `cargo run --release -p nanoinfer-metal --example quant_quality`
#[cfg(target_os = "macos")]
fn main() {
    use nanoinfer_metal::{DType, MetalEngine, MetalStage};
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model =
        nanoinfer_core::Model::from_safetensors(&std::fs::read(root.join("weights/gpt2/model.safetensors")).unwrap())
            .unwrap();
    let r: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("tests/fixtures/reference.json")).unwrap()).unwrap();
    let kl = |p: &[f32], q: &[f32]| -> f64 {
        let lse = |l: &[f32]| {
            let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            m + l.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
        };
        let (lp, lq) = (lse(p), lse(q));
        p.iter().zip(q).map(|(&a, &b)| (a as f64 - lp).exp() * ((a as f64 - lp) - (b as f64 - lq))).sum()
    };
    use DType::*;
    // WikiText-2 windows, decoded token by token so we get logits at every position.
    let text = std::fs::read_to_string(root.join("weights/wikitext-2-raw-test.txt")).ok();
    let tok = nanoinfer_core::Tokenizer::from_vocab_and_merges(
        &std::fs::read(root.join("weights/gpt2/vocab.json")).unwrap(),
        &std::fs::read(root.join("weights/gpt2/merges.txt")).unwrap(),
    )
    .unwrap();
    let (n_win, win) = (2usize, 512usize);
    let ids: Vec<u32> = text.map(|t| tok.encode(&t[..t.len().min(400_000)])).unwrap_or_default();
    let windows: Vec<&[u32]> = ids.chunks_exact(win).take(n_win).collect();
    let run_windows = |engine: &MetalEngine| -> Vec<Vec<f32>> {
        let mut all = vec![];
        for w in &windows {
            let mut s = engine.session();
            for &t in &w[..win - 1] {
                all.push(s.step(&[t]).unwrap().to_vec());
            }
        }
        all
    };
    let f32_engine = MetalEngine::new(&model, MetalStage::Mma).unwrap();
    let reference = run_windows(&f32_engine);
    drop(f32_engine);
    let nll = |logits: &[Vec<f32>]| -> f64 {
        let mut total = 0.0;
        let mut i = 0;
        for w in &windows {
            for &target in &w[1..] {
                let l = &logits[i];
                let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let lse = m + l.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln();
                total += lse - l[target as usize] as f64;
                i += 1;
            }
        }
        total / i as f64
    };
    if !windows.is_empty() {
        println!(
            "WikiText-2: {} windows x {win} tokens, f32 ppl {:.4} (token-by-token decode, metal-mma)",
            windows.len(),
            nll(&reference).exp()
        );
    }
    println!(
        "{:<6} {:<6} {:>10} {:>10} {:>6} {:>10} {:>10} {:>8}",
        "blocks", "head", "mean KL", "max KL", "top-1", "wiki ppl", "wiki KL", "wiki top1"
    );
    for (blocks, head) in
        [(F32, F32), (F16, F16), (Q8_0, F16), (Q8_0, Q8_0), (Q4_0, F16), (Q4_0, Q8_0), (Q4_0, Q4_0), (F16, Q4_0)]
    {
        let recipe = nanoinfer_metal::Recipe { dtype: blocks, head_dtype: head, ..MetalStage::Mma.recipe() };
        let engine = MetalEngine::with_recipe(&model, MetalStage::Mma, recipe).unwrap();
        let (mut sum, mut max, mut agree, mut n) = (0.0, 0.0f64, 0, 0);
        for p in r["prompts"].as_array().unwrap() {
            let ids: Vec<u32> = p["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
            let b = std::fs::read(root.join("tests/fixtures").join(p["logits_file"].as_str().unwrap())).unwrap();
            let expected: Vec<f32> = b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            let mut s = engine.session();
            let logits = s.step(&ids).unwrap();
            let k = kl(&expected, logits);
            sum += k;
            max = max.max(k);
            agree += (nanoinfer_core::sampler::argmax(logits) as u64 == p["argmax"].as_u64().unwrap()) as usize;
            n += 1;
        }
        let (mut ppl, mut wkl, mut wtop) = (f64::NAN, f64::NAN, f64::NAN);
        if !windows.is_empty() {
            let logits = run_windows(&engine);
            ppl = nll(&logits).exp();
            wkl = logits.iter().zip(&reference).map(|(q, p)| kl(p, q)).sum::<f64>() / logits.len() as f64;
            wtop = logits
                .iter()
                .zip(&reference)
                .filter(|(q, p)| nanoinfer_core::sampler::argmax(q) == nanoinfer_core::sampler::argmax(p))
                .count() as f64
                / logits.len() as f64;
        }
        println!(
            "{:<6} {:<6} {:>10.2e} {:>10.2e} {agree:>3}/{n} {ppl:>10.3} {wkl:>10.2e} {:>7.1}%",
            blocks.name(),
            head.name(),
            sum / n as f64,
            max,
            100.0 * wtop
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {}
