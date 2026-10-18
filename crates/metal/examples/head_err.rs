//! How far are Q4_0 LM-head logits from Q8_0 ones? Same body (metal-lanes,
//! Q4 blocks), only the head differs, so the logit difference is exactly the
//! head's extra quantisation error. Picks the margin for metal-rescore: the
//! true (Q8) argmax is guaranteed to be a candidate if every logit's error is
//! below delta / 2.
//!
//! `cargo run --release -p nanoinfer-metal --example head_err`
#[cfg(target_os = "macos")]
fn main() {
    use nanoinfer_metal::{DType, MetalEngine, MetalStage};
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model =
        nanoinfer_core::Model::from_safetensors(&std::fs::read(root.join("weights/gpt2/model.safetensors")).unwrap())
            .unwrap();
    let tok = nanoinfer_core::Tokenizer::from_vocab_and_merges(
        &std::fs::read(root.join("weights/gpt2/vocab.json")).unwrap(),
        &std::fs::read(root.join("weights/gpt2/merges.txt")).unwrap(),
    )
    .unwrap();
    let text = std::fs::read_to_string(root.join("weights/wikitext-2-raw-test.txt")).unwrap();
    let ids = tok.encode(&text[..200_000]);
    let windows: Vec<&[u32]> = ids.chunks_exact(256).take(4).collect();
    let q8 = MetalEngine::new(&model, MetalStage::Lanes).unwrap();
    let rec = nanoinfer_metal::Recipe { head_dtype: DType::Q4_0, ..MetalStage::Lanes.recipe() };
    let q4 = MetalEngine::with_recipe(&model, MetalStage::Lanes, rec).unwrap();
    let (mut errs, mut gaps, mut agree, mut n) = (vec![], vec![], 0, 0);
    let mut cands = vec![];
    for w in &windows {
        let (mut a, mut b) = (q8.session(), q4.session());
        for &t in w.iter() {
            let la = a.step(&[t]).unwrap().to_vec();
            let lb = b.step(&[t]).unwrap().to_vec();
            let e = la.iter().zip(&lb).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            errs.push(e);
            let ta = nanoinfer_core::sampler::argmax(&la);
            let tb = nanoinfer_core::sampler::argmax(&lb);
            agree += (ta == tb) as usize;
            n += 1;
            let mut s = la.clone();
            s.sort_by(|x, y| y.total_cmp(x));
            gaps.push(s[0] - s[1]);
            let mb = lb[tb as usize];
            for d in [0.5f32, 1.0, 1.5, 2.0] {
                cands.push((d, lb.iter().filter(|&&v| v >= mb - d).count()));
            }
        }
    }
    errs.sort_by(f32::total_cmp);
    let p = |v: &[f32], q: f64| v[((v.len() - 1) as f64 * q) as usize];
    println!("{n} positions: max |q4 - q8| logit error p50 {:.3}, p99 {:.3}, max {:.3}", p(&errs, 0.5), p(&errs, 0.99), errs[errs.len() - 1]);
    println!("argmax agreement q4 head vs q8 head: {:.1}%", 100.0 * agree as f64 / n as f64);
    gaps.sort_by(f32::total_cmp);
    println!("q8 top-1 minus top-2 gap: p10 {:.3}, p50 {:.3}", p(&gaps, 0.1), p(&gaps, 0.5));
    for d in [0.5f32, 1.0, 1.5, 2.0] {
        let mut c: Vec<f32> = cands.iter().filter(|x| x.0 == d).map(|x| x.1 as f32).collect();
        c.sort_by(f32::total_cmp);
        println!("delta {d}: candidates p50 {}, p99 {}, max {}", p(&c, 0.5), p(&c, 0.99), c[c.len() - 1]);
    }
}
#[cfg(not(target_os = "macos"))]
fn main() {}
