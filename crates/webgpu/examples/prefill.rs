//! Prefill timing breakdown: first vs repeated prefill in one session.
//! `cargo run --release -p nanoinfer-webgpu --example prefill -- webgpu-tiled 64`
use nanoinfer_core::Model;
use nanoinfer_webgpu::{gpu::block_on, Engine, Gpu, WgpuStage};
use std::rc::Rc;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let stage: WgpuStage = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(WgpuStage::Tiled);
    let p: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(64);
    let model = Model::from_safetensors(&std::fs::read("weights/gpt2/model.safetensors").unwrap()).unwrap();
    let gpu = Rc::new(block_on(Gpu::new()).unwrap());
    let e = Rc::new(Engine::new(gpu, &model, stage));
    let t0 = Instant::now();
    let now = move || t0.elapsed().as_secs_f64() * 1e3;
    let prompt: Vec<u32> = (0..p as u32).map(|i| 100 + i * 7).collect();
    let mut s = e.session(p + 8);
    let ops = block_on(s.profile_ops(&prompt, 5, &now)).unwrap();
    let mut by: Vec<(&str, f64)> = vec![];
    for (n, _, ms) in ops {
        match by.iter_mut().find(|x| x.0 == n) { Some(x) => x.1 += ms, None => by.push((n, ms)) }
    }
    by.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (n, ms) in by { println!("  {n:<12} {:>9.1} us", ms * 1e3); }
    for sess in 0..1 {
        let mut s = e.session(p + 8);
        for rep in 0..3 {
            s.reset();
            let t = Instant::now();
            block_on(s.step_greedy(&prompt, &now)).unwrap();
            println!("session {sess} prefill {rep}: {:.2} ms ({:.0} tok/s)", t.elapsed().as_secs_f64() * 1e3, p as f64 / t.elapsed().as_secs_f64());
        }
    }
}
