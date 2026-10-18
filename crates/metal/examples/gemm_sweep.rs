//! Prefill / big-batch GEMM kernels: GFLOP/s of `matmul_mma_*` (32 x 64
//! tiles, f32 staging) vs `matmul_mma2_*` (64 x 64 tiles, half weight tile)
//! on GPT-2's matrix shapes at several row counts, plus max |diff| between
//! the two.
//!
//! `cargo run --release -p nanoinfer-metal --example gemm_sweep [-- dtype]`
#[cfg(target_os = "macos")]
fn main() {
    use nanoinfer_metal::gpu::{Arg, Gpu, Sync};
    use nanoinfer_metal::weights::{quantize_q4_0, quantize_q8_0};
    let filter = std::env::args().nth(1).unwrap_or_default();
    let gpu = Gpu::new().expect("metal");
    let shapes = [("qkv", 2304usize, 768usize), ("fc", 3072, 768), ("fc_proj", 768, 3072), ("lm_head", 50257, 768)];
    let mut seed = 0x2545f4914f6cdd1du64;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
            })
            .collect()
    };
    let x = gpu.buffer_from(&rnd(512 * 3072));
    let out_a = gpu.buffer(512 * 50257 * 4);
    let out_b = gpu.buffer(512 * 50257 * 4);
    for dt in ["q4_0", "q8_0", "f16"] {
        if !dt.contains(&filter) {
            continue;
        }
        println!("{dt}: GFLOP/s old (matmul_mma) / new (matmul_mma2), max |diff|");
        for &(name, d_out, d_in) in &shapes {
            let w = rnd(d_out * d_in);
            let wb = match dt {
                "q4_0" => gpu.buffer_from(&quantize_q4_0(&w)),
                "q8_0" => gpu.buffer_from(&quantize_q8_0(&w)),
                _ => gpu.buffer_from(&w.iter().map(|&v| half::f16::from_f32(v)).collect::<Vec<_>>()),
            };
            let mut line = format!("  {name:<8}");
            for n in [32usize, 64, 128, 256, 512] {
                let run = |kernel: &str, out: &nanoinfer_metal::gpu::Buf, bm: usize| -> f64 {
                    let args = [
                        Arg::Buf(out),
                        Arg::Buf(&x),
                        Arg::Buf(&wb),
                        Arg::Buf(&x),
                        Arg::U32(n as u32),
                        Arg::U32(d_in as u32),
                        Arg::U32(d_out as u32),
                        Arg::U32(0),
                    ];
                    let mut ts = vec![];
                    for _ in 0..5 {
                        let mut r = gpu.runner(Sync::PerForward);
                        for _ in 0..4 {
                            r.dispatch_tg("mm", kernel, &args, (d_out.div_ceil(64), n.div_ceil(bm), 1), (128, 1, 1), 0);
                        }
                        ts.push(r.flush() / 4.0);
                    }
                    ts.sort_by(f64::total_cmp);
                    2.0 * (n * d_in * d_out) as f64 / ts[2] / 1e9
                };
                let a = run(&format!("matmul_mma_{dt}"), &out_a, 32);
                let b = run(&format!("matmul_mma2_{dt}"), &out_b, 64);
                let (oa, ob) = (out_a.slice::<f32>(n * d_out), out_b.slice::<f32>(n * d_out));
                let d = oa.iter().zip(ob).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
                line += &format!("  n={n}: {a:>5.0}/{b:>5.0} ({d:.1e})");
            }
            println!("{line}");
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {}
