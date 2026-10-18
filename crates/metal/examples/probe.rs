//! Microbenchmarks that set the rules of the GPU game on this machine:
//!
//! 1. **Dispatch overhead.** What does one empty kernel cost?
//!    a. N empty dispatches in one command buffer (encode cost + GPU per-dispatch gap)
//!    b. N command buffers of one dispatch each, commit + wait each (the round trip)
//! 2. **Bandwidth.** How many GB/s can a kernel actually read (and copy)? This
//!    is the real denominator of the GPU decode ceiling, not the 120 GB/s spec.
//!
//! `cargo run --release -p nanoinfer-metal --example probe`
#[cfg(target_os = "macos")]
fn main() {
    use nanoinfer_metal::gpu::{Arg, Gpu, Sync};
    use std::time::Instant;

    let gpu = Gpu::new().expect("metal");
    println!("device: {}", gpu.name());
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };

    // Warm up the pipeline / GPU clocks.
    for _ in 0..3 {
        let mut r = gpu.runner(Sync::PerForward);
        for _ in 0..1000 {
            r.dispatch("e", "empty_kernel", &[], 32, 32);
        }
        r.flush();
    }

    println!("\n1a. N empty dispatches in ONE command buffer (32 threads each)");
    println!("{:>6} {:>12} {:>12} {:>12} {:>14}", "N", "encode us", "wall us", "GPU us", "GPU us/disp");
    for &n in &[1usize, 10, 100, 135, 400, 1000, 10000] {
        let mut enc = vec![];
        let mut wall = vec![];
        let mut gpu_t = vec![];
        for _ in 0..15 {
            let t0 = Instant::now();
            let mut r = gpu.runner(Sync::PerForward);
            for _ in 0..n {
                r.dispatch("e", "empty_kernel", &[], 32, 32);
            }
            let t1 = Instant::now();
            let g = r.flush();
            enc.push((t1 - t0).as_secs_f64() * 1e6);
            wall.push(t0.elapsed().as_secs_f64() * 1e6);
            gpu_t.push(g * 1e6);
        }
        let g = median(gpu_t);
        println!("{n:>6} {:>12.1} {:>12.1} {:>12.1} {:>14.3}", median(enc), median(wall), g, g / n as f64);
    }

    println!("\n1b. N command buffers, ONE empty dispatch each, commit + wait each");
    println!("{:>6} {:>14} {:>16}", "N", "wall us total", "wall us per cb");
    for &n in &[10usize, 135, 1000] {
        let mut wall = vec![];
        for _ in 0..7 {
            let t0 = Instant::now();
            let mut r = gpu.runner(Sync::PerOp);
            for _ in 0..n {
                r.dispatch("e", "empty_kernel", &[], 32, 32);
            }
            wall.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        let w = median(wall);
        println!("{n:>6} {w:>14.1} {:>16.2}", w / n as f64);
    }

    println!("\n1c. one dispatch writing a buffer, per-dispatch cost vs buffers bound");
    let bufs: Vec<_> = (0..8).map(|_| gpu.buffer(4096)).collect();
    for &nb in &[0usize, 2, 4, 8] {
        let mut gpu_t = vec![];
        let mut enc = vec![];
        for _ in 0..15 {
            let t0 = Instant::now();
            let mut r = gpu.runner(Sync::PerForward);
            for _ in 0..1000 {
                let args: Vec<Arg> = bufs[..nb].iter().map(Arg::Buf).collect();
                r.dispatch("e", "empty_kernel", &args, 32, 32);
            }
            enc.push(t0.elapsed().as_secs_f64() * 1e6 / 1000.0);
            gpu_t.push(r.flush() * 1e6 / 1000.0);
        }
        println!("  {nb} buffers: encode {:.3} us/dispatch, GPU {:.3} us/dispatch", median(enc), median(gpu_t));
    }

    println!("\n2. Bandwidth (shared-storage buffers, float4 grid-stride loops)");
    for &mb in &[32usize, 512] {
        let bytes = mb << 20;
        let src = gpu.buffer(bytes);
        let dst = gpu.buffer(bytes);
        let len4 = (bytes / 16) as u32;
        for (kernel, factor) in [("read_bw", 1.0), ("read_bw_x4", 1.0), ("copy_bw", 2.0)] {
            for &threads in &[16384usize, 65536, 262144] {
                let sink = gpu.buffer(threads * 4);
                let out = if kernel.starts_with("read") { &sink } else { &dst };
                let mut best = 0.0f64;
                let mut gbs = vec![];
                for _ in 0..10 {
                    let mut r = gpu.runner(Sync::PerForward);
                    r.dispatch("bw", kernel, &[Arg::Buf(&src), Arg::Buf(out), Arg::U32(len4)], threads, 256);
                    let t = r.flush();
                    let g = factor * bytes as f64 / t / 1e9;
                    best = best.max(g);
                    gbs.push(g);
                }
                println!(
                    "  {kernel} {mb:>4} MB, {threads:>6} threads: median {:>6.1} GB/s, best {best:>6.1}",
                    median(gbs)
                );
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Metal probe only runs on macOS");
}
