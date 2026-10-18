//! Matvec kernel sweep: achieved GB/s for every decode matvec variant on every
//! GPT-2 matrix shape, with the cache defeated (enough copies of the matrix
//! to cover >= 256 MB, dispatched round-robin, so each dispatch reads from DRAM
//! like the real forward pass does).
//!
//! `cargo run --release -p nanoinfer-metal --example matvec_sweep [-- kernel-substring]`
#[cfg(target_os = "macos")]
fn main() {
    use nanoinfer_metal::gpu::{Arg, Gpu, Sync};
    let filter = std::env::args().nth(1).unwrap_or_default();
    let gpu = Gpu::new().expect("metal");
    // (name, d_out, d_in)
    let shapes = [
        ("qkv", 2304usize, 768usize),
        ("attn_proj", 768, 768),
        ("fc", 3072, 768),
        ("fc_proj", 768, 3072),
        ("lm_head", 50257, 768),
    ];
    // (kernel, rows per SIMD-group, SIMD-groups per threadgroup, bytes per weight, block bytes)
    let mut variants: Vec<(String, usize, usize, f64)> = vec![("matmul_naive".into(), 0, 0, 4.0)];
    for r in [1, 2, 4, 8] {
        for nsg in [1, 2, 4, 8] {
            variants.push((format!("matvec_simd_r{r}"), r, nsg, 4.0));
        }
    }
    for r in [1, 2, 4, 8] {
        for nsg in [2, 4, 8] {
            variants.push((format!("matvec_f16_r{r}"), r, nsg, 2.0));
        }
    }
    for (k, r, nsg) in [("matvec_q8_0", 1, 4), ("matvec_q8_0", 2, 4), ("matvec_q8_0", 4, 4), ("matvec_q8_0", 2, 8)] {
        variants.push((format!("{k}_r{r}"), r, nsg, 34.0 / 32.0));
    }
    for (k, r, nsg) in [("matvec_q4_0", 1, 4), ("matvec_q4_0", 2, 4), ("matvec_q4_0", 4, 4), ("matvec_q4_0", 2, 8)] {
        variants.push((format!("{k}_r{r}"), r, nsg, 18.0 / 32.0));
    }
    let x = gpu.buffer(3072 * 4);
    let out = gpu.buffer(50304 * 4);
    let bias = gpu.buffer(50304 * 4);
    println!("{:<18} {:>3} {:>3} {}", "kernel", "R", "NSG", shapes.map(|s| format!("{:>10}", s.0)).join(""));
    for (kernel, r, nsg, bpw) in variants {
        if !kernel.contains(&filter) || !gpu.has(&kernel) {
            continue;
        }
        let mut row = format!("{kernel:<18} {r:>3} {nsg:>3} ");
        for &(_, d_out, d_in) in &shapes {
            let bytes = (d_out * d_in) as f64 * bpw;
            let copies = ((256e6 / bytes).ceil() as usize).max(2);
            let mats: Vec<_> = (0..copies).map(|_| gpu.buffer(bytes as usize + 64)).collect();
            let mut best = 0.0f64;
            for _ in 0..5 {
                let mut run = gpu.runner(Sync::PerForward);
                for m in &mats {
                    let args = [
                        Arg::Buf(&out),
                        Arg::Buf(&x),
                        Arg::Buf(m),
                        Arg::Buf(&bias),
                        Arg::U32(1),
                        Arg::U32(d_in as u32),
                        Arg::U32(d_out as u32),
                        Arg::U32(1),
                    ];
                    if r == 0 {
                        run.dispatch("mv", &kernel, &args, d_out, 256);
                    } else {
                        run.dispatch_tg("mv", &kernel, &args, (d_out.div_ceil(r * nsg), 1, 1), (32 * nsg, 1, 1), 0);
                    }
                }
                let t = run.flush();
                best = best.max(bytes * copies as f64 / t / 1e9);
            }
            row += &format!("{best:>10.1}");
        }
        println!("{row}");
    }
    println!("(GB/s, best of 5, each over >= 256 MB of distinct weight copies)");
}

#[cfg(not(target_os = "macos"))]
fn main() {}
