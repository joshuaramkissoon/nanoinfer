//! Decode matvec layouts with epilogues (what the pipelined stages run):
//! `matvec_epi` (one SIMD-group per row, rows walked one after another) vs
//! `matvec_lanes` (8 lanes per row, 4 rows per SIMD-group at once), on every
//! GPT-2 matrix shape, cache defeated as in `matvec_sweep`.
//!
//! `cargo run --release -p nanoinfer-metal --example lanes_sweep [-- dtype]`
#[cfg(target_os = "macos")]
fn main() {
    use nanoinfer_metal::gpu::{Arg, Gpu, Sync};
    let filter = std::env::args().nth(1).unwrap_or_default();
    let gpu = Gpu::new().expect("metal");
    let shapes = [
        ("qkv", 2304usize, 768usize),
        ("attn_proj", 768, 768),
        ("fc", 3072, 768),
        ("fc_proj", 768, 3072),
        ("lm_head", 50257, 768),
    ];
    // (dtype, bytes per weight)
    let dtypes = [("q4_0", 18.0 / 32.0), ("q8_0", 34.0 / 32.0), ("f16", 2.0)];
    let x = gpu.buffer(3072 * 4);
    let out = gpu.buffer(50304 * 4);
    let bias = gpu.buffer(50304 * 4);
    println!("{:<18} {:>3} {}", "kernel", "NSG", shapes.map(|s| format!("{:>10}", s.0)).join(""));
    for (dt, bpw) in dtypes {
        if !dt.contains(&filter) {
            continue;
        }
        // (kernel, rows per threadgroup, threads per threadgroup)
        let epi_rows = if dt == "f16" { 4 } else { 2 };
        let mut variants = vec![(format!("mve_{dt}"), epi_rows * 4, 128)];
        for r in [1, 2] {
            for nsg in [2, 4, 8] {
                variants.push((format!("mvl_{dt}_r{r}"), r * 4 * nsg, 32 * nsg));
            }
        }
        for (kernel, rows_tg, threads) in variants {
            let mut row = format!("{kernel:<18} {:>3} ", threads / 32);
            for &(_, d_out, d_in) in &shapes {
                let bytes = (d_out * d_in) as f64 * bpw;
                let copies = ((256e6 / bytes).ceil() as usize).max(2);
                let mats: Vec<_> = (0..copies).map(|_| gpu.buffer(bytes as usize + 64)).collect();
                let mut ts = vec![];
                for _ in 0..7 {
                    let mut run = gpu.runner(Sync::PerForward);
                    for m in &mats {
                        let args = [
                            Arg::Buf(&out),
                            Arg::Buf(&x),
                            Arg::Buf(m),
                            Arg::Buf(&bias),
                            Arg::Buf(&bias),
                            Arg::Buf(&bias),
                            Arg::Buf(&out),
                            Arg::Buf(&out),
                            Arg::U32(d_in as u32),
                            Arg::U32(d_out as u32),
                            Arg::U32(1),
                            Arg::U32(0),
                        ];
                        run.dispatch_tg("mv", &kernel, &args, (d_out.div_ceil(rows_tg), 1, 1), (threads, 1, 1), 0);
                    }
                    ts.push(bytes * copies as f64 / run.flush() / 1e9);
                }
                ts.sort_by(f64::total_cmp);
                row += &format!("{:>10.1}", ts[ts.len() / 2]);
            }
            println!("{row}");
        }
    }
    println!("(GB/s, median of 7, each over >= 256 MB of distinct weight copies)");
}

#[cfg(not(target_os = "macos"))]
fn main() {}
