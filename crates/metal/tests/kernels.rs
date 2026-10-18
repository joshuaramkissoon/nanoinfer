//! Kernel-level checks that don't need the GPT-2 weights: every matvec and
//! matmul kernel, in every dtype, against a CPU dot product over the *same*
//! (dequantised) weights. Catches layout bugs separately from quantisation error.
#![cfg(target_os = "macos")]

use nanoinfer_metal::gpu::{Arg, Gpu, Sync};
use nanoinfer_metal::weights::{dequantize, quantize_q4_0, quantize_q8_0};
use nanoinfer_metal::DType;

fn rand_vec(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            (*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

#[test]
fn matvec_and_matmul_kernels_match_cpu() {
    let gpu = Gpu::new().unwrap();
    let mut seed = 0x9e3779b97f4a7c15;
    for &(rows, cols) in &[(96usize, 768usize), (70, 3072)] {
        let w = rand_vec(rows * cols, &mut seed);
        let bias = rand_vec(rows, &mut seed);
        for &n in &[1usize, 37] {
            let x = rand_vec(n * cols, &mut seed);
            let xb = gpu.buffer_from(&x);
            let bb = gpu.buffer_from(&bias);
            let out = gpu.buffer(n * rows * 4);
            let cases: Vec<(&str, DType, Vec<f32>, nanoinfer_metal::gpu::Buf)> = vec![
                ("f32", DType::F32, w.clone(), gpu.buffer_from(&w)),
                ("f16", DType::F16, w.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect(), {
                    gpu.buffer_from(&w.iter().map(|&v| half::f16::from_f32(v)).collect::<Vec<_>>())
                }),
                ("q8_0", DType::Q8_0, dequantize(&quantize_q8_0(&w), w.len(), DType::Q8_0), {
                    gpu.buffer_from(&quantize_q8_0(&w))
                }),
                ("q4_0", DType::Q4_0, dequantize(&quantize_q4_0(&w), w.len(), DType::Q4_0), {
                    gpu.buffer_from(&quantize_q4_0(&w))
                }),
            ];
            for (name, dtype, wd, wb) in &cases {
                let expect: Vec<f32> = (0..n * rows)
                    .map(|i| {
                        let (t, o) = (i / rows, i % rows);
                        bias[o] + (0..cols).map(|k| x[t * cols + k] * wd[o * cols + k]).sum::<f32>()
                    })
                    .collect();
                let mut kernels = vec![];
                if n == 1 {
                    kernels.push(match dtype {
                        DType::F32 => ("matvec_simd_r4", 4, 4),
                        DType::F16 => ("matvec_f16_r4", 4, 4),
                        DType::Q8_0 => ("matvec_q8_0_r2", 2, 4),
                        DType::Q4_0 => ("matvec_q4_0_r2", 2, 4),
                    });
                    kernels.push((&*format!("matvec_{}_r1", if *name == "f32" { "simd" } else { name }).leak(), 1, 2));
                } else {
                    kernels.push((
                        if *name == "f32" { "matmul_mma" } else { &*format!("matmul_mma_{name}").leak() },
                        0,
                        0,
                    ));
                }
                for (k, r, nsg) in kernels {
                    let args = [
                        Arg::Buf(&out),
                        Arg::Buf(&xb),
                        Arg::Buf(wb),
                        Arg::Buf(&bb),
                        Arg::U32(n as u32),
                        Arg::U32(cols as u32),
                        Arg::U32(rows as u32),
                        Arg::U32(1),
                    ];
                    let mut run = gpu.runner(Sync::PerForward);
                    if r == 0 {
                        run.dispatch_tg("mm", k, &args, (rows.div_ceil(64), n.div_ceil(32), 1), (128, 1, 1), 0);
                    } else {
                        run.dispatch_tg("mv", k, &args, (rows.div_ceil(r * nsg), n, 1), (32 * nsg, 1, 1), 0);
                    }
                    run.flush();
                    let got = out.slice::<f32>(n * rows);
                    let d = got.iter().zip(&expect).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                    assert!(d < 1e-3, "{k} ({rows}x{cols}, n={n}): max diff {d}");
                }
            }
        }
    }
}
