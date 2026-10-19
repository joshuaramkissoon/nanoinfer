//! `nanoinfer accel`: what does Apple's own BLAS get on the same forward pass?
//!
//! A reference point, not a stage: Accelerate is macOS-only (so it can't be in
//! core, which must build for wasm). It routes sgemm/sgemv to the M4's matrix
//! coprocessor (AMX / SME) and its own thread pool. We plug it in as a CLI-side
//! [`Kernels`] set whose `matmul` calls `cblas_sgemv` (decode, n = 1) or
//! `cblas_sgemm` (prefill), and run core's ordinary KV-cache forward with it.
//! Everything else (layernorm, attention, GELU) is the reference kernels.
//!
//! `VECLIB_MAXIMUM_THREADS=1` limits Accelerate to one thread.

use crate::bench::{begin, machine, median, stamp, stddev, timestamp, BENCH_TEXT};
use clap::Args;
use nanoinfer_core::kernels::Kernels;
use nanoinfer_core::stages::{kv_cache, Activations, KvCache, WeightFormat};
use nanoinfer_core::{sampler, NoTrace};
use serde_json::json;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Args)]
pub struct AccelArgs {
    #[arg(long, default_value_t = 64)]
    tokens: usize,
    #[arg(long, default_value_t = 64)]
    prompt_len: usize,
    #[arg(long, default_value_t = 5)]
    runs: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 120.0)]
    bandwidth_gbs: f64,
    #[arg(long, default_value = "bench/results.jsonl")]
    out: PathBuf,
    #[arg(long)]
    no_record: bool,
    /// Record even if the engine sources have uncommitted changes (marked dirty).
    #[arg(long)]
    allow_dirty: bool,
    #[arg(long, default_value = "")]
    note: String,
}

#[cfg(target_os = "macos")]
mod ffi {
    pub const ROW_MAJOR: i32 = 101;
    pub const NO_TRANS: i32 = 111;
    pub const TRANS: i32 = 112;
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_sgemm(
            order: i32,
            ta: i32,
            tb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_sgemv(
            order: i32,
            ta: i32,
            m: i32,
            n: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            x: *const f32,
            incx: i32,
            beta: f32,
            y: *mut f32,
            incy: i32,
        );
    }
}

/// Matmuls through Accelerate; every other op from the reference set.
pub struct Accelerate;

impl Kernels for Accelerate {
    const NAME: &'static str = "accelerate";

    #[cfg(target_os = "macos")]
    fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
        assert!(x.len() >= n * d_in && out.len() >= n * d_out && w.len() == d_in * d_out);
        // Pre-fill the bias and accumulate onto it (beta = 1).
        let beta = match bias {
            Some(b) => {
                for row in out.chunks_exact_mut(d_out).take(n) {
                    row.copy_from_slice(b);
                }
                1.0
            }
            None => 0.0,
        };
        let (n_, di, dout) = (n as i32, d_in as i32, d_out as i32);
        // Safety: dimensions and strides match the asserted slice lengths.
        unsafe {
            if n == 1 {
                // y[d_out] = W[d_out, d_in] · x
                ffi::cblas_sgemv(
                    ffi::ROW_MAJOR,
                    ffi::NO_TRANS,
                    dout,
                    di,
                    1.0,
                    w.as_ptr(),
                    di,
                    x.as_ptr(),
                    1,
                    beta,
                    out.as_mut_ptr(),
                    1,
                );
            } else {
                // out[n, d_out] = x[n, d_in] · W[d_out, d_in]ᵀ
                ffi::cblas_sgemm(
                    ffi::ROW_MAJOR,
                    ffi::NO_TRANS,
                    ffi::TRANS,
                    n_,
                    dout,
                    di,
                    1.0,
                    x.as_ptr(),
                    di,
                    w.as_ptr(),
                    di,
                    beta,
                    out.as_mut_ptr(),
                    dout,
                );
            }
        }
    }
}

pub fn run(weights: &Path, a: AccelArgs) -> Result<(), Box<dyn std::error::Error>> {
    begin(!a.no_record, a.allow_dirty)?;
    let tok = crate::load_tokenizer(weights)?;
    let model = crate::load_model(weights)?;
    let cfg = model.config;
    let mut prompt = Vec::new();
    while prompt.len() < a.prompt_len {
        prompt.extend(tok.encode(BENCH_TEXT));
    }
    prompt.truncate(a.prompt_len);
    let weight_bytes = WeightFormat::F32.decode_bytes(&cfg);
    let ceiling = a.bandwidth_gbs * 1e9 / weight_bytes;
    let veclib_threads = std::env::var("VECLIB_MAXIMUM_THREADS").unwrap_or_else(|_| "default".into());
    eprintln!("accel: Accelerate cblas_sgemv/sgemm, VECLIB_MAXIMUM_THREADS={veclib_threads}");

    let (mut pf, mut dc) = (Vec::new(), Vec::new());
    for run in 0..a.warmup + a.runs {
        let mut kv = KvCache::new(&model, a.prompt_len + a.tokens);
        let mut acts = Activations::default();
        let t = Instant::now();
        kv_cache::forward::<Accelerate, _>(&model, &prompt, &mut kv, &mut acts, &mut NoTrace);
        let mut next = sampler::argmax(&acts.logits);
        let tp = t.elapsed().as_secs_f64();
        let t = Instant::now();
        for _ in 0..a.tokens {
            kv_cache::forward::<Accelerate, _>(&model, &[next], &mut kv, &mut acts, &mut NoTrace);
            next = sampler::argmax(&acts.logits);
        }
        let td = t.elapsed().as_secs_f64();
        let (p, d) = (a.prompt_len as f64 / tp, a.tokens as f64 / td);
        eprintln!(
            "  {:<6} prefill {p:>9.2} tok/s   decode {d:>8.3} tok/s",
            if run < a.warmup { "warmup" } else { "run" }
        );
        if run >= a.warmup {
            pf.push(p);
            dc.push(d);
        }
    }
    let (pm, ps, dm, ds) = (median(&pf), stddev(&pf), median(&dc), stddev(&dc));
    let decode_gbs = dm * weight_bytes / 1e9;
    println!(
        "accelerate: prefill {pm:.2} ± {ps:.2} tok/s | decode {dm:.3} ± {ds:.3} tok/s ({decode_gbs:.1} GB/s) | {:.2}% of {ceiling:.0} tok/s ceiling",
        100.0 * dm / ceiling
    );
    if !a.no_record {
        let mut line = json!({
            "timestamp": timestamp(), "stage": "accelerate", "kernels": "accelerate (cli only)",
            "model": "gpt2-124m", "dtype": "f32", "threads": veclib_threads, "prompt_len": a.prompt_len,
            "gen_tokens": a.tokens, "runs": a.runs, "prefill_tok_s": pm, "prefill_tok_s_std": ps,
            "decode_tok_s": dm, "decode_tok_s_std": ds, "decode_ceiling_tok_s": ceiling, "decode_gbs": decode_gbs,
            "pct_of_ceiling": 100.0 * dm / ceiling, "bandwidth_gbs": a.bandwidth_gbs, "machine": machine(),
            "traced": false, "note": a.note,
        });
        stamp(&mut line, "main thread interactive, vecLib threads default");
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&a.out)?;
        writeln!(f, "{line}")?;
    }
    Ok(())
}
