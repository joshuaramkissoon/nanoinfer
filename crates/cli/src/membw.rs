//! `nanoinfer membw`: how many GB/s can the CPU actually *read*?
//!
//! The decode ceiling is bandwidth ÷ bytes per token, and Apple's 120 GB/s is a
//! spec figure for the whole SoC (CPU + GPU + everything). This measures what N
//! CPU threads can stream from DRAM: each thread sums its own contiguous slice
//! of a buffer much larger than every cache, so every byte comes from memory.
//!
//! The sum uses 8 independent u64 lanes so it vectorises without fast-math
//! (integer adds are associative) and the loop is limited by loads, not adds.

use clap::Args;
use std::sync::Barrier;
use std::time::Instant;

#[derive(Args)]
pub struct MembwArgs {
    /// Comma-separated thread counts to test.
    #[arg(long, default_value = "1,2,4,6,8,10")]
    threads: String,
    /// Buffer size in MB (default ≈ GPT-2 f32 weights; far larger than the 16 MB L2 / ~8 MB SLC).
    #[arg(long, default_value_t = 512)]
    mb: usize,
    /// Timed passes over the buffer per thread count.
    #[arg(long, default_value_t = 10)]
    passes: usize,
}

#[inline(never)]
fn sum_slice(s: &[u64]) -> u64 {
    let mut acc = [0u64; 8];
    let (chunks, rest) = s.as_chunks::<8>();
    for c in chunks {
        for j in 0..8 {
            acc[j] = acc[j].wrapping_add(c[j]);
        }
    }
    acc.iter().chain(rest).fold(0u64, |a, &b| a.wrapping_add(b))
}

/// Seconds for `threads` threads to each read their 1/threads of `buf` once:
/// from the first thread starting to the last one finishing. Each worker takes
/// its own timestamps, because with every core busy the coordinating thread can
/// be descheduled after the barrier and would start its clock late.
fn timed_pass(buf: &[u64], threads: usize) -> f64 {
    let chunk = buf.len().div_ceil(threads);
    let barrier = Barrier::new(threads);
    let spans: Vec<(Instant, Instant)> = std::thread::scope(|s| {
        let handles: Vec<_> = buf
            .chunks(chunk)
            .map(|part| {
                let barrier = &barrier;
                s.spawn(move || {
                    crate::bench::prefer_performance_cores();
                    barrier.wait();
                    let t0 = Instant::now();
                    std::hint::black_box(sum_slice(part));
                    (t0, Instant::now())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let start = spans.iter().map(|s| s.0).min().unwrap();
    let end = spans.iter().map(|s| s.1).max().unwrap();
    (end - start).as_secs_f64()
}

pub fn run(a: MembwArgs) -> Result<(), Box<dyn std::error::Error>> {
    crate::bench::prefer_performance_cores();
    let n = a.mb * (1 << 20) / 8;
    // Touch every page so we measure DRAM, not page faults.
    let buf: Vec<u64> = (0..n as u64).collect();
    let bytes = (n * 8) as f64;
    println!("membw: {} MB buffer, {} passes per thread count (read-only, u64 sum)", a.mb, a.passes);
    println!("{:>7} {:>10} {:>10} {:>10}", "threads", "best GB/s", "median", "min");
    for t in a.threads.split(',') {
        let t: usize = t.trim().parse()?;
        timed_pass(&buf, t); // warm
        let mut gbs: Vec<f64> = (0..a.passes).map(|_| bytes / timed_pass(&buf, t) / 1e9).collect();
        gbs.sort_by(f64::total_cmp);
        println!("{t:>7} {:>10.1} {:>10.1} {:>10.1}", gbs[gbs.len() - 1], gbs[gbs.len() / 2], gbs[0]);
    }
    Ok(())
}
