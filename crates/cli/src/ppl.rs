//! `nanoinfer ppl`: what did a speed-up cost in quality?
//!
//! Runs a stage over WikiText-2 (test split) in non-overlapping windows of
//! `--window` tokens and reports perplexity, i.e. exp(mean negative log-likelihood
//! of each next token). A reference stage (f32) runs in lockstep on the same
//! tokens, so we can also report, per position:
//!
//! - **KL(ref ‖ stage)**: how far the stage's next-token distribution drifted, in nats;
//! - **top-1 agreement**: how often both pick the same most likely token;
//! - **max |Δlogit|**.
//!
//! Tokens are fed one at a time through the decode path (that's the path the
//! quantised kernels optimise), so a window of 1024 costs 1024 decode steps.

use clap::Args;
use nanoinfer_core::{sampler, NoTrace, Session, Stage};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Args)]
pub struct PplArgs {
    #[arg(long, default_value = "kv-cache")]
    stage: Stage,
    /// f32 stage to compare against (KL, top-1 agreement).
    #[arg(long, default_value = "unrolled")]
    reference: Stage,
    /// Plain-text evaluation file (scripts/fetch_wikitext.sh).
    #[arg(long, default_value = "weights/wikitext-2-raw-test.txt")]
    text: PathBuf,
    /// Tokens per window (context is reset between windows).
    #[arg(long, default_value_t = 512)]
    window: usize,
    /// Number of windows to evaluate.
    #[arg(long, default_value_t = 4)]
    windows: usize,
}

fn log_softmax_at(logits: &[f32], target: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = max + logits.iter().map(|&l| (l as f64 - max).exp()).sum::<f64>().ln();
    logits[target] as f64 - lse
}

/// KL(p ‖ q) for p = softmax(a), q = softmax(b), in nats.
fn kl(a: &[f32], b: &[f32]) -> f64 {
    let lse = |x: &[f32]| {
        let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + x.iter().map(|&l| (l as f64 - m).exp()).sum::<f64>().ln()
    };
    let (la, lb) = (lse(a), lse(b));
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let lp = x as f64 - la;
            lp.exp() * (lp - (y as f64 - lb))
        })
        .sum()
}

pub fn run(weights: &Path, a: PplArgs) -> Result<(), Box<dyn std::error::Error>> {
    crate::bench::prefer_performance_cores();
    let tok = crate::load_tokenizer(weights)?;
    let model = crate::load_model(weights)?;
    let text = std::fs::read_to_string(&a.text)
        .map_err(|e| format!("{}: {e} (run scripts/fetch_wikitext.sh)", a.text.display()))?;
    let ids = tok.encode(&text);
    let window = a.window.min(model.config.n_ctx);
    let n_win = a.windows.min(ids.len() / window);
    eprintln!("ppl: stage={} vs reference={} | {n_win} windows × {window} tokens", a.stage, a.reference);

    let compare = a.stage != a.reference;
    let (mut nll, mut nll_ref, mut kl_sum, mut agree, mut max_diff, mut count) = (0.0, 0.0, 0.0, 0usize, 0f32, 0usize);
    let t0 = Instant::now();
    for w in 0..n_win {
        let chunk = &ids[w * window..(w + 1) * window];
        let mut s = Session::with_max_seq(&model, a.stage, window);
        let mut r = Session::with_max_seq(&model, a.reference, window);
        for i in 0..window - 1 {
            let target = chunk[i + 1] as usize;
            let logits = s.step(&model, &chunk[i..i + 1], &mut NoTrace)?.to_vec();
            nll -= log_softmax_at(&logits, target);
            if compare {
                let lr = r.step(&model, &chunk[i..i + 1], &mut NoTrace)?;
                nll_ref -= log_softmax_at(lr, target);
                kl_sum += kl(lr, &logits);
                agree += usize::from(sampler::argmax(lr) == sampler::argmax(&logits));
                max_diff = lr.iter().zip(&logits).map(|(x, y)| (x - y).abs()).fold(max_diff, f32::max);
            }
            count += 1;
        }
        eprintln!("  window {}: running ppl {:.3}", w + 1, (nll / count as f64).exp());
    }
    let ppl = (nll / count as f64).exp();
    println!("{}: ppl {ppl:.4} over {count} tokens ({:.1}s)", a.stage, t0.elapsed().as_secs_f64());
    if compare {
        let ppl_ref = (nll_ref / count as f64).exp();
        println!(
            "  vs {}: ppl {ppl_ref:.4} (Δppl {:+.4}, {:+.2}%) | mean KL {:.2e} nats | top-1 agreement {:.2}% | max |Δlogit| {:.3}",
            a.reference,
            ppl - ppl_ref,
            100.0 * (ppl / ppl_ref - 1.0),
            kl_sum / count as f64,
            100.0 * agree as f64 / count as f64,
            max_diff
        );
    }
    Ok(())
}
