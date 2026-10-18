//! `nanoinfer`: command-line driver for the engine.
//!
//! ```text
//! nanoinfer generate --stage kv-cache --prompt "Hello" --tokens 40
//! nanoinfer bench --stage naive --tokens 16
//! nanoinfer tokenize "Hello world"
//! ```
//!
//! The CLI owns everything the core crate refuses to touch: files (weights are
//! memory-mapped), the clock, and the process environment.

mod bench;
mod membw;
#[cfg(target_os = "macos")]
mod metal_bench;
mod ppl;
mod report;

use clap::{Parser, Subcommand};
use nanoinfer_core::{sampler::Sampler, Model, Stage, Tokenizer};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "nanoinfer", about = "A small LLM inference engine, optimised one measured step at a time")]
struct Cli {
    /// Directory containing model.safetensors, vocab.json and merges.txt.
    #[arg(long, global = true, default_value = "weights/gpt2")]
    weights: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate text from a prompt.
    Generate {
        #[arg(long, default_value = "kv-cache")]
        stage: Stage,
        #[arg(long, default_value = "The meaning of life is")]
        prompt: String,
        /// Number of tokens to generate.
        #[arg(long, default_value_t = 40)]
        tokens: usize,
        /// 0 = greedy.
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Print a per-op time breakdown afterwards.
        #[arg(long)]
        profile: bool,
        /// Write a JSONL event trace (the format the web visualisations read).
        #[arg(long)]
        trace_out: Option<PathBuf>,
        /// With --trace-out: capture this layer/head's attention map, e.g. "5,1".
        #[arg(long)]
        attention: Option<String>,
    },
    /// Measure prefill and decode throughput.
    Bench(bench::BenchArgs),
    /// Measure how fast the CPU can stream memory (sets a realistic decode ceiling).
    Membw(membw::MembwArgs),
    /// Perplexity on WikiText-2, plus KL / top-1 agreement against an f32 stage.
    Ppl(ppl::PplArgs),
    /// Show how text is split into tokens.
    Tokenize { text: String },
}

pub fn load_tokenizer(dir: &Path) -> Result<Tokenizer, Box<dyn std::error::Error>> {
    let vocab = std::fs::read(dir.join("vocab.json"))?;
    let merges = std::fs::read(dir.join("merges.txt"))?;
    Ok(Tokenizer::from_vocab_and_merges(&vocab, &merges)?)
}

pub fn load_model(dir: &Path) -> Result<Model, Box<dyn std::error::Error>> {
    let path = dir.join("model.safetensors");
    let file =
        std::fs::File::open(&path).map_err(|e| format!("{}: {e} (run scripts/fetch_weights.sh)", path.display()))?;
    // Safety: the file is not modified while mapped (it's our own weights dir).
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let t = Instant::now();
    let model = Model::from_safetensors(&mmap)?;
    eprintln!(
        "loaded {} ({:.1}M params) in {:.2}s",
        path.display(),
        model.config.n_params() as f64 / 1e6,
        t.elapsed().as_secs_f64()
    );
    Ok(model)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `bench --stage metal-*` runs on the GPU (crates/metal).
    #[cfg(target_os = "macos")]
    if let Some(result) = metal_bench::intercept() {
        return result;
    }
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Tokenize { text } => {
            let tok = load_tokenizer(&cli.weights)?;
            let ids = tok.encode(&text);
            for &id in &ids {
                println!("{id:>6}  {:?}", String::from_utf8_lossy(tok.token_bytes(id)));
            }
            println!("{} tokens", ids.len());
        }
        Cmd::Bench(args) => bench::run(&cli.weights, args)?,
        Cmd::Membw(args) => membw::run(args)?,
        Cmd::Ppl(args) => ppl::run(&cli.weights, args)?,
        Cmd::Generate { stage, prompt, tokens, temperature, top_k, seed, profile, trace_out, attention } => {
            let tok = load_tokenizer(&cli.weights)?;
            let model = load_model(&cli.weights)?;
            let mut sampler = Sampler::new(temperature, top_k, seed);
            let attention = match attention {
                Some(s) => {
                    let (l, h) = s.split_once(',').ok_or("--attention expects LAYER,HEAD")?;
                    Some((l.trim().parse()?, h.trim().parse()?))
                }
                None => None,
            };
            generate(&model, &tok, stage, &prompt, tokens, &mut sampler, profile, trace_out, attention)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn generate(
    model: &Model,
    tok: &Tokenizer,
    stage: Stage,
    prompt: &str,
    n: usize,
    sampler: &mut Sampler,
    profile: bool,
    trace_out: Option<PathBuf>,
    attention: Option<(usize, usize)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let prompt_ids = tok.encode(prompt);
    let mut session = nanoinfer_core::Session::new(model, stage);
    let start = Instant::now();
    let clock = move || start.elapsed().as_secs_f64();
    let mut rec = nanoinfer_core::trace::Recorder::new(clock);
    rec.top_k = if trace_out.is_some() { 10 } else { 0 };
    rec.attention_head = attention;
    let mut jsonl = trace_out.as_ref().map(|_| report::TraceWriter::new(model.config, stage, &prompt_ids));

    print!("{prompt}");
    let mut pending: Vec<u8> = Vec::new(); // bytes of a partial UTF-8 char
    let mut input = prompt_ids.clone();
    let t0 = Instant::now();
    let mut t_first = 0.0;
    for i in 0..n {
        let logits = session.step(model, &input, &mut rec)?;
        let next = sampler.sample(logits);
        if i == 0 {
            t_first = t0.elapsed().as_secs_f64();
        }
        if let Some(w) = jsonl.as_mut() {
            w.forward(&rec, tok, next);
        }
        pending.extend_from_slice(tok.token_bytes(next));
        if let Ok(s) = std::str::from_utf8(&pending) {
            print!("{s}");
            pending.clear();
            std::io::stdout().flush()?;
        }
        input = vec![next];
        if session.len() >= session.max_seq() {
            break;
        }
    }
    let total = t0.elapsed().as_secs_f64();
    println!();
    eprintln!(
        "[{stage}] prompt {} tok, prefill {:.0} ms ({:.1} tok/s), decode {:.2} tok/s",
        prompt_ids.len(),
        t_first * 1e3,
        prompt_ids.len() as f64 / t_first,
        (n.saturating_sub(1)) as f64 / (total - t_first)
    );
    if profile {
        report::print_profile(&rec);
    }
    if let (Some(path), Some(w)) = (trace_out, jsonl) {
        w.finish(&rec, &path)?;
        eprintln!("wrote trace {}", path.display());
    }
    Ok(())
}
