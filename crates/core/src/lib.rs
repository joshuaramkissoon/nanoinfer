//! # nanoinfer-core
//!
//! A small, readable LLM inference engine for GPT-2, written to be *optimised in
//! public*: every speed-up is a separate, selectable [`Stage`] so you can compare
//! them side by side (in the CLI, or live in a browser via WebAssembly).
//!
//! The core crate deliberately has **no filesystem, threads or clocks**. Weights and
//! tokenizer files arrive as `&[u8]`, and timing comes from a [`trace::Clock`] the
//! caller injects. That is what lets the exact same code compile to
//! `wasm32-unknown-unknown`.
//!
//! Map of the crate:
//!
//! - [`safetensors`] / [`weights`]: parse the checkpoint into plain `Vec<f32>`s.
//! - [`tokenizer`]: GPT-2 byte-level BPE, from scratch.
//! - [`kernels`]: the math (matmul, layernorm, gelu, softmax, attention). Later
//!   stages swap these for faster versions.
//! - [`stages`]: the forward passes. `naive` recomputes everything every token,
//!   `kv_cache` remembers keys and values.
//! - [`trace`]: a zero-cost-when-off hook the forward pass reports to (op timings,
//!   FLOPs, bytes, KV length, logits, attention maps). Drives the visualisations.
//! - [`sampler`]: turning logits into the next token.

pub mod config;
pub mod error;
pub mod kernels;
pub mod safetensors;
pub mod sampler;
pub mod stages;
pub mod tokenizer;
pub mod trace;
pub mod weights;

pub use config::Config;
pub use error::{Error, Result};
pub use sampler::Sampler;
pub use stages::{Session, Stage};
pub use tokenizer::Tokenizer;
pub use trace::{NoTrace, Trace};
pub use weights::Model;
