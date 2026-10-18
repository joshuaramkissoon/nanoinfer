//! # nanoinfer-metal
//!
//! GPT-2 on the Apple GPU, hillclimbed one measured stage at a time, exactly
//! like the CPU engine: every [`MetalStage`] stays runnable, and each one is
//! a [`Recipe`] (which kernels, which dtype, how often the CPU waits) rather
//! than an edit of the previous stage's code.
//!
//! Reuses `nanoinfer-core` for everything that isn't GPU maths: the
//! safetensors loader ([`nanoinfer_core::Model`], already transposed to
//! `[out, in]`), the tokenizer and the config.
//!
//! ```ignore
//! let engine = MetalEngine::new(&model, MetalStage::Naive)?;
//! let mut s = engine.session();
//! let logits = s.step(&prompt)?;       // prefill
//! let next = s.step_greedy(&[tok])?;   // decode, argmax on CPU or GPU
//! ```
//!
//! macOS only: on other targets this crate is empty.
#![cfg(target_os = "macos")]

mod forward;
pub mod gpu;
mod weights;

pub use forward::{MetalEngine, MetalSession};
pub use gpu::{RunStats, Sync};

use std::fmt;
use std::str::FromStr;

/// Weight storage format on the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    /// Blocks of 32 int8 weights + one f16 scale (34 bytes / 32 weights).
    Q8_0,
    /// Blocks of 32 int4 weights + one f16 scale (18 bytes / 32 weights).
    Q4_0,
}

impl DType {
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::Q8_0 => "q8_0",
            DType::Q4_0 => "q4_0",
        }
    }
    /// Bytes per weight of a matrix in this format (block scales included).
    pub fn bytes_per_weight(self) -> f64 {
        match self {
            DType::F32 => 4.0,
            DType::F16 => 2.0,
            DType::Q8_0 => 34.0 / 32.0,
            DType::Q4_0 => 18.0 / 32.0,
        }
    }
}

/// Decode (n = 1) matrix-vector kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Matvec {
    /// One thread per output, serial dot product (uncoalesced).
    Naive,
    /// One SIMD-group per 4 rows, float4 loads, `simd_sum` reduction.
    Simd,
}

/// Prefill (n > 1) matrix-matrix kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Matmul {
    Naive,
    /// Run the decode matvec once per row (re-reads the weights n times).
    Matvec,
    /// 32x32 output tiles staged through threadgroup memory.
    Tiled,
    /// `simdgroup_float8x8` multiply-accumulate.
    SimdMatrix,
}

/// LayerNorm / attention style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Par {
    /// One thread per row / per (query, head).
    Naive,
    /// A threadgroup cooperates on each row / (query, head).
    Simd,
}

/// Everything that distinguishes one Metal stage from another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Recipe {
    pub sync: Sync,
    pub dtype: DType,
    pub matvec: Matvec,
    pub matmul: Matmul,
    pub norm: Par,
    pub attn: Par,
    /// Fused kernels (layernorm into matvec, bias+gelu, residual into the
    /// projection) instead of one dispatch per op.
    pub fused: bool,
    /// Argmax over the logits on the GPU (only 4 bytes read back).
    pub gpu_argmax: bool,
}

/// The Metal hillclimb, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetalStage {
    /// Literal port of the CPU kernels, one command buffer + wait per op.
    Naive,
    /// Same kernels, the whole forward in one command buffer.
    OneCb,
    /// + coalesced SIMD-group matvec for decode.
    Simd,
    /// + threadgroup-parallel layernorm and attention.
    Par,
}

impl MetalStage {
    pub const ALL: &'static [MetalStage] = &[MetalStage::Naive, MetalStage::OneCb, MetalStage::Simd, MetalStage::Par];

    pub fn name(self) -> &'static str {
        match self {
            MetalStage::Naive => "metal-naive",
            MetalStage::OneCb => "metal-onecb",
            MetalStage::Simd => "metal-simd",
            MetalStage::Par => "metal-par",
        }
    }

    pub fn recipe(self) -> Recipe {
        let naive = Recipe {
            sync: Sync::PerOp,
            dtype: DType::F32,
            matvec: Matvec::Naive,
            matmul: Matmul::Naive,
            norm: Par::Naive,
            attn: Par::Naive,
            fused: false,
            gpu_argmax: false,
        };
        match self {
            MetalStage::Naive => naive,
            MetalStage::OneCb => Recipe { sync: Sync::PerForward, ..naive },
            MetalStage::Simd => Recipe { sync: Sync::PerForward, matvec: Matvec::Simd, ..naive },
            MetalStage::Par => {
                Recipe { sync: Sync::PerForward, matvec: Matvec::Simd, norm: Par::Simd, attn: Par::Simd, ..naive }
            }
        }
    }

    /// Name of the kernel family, for the bench ledger's `kernels` field.
    pub fn kernels(self) -> String {
        let r = self.recipe();
        format!("metal/{:?}-matvec/{:?}-matmul/{}", r.matvec, r.matmul, r.dtype.name()).to_lowercase()
    }

    /// Bytes of weights read from memory per decoded token (the ceiling's denominator).
    pub fn decode_weight_bytes(self, cfg: &nanoinfer_core::Config) -> f64 {
        let c = cfg.n_embd;
        let matrices = cfg.n_layer * 12 * c * c + cfg.n_vocab * c; // 3C*C + C*C + 4C*C + 4C*C, + wte
        let small = cfg.n_params() - cfg.n_ctx * c - matrices; // biases + layernorms, kept f32
        matrices as f64 * self.recipe().dtype.bytes_per_weight() + small as f64 * 4.0
    }
}

impl fmt::Display for MetalStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for MetalStage {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        MetalStage::ALL.iter().copied().find(|st| st.name() == s).ok_or_else(|| format!("unknown metal stage {s}"))
    }
}
