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

pub mod batch;
mod forward;
pub mod gpu;
pub mod weights;

pub use batch::{Batch, BatchEngine, BatchOpts};
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

/// Attention kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attn {
    /// One thread per (query, head).
    Naive,
    /// One 256-thread threadgroup per (query, head).
    Simd,
    /// Decode: keys split into 64-key chunks across threadgroups, then merged
    /// (flash-decoding). Prefill: as `Simd`.
    SplitK,
    /// `SplitK` for decode + a simdgroup-matrix flash-attention kernel for prefill.
    Flash,
}

/// How much of a decode step is fused into the matvecs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fusion {
    /// One dispatch per op (11 per layer).
    None,
    /// LayerNorm in every matvec's prologue + epilogues: 5 per layer
    /// (slower: each threadgroup re-stages and re-normalises the input row).
    Full,
    /// Epilogues only (bias + GELU, residual add, KV-cache write); LayerNorm
    /// stays its own small kernel: 7 per layer.
    Epilogue,
}

/// Everything that distinguishes one Metal stage from another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Recipe {
    pub sync: Sync,
    pub dtype: DType,
    /// dtype of the tied LM head (wte), which small models are most sensitive to.
    pub head_dtype: DType,
    pub matvec: Matvec,
    pub matmul: Matmul,
    pub norm: Par,
    pub attn: Attn,
    /// Decode-time kernel fusion.
    pub fused: Fusion,
    /// Argmax over the logits on the GPU (only 4 bytes read back).
    pub gpu_argmax: bool,
    /// K/V cache stored as f16 (halves the bytes attention reads; needs
    /// `Attn::Flash` and `Fusion::Epilogue`).
    pub kv_f16: bool,
    /// Greedy decode commits every token's command buffer up front; each
    /// forward reads the token the previous one's argmax wrote on the GPU.
    pub pipelined: bool,
    /// Decode matvecs give each row 8 lanes (4 rows per SIMD-group at once)
    /// instead of a whole SIMD-group (needs `Fusion::Epilogue`).
    pub lanes: bool,
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
    /// + shared-memory tiled matmul for prefill.
    Tiled,
    /// + simdgroup_matrix (8x8 hardware MMA) matmul for prefill.
    Mma,
    /// metal-mma with f16 weights (f32 activations and accumulation).
    F16,
    /// metal-mma with Q8_0 weights, dequantised in registers.
    Q8,
    /// metal-mma with Q4_0 weights, dequantised in registers.
    Q4,
    /// metal-q4 + argmax on the GPU.
    Argmax,
    /// + LayerNorm fused into the matvec prologues and epilogue fusion (dead end, kept runnable).
    FusedLn,
    /// + epilogue fusion: bias+GELU, residual add, KV write inside the matvecs.
    Fused,
    /// + the CPU stops waiting between tokens (GPU-resident greedy loop).
    Pipelined,
    /// metal-pipelined with f16 weights (the fastest lossless-ish stage).
    PipelinedF16,
    /// metal-pipelined with Q8_0 weights everywhere.
    PipelinedQ8,
    /// metal-pipelined + split-K (flash-decoding) attention.
    FlashDec,
    /// + flash-attention prefill (simdgroup matrices, online softmax).
    Flash,
    /// + f16 KV cache.
    Kv16,
    /// + 8 lanes per row in the decode matvecs (more loads in flight on narrow rows).
    Lanes,
}

impl MetalStage {
    pub const ALL: &'static [MetalStage] = &[
        MetalStage::Naive,
        MetalStage::OneCb,
        MetalStage::Simd,
        MetalStage::Par,
        MetalStage::Tiled,
        MetalStage::Mma,
        MetalStage::F16,
        MetalStage::Q8,
        MetalStage::Q4,
        MetalStage::Argmax,
        MetalStage::FusedLn,
        MetalStage::Fused,
        MetalStage::Pipelined,
        MetalStage::PipelinedF16,
        MetalStage::PipelinedQ8,
        MetalStage::FlashDec,
        MetalStage::Flash,
        MetalStage::Kv16,
        MetalStage::Lanes,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MetalStage::Naive => "metal-naive",
            MetalStage::OneCb => "metal-onecb",
            MetalStage::Simd => "metal-simd",
            MetalStage::Par => "metal-par",
            MetalStage::Tiled => "metal-tiled",
            MetalStage::Mma => "metal-mma",
            MetalStage::F16 => "metal-f16",
            MetalStage::Q8 => "metal-q8",
            MetalStage::Q4 => "metal-q4",
            MetalStage::Argmax => "metal-argmax",
            MetalStage::FusedLn => "metal-fused-ln",
            MetalStage::Fused => "metal-fused",
            MetalStage::Pipelined => "metal-pipelined",
            MetalStage::PipelinedF16 => "metal-pipelined-f16",
            MetalStage::PipelinedQ8 => "metal-pipelined-q8",
            MetalStage::FlashDec => "metal-flashdec",
            MetalStage::Flash => "metal-flash",
            MetalStage::Kv16 => "metal-kv16",
            MetalStage::Lanes => "metal-lanes",
        }
    }

    pub fn recipe(self) -> Recipe {
        let naive = Recipe {
            sync: Sync::PerOp,
            dtype: DType::F32,
            head_dtype: DType::F32,
            matvec: Matvec::Naive,
            matmul: Matmul::Naive,
            norm: Par::Naive,
            attn: Attn::Naive,
            fused: Fusion::None,
            gpu_argmax: false,
            pipelined: false,
            kv_f16: false,
            lanes: false,
        };
        let par = Recipe { sync: Sync::PerForward, matvec: Matvec::Simd, norm: Par::Simd, attn: Attn::Simd, ..naive };
        let mma = Recipe { matmul: Matmul::SimdMatrix, ..par };
        let q4 = Recipe { dtype: DType::Q4_0, head_dtype: DType::Q8_0, ..mma };
        let pipelined = Recipe { gpu_argmax: true, fused: Fusion::Epilogue, pipelined: true, ..q4 };
        match self {
            MetalStage::Naive => naive,
            MetalStage::OneCb => Recipe { sync: Sync::PerForward, ..naive },
            MetalStage::Simd => Recipe { sync: Sync::PerForward, matvec: Matvec::Simd, ..naive },
            MetalStage::Par => par,
            MetalStage::Tiled => Recipe { matmul: Matmul::Tiled, ..par },
            MetalStage::Mma => mma,
            MetalStage::F16 => Recipe { dtype: DType::F16, head_dtype: DType::F16, ..mma },
            MetalStage::Q8 => Recipe { dtype: DType::Q8_0, head_dtype: DType::Q8_0, ..mma },
            MetalStage::Q4 => q4,
            MetalStage::Argmax => Recipe { gpu_argmax: true, ..q4 },
            MetalStage::FusedLn => Recipe { gpu_argmax: true, fused: Fusion::Full, ..q4 },
            MetalStage::Fused => Recipe { gpu_argmax: true, fused: Fusion::Epilogue, ..q4 },
            MetalStage::Pipelined => pipelined,
            MetalStage::PipelinedF16 => Recipe { dtype: DType::F16, head_dtype: DType::F16, ..pipelined },
            MetalStage::PipelinedQ8 => Recipe { dtype: DType::Q8_0, head_dtype: DType::Q8_0, ..pipelined },
            MetalStage::FlashDec => Recipe { attn: Attn::SplitK, ..pipelined },
            MetalStage::Flash => Recipe { attn: Attn::Flash, ..pipelined },
            MetalStage::Kv16 => Recipe { attn: Attn::Flash, kv_f16: true, ..pipelined },
            MetalStage::Lanes => Recipe { attn: Attn::Flash, kv_f16: true, lanes: true, ..pipelined },
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
        let r = self.recipe();
        let blocks = cfg.n_layer * 12 * c * c; // 3C*C + C*C + 4C*C + 4C*C per layer
        let head = cfg.n_vocab * c;
        let small = cfg.n_params() - cfg.n_ctx * c - blocks - head; // biases + layernorms, kept f32
        blocks as f64 * r.dtype.bytes_per_weight() + head as f64 * r.head_dtype.bytes_per_weight() + small as f64 * 4.0
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
