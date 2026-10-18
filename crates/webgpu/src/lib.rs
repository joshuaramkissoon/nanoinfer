//! # nanoinfer-webgpu
//!
//! GPT-2 on WebGPU through `wgpu`: the same WGSL kernels and the same Rust
//! host code run natively (wgpu's Metal backend, for a native comparison
//! bench) and in a browser tab (wgpu's WebGPU backend, compiled to wasm).
//! Hillclimbed like the CPU and Metal engines: every [`WgpuStage`] stays
//! runnable and is a [`Recipe`], not an edit of the previous stage.
//!
//! Everything that touches the GPU is `async` (a browser can't block on the
//! GPU); natively, wrap calls in [`gpu::block_on`].

pub mod forward;
pub mod gpu;
pub mod probe;
pub mod weights;

pub use forward::{Engine, RunStats, Session};
pub use gpu::Gpu;
pub use wgpu;

use std::fmt;
use std::str::FromStr;

/// Weight storage format on the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    /// IEEE half, packed two per u32 (`unpack2x16float`, no `shader-f16` needed).
    F16,
    /// Blocks of 32 int8 + one f16 scale (34 bytes / 32 weights).
    Q8_0,
}

impl DType {
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::Q8_0 => "q8_0",
        }
    }
    /// The lossier of two formats (for test tolerances).
    pub fn max_lossy(self, other: DType) -> DType {
        if self.bytes_per_weight() < other.bytes_per_weight() { self } else { other }
    }
    pub fn bytes_per_weight(self) -> f64 {
        match self {
            DType::F32 => 4.0,
            DType::F16 => 2.0,
            DType::Q8_0 => 34.0 / 32.0,
        }
    }
}

/// How often the CPU submits and waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sync {
    /// Every op: its own command encoder + submit, then wait for the GPU.
    PerOp,
    /// One command encoder, one compute pass, one submit per forward.
    PerForward,
}

/// Decode (n = 1) matrix-vector kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Matvec {
    /// One thread per output, serial dot product.
    Naive,
    /// A group of threads per row, vector loads, workgroup-memory reduction.
    Workgroup,
    /// Same, reduced with `subgroupAdd` (needs the `subgroups` feature).
    Subgroup,
}

/// Prefill (n > 1) matrix-matrix kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Matmul {
    Naive,
    /// 64x64 tiles through workgroup memory, 4x4 register blocks.
    Tiled,
    /// Same tiles, staged and read as vec4s (4x fewer workgroup accesses).
    Tiled4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Recipe {
    pub sync: Sync,
    pub dtype: DType,
    /// The tied LM head / embedding table (small models are most sensitive to it).
    pub head_dtype: DType,
    pub matvec: Matvec,
    pub matmul: Matmul,
    /// Workgroup-parallel LayerNorm and attention (else one thread per row / head).
    pub par: bool,
    /// LayerNorm on 64 threads with the row held in registers (needs n_embd = 768).
    pub ln64: bool,
    /// Bias + GELU, residual adds and KV writes fused into the matmul epilogues.
    pub fused: bool,
    /// ln_1 / ln_2 folded into the qkv / fc weights; decode skips those
    /// LayerNorm kernels (the matvec computes the row's mean and variance).
    pub ln_fold: bool,
    /// Attention with vec4 loads (q in registers, 4 dims per thread in P·V).
    pub attn4: bool,
    /// Decode attention split over chunks of 64 keys + a merge (flash-decoding).
    pub flashdec: bool,
    /// K/V cache stored as f16 (packed in u32); needs `attn4`.
    pub kv16: bool,
    /// Q8 decode matvecs load 16 weights (one vec4<u32>) at a time, 16
    /// threads per 768-wide row.
    pub wide_q8: bool,
    /// Argmax on the GPU, written straight into the next forward's input slot.
    pub gpu_argmax: bool,
    /// Greedy decode encodes many forwards into one submit and reads the
    /// tokens back once (the GPU-resident loop).
    pub resident: bool,
}

/// The WebGPU hillclimb, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WgpuStage {
    Naive,
    OneSubmit,
    Matvec,
    Par,
    Subgroup,
    Fused,
    F16,
    Q8,
    Argmax,
    Resident,
    Tiled,
    Ln64,
    Q8Head,
    LnFold,
    Attn4,
    FlashDec,
    Kv16,
    Vec,
}

impl WgpuStage {
    pub const ALL: &'static [WgpuStage] = &[
        WgpuStage::Naive,
        WgpuStage::OneSubmit,
        WgpuStage::Matvec,
        WgpuStage::Par,
        WgpuStage::Subgroup,
        WgpuStage::Fused,
        WgpuStage::F16,
        WgpuStage::Q8,
        WgpuStage::Argmax,
        WgpuStage::Resident,
        WgpuStage::Tiled,
        WgpuStage::Ln64,
        WgpuStage::Q8Head,
        WgpuStage::LnFold,
        WgpuStage::Attn4,
        WgpuStage::FlashDec,
        WgpuStage::Kv16,
        WgpuStage::Vec,
    ];

    pub fn name(self) -> &'static str {
        match self {
            WgpuStage::Naive => "webgpu-naive",
            WgpuStage::OneSubmit => "webgpu-onesubmit",
            WgpuStage::Matvec => "webgpu-matvec",
            WgpuStage::Par => "webgpu-par",
            WgpuStage::Subgroup => "webgpu-subgroup",
            WgpuStage::Fused => "webgpu-fused",
            WgpuStage::F16 => "webgpu-f16",
            WgpuStage::Q8 => "webgpu-q8",
            WgpuStage::Argmax => "webgpu-argmax",
            WgpuStage::Resident => "webgpu-resident",
            WgpuStage::Tiled => "webgpu-tiled",
            WgpuStage::Ln64 => "webgpu-ln64",
            WgpuStage::Q8Head => "webgpu-q8head",
            WgpuStage::LnFold => "webgpu-lnfold",
            WgpuStage::Attn4 => "webgpu-attn4",
            WgpuStage::FlashDec => "webgpu-flashdec",
            WgpuStage::Kv16 => "webgpu-kv16",
            WgpuStage::Vec => "webgpu-vec",
        }
    }

    /// One line: what this stage changes.
    pub fn kernels(self) -> &'static str {
        match self {
            WgpuStage::Naive => "WGSL port of the CPU kernels, one submit + wait per op",
            WgpuStage::OneSubmit => "same kernels, one command encoder + one submit per token",
            WgpuStage::Matvec => "64 threads per row, vec4 loads, workgroup reduction",
            WgpuStage::Par => "256-thread workgroups for LayerNorm and attention",
            WgpuStage::Subgroup => "matvec reduced with subgroupAdd (native wgpu only)",
            WgpuStage::Fused => "bias+GELU, residual and KV write fused into matmul epilogues",
            WgpuStage::F16 => "f16 weights packed in u32 (unpack2x16float)",
            WgpuStage::Q8 => "Q8_0 weights dequantised in the kernel, f16 LM head",
            WgpuStage::Argmax => "argmax on the GPU (4 bytes read back, not 200 KB)",
            WgpuStage::Resident => "GPU-resident greedy loop: 16 tokens per submit, one readback",
            WgpuStage::Tiled => "64x64 tiled prefill matmul",
            WgpuStage::Ln64 => "LayerNorm on 64 threads, row in registers (12 barriers, not 16)",
            WgpuStage::Q8Head => "Q8_0 LM head too (half the head's bytes; costs quality)",
            WgpuStage::LnFold => "LayerNorm gains folded into qkv/fc weights; decode matvec computes mean/var (no LN kernels)",
            WgpuStage::Attn4 => "attention with vec4 loads: q in registers, 4 dims x 1/16 of the keys per thread in P.V",
            WgpuStage::FlashDec => "decode attention split over 64-key chunks + merge (flash-decoding)",
            WgpuStage::Kv16 => "f16 KV cache (half the attention bytes; K/V packed by a kv_write kernel)",
            WgpuStage::Vec => "vec4 traffic: prefill tiles staged/read as vec4s, Q8 matvec 16 weights per load",
        }
    }

    pub fn recipe(self) -> Recipe {
        let naive = Recipe {
            sync: Sync::PerOp,
            dtype: DType::F32,
            head_dtype: DType::F32,
            matvec: Matvec::Naive,
            matmul: Matmul::Naive,
            par: false,
            ln64: false,
            fused: false,
            ln_fold: false,
            attn4: false,
            flashdec: false,
            kv16: false,
            wide_q8: false,
            gpu_argmax: false,
            resident: false,
        };
        let onesubmit = Recipe { sync: Sync::PerForward, ..naive };
        let matvec = Recipe { matvec: Matvec::Workgroup, ..onesubmit };
        let par = Recipe { par: true, ..matvec };
        let fused = Recipe { fused: true, ..par };
        let f16 = Recipe { dtype: DType::F16, head_dtype: DType::F16, ..fused };
        let q8 = Recipe { dtype: DType::Q8_0, head_dtype: DType::F16, ..fused };
        let argmax = Recipe { gpu_argmax: true, ..q8 };
        let resident = Recipe { resident: true, ..argmax };
        match self {
            WgpuStage::Naive => naive,
            WgpuStage::OneSubmit => onesubmit,
            WgpuStage::Matvec => matvec,
            WgpuStage::Par => par,
            WgpuStage::Subgroup => Recipe { matvec: Matvec::Subgroup, ..par },
            WgpuStage::Fused => fused,
            WgpuStage::F16 => f16,
            WgpuStage::Q8 => q8,
            WgpuStage::Argmax => argmax,
            WgpuStage::Resident => resident,
            WgpuStage::Tiled => Recipe { matmul: Matmul::Tiled, ..resident },
            WgpuStage::Ln64 => Recipe { matmul: Matmul::Tiled, ln64: true, ..resident },
            WgpuStage::Q8Head => Recipe { matmul: Matmul::Tiled, ln64: true, head_dtype: DType::Q8_0, ..resident },
            WgpuStage::LnFold => Recipe { matmul: Matmul::Tiled, ln64: true, head_dtype: DType::Q8_0, ln_fold: true, ..resident },
            WgpuStage::Attn4 => Recipe { matmul: Matmul::Tiled, ln64: true, head_dtype: DType::Q8_0, ln_fold: true, attn4: true, ..resident },
            WgpuStage::Kv16 => Recipe { matmul: Matmul::Tiled, ln64: true, head_dtype: DType::Q8_0, ln_fold: true, attn4: true, kv16: true, ..resident },
            WgpuStage::Vec => Recipe {
                matmul: Matmul::Tiled4,
                ln64: true,
                head_dtype: DType::Q8_0,
                ln_fold: true,
                attn4: true,
                kv16: true,
                wide_q8: true,
                ..resident
            },
            WgpuStage::FlashDec => Recipe { matmul: Matmul::Tiled, ln64: true, head_dtype: DType::Q8_0, ln_fold: true, attn4: true, flashdec: true, ..resident },
        }
    }

    /// Can this device run the stage as written? (`webgpu-subgroup` needs the
    /// `subgroups` feature, which wgpu only exposes natively.)
    pub fn supported(self, gpu: &Gpu) -> bool {
        self.recipe().matvec != Matvec::Subgroup || gpu.has(wgpu::Features::SUBGROUP)
    }

    /// Bytes of weights one decode step streams (matrices + head; biases and
    /// LayerNorms are negligible).
    pub fn decode_weight_bytes(self, cfg: &nanoinfer_core::Config) -> f64 {
        let r = self.recipe();
        let c = cfg.n_embd as f64;
        let body = cfg.n_layer as f64 * 12.0 * c * c;
        body * r.dtype.bytes_per_weight() + cfg.n_vocab as f64 * c * r.head_dtype.bytes_per_weight()
    }
}

impl fmt::Display for WgpuStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for WgpuStage {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        WgpuStage::ALL.iter().copied().find(|st| st.name() == s).ok_or_else(|| {
            let names: Vec<_> = WgpuStage::ALL.iter().map(|s| s.name()).collect();
            format!("unknown WebGPU stage '{s}' (have: {})", names.join(", "))
        })
    }
}
