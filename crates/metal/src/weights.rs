//! Copy a [`nanoinfer_core::Model`] into GPU buffers in the stage's dtype.
//!
//! Unified memory means "upload" is a memcpy into a shared-storage buffer: no
//! PCIe, no staging. The copy happens once at load; per token the GPU reads
//! these pages directly.

use crate::gpu::{Buf, Gpu};
use crate::DType;
use half::f16;
use nanoinfer_core::Model;

/// Q8_0, structure-of-arrays: `[n] i8` quants, then `[n / 32]` f16 scales.
/// Each block of 32 weights gets scale `d = max|w| / 127`, `q = round(w / d)`.
pub fn quantize_q8_0(w: &[f32]) -> Vec<u8> {
    assert_eq!(w.len() % 32, 0);
    let mut qs = Vec::with_capacity(w.len() + w.len() / 16);
    let mut scales = Vec::with_capacity(w.len() / 32);
    for block in w.as_chunks::<32>().0 {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        qs.extend(block.iter().map(|&v| (v * id).round() as i8 as u8));
        scales.push(f16::from_f32(d));
    }
    qs.extend(scales.iter().flat_map(|s| s.to_le_bytes()));
    qs
}

/// CPU dequantisation of [`quantize_q8_0`] / [`quantize_q4_0`] output, for tests.
pub fn dequantize(q: &[u8], n: usize, dtype: DType) -> Vec<f32> {
    let nb = n / 32;
    let qbytes = match dtype {
        DType::Q8_0 => n,
        DType::Q4_0 => n / 2,
        _ => panic!("not a block format"),
    };
    let scale = |b: usize| f16::from_le_bytes([q[qbytes + 2 * b], q[qbytes + 2 * b + 1]]).to_f32();
    (0..n)
        .map(|i| {
            let (b, j) = (i / 32, i % 32);
            let v = match dtype {
                DType::Q8_0 => q[i] as i8 as f32,
                _ => {
                    let byte = q[b * 16 + j % 16];
                    (if j < 16 { byte & 15 } else { byte >> 4 }) as f32 - 8.0
                }
            };
            debug_assert!(b < nb);
            v * scale(b)
        })
        .collect()
}

/// Q4_0 (GGML's rounding), structure-of-arrays: `[n / 2]` bytes of nibbles,
/// then `[n / 32]` f16 scales. In a block of 32, byte i holds element i (low
/// nibble) and element i + 16 (high nibble); value = (nibble - 8) * d where
/// d = (the signed value with the largest magnitude) / -8.
pub fn quantize_q4_0(w: &[f32]) -> Vec<u8> {
    assert_eq!(w.len() % 32, 0);
    let mut qs = Vec::with_capacity(w.len() / 2 + w.len() / 16);
    let mut scales = Vec::with_capacity(w.len() / 32);
    for block in w.as_chunks::<32>().0 {
        let max = block.iter().copied().fold(0.0f32, |m, v| if v.abs() > m.abs() { v } else { m });
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let q = |v: f32| ((v * id + 8.5) as i32).clamp(0, 15) as u8;
        for i in 0..16 {
            qs.push(q(block[i]) | (q(block[i + 16]) << 4));
        }
        scales.push(f16::from_f32(d));
    }
    qs.extend(scales.iter().flat_map(|s| s.to_le_bytes()));
    qs
}

/// A weight matrix `[rows, cols]` (rows = out features) in some dtype.
pub struct Mat {
    pub buf: Buf,
    pub dtype: DType,
    pub rows: usize,
    pub cols: usize,
}

pub struct LayerW {
    pub ln1_w: Buf,
    pub ln1_b: Buf,
    pub attn_w: Mat,
    pub attn_b: Buf,
    pub proj_w: Mat,
    pub proj_b: Buf,
    pub ln2_w: Buf,
    pub ln2_b: Buf,
    pub fc_w: Mat,
    pub fc_b: Buf,
    pub fcp_w: Mat,
    pub fcp_b: Buf,
}

pub struct Weights {
    /// Token embedding table, f32, for the embedding lookup.
    pub wte: Buf,
    /// The same table as the LM head matrix, in the stage dtype.
    pub lm_head: Mat,
    pub wpe: Buf,
    pub layers: Vec<LayerW>,
    pub ln_f_w: Buf,
    pub ln_f_b: Buf,
}

fn mat(gpu: &Gpu, w: &[f32], rows: usize, cols: usize, dtype: DType) -> Mat {
    assert_eq!(w.len(), rows * cols);
    let buf = match dtype {
        DType::F32 => gpu.buffer_from(w),
        DType::F16 => gpu.buffer_from(&w.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()),
        DType::Q8_0 => gpu.buffer_from(&quantize_q8_0(w)),
        DType::Q4_0 => gpu.buffer_from(&quantize_q4_0(w)),
    };
    Mat { buf, dtype, rows, cols }
}

impl Weights {
    pub fn upload(gpu: &Gpu, model: &Model, dtype: DType, head_dtype: DType) -> Weights {
        let cfg = model.config;
        let c = cfg.n_embd;
        let layers = model
            .layers
            .iter()
            .map(|l| LayerW {
                ln1_w: gpu.buffer_from(&l.ln_1_w),
                ln1_b: gpu.buffer_from(&l.ln_1_b),
                attn_w: mat(gpu, &l.attn_w, 3 * c, c, dtype),
                attn_b: gpu.buffer_from(&l.attn_b),
                proj_w: mat(gpu, &l.proj_w, c, c, dtype),
                proj_b: gpu.buffer_from(&l.proj_b),
                ln2_w: gpu.buffer_from(&l.ln_2_w),
                ln2_b: gpu.buffer_from(&l.ln_2_b),
                fc_w: mat(gpu, &l.fc_w, 4 * c, c, dtype),
                fc_b: gpu.buffer_from(&l.fc_b),
                fcp_w: mat(gpu, &l.fc_proj_w, c, 4 * c, dtype),
                fcp_b: gpu.buffer_from(&l.fc_proj_b),
            })
            .collect();
        let wte = gpu.buffer_from(&model.wte);
        let lm_head = if head_dtype == DType::F32 {
            // Same bytes, second handle: no need to hold two f32 copies.
            Mat { buf: Buf { raw: wte.raw.clone(), len: wte.len }, dtype: DType::F32, rows: cfg.n_vocab, cols: c }
        } else {
            mat(gpu, &model.wte, cfg.n_vocab, c, head_dtype)
        };
        Weights {
            wte,
            lm_head,
            wpe: gpu.buffer_from(&model.wpe),
            layers,
            ln_f_w: gpu.buffer_from(&model.ln_f_w),
            ln_f_b: gpu.buffer_from(&model.ln_f_b),
        }
    }
}
