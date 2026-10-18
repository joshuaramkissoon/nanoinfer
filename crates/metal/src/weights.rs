//! Copy a [`nanoinfer_core::Model`] into GPU buffers in the stage's dtype.
//!
//! Unified memory means "upload" is a memcpy into a shared-storage buffer: no
//! PCIe, no staging. The copy happens once at load; per token the GPU reads
//! these pages directly.

use crate::gpu::{Buf, Gpu};
use crate::DType;
use nanoinfer_core::Model;

/// A weight matrix `[rows, cols]` (rows = out features) in some dtype.
pub struct Mat {
    pub buf: Buf,
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
        DType::F16 | DType::Q8_0 | DType::Q4_0 => unimplemented!("dtype {dtype:?}"),
    };
    Mat { buf, rows, cols }
}

impl Weights {
    pub fn upload(gpu: &Gpu, model: &Model, dtype: DType) -> Weights {
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
        let lm_head = if dtype == DType::F32 {
            // Same bytes, second handle: no need to hold two f32 copies.
            Mat { buf: Buf { raw: wte.raw.clone(), len: wte.len }, rows: cfg.n_vocab, cols: c }
        } else {
            mat(gpu, &model.wte, cfg.n_vocab, c, dtype)
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
