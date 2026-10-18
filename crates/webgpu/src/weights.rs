//! Copy a [`nanoinfer_core::Model`] into GPU buffers in a stage's dtype.
//!
//! The tied embedding / LM head (50257 x 768) is split into parts that each
//! fit WebGPU's *default* `maxStorageBufferBindingSize` (128 MiB): f32 is
//! 154 MB, so it becomes two bindings; f16 and Q8 fit in one.

use crate::gpu::Gpu;
use crate::DType;
use half::f16;
use nanoinfer_core::Model;

/// The default WebGPU binding limit; we split to it even when the adapter
/// allows more, so the same code runs on every browser.
pub const BINDING_LIMIT: u64 = 128 << 20;

/// A weight matrix `[rows, cols]` (rows = output features) in some dtype.
pub struct Mat {
    pub w: wgpu::Buffer,
    /// Q8_0 block scales (f16, two per u32).
    pub scales: Option<wgpu::Buffer>,
    pub dtype: DType,
    pub rows: usize,
    pub cols: usize,
}

pub struct LayerW {
    pub ln_1_w: wgpu::Buffer,
    pub ln_1_b: wgpu::Buffer,
    pub attn: Mat,
    pub attn_b: wgpu::Buffer,
    pub proj: Mat,
    pub proj_b: wgpu::Buffer,
    pub ln_2_w: wgpu::Buffer,
    pub ln_2_b: wgpu::Buffer,
    pub fc: Mat,
    pub fc_b: wgpu::Buffer,
    pub fc_proj: Mat,
    pub fc_proj_b: wgpu::Buffer,
}

pub struct Weights {
    pub dtype: DType,
    pub head_dtype: DType,
    /// (first row, rows of wte) per part
    pub head: Vec<(usize, Mat)>,
    pub wpe: wgpu::Buffer,
    pub layers: Vec<LayerW>,
    pub ln_f_w: wgpu::Buffer,
    pub ln_f_b: wgpu::Buffer,
    /// bytes uploaded
    pub bytes: u64,
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    bytemuck::cast_slice(v)
}

pub fn to_f16(w: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(w.len() * 2);
    for &v in w {
        out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    out
}

/// Q8_0 (same rounding as the Metal and core engines): per block of 32,
/// d = max|w| / 127, q = round(w / d). Returns (quants as bytes, f16 scales).
pub fn quantize_q8_0(w: &[f32]) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(w.len() % 32, 0);
    let mut qs = Vec::with_capacity(w.len());
    let mut sc = Vec::with_capacity(w.len() / 16);
    for block in w.chunks_exact(32) {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        qs.extend(block.iter().map(|&v| (v * id).round() as i8 as u8));
        sc.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    }
    (qs, sc)
}

impl Mat {
    pub fn upload(gpu: &Gpu, label: &str, w: &[f32], rows: usize, cols: usize, dtype: DType) -> Mat {
        assert_eq!(w.len(), rows * cols);
        let (buf, scales) = match dtype {
            DType::F32 => (gpu.buffer_init(label, f32_bytes(w)), None),
            DType::F16 => (gpu.buffer_init(label, &to_f16(w)), None),
            DType::Q8_0 => {
                let (q, s) = quantize_q8_0(w);
                (gpu.buffer_init(label, &q), Some(gpu.buffer_init(&format!("{label}.scales"), &s)))
            }
        };
        Mat { w: buf, scales, dtype, rows, cols }
    }
    pub fn bytes(&self) -> u64 {
        (self.rows * self.cols) as u64 * self.dtype.bytes_per_weight() as u64
    }
}

impl Weights {
    pub fn upload(gpu: &Gpu, m: &Model, dtype: DType, head_dtype: DType) -> Weights {
        let cfg = m.config;
        let c = cfg.n_embd;
        let v = |label: &str, x: &[f32]| gpu.buffer_init(label, f32_bytes(x));
        let mut bytes = 0u64;
        // LM head parts
        let row_bytes = (c as f64 * head_dtype.bytes_per_weight()) as u64;
        let per_part = (BINDING_LIMIT / row_bytes) as usize;
        let mut head = vec![];
        let mut r0 = 0;
        while r0 < cfg.n_vocab {
            let rows = per_part.min(cfg.n_vocab - r0);
            // keep Q8 blocks whole: c is a multiple of 32, so any row split is
            let part = Mat::upload(gpu, "wte", &m.wte[r0 * c..(r0 + rows) * c], rows, c, head_dtype);
            bytes += part.bytes();
            head.push((r0, part));
            r0 += rows;
        }
        let layers = m
            .layers
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let mat = |name: &str, w: &[f32], rows, cols| Mat::upload(gpu, &format!("h{i}.{name}"), w, rows, cols, dtype);
                let lw = LayerW {
                    ln_1_w: v("ln_1_w", &l.ln_1_w),
                    ln_1_b: v("ln_1_b", &l.ln_1_b),
                    attn: mat("attn", &l.attn_w, 3 * c, c),
                    attn_b: v("attn_b", &l.attn_b),
                    proj: mat("proj", &l.proj_w, c, c),
                    proj_b: v("proj_b", &l.proj_b),
                    ln_2_w: v("ln_2_w", &l.ln_2_w),
                    ln_2_b: v("ln_2_b", &l.ln_2_b),
                    fc: mat("fc", &l.fc_w, 4 * c, c),
                    fc_b: v("fc_b", &l.fc_b),
                    fc_proj: mat("fc_proj", &l.fc_proj_w, c, 4 * c),
                    fc_proj_b: v("fc_proj_b", &l.fc_proj_b),
                };
                bytes += lw.attn.bytes() + lw.proj.bytes() + lw.fc.bytes() + lw.fc_proj.bytes();
                lw
            })
            .collect();
        Weights {
            dtype,
            head_dtype,
            head,
            wpe: v("wpe", &m.wpe),
            layers,
            ln_f_w: v("ln_f_w", &m.ln_f_w),
            ln_f_b: v("ln_f_b", &m.ln_f_b),
            bytes,
        }
    }
}
