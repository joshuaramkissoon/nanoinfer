//! Stage 14: a head-major f16 KV cache.
//!
//! Stage 13's profile: at ~600 tokens of context attention is 37% of a decode
//! token, moving only ~34 GB/s. Two problems with the stage 1 cache layout
//! `[layer][position][C]` in f32:
//!
//! 1. **Layout.** One head's keys are 64 floats per position, 3 KB apart:
//!    hundreds of short strided reads, the access pattern the prefetcher handles
//!    worst (stage 3). Head-major `[layer][head][position][64]` makes each head's
//!    keys (and values) one contiguous stream.
//! 2. **Bytes.** At 512 positions the f32 cache is 38 MB per decoded token (2 ×
//!    12 layers × 512 × 768 × 4 B): law 5, the KV cache becomes a weight you read
//!    every token. f16 halves it. Keys/values are converted with `fcvtn` on write
//!    and widened with `fcvtl` on read; all math stays f32.
//!
//! The rest of the forward pass is stage 12/13's ([`super::packed`]): packed
//! weights, rescored head, fast pool.

use super::packed::{Head, PackedModel};
use super::Activations;
use crate::config::Config;
use crate::kernels::packed::{PackedMatrix, Par, Scratch};
use crate::kernels::tiled::OutPtr;
use crate::kernels::{fastops, lowlat, Kernels};
use crate::quant::{f16_to_f32, f32_to_f16};
use crate::trace::{traced, Op, OpKind, Trace};
use crate::weights::Model;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// `[n_layer][n_head][capacity][head_dim]` keys and values as f16 bits.
#[derive(Debug, Default)]
pub struct KvCache16 {
    pub k: Vec<u16>,
    pub v: Vec<u16>,
    pub capacity: usize,
    n_head: usize,
    head_dim: usize,
}

impl KvCache16 {
    pub fn new(model: &Model, capacity: usize) -> Self {
        let cfg = &model.config;
        let n = cfg.n_layer * capacity * cfg.n_embd;
        KvCache16 { k: vec![0; n], v: vec![0; n], capacity, n_head: cfg.n_head, head_dim: cfg.head_dim() }
    }

    /// Bytes when full (half of [`super::KvCache::bytes`]).
    pub fn bytes(&self) -> usize {
        (self.k.len() + self.v.len()) * 2
    }

    /// Offset of (layer, head, position 0).
    fn base(&self, l: usize, h: usize) -> usize {
        ((l * self.n_head + h) * self.capacity) * self.head_dim
    }
}

/// f32 → f16 for a slice (NEON `fcvtn` 4 at a time on aarch64).
fn narrow(src: &[f32], dst: &mut [u16]) {
    #[cfg(target_arch = "aarch64")]
    {
        let n4 = src.len() / 4 * 4;
        for i in (0..n4).step_by(4) {
            // Safety: i + 4 <= len of both; pure register conversion.
            unsafe {
                let v = vld1q_f32(src.as_ptr().add(i));
                let h: uint16x4_t;
                std::arch::asm!("fcvtn {h:v}.4h, {v:v}.4s", v = in(vreg) v, h = out(vreg) h, options(pure, nomem, nostack, preserves_flags));
                vst1_u16(dst.as_mut_ptr().add(i), h);
            }
        }
        for i in n4..src.len() {
            dst[i] = f32_to_f16(src[i]);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = f32_to_f16(s);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn widen(h: uint16x8_t) -> (float32x4_t, float32x4_t) {
    let lo: float32x4_t;
    let hi: float32x4_t;
    std::arch::asm!(
        "fcvtl {lo:v}.4s, {h:v}.4h",
        "fcvtl2 {hi:v}.4s, {h:v}.8h",
        h = in(vreg) h,
        lo = out(vreg) lo,
        hi = lateout(vreg) hi,
        options(pure, nomem, nostack, preserves_flags)
    );
    (lo, hi)
}

/// One (query, head) unit, head_dim 64: `k`, `v` = this head's `n_kv` contiguous
/// 64-half rows.
///
/// # Safety
/// `k`, `v` have ≥ n_kv·64 halves, `q`, `out` 64 floats, `scores` ≥ n_kv.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop)]
unsafe fn attend64_f16(out: &mut [f32], q: &[f32], k: &[u16], v: &[u16], n_kv: usize, scores: &mut [f32]) {
    let qv: [float32x4_t; 16] = std::array::from_fn(|c| vld1q_f32(q.as_ptr().add(4 * c)));
    let s = &mut scores[..n_kv];
    let mut max = f32::NEG_INFINITY;
    for (j, sj) in s.iter_mut().enumerate() {
        let kp = k.as_ptr().add(j * 64);
        let mut a = [vdupq_n_f32(0.0); 4];
        for c in 0..8 {
            let (lo, hi) = widen(vld1q_u16(kp.add(8 * c)));
            a[(2 * c) & 3] = vfmaq_f32(a[(2 * c) & 3], qv[2 * c], lo);
            a[(2 * c + 1) & 3] = vfmaq_f32(a[(2 * c + 1) & 3], qv[2 * c + 1], hi);
        }
        *sj = vaddvq_f32(vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3]))) * 0.125;
        max = max.max(*sj);
    }
    let mut total = 0.0;
    for x in s.iter_mut() {
        *x = fastops::exp(*x - max);
        total += *x;
    }
    let inv = 1.0 / total;
    let mut o = [vdupq_n_f32(0.0); 16];
    for (j, &p) in s.iter().enumerate() {
        let vp = v.as_ptr().add(j * 64);
        let p = p * inv;
        for c in 0..8 {
            let (lo, hi) = widen(vld1q_u16(vp.add(8 * c)));
            o[2 * c] = vfmaq_n_f32(o[2 * c], lo, p);
            o[2 * c + 1] = vfmaq_n_f32(o[2 * c + 1], hi, p);
        }
    }
    for c in 0..16 {
        vst1q_f32(out.as_mut_ptr().add(4 * c), o[c]);
    }
}

/// Portable unit (any head_dim): scalar f16 widening.
fn attend_f16_portable(out: &mut [f32], q: &[f32], k: &[u16], v: &[u16], hd: usize, n_kv: usize, scores: &mut [f32]) {
    let scale = 1.0 / (hd as f32).sqrt();
    let s = &mut scores[..n_kv];
    let mut max = f32::NEG_INFINITY;
    for (j, sj) in s.iter_mut().enumerate() {
        *sj = q.iter().zip(&k[j * hd..(j + 1) * hd]).map(|(a, &b)| a * f16_to_f32(b)).sum::<f32>() * scale;
        max = max.max(*sj);
    }
    let mut total = 0.0;
    for x in s.iter_mut() {
        *x = fastops::exp(*x - max);
        total += *x;
    }
    out.fill(0.0);
    for (j, &p) in s.iter().enumerate() {
        let p = p / total;
        for (o, &vv) in out.iter_mut().zip(&v[j * hd..(j + 1) * hd]) {
            *o += p * f16_to_f32(vv);
        }
    }
}

/// Causal attention for `n` query rows (in `qkv`, stride 3C) at positions
/// `pos0..`, over layer `l`'s f16 cache. Units split on the fast pool.
fn attention(out: &mut [f32], qkv: &[f32], kv: &KvCache16, l: usize, n: usize, pos0: usize, c: usize) {
    let (nh, hd) = (kv.n_head, kv.head_dim);
    let units = n * nh;
    let n_kv_max = pos0 + n;
    let ptr = OutPtr(out.as_mut_ptr());
    let unit = |u: usize, scratch: &mut [f32]| {
        let (i, h) = (u / nh, u % nh);
        let p = ptr;
        // Safety: each unit writes its own disjoint head_dim slice of `out`.
        let o = unsafe { std::slice::from_raw_parts_mut(p.0.add(i * c + h * hd), hd) };
        let q = &qkv[i * 3 * c + h * hd..][..hd];
        let n_kv = pos0 + i + 1;
        let base = kv.base(l, h);
        let (k, v) = (&kv.k[base..base + n_kv * hd], &kv.v[base..base + n_kv * hd]);
        #[cfg(target_arch = "aarch64")]
        if hd == 64 {
            // Safety: slices sized above; scratch ≥ n_kv_max ≥ n_kv.
            return unsafe { attend64_f16(o, q, k, v, n_kv, scratch) };
        }
        attend_f16_portable(o, q, k, v, hd, n_kv, scratch);
    };
    let threads = lowlat::num_threads();
    let per = units.div_ceil(threads * lowlat::CHUNKS_PER_THREAD).max(1);
    lowlat::par_for(units.div_ceil(per), &|ch| {
        let mut stack = [0f32; 1024];
        let mut heap = Vec::new();
        let scratch: &mut [f32] = if n_kv_max <= stack.len() {
            &mut stack
        } else {
            heap.resize(n_kv_max, 0.0);
            &mut heap
        };
        for u in ch * per..((ch + 1) * per).min(units) {
            unit(u, scratch);
        }
    });
}

fn op_matmul<M: PackedMatrix>(name: &'static str, layer: Option<usize>, n: usize, m: &M) -> Op {
    let base = Op::matmul(name, layer, n, m.d_in(), m.d_out());
    Op { bytes: base.bytes - 4 * (m.d_in() * m.d_out()) as u64 + m.bytes() as u64, ..base }
}

/// Stage 12/13's forward with the f16 head-major cache. `len` is the number of
/// cached positions (kept by the caller).
#[allow(clippy::too_many_arguments)]
pub fn forward<K: Kernels, M: PackedMatrix, H: Head, P: Par, T: Trace>(
    model: &Model,
    packed: &PackedModel<M, H>,
    new_tokens: &[u32],
    kv: &mut KvCache16,
    len: &mut usize,
    acts: &mut Activations,
    scratch: &mut Scratch,
    trace: &mut T,
) {
    let cfg = &model.config;
    let n = new_tokens.len();
    let (c, hd, nh) = (cfg.n_embd, cfg.head_dim(), cfg.n_head);
    let rc = n * c;
    let pos0 = *len;
    debug_assert!(pos0 + n <= kv.capacity);
    acts.ensure(model, n);

    super::blocks::embed::<K, T>(model, new_tokens, pos0, acts, trace);

    for (l, (layer, pl)) in model.layers.iter().zip(&packed.layers).enumerate() {
        traced(
            trace,
            || Op::elementwise(OpKind::LayerNorm, "ln_1", Some(l), n, c, 8, 8),
            || K::layernorm(&mut acts.xn, &acts.x, &layer.ln_1_w, &layer.ln_1_b, n, c, Config::LN_EPS),
        );
        traced(
            trace,
            || op_matmul("qkv", Some(l), n, &pl.attn),
            || pl.attn.matmul::<P>(&mut acts.qkv, &acts.xn, Some(&layer.attn_b), n, scratch),
        );
        traced(
            trace,
            || Op::elementwise(OpKind::KvWrite, "kv_write", Some(l), n, 2 * c, 0, 6),
            || {
                for i in 0..n {
                    let row = &acts.qkv[i * 3 * c..][..3 * c];
                    for h in 0..nh {
                        let dst = kv.base(l, h) + (pos0 + i) * hd;
                        narrow(&row[c + h * hd..c + (h + 1) * hd], &mut kv.k[dst..dst + hd]);
                        narrow(&row[2 * c + h * hd..2 * c + (h + 1) * hd], &mut kv.v[dst..dst + hd]);
                    }
                }
            },
        );
        traced(
            trace,
            || Op { bytes: Op::attention(l, n, pos0, c, nh).bytes / 2, ..Op::attention(l, n, pos0, c, nh) },
            || attention(&mut acts.att, &acts.qkv[..n * 3 * c], kv, l, n, pos0, c),
        );
        traced(
            trace,
            || op_matmul("attn_proj", Some(l), n, &pl.proj),
            || pl.proj.matmul::<P>(&mut acts.proj, &acts.att, Some(&layer.proj_b), n, scratch),
        );
        traced(
            trace,
            || Op::elementwise(OpKind::Residual, "residual_1", Some(l), n, c, 1, 12),
            || K::residual_add(&mut acts.x[..rc], &acts.proj[..rc]),
        );
        traced(
            trace,
            || Op::elementwise(OpKind::LayerNorm, "ln_2", Some(l), n, c, 8, 8),
            || K::layernorm(&mut acts.xn, &acts.x, &layer.ln_2_w, &layer.ln_2_b, n, c, Config::LN_EPS),
        );
        traced(
            trace,
            || op_matmul("fc", Some(l), n, &pl.fc),
            || pl.fc.matmul::<P>(&mut acts.fc, &acts.xn, Some(&layer.fc_b), n, scratch),
        );
        traced(
            trace,
            || Op::elementwise(OpKind::Gelu, "gelu", Some(l), n, 4 * c, 15, 8),
            || K::gelu(&mut acts.fc[..4 * rc]),
        );
        traced(
            trace,
            || op_matmul("fc_proj", Some(l), n, &pl.fc_proj),
            || pl.fc_proj.matmul::<P>(&mut acts.proj, &acts.fc, Some(&layer.fc_proj_b), n, scratch),
        );
        traced(
            trace,
            || Op::elementwise(OpKind::Residual, "residual_2", Some(l), n, c, 1, 12),
            || K::residual_add(&mut acts.x[..rc], &acts.proj[..rc]),
        );
    }
    *len += n;
    if T::ENABLED {
        trace.kv_cache(*len, kv.capacity);
    }

    let last = &acts.x[(n - 1) * c..n * c];
    let v = cfg.n_vocab as u64;
    traced(
        trace,
        || Op {
            kind: OpKind::LmHead,
            name: "lm_head",
            layer: None,
            dims: [1, c, cfg.n_vocab],
            flops: 2 * c as u64 * v + 8 * c as u64,
            bytes: packed.head.bytes() as u64 + 4 * (c as u64 + v),
        },
        || packed.head.logits::<K, P>(model, last, &mut acts.xn[..c], &mut acts.logits, scratch),
    );
}
