//! Stage 7 kernels: make the "small" ops fast too.
//!
//! Once the matmuls ran on every core, the ops that used to be <1% of the time
//! became the long pole. In prefill at 10 threads, scalar GELU and attention were
//! about half the time (Amdahl's law). They were slow for two separate reasons:
//!
//! 1. **They don't vectorise.** GELU calls `tanhf` (a libm function call per
//!    element); softmax calls `expf`; attention's dot products are
//!    `iter().sum()` chains. Fix: [`tanh`] and [`exp`] as branch-free polynomial
//!    approximations that LLVM vectorises, plus the stage 2 multi-accumulator dot.
//! 2. **They run on one thread.** Fix: split GELU into chunks and attention into
//!    (query, head) units on the same pool the matmuls use, when there's enough
//!    work to pay for a dispatch.
//!
//! The approximations are accurate to a few ULP (tested against `std` below), far
//! below the 1e-3 logit tolerance.

use super::tiled::OutPtr;
use super::{spin, unrolled, Rows};

/// Below this many elements an op isn't worth a parallel dispatch (~1–5 µs).
const PAR_MIN: usize = 1 << 15;

/// `tanh(x)` as a rational function (Eigen's `generic_fast_tanh_float`):
/// odd degree-13 numerator over even degree-6 denominator, input clamped to
/// ±7.9 where tanh is ±1 in f32. Branch-free, so it vectorises.
#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    const CLAMP: f32 = 7.905_311;
    const A1: f32 = 4.893_524_6e-3;
    const A3: f32 = 6.372_619e-4;
    const A5: f32 = 1.485_722_4e-5;
    const A7: f32 = 5.122_297e-8;
    const A9: f32 = -8.604_672e-11;
    const A11: f32 = 2.000_188e-13;
    const A13: f32 = -2.760_768_5e-16;
    const B0: f32 = 4.893_525e-3;
    const B2: f32 = 2.268_434_6e-3;
    const B4: f32 = 1.185_347_1e-4;
    const B6: f32 = 1.198_258_4e-6;
    let x = x.clamp(-CLAMP, CLAMP);
    let x2 = x * x;
    let p = x * (A1 + x2 * (A3 + x2 * (A5 + x2 * (A7 + x2 * (A9 + x2 * (A11 + x2 * A13))))));
    let q = B0 + x2 * (B2 + x2 * (B4 + x2 * B6));
    p / q
}

/// `exp(x)`: write x = n·ln2 + r with integer n and |r| ≤ ln2/2, evaluate e^r
/// with a degree-6 Taylor polynomial, and build 2^n directly in the exponent bits.
/// ln2 is split into a short "hi" part (exact when multiplied by n) and a "lo"
/// correction (Cody–Waite), so r stays accurate even for large |x|. Inputs are
/// clamped to [-87, 88]; below -87 it flushes to ~1e-38 (fine for softmax, where
/// we only exponentiate `score - max ≤ 0`). Branch-free, so it vectorises.
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    const LN2_HI: f32 = 0.693_359_4; // 0.693359375 = 355/512, 9 significant bits
    const LN2_LO: f32 = -2.121_944_4e-4;
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round_ties_even();
    let r = (x - n * LN2_HI) - n * LN2_LO;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0))))));
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// GELU (tanh approximation, as GPT-2 was trained), with the fast tanh.
#[inline(always)]
fn gelu1(u: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * u * (1.0 + tanh(SQRT_2_OVER_PI * (u + 0.044715 * u * u * u)))
}

pub(crate) fn gelu_serial(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = gelu1(*v);
    }
}

/// In-place GELU, parallel over chunks when large (prefill).
pub fn gelu(x: &mut [f32]) {
    let n = x.len();
    let threads = spin::num_threads();
    if n < PAR_MIN || threads == 1 {
        return gelu_serial(x);
    }
    let chunk = n.div_ceil(threads * 2).next_multiple_of(64);
    let ptr = OutPtr(x.as_mut_ptr());
    spin::par_for(n.div_ceil(chunk), &|c| {
        let (b, e) = (c * chunk, ((c + 1) * chunk).min(n));
        let p = ptr; // capture the Sync wrapper, not its raw-pointer field
                     // Safety: chunks are disjoint ranges of `x`.
        gelu_serial(unsafe { std::slice::from_raw_parts_mut(p.0.add(b), e - b) });
    });
}

/// LayerNorm with vectorised (multi-accumulator) mean and variance.
pub fn layernorm(out: &mut [f32], x: &[f32], weight: &[f32], bias: &[f32], n: usize, d: usize, eps: f32) {
    for (x_row, out_row) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)).take(n) {
        let mean = sum(x_row) / d as f32;
        let mut acc = [0f32; 16];
        let (c, r) = x_row.as_chunks::<16>();
        for ch in c {
            for j in 0..16 {
                let v = ch[j] - mean;
                acc[j] += v * v;
            }
        }
        let var = (acc.iter().sum::<f32>() + r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()) / d as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        for (((o, &v), &w), &b) in out_row.iter_mut().zip(x_row).zip(weight).zip(bias) {
            *o = (v - mean) * rstd * w + b;
        }
    }
}

fn sum(x: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let (c, r) = x.as_chunks::<16>();
    for ch in c {
        for j in 0..16 {
            acc[j] += ch[j];
        }
    }
    acc.iter().sum::<f32>() + r.iter().sum::<f32>()
}

/// One (query, head) unit of causal attention: scores over keys `0..n_kv`,
/// softmax, weighted sum of values into `out` (`head_dim` floats).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn attend_one(
    out: &mut [f32],
    q: &[f32],
    k: Rows,
    v: Rows,
    h: usize,
    hd: usize,
    n_kv: usize,
    scores: &mut [f32],
) {
    let scale = 1.0 / (hd as f32).sqrt();
    let s = &mut scores[..n_kv];
    let mut max = f32::NEG_INFINITY;
    for (j, sj) in s.iter_mut().enumerate() {
        *sj = unrolled::dot(q, k.head(j, h, hd)) * scale;
        max = max.max(*sj);
    }
    let mut total = 0.0;
    for sj in s.iter_mut() {
        *sj = exp(*sj - max);
        total += *sj;
    }
    let inv = 1.0 / total;
    out.fill(0.0);
    for (j, &p) in s.iter().enumerate() {
        let p = p * inv;
        // axpy: elementwise, so it vectorises without reassociation.
        for (o, &vv) in out.iter_mut().zip(v.head(j, h, hd)) {
            *o += p * vv;
        }
    }
}

/// Same contract as [`super::reference::attention`], parallel over
/// (query, head) units when there's enough work. `scores` is unused (each unit
/// keeps its own on the stack, so units can run concurrently).
#[allow(clippy::too_many_arguments)]
pub fn attention(
    out: &mut [f32],
    q: Rows,
    k: Rows,
    v: Rows,
    n_q: usize,
    q_pos0: usize,
    n_head: usize,
    head_dim: usize,
    scores: &mut [f32],
) {
    let c = n_head * head_dim;
    let units = n_q * n_head;
    let n_kv_max = q_pos0 + n_q;
    // Work ≈ units × keys × 2·head_dim multiply-adds.
    let work = units * n_kv_max * head_dim;
    let ptr = OutPtr(out.as_mut_ptr());
    let unit = |u: usize, scratch: &mut [f32]| {
        let (i, h) = (u / n_head, u % n_head);
        let p = ptr; // capture the Sync wrapper, not its raw-pointer field
                     // Safety: each unit writes its own disjoint head_dim slice of `out`.
        let o = unsafe { std::slice::from_raw_parts_mut(p.0.add(i * c + h * head_dim), head_dim) };
        attend_one(o, q.head(i, h, head_dim), k, v, h, head_dim, q_pos0 + i + 1, scratch);
    };
    let threads = spin::num_threads();
    if threads == 1 || work < PAR_MIN * 4 {
        for u in 0..units {
            unit(u, scores);
        }
        return;
    }
    // Contiguous runs of units per chunk; each chunk gets one scratch buffer.
    let per = units.div_ceil(threads * 4);
    spin::par_for(units.div_ceil(per), &|ch| {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tanh_and_exp_match_std() {
        let mut worst_t = 0f32;
        let mut worst_e = 0f32;
        for i in -20000..=20000 {
            let x = i as f32 / 1000.0; // -20..20
            worst_t = worst_t.max((tanh(x) - x.tanh()).abs());
            let e = x.exp();
            worst_e = worst_e.max((exp(x) - e).abs() / e);
        }
        for i in -860..=0 {
            let x = i as f32 / 10.0;
            let e = x.exp();
            worst_e = worst_e.max((exp(x) - e).abs() / e.max(f32::MIN_POSITIVE));
        }
        assert!(worst_t < 2e-6, "tanh abs err {worst_t}");
        assert!(worst_e < 2e-6, "exp rel err {worst_e}");
    }

    #[test]
    fn attention_matches_reference() {
        let (n_head, hd, n_q, pos0) = (3, 8, 5, 7);
        let c = n_head * hd;
        let n_kv = pos0 + n_q;
        let f = |i: usize| ((i * 7919) % 13) as f32 / 6.0 - 1.0;
        let qd: Vec<f32> = (0..n_q * c).map(f).collect();
        let kd: Vec<f32> = (0..n_kv * c).map(|i| f(i + 3)).collect();
        let vd: Vec<f32> = (0..n_kv * c).map(|i| f(i + 11)).collect();
        let r = |d| Rows { data: d, stride: c };
        let (mut a, mut b, mut s) = (vec![0.0; n_q * c], vec![0.0; n_q * c], vec![0.0; n_kv]);
        super::super::reference::attention(&mut a, r(&qd), r(&kd), r(&vd), n_q, pos0, n_head, hd, &mut s);
        attention(&mut b, r(&qd), r(&kd), r(&vd), n_q, pos0, n_head, hd, &mut s);
        let d = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        assert!(d < 1e-5, "max diff {d}");
    }
}
