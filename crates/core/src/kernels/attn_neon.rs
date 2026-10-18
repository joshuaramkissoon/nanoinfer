//! Stage 13: decode attention as a register-resident NEON kernel.
//!
//! Stage 7's attention unit is portable: each score is a call to the stage 2
//! dot (16 partial sums, tree-reduced, for only 64 products), then softmax, then
//! an axpy per key that re-loads and re-stores the 64-float output. At decode
//! it ran ~34 µs per layer at a few hundred tokens of context, ~15% of a token.
//!
//! For GPT-2's head_dim = 64:
//! - **scores:** q lives in 16 NEON registers for the whole pass; two keys per
//!   iteration (8 accumulators, so the FMA latency is hidden), one horizontal add
//!   per key;
//! - **softmax:** the vectorised `exp` from stage 7;
//! - **values:** the 64-float output lives in 16 registers across *all* keys:
//!   each key is 16 independent FMAs, `o += p_j · v_j`, and one store at the end.
//!
//! Other head sizes, and non-aarch64 targets, use stage 7's unit.

use super::tiled::OutPtr;
use super::{fastops, lowlat, Rows};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// One (query, head) unit for head_dim = 64.
///
/// # Safety
/// `k`/`v` hold `n_kv` positions; `q`, `out` have 64 floats; `scores` ≥ n_kv.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn attend64(out: &mut [f32], q: &[f32], k: Rows, v: Rows, h: usize, n_kv: usize, scores: &mut [f32]) {
    const HD: usize = 64;
    let scale = 0.125; // 1/sqrt(64)
    let qp = q.as_ptr();
    let qv: [float32x4_t; 16] = std::array::from_fn(|c| vld1q_f32(qp.add(4 * c)));
    let kb = k.data.as_ptr().add(h * HD);
    let s = &mut scores[..n_kv];
    let mut j = 0;
    let mut max = f32::NEG_INFINITY;
    while j + 2 <= n_kv {
        let k0 = kb.add(j * k.stride);
        let k1 = kb.add((j + 1) * k.stride);
        let mut a0 = [vdupq_n_f32(0.0); 4];
        let mut a1 = [vdupq_n_f32(0.0); 4];
        for c in 0..16 {
            a0[c & 3] = vfmaq_f32(a0[c & 3], qv[c], vld1q_f32(k0.add(4 * c)));
            a1[c & 3] = vfmaq_f32(a1[c & 3], qv[c], vld1q_f32(k1.add(4 * c)));
        }
        let s0 = vaddvq_f32(vaddq_f32(vaddq_f32(a0[0], a0[1]), vaddq_f32(a0[2], a0[3]))) * scale;
        let s1 = vaddvq_f32(vaddq_f32(vaddq_f32(a1[0], a1[1]), vaddq_f32(a1[2], a1[3]))) * scale;
        s[j] = s0;
        s[j + 1] = s1;
        max = max.max(s0).max(s1);
        j += 2;
    }
    if j < n_kv {
        let k0 = kb.add(j * k.stride);
        let mut a0 = [vdupq_n_f32(0.0); 4];
        for c in 0..16 {
            a0[c & 3] = vfmaq_f32(a0[c & 3], qv[c], vld1q_f32(k0.add(4 * c)));
        }
        s[j] = vaddvq_f32(vaddq_f32(vaddq_f32(a0[0], a0[1]), vaddq_f32(a0[2], a0[3]))) * scale;
        max = max.max(s[j]);
    }
    let mut total = 0.0;
    for x in s.iter_mut() {
        *x = fastops::exp(*x - max);
        total += *x;
    }
    let inv = 1.0 / total;
    let vb = v.data.as_ptr().add(h * HD);
    let mut o = [vdupq_n_f32(0.0); 16];
    for (j, &p) in s.iter().enumerate() {
        let vp = vb.add(j * v.stride);
        let p = p * inv;
        for c in 0..16 {
            o[c] = vfmaq_n_f32(o[c], vld1q_f32(vp.add(4 * c)), p);
        }
    }
    let op = out.as_mut_ptr();
    for c in 0..16 {
        vst1q_f32(op.add(4 * c), o[c]);
    }
}

#[allow(clippy::too_many_arguments)]
fn attend(out: &mut [f32], q: &[f32], k: Rows, v: Rows, h: usize, hd: usize, n_kv: usize, scores: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    if hd == 64 {
        assert!(q.len() == 64 && out.len() == 64 && scores.len() >= n_kv);
        assert!(
            k.data.len() >= (n_kv - 1) * k.stride + (h + 1) * 64
                && v.data.len() >= (n_kv - 1) * v.stride + (h + 1) * 64
        );
        // Safety: bounds asserted above.
        return unsafe { attend64(out, q, k, v, h, n_kv, scores) };
    }
    fastops::attend_one(out, q, k, v, h, hd, n_kv, scores)
}

/// Same contract as [`super::reference::attention`]; (query, head) units split
/// over the fast pool as in stage 11.
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
    let ptr = OutPtr(out.as_mut_ptr());
    let unit = |u: usize, scratch: &mut [f32]| {
        let (i, h) = (u / n_head, u % n_head);
        let p = ptr;
        // Safety: each unit writes its own disjoint head_dim slice of `out`.
        let o = unsafe { std::slice::from_raw_parts_mut(p.0.add(i * c + h * head_dim), head_dim) };
        attend(o, q.head(i, h, head_dim), k, v, h, head_dim, q_pos0 + i + 1, scratch);
    };
    let threads = lowlat::num_threads();
    if threads == 1 || units == 1 {
        return (0..units).for_each(|u| unit(u, scores));
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_attention() {
        for (n_q, pos0) in [(1, 0), (1, 37), (3, 10)] {
            let (n_head, hd) = (3, 64);
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
            assert!(d < 1e-5, "n_q={n_q} pos0={pos0}: max diff {d}");
        }
    }
}
