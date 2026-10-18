//! Reference kernels: every op the transformer needs, as a plain function over
//! slices. Scalar f32, one thread, written to be obviously correct rather than
//! fast. Used by stages 0 and 1, and kept forever as the ground truth.
//!
//! Conventions:
//! - Everything is row-major. A "matrix `[n, d]`" is `n` rows of `d` floats.
//! - Weight matrices are `[out, in]` (see [`crate::weights`]).
//! - Functions write into caller-provided output buffers and never allocate,
//!   so the forward pass controls all memory.

use super::Rows;

/// `out[n, d_out] = x[n, d_in] · wᵀ + bias`, with `w` stored `[d_out, d_in]`.
///
/// This is where ~all the time goes. For one token (n = 1) it is a matrix-vector
/// product that reads every weight exactly once and does just 2 FLOPs per weight
/// (one multiply, one add), so it is limited by memory bandwidth, not arithmetic.
/// For many tokens at once (prefill), each weight row is reused `n` times, so
/// arithmetic becomes the limit. That asymmetry is the central fact of LLM
/// inference performance.
///
/// The inner loop is a dot product with a single accumulator. Because float
/// addition is not associative, the compiler is not allowed to reorder it into
/// SIMD lanes, so this runs one fused-ish multiply-add per few cycles. That is
/// intentionally left on the table for a later stage.
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
    debug_assert!(x.len() >= n * d_in && out.len() >= n * d_out && w.len() == d_in * d_out);
    for (x_row, out_row) in x.chunks_exact(d_in).zip(out.chunks_exact_mut(d_out)).take(n) {
        for (o, (y, w_row)) in out_row.iter_mut().zip(w.chunks_exact(d_in)).enumerate() {
            let mut acc = bias.map_or(0.0, |b| b[o]);
            for (a, b) in x_row.iter().zip(w_row) {
                acc += a * b;
            }
            *y = acc;
        }
    }
}

/// LayerNorm over the last dimension: for each row, subtract the mean, divide by
/// the standard deviation, then apply a learned per-channel scale and shift.
/// `out` and `x` are `[n, d]`.
pub fn layernorm(out: &mut [f32], x: &[f32], weight: &[f32], bias: &[f32], n: usize, d: usize, eps: f32) {
    for (x_row, out_row) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)).take(n) {
        let mean = x_row.iter().sum::<f32>() / d as f32;
        let var = x_row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let rstd = 1.0 / (var + eps).sqrt();
        for i in 0..d {
            out_row[i] = (x_row[i] - mean) * rstd * weight[i] + bias[i];
        }
    }
}

/// GELU activation, the tanh approximation GPT-2 was trained with ("gelu_new"):
/// `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))`. In place.
pub fn gelu(x: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    for v in x.iter_mut() {
        let u = *v;
        *v = 0.5 * u * (1.0 + (SQRT_2_OVER_PI * (u + 0.044715 * u * u * u)).tanh());
    }
}

/// Numerically stable softmax, in place. Subtracting the max first keeps `exp`
/// from overflowing and doesn't change the result.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// `x += y`, the residual connection.
pub fn residual_add(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// Token + position embedding: `out[t] = wte[tokens[t]] + wpe[pos0 + t]`.
pub fn embed(out: &mut [f32], wte: &[f32], wpe: &[f32], tokens: &[u32], pos0: usize, d: usize) {
    for (t, (&tok, row)) in tokens.iter().zip(out.chunks_exact_mut(d)).enumerate() {
        let te = &wte[tok as usize * d..][..d];
        let pe = &wpe[(pos0 + t) * d..][..d];
        for i in 0..d {
            row[i] = te[i] + pe[i];
        }
    }
}

/// Causal multi-head self-attention, for `n_q` queries at absolute positions
/// `q_pos0 .. q_pos0 + n_q`. Query at position `p` attends to keys/values at
/// positions `0..=p` (the causal mask is just "don't look past yourself").
///
/// For each head `h` and query `i`:
/// ```text
/// scores[j] = q_i · k_j / √head_dim      for j in 0..=p
/// probs     = softmax(scores)
/// out_i     = Σ_j probs[j] · v_j
/// ```
/// `out` is `[n_q, C]` with heads concatenated, ready for the output projection.
/// `scores` is scratch space of at least `q_pos0 + n_q` floats.
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
    let scale = 1.0 / (head_dim as f32).sqrt();
    for i in 0..n_q {
        let n_kv = q_pos0 + i + 1; // causal: keys 0..=p
        for h in 0..n_head {
            let qh = q.head(i, h, head_dim);
            let att = &mut scores[..n_kv];
            for (j, s) in att.iter_mut().enumerate() {
                let kh = k.head(j, h, head_dim);
                *s = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            softmax(att);
            let oh = &mut out[i * c + h * head_dim..][..head_dim];
            oh.fill(0.0);
            for (j, &p) in att.iter().enumerate() {
                let vh = v.head(j, h, head_dim);
                for (o, &vv) in oh.iter_mut().zip(vh) {
                    *o += p * vv;
                }
            }
        }
    }
}

/// The attention probabilities for a single (query, head): `softmax(q·Kᵀ/√d)`
/// over keys `0..n_kv`. The fast path never materialises this for inspection;
/// we recompute it only when a [`crate::trace::Trace`] asks for an attention map.
pub fn attention_probs(out: &mut [f32], q_head: &[f32], k: Rows, h: usize, head_dim: usize, n_kv: usize) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    for (j, s) in out[..n_kv].iter_mut().enumerate() {
        *s = q_head.iter().zip(k.head(j, h, head_dim)).map(|(a, b)| a * b).sum::<f32>() * scale;
    }
    softmax(&mut out[..n_kv]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_small() {
        // x = [[1,2],[3,4]], w = [[1,0],[0,1],[1,1]] ([out=3,in=2]), b = [0,0,10]
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let b = [0.0, 0.0, 10.0];
        let mut out = [0.0; 6];
        matmul(&mut out, &x, &w, Some(&b), 2, 2, 3);
        assert_eq!(out, [1.0, 2.0, 13.0, 3.0, 4.0, 17.0]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0, 2.0, 3.0, 1000.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[3] > 0.999);
    }

    #[test]
    fn layernorm_normalises() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0; 4];
        layernorm(&mut out, &x, &[1.0; 4], &[0.0; 4], 1, 4, 1e-5);
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        let var: f32 = out.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-6 && (var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn attention_single_key_copies_value() {
        // With one position the only key gets probability 1, so out == v.
        let q = [0.3, -0.2];
        let k = [0.5, 0.5];
        let v = [7.0, 8.0];
        let mut out = [0.0; 2];
        let mut s = [0.0; 1];
        let r = |d| Rows { data: d, stride: 2 };
        attention(&mut out, r(&q), r(&k), r(&v), 1, 0, 1, 2, &mut s);
        assert_eq!(out, v);
    }
}
