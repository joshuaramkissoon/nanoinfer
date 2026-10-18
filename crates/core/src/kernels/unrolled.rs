//! Stage 2 kernels: break the FMA dependency chain with independent accumulators.
//!
//! The reference dot product is `acc += a[i] * b[i]` with ONE accumulator. Every
//! add has to wait for the previous one to finish (~4 cycles of FMA latency on
//! the M4), and the compiler may not reorder the sum (float addition is not
//! associative, and Rust has no fast-math), so it can't split it into SIMD
//! lanes either. The core sits mostly idle, waiting on itself.
//!
//! Fix: keep 16 *separate* partial sums. `acc[j]` only ever sees elements
//! `i ≡ j (mod 16)`, so the 16 chains are independent, and we choose the
//! summation order explicitly. Nothing is being reordered behind our back, which
//! means LLVM is now *allowed* to put `acc[0..4]` in one 128-bit NEON register,
//! `acc[4..8]` in another, and so on: the scalar code below compiles to four
//! vector FMAs per 16 elements (see the LOG for the disassembly). Same Rust, no
//! intrinsics, portable to wasm simd128.
//!
//! The result differs from the reference in the last bits (different summation
//! order), well inside the test tolerance.

/// Independent partial sums per dot product. 16 = 4 NEON registers of 4 lanes:
/// enough independent FMAs in flight to cover 4-cycle latency on ~4 FMA pipes.
pub const LANES: usize = 16;

/// Dot product of two equal-length slices with [`LANES`] independent accumulators.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; LANES];
    let (ac, ar) = a.as_chunks::<LANES>();
    let (bc, br) = b.as_chunks::<LANES>();
    for (x, y) in ac.iter().zip(bc) {
        for j in 0..LANES {
            acc[j] += x[j] * y[j];
        }
    }
    // Pairwise tree reduction of the partial sums (also vectorises).
    let mut width = LANES / 2;
    while width > 0 {
        for j in 0..width {
            acc[j] += acc[j + width];
        }
        width /= 2;
    }
    let mut s = acc[0];
    for (x, y) in ar.iter().zip(br) {
        s += x * y;
    }
    s
}

/// Same contract as [`super::reference::matmul`], with the multi-accumulator dot.
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
    debug_assert!(x.len() >= n * d_in && out.len() >= n * d_out && w.len() == d_in * d_out);
    for (x_row, out_row) in x.chunks_exact(d_in).zip(out.chunks_exact_mut(d_out)).take(n) {
        for (o, (y, w_row)) in out_row.iter_mut().zip(w.chunks_exact(d_in)).enumerate() {
            *y = dot(x_row, w_row) + bias.map_or(0.0, |b| b[o]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_scalar_with_remainder() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.5 - 3.0).collect();
        let b: Vec<f32> = (0..37).map(|i| (i % 7) as f32 - 2.0).collect();
        let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - want).abs() < 1e-3);
    }
}
