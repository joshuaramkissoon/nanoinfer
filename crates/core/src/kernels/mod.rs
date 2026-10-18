//! Kernels: the math, behind a swappable interface.
//!
//! ## The rule: never overwrite an old kernel
//!
//! Each optimisation adds a *new* kernel set next to the old ones instead of
//! editing them, so every historical version stays runnable side by side (in
//! benchmarks, tests, and the live blog figures). A kernel set is a zero-sized
//! type implementing [`Kernels`]. Every method has a default that calls the
//! [`reference`] implementation, so a new set overrides only the ops it speeds
//! up:
//!
//! ```ignore
//! pub struct Simd;
//! impl Kernels for Simd {
//!     const NAME: &'static str = "simd";
//!     fn matmul(out: &mut [f32], x: &[f32], w: &[f32], b: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
//!         simd::matmul(out, x, w, b, n, d_in, d_out)
//!     }
//!     // layernorm, gelu, ... inherited from the reference set
//! }
//! ```
//!
//! The forward passes are generic over `K: Kernels`, so the choice is resolved at
//! compile time per stage (no dynamic dispatch in inner loops) while still being
//! selectable at runtime through [`crate::Stage`].

pub mod f16;
pub mod fastops;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod packed;
pub mod reference;
pub mod spin;
pub mod threaded;
pub mod tiled;
pub mod unrolled;

/// Where queries, keys and values live in memory. Attention doesn't care whether
/// K/V were just computed (stage 0: a fused `[n, 3C]` QKV buffer) or come from a
/// cache (stage 1: `[pos, C]`), only how to index them, so we describe each as a
/// slice plus a row stride.
#[derive(Clone, Copy)]
pub struct Rows<'a> {
    pub data: &'a [f32],
    /// Floats between the start of consecutive positions.
    pub stride: usize,
}

impl Rows<'_> {
    /// The `hd` floats of head `h` at position `pos`.
    #[inline]
    pub fn head(&self, pos: usize, h: usize, hd: usize) -> &[f32] {
        &self.data[pos * self.stride + h * hd..][..hd]
    }
}

/// A complete set of ops. See the module docs; see [`reference`] for what each
/// op computes.
#[allow(clippy::too_many_arguments)]
pub trait Kernels {
    /// Short name for traces and bench output.
    const NAME: &'static str;

    fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
        reference::matmul(out, x, w, bias, n, d_in, d_out)
    }
    fn layernorm(out: &mut [f32], x: &[f32], weight: &[f32], bias: &[f32], n: usize, d: usize, eps: f32) {
        reference::layernorm(out, x, weight, bias, n, d, eps)
    }
    fn gelu(x: &mut [f32]) {
        reference::gelu(x)
    }
    fn residual_add(x: &mut [f32], y: &[f32]) {
        reference::residual_add(x, y)
    }
    fn embed(out: &mut [f32], wte: &[f32], wpe: &[f32], tokens: &[u32], pos0: usize, d: usize) {
        reference::embed(out, wte, wpe, tokens, pos0, d)
    }
    fn attention(
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
        reference::attention(out, q, k, v, n_q, q_pos0, n_head, head_dim, scores)
    }
}

/// The scalar reference kernels (stages 0 and 1).
#[derive(Debug, Clone, Copy, Default)]
pub struct Reference;

impl Kernels for Reference {
    const NAME: &'static str = "reference";
}

/// Stage 2: multiple independent accumulators in the matmul (autovectorised).
#[derive(Debug, Clone, Copy, Default)]
pub struct Unrolled;

impl Kernels for Unrolled {
    const NAME: &'static str = "unrolled";
    fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
        unrolled::matmul(out, x, w, bias, n, d_in, d_out)
    }
}

/// Stage 3: explicit NEON matmul (fused FMA, one weight row at a time). Falls back
/// to [`Unrolled`]'s matmul where NEON isn't available (wasm32).
#[derive(Debug, Clone, Copy, Default)]
pub struct Neon;

impl Kernels for Neon {
    const NAME: &'static str = "neon";
    fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
        #[cfg(target_arch = "aarch64")]
        neon::matmul(out, x, w, bias, n, d_in, d_out);
        #[cfg(not(target_arch = "aarch64"))]
        unrolled::matmul(out, x, w, bias, n, d_in, d_out);
    }
}

/// Stage 4: 4×4 register-blocked matmul for prefill (decode uses [`Neon`]'s kernel).
#[derive(Debug, Clone, Copy, Default)]
pub struct Tiled;

impl Kernels for Tiled {
    const NAME: &'static str = "tiled";
    fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
        tiled::matmul(out, x, w, bias, n, d_in, d_out)
    }
}

/// Stage 5: matmuls split across cores (rayon, `threads` feature), over stage 4's kernels.
#[derive(Debug, Clone, Copy, Default)]
pub struct Threaded;

impl Kernels for Threaded {
    const NAME: &'static str = "threaded";
    fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
        threaded::matmul(out, x, w, bias, n, d_in, d_out)
    }
}

/// Stage 6: stage 5's matmul split, dispatched on a persistent spinning pool.
#[derive(Debug, Clone, Copy, Default)]
pub struct Spin;

impl Kernels for Spin {
    const NAME: &'static str = "spin";
    fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
        spin::matmul(out, x, w, bias, n, d_in, d_out)
    }
}

/// Stage 7: stage 6 matmuls + vectorised, parallel GELU / softmax / attention / layernorm.
#[derive(Debug, Clone, Copy, Default)]
pub struct FastOps;

impl Kernels for FastOps {
    const NAME: &'static str = "fast-ops";
    fn matmul(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>, n: usize, d_in: usize, d_out: usize) {
        spin::matmul(out, x, w, bias, n, d_in, d_out)
    }
    fn layernorm(out: &mut [f32], x: &[f32], weight: &[f32], bias: &[f32], n: usize, d: usize, eps: f32) {
        fastops::layernorm(out, x, weight, bias, n, d, eps)
    }
    fn gelu(x: &mut [f32]) {
        fastops::gelu(x)
    }
    fn attention(
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
        fastops::attention(out, q, k, v, n_q, q_pos0, n_head, head_dim, scores)
    }
}
