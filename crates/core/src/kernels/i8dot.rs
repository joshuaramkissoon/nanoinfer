//! The int8 block dot products behind stages 9+ (Q8_0 / Q4_0 decode), off aarch64.
//!
//! On aarch64 the quantised kernels use `sdot` directly ([`super::q8`], [`super::q4`],
//! [`super::q8hilo`]) and never call this module. Everywhere else they sum each block
//! here, as an exact `i32`, then scale it in f32 in the same order as the scalar
//! reference, so the result is bit-identical to the scalar fallback whatever path ran.
//!
//! * **wasm simd128**: `i16x8_extmul_{low,high}_i8x16` (16 exact i8 × i8 → i16
//!   products) + `i32x4_extadd_pairwise_i16x8`. 32 products in ~10 instructions instead
//!   of 32 scalar multiply-adds (each has a one-instruction arm64 equivalent: `smull`/`smull2`/`saddlp`).
//! * **wasm relaxed-SIMD, Q4 only**: `i32x4_relaxed_dot_i8x16_i7x16_add`. Its second
//!   operand must be in 0..=127 (the "i7"); with the top bit set the result is
//!   implementation-defined (an engine may read it as signed or unsigned). The
//!   raw nibbles `n` (0..=15) are always valid i7, so Σ (n − 8)·x = Σ n·x − Σ 8·x is two
//!   relaxed dots per half-block (the second against a splat of 8), exact everywhere and
//!   no unpack-and-subtract: +13–15% Q4 decode on V8 / M4 over the simd128 path.
//!   Q8 has no relaxed variant: both operands are signed, and making one of them i7
//!   (move x's sign onto w: Σ w·x = Σ (±w)·|x|, valid for −127..=127) costs four extra
//!   ops per 16 bytes. Measured, it was 9% *slower* than extmul on V8 / M4, so the
//!   relaxed build uses the simd128 Q8 path.
//! * anything else: the scalar loop (unchanged from before).
#![cfg_attr(target_arch = "aarch64", allow(dead_code))]

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32::*;

/// Σ_j w[j]·x[j] over one 32-wide block, scalar. The reference every path must equal.
#[inline]
pub fn isum_q8_scalar(w: &[i8; 32], x: &[i8; 32]) -> i32 {
    w.iter().zip(x).map(|(&a, &b)| a as i32 * b as i32).sum()
}

/// Σ_j (nibble_j − 8)·x[j] for a Q4_0 block (`w[j]` holds weight j low, j + 16 high).
#[inline]
pub fn isum_q4_scalar(w: &[u8; 16], x: &[i8; 32]) -> i32 {
    let mut isum = 0i32;
    for j in 0..16 {
        isum += ((w[j] & 0xf) as i32 - 8) * x[j] as i32;
        isum += ((w[j] >> 4) as i32 - 8) * x[j + 16] as i32;
    }
    isum
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn hsum(v: v128) -> i32 {
    // Integer adds are associative: any order gives the exact sum.
    let v = i32x4_add(v, i32x4_shuffle::<2, 3, 0, 1>(v, v));
    let v = i32x4_add(v, i32x4_shuffle::<1, 0, 3, 2>(v, v));
    i32x4_extract_lane::<0>(v)
}

/// Σ w·x over one Q8_0 block (exact).
#[inline(always)]
pub fn isum_q8(w: &[i8; 32], x: &[i8; 32]) -> i32 {
    // (No relaxed variant: see the module docs.)
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    // Safety: 32-byte arrays, unaligned 16-byte loads in bounds.
    unsafe {
        let (w0, w1) = (v128_load(w.as_ptr() as *const v128), v128_load(w.as_ptr().add(16) as *const v128));
        let (x0, x1) = (v128_load(x.as_ptr() as *const v128), v128_load(x.as_ptr().add(16) as *const v128));
        hsum(mul_add_i8(w1, x1, mul_add_i8(w0, x0, i32x4_splat(0))))
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    isum_q8_scalar(w, x)
}

/// Σ (n − 8)·x over one Q4_0 block (exact).
#[inline(always)]
pub fn isum_q4(w: &[u8; 16], x: &[i8; 32]) -> i32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", target_feature = "relaxed-simd"))]
    // Safety: 16- and 32-byte arrays, unaligned loads in bounds.
    unsafe {
        let v = v128_load(w.as_ptr() as *const v128);
        let (x0, x1) = (v128_load(x.as_ptr() as *const v128), v128_load(x.as_ptr().add(16) as *const v128));
        let (lo, hi) = (v128_and(v, u8x16_splat(0x0f)), u8x16_shr(v, 4)); // 0..=15: valid i7
        let eight = i8x16_splat(8);
        let n = i32x4_relaxed_dot_i8x16_i7x16_add(x1, hi, i32x4_relaxed_dot_i8x16_i7x16_add(x0, lo, i32x4_splat(0)));
        let c = i32x4_relaxed_dot_i8x16_i7x16_add(x1, eight, i32x4_relaxed_dot_i8x16_i7x16_add(x0, eight, i32x4_splat(0)));
        hsum(i32x4_sub(n, c))
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(target_feature = "relaxed-simd")))]
    // Safety: as above.
    unsafe {
        let v = v128_load(w.as_ptr() as *const v128);
        let (x0, x1) = (v128_load(x.as_ptr() as *const v128), v128_load(x.as_ptr().add(16) as *const v128));
        let eight = i8x16_splat(8);
        let lo = i8x16_sub(v128_and(v, u8x16_splat(0x0f)), eight);
        let hi = i8x16_sub(u8x16_shr(v, 4), eight);
        hsum(mul_add_i8(hi, x1, mul_add_i8(lo, x0, i32x4_splat(0))))
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    isum_q4_scalar(w, x)
}

/// acc + pairwise sums of a·b (16 exact i16 products → 4 i32 lanes).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn mul_add_i8(a: v128, b: v128, acc: v128) -> v128 {
    let lo = i32x4_extadd_pairwise_i16x8(i16x8_extmul_low_i8x16(a, b));
    let hi = i32x4_extadd_pairwise_i16x8(i16x8_extmul_high_i8x16(a, b));
    i32x4_add(acc, i32x4_add(lo, hi))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64*: deterministic, no dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }
        fn i8(&mut self, lo: i32, hi: i32) -> i8 {
            (lo + (self.next() >> 33) as i32 % (hi - lo + 1)) as i8
        }
    }

    // Every path takes the full i8 range (quantize_q8 itself only emits -127..=127).
    const Q8_MIN: i32 = -128;

    #[test]
    fn int8_block_dots_bit_exact_vs_scalar() {
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        for case in 0..20_000 {
            let mut w = [0i8; 32];
            let mut x = [0i8; 32];
            let mut n = [0u8; 16];
            // Mix uniform data with saturated extremes (worst case for i16 / i32 overflow).
            let ext = case % 4 == 0;
            for j in 0..32 {
                w[j] = if ext { [Q8_MIN, 127][(r.next() & 1) as usize] as i8 } else { r.i8(Q8_MIN, 127) };
                x[j] = if ext { [Q8_MIN, 127][(r.next() & 1) as usize] as i8 } else { r.i8(Q8_MIN, 127) };
            }
            for b in &mut n {
                *b = if ext { [0x00, 0xff, 0x0f, 0xf0][(r.next() & 3) as usize] } else { r.next() as u8 };
            }
            assert_eq!(isum_q8(&w, &x), isum_q8_scalar(&w, &x), "q8 case {case}");
            assert_eq!(isum_q4(&n, &x), isum_q4_scalar(&n, &x), "q4 case {case}");
        }
    }

    #[test]
    fn dot_q8_and_q4_bit_exact_vs_scalar_on_random_rows() {
        use crate::quant::{quantize_q4, quantize_q8, BlockQ4, BlockQ8};
        let mut r = Rng(42);
        let mut f = || (r.next() >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0;
        for nb in [1, 2, 3, 24, 96] {
            let blk = |f: &mut dyn FnMut() -> f32| std::array::from_fn::<f32, 32, _>(|_| f());
            let w8: Vec<BlockQ8> = (0..nb).map(|_| quantize_q8(&blk(&mut f))).collect();
            let w4: Vec<BlockQ4> = (0..nb).map(|_| quantize_q4(&blk(&mut f))).collect();
            let x: Vec<BlockQ8> = (0..nb).map(|_| quantize_q8(&blk(&mut f))).collect();
            let s8: f32 = w8.iter().zip(&x).map(|(a, b)| isum_q8_scalar(&a.qs, &b.qs) as f32 * a.d * b.d).sum();
            let s4: f32 = w4.iter().zip(&x).map(|(a, b)| isum_q4_scalar(&a.qs, &b.qs) as f32 * a.d * b.d).sum();
            // Safety: equal lengths.
            let (g8, g4) = unsafe { (super::super::q8::dot_q8(&w8, &x), super::super::q4::dot_q4(&w4, &x)) };
            if cfg!(target_arch = "aarch64") {
                // NEON accumulates in f32 lanes (different order): close, not identical.
                assert!((g8 - s8).abs() <= 1e-4 * s8.abs().max(1.0), "nb {nb}: {g8} vs {s8}");
                assert!((g4 - s4).abs() <= 1e-4 * s4.abs().max(1.0), "nb {nb}: {g4} vs {s4}");
            } else {
                assert_eq!(g8.to_bits(), s8.to_bits(), "q8 nb {nb}");
                assert_eq!(g4.to_bits(), s4.to_bits(), "q4 nb {nb}");
            }
        }
    }
}
