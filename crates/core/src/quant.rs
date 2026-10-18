//! Weight formats smaller than f32: f16, and ggml-style block quantisation.
//!
//! Decode reads every weight once per token, so tok/s ≈ bandwidth ÷ bytes. Halve
//! the bytes, double the ceiling. The formats here, all *portable* (no SIMD;
//! the kernels that read them live in `kernels/`):
//!
//! - **f16**: IEEE half precision, 2 bytes. 11 bits of mantissa (f32 has 24).
//!   Converted with round-to-nearest-even, accumulated in f32.
//! - **Q8_0**: blocks of 32 weights share one f32 scale `d = max|w| / 127`; each
//!   weight is `round(w / d)` as an `i8`. 36 bytes per 32 weights (9 bits each).
//!   The per-block scale is the trick: an outlier only costs precision in its own
//!   block of 32, not the whole row.
//! - **Q4_0**: same idea with 4-bit values: `d = max / -8` (sign of the
//!   largest-magnitude weight, so it maps to exactly -8), `q = round(w / d) + 8`
//!   clamped to 0..=15, two per byte: the low nibbles hold weights 0–15 of the
//!   block, the high nibbles 16–31 (ggml's layout). 20 bytes per 32 weights.
//!
//! (ggml stores the scale as f16; we keep f32 so the kernels don't convert scales.
//! It costs 0.5 bits per weight.)

/// Weights per quantisation block.
pub const QK: usize = 32;

/// f32 → IEEE f16 bits, round to nearest even (handles subnormals, inf, NaN).
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        // inf / NaN (keep NaN quiet)
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        // Subnormal f16 (or zero): shift the full mantissa (with implicit 1) right.
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rounded = (m + half - 1 + ((m >> shift) & 1)) >> shift;
        return sign | rounded as u16;
    }
    // Normal: keep 10 mantissa bits, round to nearest even on the 13 dropped bits.
    let v = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let v = if rem > 0x1000 || (rem == 0x1000 && (v & 1) == 1) { v + 1 } else { v };
    sign | v as u16 // a carry into the exponent is the correct rounding (up to inf)
}

/// IEEE f16 bits → f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // subnormal: renormalise
            let lz = man.leading_zeros() - 22; // zeros within the 10-bit field
            let man = (man << (lz + 1)) & 0x3ff;
            sign | ((127 - 15 - lz) << 23) | (man << 13)
        }
        0x1f => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// One Q8_0 block: 32 × int8 and their shared scale.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BlockQ8 {
    pub d: f32,
    pub qs: [i8; QK],
}

/// One Q4_0 block: 32 × 4-bit (offset by 8) and their shared scale.
/// `qs[j] & 0xf` is weight `j`, `qs[j] >> 4` is weight `j + 16`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BlockQ4 {
    pub d: f32,
    pub qs: [u8; QK / 2],
}

/// Quantise 32 values to Q8_0.
pub fn quantize_q8(x: &[f32; QK]) -> BlockQ8 {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut qs = [0i8; QK];
    for (q, v) in qs.iter_mut().zip(x) {
        *q = (v * id).round() as i8;
    }
    BlockQ8 { d, qs }
}

/// Quantise 32 values to Q4_0 (ggml's reference rule).
pub fn quantize_q4(x: &[f32; QK]) -> BlockQ4 {
    // The value with the largest magnitude, with its sign, maps to -8.
    let max = x.iter().copied().fold(0f32, |m, v| if v.abs() > m.abs() { v } else { m });
    let d = max / -8.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let q = |v: f32| ((v * id + 8.5) as i32).clamp(0, 15) as u8;
    let mut qs = [0u8; QK / 2];
    for j in 0..QK / 2 {
        qs[j] = q(x[j]) | (q(x[j + QK / 2]) << 4);
    }
    BlockQ4 { d, qs }
}

/// Quantise each row of `w[rows, cols]` (cols a multiple of 32) to Q8_0 blocks,
/// row-major: block `b` of row `r` is at `r * cols/32 + b`.
pub fn quantize_rows_q8(w: &[f32], cols: usize) -> Vec<BlockQ8> {
    assert!(cols.is_multiple_of(QK) && w.len().is_multiple_of(cols));
    w.as_chunks::<QK>().0.iter().map(quantize_q8).collect()
}

/// As [`quantize_rows_q8`], for Q4_0.
pub fn quantize_rows_q4(w: &[f32], cols: usize) -> Vec<BlockQ4> {
    assert!(cols.is_multiple_of(QK) && w.len().is_multiple_of(cols));
    w.as_chunks::<QK>().0.iter().map(quantize_q4).collect()
}

pub fn dequantize_q8(b: &BlockQ8) -> [f32; QK] {
    b.qs.map(|q| q as f32 * b.d)
}

pub fn dequantize_q4(b: &BlockQ4) -> [f32; QK] {
    let mut out = [0f32; QK];
    for j in 0..QK / 2 {
        out[j] = ((b.qs[j] & 0xf) as i32 - 8) as f32 * b.d;
        out[j + QK / 2] = ((b.qs[j] >> 4) as i32 - 8) as f32 * b.d;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip_and_rounding() {
        for &v in &[0.0f32, -0.0, 1.0, -2.5, 65504.0, 6.1035156e-5, 5.9604645e-8, 1e-3, 1.234_567] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert!((back - v).abs() <= v.abs() * 1e-3 + 6e-8, "{v} -> {h:#x} -> {back}");
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(65520.0), 0x7c00); // rounds up to inf
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3c00); // tie -> even
        assert_eq!(f32_to_f16(1.0 + 3.0 / 2048.0), 0x3c02); // tie -> even (up)
                                                            // every f16 survives f16 -> f32 -> f16
        for h in 0..=0xffffu16 {
            let f = f16_to_f32(h);
            if !f.is_nan() {
                assert_eq!(f32_to_f16(f), h, "{h:#x}");
            }
        }
    }

    #[test]
    fn q8_and_q4_errors_are_bounded() {
        let x: [f32; QK] = std::array::from_fn(|i| ((i * 37) % 17) as f32 / 8.0 - 1.0);
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let q8 = dequantize_q8(&quantize_q8(&x));
        assert!(x.iter().zip(&q8).all(|(a, b)| (a - b).abs() <= amax / 127.0 * 0.5 + 1e-6));
        // Q4_0 is asymmetric (-8..=7 steps): -max would need +8 and is clamped to +7,
        // so the worst case is one full step.
        let q4 = dequantize_q4(&quantize_q4(&x));
        assert!(x.iter().zip(&q4).all(|(a, b)| (a - b).abs() <= amax / 8.0 + 1e-6));
    }
}
