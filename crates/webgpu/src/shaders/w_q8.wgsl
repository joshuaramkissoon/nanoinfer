// Q8_0: int8 quants packed four per u32 (row-major), plus one f16 scale per
// block of 32 weights (packed two per u32). value = q * d.
@group(0) @binding(1) var<storage, read> W: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> SC: array<u32>;
const WPL: u32 = 8u;
fn qscale(b: u32) -> f32 {
  return unpack2x16float(SC[b / 2u])[b % 2u];
}
fn wget(row: u32, k: u32) -> f32 {
  let i = row * P.cols + k;
  let v = W[i / 8u];
  let u = v[(i / 4u) % 2u];
  let q = extractBits(bitcast<i32>(u), (i % 4u) * 8u, 8u);
  return f32(q) * qscale(i / 32u);
}
