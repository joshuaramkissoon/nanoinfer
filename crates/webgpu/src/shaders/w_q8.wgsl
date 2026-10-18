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
// 4 consecutive weights of a row (k a multiple of 4): one u32 of int8s.
fn wget4(row: u32, k: u32) -> vec4<f32> {
  let i = row * P.cols + k;
  let u = W[i / 8u][(i / 4u) % 2u];
  return unpack4x8snorm(u) * (127.0 * qscale(i / 32u));
}
