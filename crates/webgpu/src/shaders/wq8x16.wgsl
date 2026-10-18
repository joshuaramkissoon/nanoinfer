// Q8_0 for the decode matvecs, 16 int8 weights per load (one vec4<u32>, half
// a block, one scale): twice the bytes per load of w_q8 + wdot_q8 and half
// the scale fetches per weight.
@group(0) @binding(1) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> SC: array<u32>;
const WPL: u32 = 16u;
fn qscale(b: u32) -> f32 {
  return unpack2x16float(SC[b / 2u])[b % 2u];
}
fn wdot(row: u32, j: u32, xb: u32) -> f32 {
  let w = W[row * (P.cols / 16u) + j];
  let xi = xb / 4u + 4u * j;
  let d = qscale((row * P.cols) / 32u + j / 2u) * 127.0;
  return (dot(unpack4x8snorm(w.x), X[xi]) + dot(unpack4x8snorm(w.y), X[xi + 1u])
        + dot(unpack4x8snorm(w.z), X[xi + 2u]) + dot(unpack4x8snorm(w.w), X[xi + 3u])) * d;
}
