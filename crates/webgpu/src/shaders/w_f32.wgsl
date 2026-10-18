@group(0) @binding(1) var<storage, read> W: array<vec4<f32>>;
const WPL: u32 = 4u; // weights per vector load
fn wget(row: u32, k: u32) -> f32 {
  let i = row * P.cols + k;
  return W[i / 4u][i % 4u];
}
// 4 consecutive weights of a row (k a multiple of 4).
fn wget4(row: u32, k: u32) -> vec4<f32> {
  return W[(row * P.cols + k) / 4u];
}
