@group(0) @binding(1) var<storage, read> W: array<vec4<f32>>;
const WPL: u32 = 4u; // weights per vector load
fn wget(row: u32, k: u32) -> f32 {
  let i = row * P.cols + k;
  return W[i / 4u][i % 4u];
}
