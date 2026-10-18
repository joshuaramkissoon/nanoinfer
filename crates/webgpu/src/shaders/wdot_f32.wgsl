// j-th vector load of `row` against x (X is array<vec4<f32>>, xb in floats).
fn wdot(row: u32, j: u32, xb: u32) -> f32 {
  return dot(W[row * (P.cols / 4u) + j], X[xb / 4u + j]);
}
