// 8 int8 weights per load; unpack4x8snorm gives q/127 exactly for q in [-127, 127].
fn wdot(row: u32, j: u32, xb: u32) -> f32 {
  let w = W[row * (P.cols / 8u) + j];
  let x0 = X[xb / 4u + 2u * j];
  let x1 = X[xb / 4u + 2u * j + 1u];
  let d = qscale((row * P.cols) / 32u + j / 4u) * 127.0;
  return (dot(unpack4x8snorm(w.x), x0) + dot(unpack4x8snorm(w.y), x1)) * d;
}
