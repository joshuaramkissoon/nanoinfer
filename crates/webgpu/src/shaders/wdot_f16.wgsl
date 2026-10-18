fn wdot(row: u32, j: u32, xb: u32) -> f32 {
  let w = W[row * (P.cols / 8u) + j];
  let x0 = X[xb / 4u + 2u * j];
  let x1 = X[xb / 4u + 2u * j + 1u];
  return dot(vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)), x0)
       + dot(vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w)), x1);
}
