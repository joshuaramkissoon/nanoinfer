// One thread per output element, serial dot product: the literal port.
@group(0) @binding(0) var<storage, read> X: array<f32>;

@compute @workgroup_size(64)
fn matmul_naive(@builtin(global_invocation_id) g: vec3<u32>) {
  let o = g.x;
  let t = g.y;
  if (o >= P.rows) { return; }
  let xt = select(t, S.n - 1u, P.x_last != 0u);
  var acc = 0.0;
  for (var k = 0u; k < P.cols; k++) { acc += X[xt * P.cols + k] * wget(o, k); }
  epilogue(t, o, acc);
}
