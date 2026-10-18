// Decode matvec with the LayerNorm folded in (webgpu-lnfold). The LN gain is
// already multiplied into W and its bias into B (weights.rs::fold_ln), so
//   y = W'·x̂ + b' with x̂ = (x - mean) * rstd
//     = rstd * (W'·x - mean * rowsum(W')) + b'.
// X is the raw residual stream. The threads of the first row of every
// workgroup also sum x and x² (over the same vec4s), reduced in the same
// barrier steps as the dot products: no LayerNorm kernel, no xn buffer, and
// no workgroup re-normalises x before its first weight load. The row sums
// sit after the biases in B (B[rows + o]).
@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
override TPR: u32 = 64u;
const WG: u32 = 256u;
var<workgroup> red: array<f32, 256>;
var<workgroup> r1: array<f32, 64>;
var<workgroup> r2: array<f32, 64>;

fn row_partial(li: u32, wg: vec3<u32>) -> f32 {
  let row = wg.x * (WG / TPR) + li / TPR;
  let xt = select(wg.y, S.n - 1u, P.x_last != 0u);
  var acc = 0.0;
  if (row < P.rows) {
    let nl = P.cols / WPL;
    for (var j = li % TPR; j < nl; j += TPR) { acc += wdot(row, j, xt * P.cols); }
  }
  return acc;
}

@compute @workgroup_size(256)
fn matvec_ln(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let lane = li % TPR;
  let row = wg.x * (WG / TPR) + li / TPR;
  red[li] = row_partial(li, wg);
  if (li < TPR) {
    let xt = select(wg.y, S.n - 1u, P.x_last != 0u);
    let base = xt * P.cols / 4u;
    var s1 = 0.0;
    var s2 = 0.0;
    for (var j = lane; j < P.cols / 4u; j += TPR) {
      let v = X[base + j];
      s1 += dot(v, vec4<f32>(1.0));
      s2 += dot(v, v);
    }
    r1[li] = s1;
    r2[li] = s2;
  }
  workgroupBarrier();
  for (var s = TPR / 2u; s > 0u; s >>= 1u) {
    if (lane < s) {
      red[li] += red[li + s];
      if (li < TPR) {
        r1[li] += r1[li + s];
        r2[li] += r2[li + s];
      }
    }
    workgroupBarrier();
  }
  if (lane == 0u && row < P.rows) {
    let inv_c = 1.0 / f32(P.cols);
    let mean = r1[0] * inv_c;
    let var_ = max(r2[0] * inv_c - mean * mean, 0.0);
    let rstd = inverseSqrt(var_ + 1e-5);
    epilogue(wg.y, row, rstd * (red[li] - mean * B[P.rows + row]));
  }
}
