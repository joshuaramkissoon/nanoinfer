// Decode matvec: TPR threads share each row (coalesced vector loads, a few
// in flight per thread), 256 / TPR rows per workgroup, workgroup_id.y = token.
@group(0) @binding(0) var<storage, read> X: array<vec4<f32>>;
override TPR: u32 = 64u;
const WG: u32 = 256u;
var<workgroup> red: array<f32, 256>;

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

// Reduction through workgroup memory (works everywhere).
@compute @workgroup_size(256)
fn matvec(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let lane = li % TPR;
  let row = wg.x * (WG / TPR) + li / TPR;
  red[li] = row_partial(li, wg);
  workgroupBarrier();
  for (var s = TPR / 2u; s > 0u; s >>= 1u) {
    if (lane < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  if (lane == 0u && row < P.rows) { epilogue(wg.y, row, red[li]); }
}
