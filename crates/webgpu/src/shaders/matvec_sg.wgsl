// Same, reduced with subgroupAdd (assumes subgroup size divides TPR).
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

@compute @workgroup_size(256)
fn matvec_sg(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>,
             @builtin(subgroup_invocation_id) sl: u32, @builtin(subgroup_size) ss: u32) {
  let row = wg.x * (WG / TPR) + li / TPR;
  let s = subgroupAdd(row_partial(li, wg));
  if (sl == 0u) { red[li / ss] = s; }
  workgroupBarrier();
  if (li % TPR == 0u && row < P.rows) {
    var tot = 0.0;
    for (var i = 0u; i < TPR / ss; i++) { tot += red[li / ss + i]; }
    epilogue(wg.y, row, tot);
  }
}
