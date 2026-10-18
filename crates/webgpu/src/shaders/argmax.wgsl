// Greedy pick on the GPU: partial (value, index) per workgroup, then one
// workgroup reduces them and writes the token to gen[pos0 + n] (where the
// next forward's embed reads it). Ties go to the lowest index, like core.
struct Q { n: u32, groups: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32 }
@group(0) @binding(0) var<storage, read> LG: array<f32>;
@group(0) @binding(1) var<storage, read_write> PV: array<f32>;
@group(0) @binding(2) var<storage, read_write> PI: array<u32>;
@group(0) @binding(3) var<storage, read_write> GEN: array<u32>;
@group(0) @binding(4) var<uniform> P: Q;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;

fn better(v: f32, i: u32, w: f32, j: u32) -> bool {
  return v > w || (v == w && i < j);
}

fn reduce(li: u32) {
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s && better(rv[li + s], ri[li + s], rv[li], ri[li])) {
      rv[li] = rv[li + s];
      ri[li] = ri[li + s];
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(256)
fn argmax_partial(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  var bv = -3.0e38;
  var bi = 0xffffffffu;
  for (var i = wg.x * 256u + li; i < P.n; i += P.groups * 256u) {
    let v = LG[i];
    if (v > bv) { bv = v; bi = i; }
  }
  rv[li] = bv;
  ri[li] = bi;
  reduce(li);
  if (li == 0u) { PV[wg.x] = rv[0]; PI[wg.x] = ri[0]; }
}

@compute @workgroup_size(256)
fn argmax_final(@builtin(local_invocation_index) li: u32) {
  var bv = -3.0e38;
  var bi = 0xffffffffu;
  if (li < P.groups) { bv = PV[li]; bi = PI[li]; }
  rv[li] = bv;
  ri[li] = bi;
  reduce(li);
  if (li == 0u) { GEN[S.pos0 + S.n] = ri[0]; }
}
