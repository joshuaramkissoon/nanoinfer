// Causal self-attention for the n new tokens against this layer's KV cache.
struct Q { c: u32, nh: u32, layer_off: u32, max_ctx: u32, e: u32, f: u32, g: u32, h: u32 }
@group(0) @binding(0) var<storage, read> QKV: array<f32>;
@group(0) @binding(1) var<storage, read> KC: array<f32>;
@group(0) @binding(2) var<storage, read> VC: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<storage, read_write> SCR: array<f32>;
@group(0) @binding(5) var<uniform> P: Q;
const HD: u32 = 64u;
const NEG: f32 = -3.0e38;

// Literal port: one thread per (token, head), scores in device memory.
@compute @workgroup_size(64)
fn attn_naive(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x;
  let h = g.y;
  if (t >= S.n) { return; }
  let c = P.c;
  let L = S.pos0 + t + 1u;
  let qb = t * 3u * c + h * HD;
  let sb = (t * P.nh + h) * P.max_ctx;
  var m = NEG;
  for (var j = 0u; j < L; j++) {
    let kb = P.layer_off + j * c + h * HD;
    var d = 0.0;
    for (var e = 0u; e < HD; e++) { d += QKV[qb + e] * KC[kb + e]; }
    d *= 0.125;
    SCR[sb + j] = d;
    m = max(m, d);
  }
  var s = 0.0;
  for (var j = 0u; j < L; j++) { let e = exp(SCR[sb + j] - m); SCR[sb + j] = e; s += e; }
  for (var e = 0u; e < HD; e++) {
    var a = 0.0;
    for (var j = 0u; j < L; j++) { a += SCR[sb + j] * VC[P.layer_off + j * c + h * HD + e]; }
    O[t * c + h * HD + e] = a / s;
  }
}

var<workgroup> sc: array<f32, 1024>;
var<workgroup> red: array<f32, 256>;

fn wg_reduce(li: u32, v: f32, is_max: bool) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) {
      if (is_max) { red[li] = max(red[li], red[li + s]); } else { red[li] += red[li + s]; }
    }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

// One 256-thread workgroup per (token, head): scores in workgroup memory,
// workgroup max/sum, then 4 groups of 64 threads split the keys for P·V.
@compute @workgroup_size(256)
fn attn_par(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let t = wg.x;
  let h = wg.y;
  let c = P.c;
  let L = S.pos0 + t + 1u;
  let qb = t * 3u * c + h * HD;
  for (var j = li; j < L; j += 256u) {
    let kb = P.layer_off + j * c + h * HD;
    var d = 0.0;
    for (var e = 0u; e < HD; e += 4u) {
      d += dot(vec4<f32>(QKV[qb + e], QKV[qb + e + 1u], QKV[qb + e + 2u], QKV[qb + e + 3u]),
               vec4<f32>(KC[kb + e], KC[kb + e + 1u], KC[kb + e + 2u], KC[kb + e + 3u]));
    }
    sc[j] = d * 0.125;
  }
  workgroupBarrier();
  var m = NEG;
  for (var j = li; j < L; j += 256u) { m = max(m, sc[j]); }
  m = wg_reduce(li, m, true);
  var s = 0.0;
  for (var j = li; j < L; j += 256u) { let e = exp(sc[j] - m); sc[j] = e; s += e; }
  s = wg_reduce(li, s, false); // (its barriers also publish sc)
  let d = li % HD;
  let grp = li / HD;
  var a = 0.0;
  for (var j = grp; j < L; j += 4u) { a += sc[j] * VC[P.layer_off + j * c + h * HD + d]; }
  red[li] = a;
  workgroupBarrier();
  if (li < HD) {
    O[t * c + h * HD + li] = (red[li] + red[li + 64u] + red[li + 128u] + red[li + 192u]) / s;
  }
}
