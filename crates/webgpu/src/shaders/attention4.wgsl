// webgpu-attn: attn_par with vector loads. The same buffers, viewed as
// vec4s (every head slice starts at a multiple of 4 floats): a key's score is
// 16 vec4 dots with q held in registers (not 64 scalar loads of q and k),
// and P·V gives each thread 4 dims of 1/16 of the keys (16 vec4 loads in
// flight per thread, not 64 dependent scalar ones), then one 16-way sum;
// the softmax denominator falls out of the same pass (12 barriers, not 20).
// Used for prefill too when the KV cache is f16 (webgpu-kv16).
struct Q { c: u32, nh: u32, layer_off: u32, max_ctx: u32, e: u32, f: u32, g: u32, h: u32 }
@group(0) @binding(0) var<storage, read> QKV4: array<vec4<f32>>;
// KC/VC bindings 1, 2 and the loaders kld/vld (4 values at a vec4 index)
// come from kv_f32.wgsl or kv_f16.wgsl.
@group(0) @binding(3) var<storage, read_write> O4: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> SCR: array<f32>;
@group(0) @binding(5) var<uniform> P: Q;
const HD: u32 = 64u;
const HD4: u32 = 16u;
const NEG: f32 = -3.0e38;

var<workgroup> sc: array<f32, 1024>;
var<workgroup> red: array<f32, 256>;
var<workgroup> red4: array<vec4<f32>, 256>;

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

@compute @workgroup_size(256)
fn attn_par4(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let t = wg.x;
  let h = wg.y;
  let c4 = P.c / 4u;
  let L = S.pos0 + t + 1u;
  let qb = t * 3u * c4 + h * HD4;
  var q: array<vec4<f32>, 16>;
  for (var e = 0u; e < HD4; e++) { q[e] = QKV4[qb + e]; }
  let lo4 = P.layer_off / 4u;
  for (var j = li; j < L; j += 256u) {
    let kb = lo4 + j * c4 + h * HD4;
    var d = 0.0;
    for (var e = 0u; e < HD4; e++) { d += dot(q[e], kld(kb + e)); }
    sc[j] = d * 0.125;
  }
  workgroupBarrier();
  var m = NEG;
  for (var j = li; j < L; j += 256u) { m = max(m, sc[j]); }
  m = wg_reduce(li, m, true);
  for (var j = li; j < L; j += 256u) { sc[j] = exp(sc[j] - m); }
  workgroupBarrier();
  // The softmax sum comes out of P·V: each thread also sums the weights of
  // its keys (one reduction and 8 barriers fewer than a separate sum).
  let dg = li % HD4;
  let kg = li / HD4;
  var a = vec4<f32>(0.0);
  var se = 0.0;
  for (var j = kg; j < L; j += 16u) {
    let e = sc[j];
    a += e * vld(lo4 + j * c4 + h * HD4 + dg);
    se += e;
  }
  red4[li] = a;
  red[li] = se;
  workgroupBarrier();
  if (li < HD4) {
    var o = vec4<f32>(0.0);
    var s = 0.0;
    for (var g = 0u; g < 16u; g++) {
      o += red4[g * HD4 + li];
      s += red[g * HD4];
    }
    O4[t * c4 + h * HD4 + li] = o / s;
  }
}
