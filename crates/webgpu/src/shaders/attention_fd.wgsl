// webgpu-flashdec: decode attention split over the keys (flash-decoding).
// attn_fd_split: one 64-thread workgroup per (chunk of 64 keys, head):
// scores (q in registers, vec4 keys), chunk max, exp, P·V on vec4s, and a
// partial (o[64], m, s) written to SCR. attn_fd_merge: one workgroup per
// head rescales the partials by exp(m_c - M) and sums them. Workgroups past
// the context exit at once, so the grid can be sized for the session's
// max length and reused by every step of a GPU-resident submit.
struct Q { c: u32, nh: u32, layer_off: u32, max_ctx: u32, e: u32, f: u32, g: u32, h: u32 }
@group(0) @binding(0) var<storage, read> QKV4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> KC4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> VC4: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> O4: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> SCR: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> P: Q;
const HD4: u32 = 16u;
const CH: u32 = 64u;
const NEG: f32 = -3.0e38;
// per (head, chunk): 16 vec4 of o, then one vec4 (m, s, 0, 0)
const PART: u32 = 17u;

var<workgroup> sc: array<f32, 64>;
var<workgroup> red: array<f32, 64>;
var<workgroup> red4: array<vec4<f32>, 64>;

@compute @workgroup_size(64)
fn attn_fd_split(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let chunk = wg.x;
  let h = wg.y;
  let L = S.pos0 + 1u;
  let j0 = chunk * CH;
  if (j0 >= L) { return; }
  let c4 = P.c / 4u;
  let lo4 = P.layer_off / 4u;
  let qb = h * HD4;
  var q: array<vec4<f32>, 16>;
  for (var e = 0u; e < HD4; e++) { q[e] = QKV4[qb + e]; }
  let j = j0 + li;
  var d = NEG;
  if (j < L) {
    let kb = lo4 + j * c4 + h * HD4;
    var acc = 0.0;
    for (var e = 0u; e < HD4; e++) { acc += dot(q[e], KC4[kb + e]); }
    d = acc * 0.125;
  }
  red[li] = d;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] = max(red[li], red[li + s]); }
    workgroupBarrier();
  }
  let m = red[0];
  sc[li] = select(0.0, exp(d - m), j < L);
  workgroupBarrier();
  // P·V: thread = (4-dim group dg, key group kg of 4); 16 keys each
  let dg = li % HD4;
  let kg = li / HD4;
  let nk = min(CH, L - j0);
  var a = vec4<f32>(0.0);
  var se = 0.0;
  for (var k = kg; k < nk; k += 4u) {
    let e = sc[k];
    a += e * VC4[lo4 + (j0 + k) * c4 + h * HD4 + dg];
    se += e;
  }
  red4[li] = a;
  workgroupBarrier();
  let pb = (h * P.max_ctx + chunk) * PART;
  if (li < HD4) {
    SCR[pb + li] = red4[li] + red4[li + 16u] + red4[li + 32u] + red4[li + 48u];
  }
  // se is the same for every dg of a key group: sum the 4 groups' first lanes
  red[li] = se;
  workgroupBarrier();
  if (li == 0u) {
    SCR[pb + 16u] = vec4<f32>(m, red[0] + red[16] + red[32] + red[48], 0.0, 0.0);
  }
}

// One 16-thread workgroup per head, thread = 4 output dims.
@compute @workgroup_size(16)
fn attn_fd_merge(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let h = wg.y;
  let L = S.pos0 + 1u;
  let nc = (L + CH - 1u) / CH;
  let base = h * P.max_ctx * PART;
  var mx = NEG;
  for (var ch = 0u; ch < nc; ch++) { mx = max(mx, SCR[base + ch * PART + 16u].x); }
  var o = vec4<f32>(0.0);
  var s = 0.0;
  for (var ch = 0u; ch < nc; ch++) {
    let ms = SCR[base + ch * PART + 16u];
    let w = exp(ms.x - mx);
    o += w * SCR[base + ch * PART + li];
    s += w * ms.y;
  }
  O4[h * HD4 + li] = o / s;
}
