// webgpu-kv16: K and V rows of the qkv buffer into the f16 cache, two
// values per u32 (pack2x16float). layer_off is in values, as for f32.
struct Q { c: u32, layer_off: u32, cc: u32, d: u32, e: u32, f: u32, g: u32, h: u32 }
@group(0) @binding(0) var<storage, read> QKV: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> KC: array<u32>;
@group(0) @binding(2) var<storage, read_write> VC: array<u32>;
@group(0) @binding(3) var<uniform> P: Q;
@compute @workgroup_size(256)
fn kv_write16(@builtin(global_invocation_id) g: vec3<u32>) {
  let c2 = P.c / 2u;
  let i = g.x;
  if (i >= S.n * c2) { return; }
  let t = i / c2;
  let k = i % c2;
  let dst = P.layer_off / 2u + (S.pos0 + t) * c2 + k;
  KC[dst] = pack2x16float(QKV[t * 3u * c2 + c2 + k]);
  VC[dst] = pack2x16float(QKV[t * 3u * c2 + 2u * c2 + k]);
}
