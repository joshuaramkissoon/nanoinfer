struct Q { c: u32, layer_off: u32, cc: u32, d: u32, e: u32, f: u32, g: u32, h: u32 }
@group(0) @binding(0) var<storage, read> QKV: array<f32>;
@group(0) @binding(1) var<storage, read_write> KC: array<f32>;
@group(0) @binding(2) var<storage, read_write> VC: array<f32>;
@group(0) @binding(3) var<uniform> P: Q;
@compute @workgroup_size(256)
fn kv_write(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = P.c;
  let i = g.x;
  if (i >= S.n * c) { return; }
  let t = i / c;
  let k = i % c;
  let dst = P.layer_off + (S.pos0 + t) * c + k;
  KC[dst] = QKV[t * 3u * c + c + k];
  VC[dst] = QKV[t * 3u * c + 2u * c + k];
}
