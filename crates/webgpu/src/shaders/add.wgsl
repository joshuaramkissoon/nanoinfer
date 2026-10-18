struct Q { len: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32 }
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> R: array<f32>;
@group(0) @binding(2) var<uniform> P: Q;
@compute @workgroup_size(256)
fn add_inplace(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= S.n * P.len) { return; }
  X[g.x] = X[g.x] + R[g.x];
}
