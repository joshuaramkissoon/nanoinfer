struct Q { c: u32, b: u32, cc: u32, d: u32, e: u32, f: u32, g: u32, h: u32 }
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> LW: array<f32>;
@group(0) @binding(2) var<storage, read> LB: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@group(0) @binding(4) var<uniform> P: Q;
var<workgroup> red: array<f32, 256>;

// Literal port: one thread per row, two serial passes.
@compute @workgroup_size(64)
fn layernorm_naive(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x;
  if (t >= S.n) { return; }
  let c = P.c;
  let base = t * c;
  var s = 0.0;
  for (var i = 0u; i < c; i++) { s += X[base + i]; }
  let mean = s / f32(c);
  var v = 0.0;
  for (var i = 0u; i < c; i++) { let d = X[base + i] - mean; v += d * d; }
  let rstd = 1.0 / sqrt(v / f32(c) + 1e-5);
  for (var i = 0u; i < c; i++) { Y[base + i] = (X[base + i] - mean) * rstd * LW[i] + LB[i]; }
}

fn wg_sum(li: u32, v: f32) -> f32 {
  red[li] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

// One 256-thread workgroup per row; same two-pass maths.
@compute @workgroup_size(256)
fn layernorm_par(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let c = P.c;
  let base = wg.x * c;
  var s = 0.0;
  for (var i = li; i < c; i += 256u) { s += X[base + i]; }
  let mean = wg_sum(li, s) / f32(c);
  var v = 0.0;
  for (var i = li; i < c; i += 256u) { let d = X[base + i] - mean; v += d * d; }
  let rstd = 1.0 / sqrt(wg_sum(li, v) / f32(c) + 1e-5);
  for (var i = li; i < c; i += 256u) { Y[base + i] = (X[base + i] - mean) * rstd * LW[i] + LB[i]; }
}

var<workgroup> red64: array<f32, 64>;

fn sum64(li: u32, v: f32) -> f32 {
  red64[li] = v;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) { red64[li] += red64[li + s]; }
    workgroupBarrier();
  }
  let r = red64[0];
  workgroupBarrier();
  return r;
}

// 64 threads per row, 12 elements each read as 3 vec4s held in registers:
// x is read from memory once, and each reduction is 6 barrier steps, not 8.
@compute @workgroup_size(64)
fn layernorm_wg64(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let c = P.c;
  let base = wg.x * c;
  // c = 768 = 64 threads x 3 vec4 (GPT-2 small); other widths use the strided loop
  let i0 = base + 4u * li;
  let i1 = i0 + 256u;
  let i2 = i0 + 512u;
  let a = vec4<f32>(X[i0], X[i0 + 1u], X[i0 + 2u], X[i0 + 3u]);
  let b = vec4<f32>(X[i1], X[i1 + 1u], X[i1 + 2u], X[i1 + 3u]);
  let d = vec4<f32>(X[i2], X[i2 + 1u], X[i2 + 2u], X[i2 + 3u]);
  let mean = sum64(li, dot(a + b + d, vec4<f32>(1.0))) / f32(c);
  let da = a - mean;
  let db = b - mean;
  let dd = d - mean;
  let rstd = 1.0 / sqrt(sum64(li, dot(da, da) + dot(db, db) + dot(dd, dd)) / f32(c) + 1e-5);
  for (var k = 0u; k < 4u; k++) {
    let j0 = 4u * li + k;
    Y[base + j0] = da[k] * rstd * LW[j0] + LB[j0];
    Y[base + j0 + 256u] = db[k] * rstd * LW[j0 + 256u] + LB[j0 + 256u];
    Y[base + j0 + 512u] = dd[k] * rstd * LW[j0 + 512u] + LB[j0 + 512u];
  }
}
