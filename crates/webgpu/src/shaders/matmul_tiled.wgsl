// Prefill: 64x64 output tiles, K in steps of 16 staged through workgroup
// memory, each of 256 threads accumulates a 4x4 block in registers.
@group(0) @binding(0) var<storage, read> X: array<f32>;
var<workgroup> As: array<f32, 1024>; // [16][64] token rows
var<workgroup> Bs: array<f32, 1024>; // [16][64] weight rows

@compute @workgroup_size(256)
fn matmul_tiled(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let o0 = wg.x * 64u;
  let t0 = wg.y * 64u;
  let tx = li % 16u;
  let ty = li / 16u;
  // four vec4 accumulators, not array<f32, 16>: naga keeps a dynamically
  // indexed local array in (bounds-checked) memory, which made this kernel
  // ~10x slower natively than the same WGSL under Chrome's Tint.
  var c0 = vec4<f32>(0.0);
  var c1 = vec4<f32>(0.0);
  var c2 = vec4<f32>(0.0);
  var c3 = vec4<f32>(0.0);
  for (var k0 = 0u; k0 < P.cols; k0 += 16u) {
    for (var e = 0u; e < 4u; e++) {
      let idx = li + e * 256u;
      let kk = idx % 16u;
      let r = idx / 16u;
      let t = t0 + r;
      var a = 0.0;
      if (t < S.n) { a = X[t * P.cols + k0 + kk]; }
      As[kk * 64u + r] = a;
      let o = o0 + r;
      var b = 0.0;
      if (o < P.rows) { b = wget(o, k0 + kk); }
      Bs[kk * 64u + r] = b;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      let ab = kk * 64u + ty * 4u;
      let bb = kk * 64u + tx * 4u;
      let b = vec4<f32>(Bs[bb], Bs[bb + 1u], Bs[bb + 2u], Bs[bb + 3u]);
      c0 += As[ab] * b;
      c1 += As[ab + 1u] * b;
      c2 += As[ab + 2u] * b;
      c3 += As[ab + 3u] * b;
    }
    workgroupBarrier();
  }
  let rows = array<vec4<f32>, 4>(c0, c1, c2, c3);
  for (var i = 0u; i < 4u; i++) {
    let t = t0 + ty * 4u + i;
    if (t >= S.n) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let o = o0 + tx * 4u + j;
      if (o < P.rows) { epilogue(t, o, rows[i][j]); }
    }
  }
}
