// webgpu-vec prefill: matmul_tiled with vector traffic. Same 64x64 output
// tile, 16-deep K steps and 4x4 register blocks, but
//  - staging: each thread loads ONE vec4 of X (4 k's of a token row) and ONE
//    wget4 (4 k's of a weight row) and stores each as a whole vec4, not 4
//    scalar X loads + 4 scalar wgets;
//  - tiles are row-major with k in vec4s ([64 rows][4 vec4], padded to 5 to
//    spread banks), so per 4 k's a thread reads 8 vec4s and does 16 vec4
//    dots: 4x fewer workgroup accesses than the scalar kernel.
// (A [k][row] vec4 layout needs component writes from different threads into
// one vec4, which races: stores are whole-vector.)
// Fewer, wider accesses also mean fewer of the bounds checks robust buffer
// access adds to every dynamic index.
@group(0) @binding(0) var<storage, read> X4: array<vec4<f32>>;
const RS: u32 = 5u; // vec4s per tile row (4 + 1 pad)
var<workgroup> As: array<vec4<f32>, 320>; // [64 token rows][RS]
var<workgroup> Bs: array<vec4<f32>, 320>; // [64 weight rows][RS]

@compute @workgroup_size(256)
fn matmul_tiled4(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let o0 = wg.x * 64u;
  let t0 = wg.y * 64u;
  let tx = li % 16u;
  let ty = li / 16u;
  var c0 = vec4<f32>(0.0);
  var c1 = vec4<f32>(0.0);
  var c2 = vec4<f32>(0.0);
  var c3 = vec4<f32>(0.0);
  // staging role: tile row r, k group kq (4 k's)
  let r = li / 4u;
  let kq = li % 4u;
  let cols4 = P.cols / 4u;
  for (var k0 = 0u; k0 < P.cols; k0 += 16u) {
    let t = t0 + r;
    var a = vec4<f32>(0.0);
    if (t < S.n) { a = X4[t * cols4 + k0 / 4u + kq]; }
    let o = o0 + r;
    var b = vec4<f32>(0.0);
    if (o < P.rows) { b = wget4(o, k0 + 4u * kq); }
    As[r * RS + kq] = a;
    Bs[r * RS + kq] = b;
    workgroupBarrier();
    for (var q = 0u; q < 4u; q++) {
      let a0 = As[(4u * ty) * RS + q];
      let a1 = As[(4u * ty + 1u) * RS + q];
      let a2 = As[(4u * ty + 2u) * RS + q];
      let a3 = As[(4u * ty + 3u) * RS + q];
      let b0 = Bs[(4u * tx) * RS + q];
      let b1 = Bs[(4u * tx + 1u) * RS + q];
      let b2 = Bs[(4u * tx + 2u) * RS + q];
      let b3 = Bs[(4u * tx + 3u) * RS + q];
      c0 += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      c1 += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      c2 += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      c3 += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
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
