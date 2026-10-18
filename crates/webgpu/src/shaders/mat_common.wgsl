// Bindings + epilogue shared by every matrix kernel (matvec, naive matmul,
// tiled matmul). y = x · Wᵀ (+ b), W row-major [rows, cols] = [out, in].
// The weight binding (1, and 2 for block scales) comes from a w_<dtype> file.
struct MatP {
  rows: u32,      // output features
  cols: u32,      // input features
  ldy: u32,       // row stride of Y
  y_off: u32,     // column offset into Y (the LM head is split in parts)
  x_last: u32,    // read only the last token's x (LM head), write Y row 0
  has_bias: u32,
  layer_off: u32, // KV cache offset of this layer (EPI 3)
  c: u32,         // n_embd (EPI 3)
}
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read_write> KC: array<f32>;
@group(0) @binding(6) var<storage, read_write> VC: array<f32>;
@group(0) @binding(7) var<uniform> P: MatP;

// 0: y = acc + b   1: GELU   2: residual (y += acc + b)   3: QKV, also
// writing K and V into this layer's cache at their absolute positions.
override EPI: u32 = 0u;

fn epilogue(t: u32, o: u32, acc: f32) {
  var v = acc;
  if (P.has_bias != 0u) { v += B[o]; }
  let yrow = select(t, 0u, P.x_last != 0u);
  let yi = yrow * P.ldy + P.y_off + o;
  if (EPI == 1u) { v = gelu(v); }
  if (EPI == 2u) { Y[yi] = Y[yi] + v; return; }
  Y[yi] = v;
  if (EPI == 3u) {
    let c = P.c;
    let pos = S.pos0 + t;
    if (o >= 2u * c) { VC[P.layer_off + pos * c + o - 2u * c] = v; }
    else if (o >= c) { KC[P.layer_off + pos * c + o - c] = v; }
  }
}
