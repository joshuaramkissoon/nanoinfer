// x[t] = wte[gen[pos]] + wpe[pos], read from one part of wte (the f32 table
// is split to fit the default 128 MiB binding limit). The first part's
// dispatch writes wpe + its rows; later parts add theirs.
struct MatP { rows: u32, cols: u32, ldy: u32, y_off: u32, x_last: u32, has_bias: u32, layer_off: u32, c: u32 }
@group(0) @binding(0) var<storage, read> GEN: array<u32>;
@group(0) @binding(3) var<storage, read> WPE: array<f32>;
@group(0) @binding(4) var<storage, read_write> X: array<f32>;
@group(0) @binding(7) var<uniform> P: MatP;

@compute @workgroup_size(256)
fn embed(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = P.cols;
  let i = g.x;
  if (i >= S.n * c) { return; }
  let t = i / c;
  let k = i % c;
  let pos = S.pos0 + t;
  let tok = GEN[pos];
  var v: f32;
  if (P.has_bias != 0u) { v = WPE[pos * c + k]; } else { v = X[i]; }
  if (tok >= P.y_off && tok < P.y_off + P.rows) { v += wget(tok - P.y_off, k); }
  X[i] = v;
}
