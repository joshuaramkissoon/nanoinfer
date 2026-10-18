// f16 weights packed two per u32 (no shader-f16 needed: unpack2x16float is core WGSL).
@group(0) @binding(1) var<storage, read> W: array<vec4<u32>>;
const WPL: u32 = 8u;
fn wget(row: u32, k: u32) -> f32 {
  let i = row * P.cols + k;
  let v = W[i / 8u];
  return unpack2x16float(v[(i / 2u) % 4u])[i % 2u];
}
