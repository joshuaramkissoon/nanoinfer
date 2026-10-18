// f32 KV cache, read 4 values at a time.
@group(0) @binding(1) var<storage, read> KC4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> VC4: array<vec4<f32>>;
fn kld(i: u32) -> vec4<f32> { return KC4[i]; }
fn vld(i: u32) -> vec4<f32> { return VC4[i]; }
