// f16 KV cache (webgpu-kv16): 4 values = two u32s of packed halves.
@group(0) @binding(1) var<storage, read> KC2: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> VC2: array<vec2<u32>>;
fn kld(i: u32) -> vec4<f32> { let u = KC2[i]; return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y)); }
fn vld(i: u32) -> vec4<f32> { let u = VC2[i]; return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y)); }
