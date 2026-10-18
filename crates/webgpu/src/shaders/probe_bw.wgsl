@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

// Grid-stride read of the whole buffer; one f32 per thread written so the
// loads can't be elided.
@compute @workgroup_size(256)
fn read_bw(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let stride = nw.x * 256u;
  var acc = vec4<f32>(0.0);
  var i = g.x;
  let n = arrayLength(&src);
  loop {
    if (i >= n) { break; }
    acc += src[i];
    i += stride;
  }
  out[g.x] = acc.x + acc.y + acc.z + acc.w;
}
