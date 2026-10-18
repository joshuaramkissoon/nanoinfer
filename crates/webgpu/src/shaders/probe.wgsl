// Dispatch-overhead and bandwidth probes.
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;

// The smallest kernel that still has a dependency on the previous dispatch.
@compute @workgroup_size(32)
fn tick(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x == 0u) { buf[0] = buf[0] + 1u; }
}
