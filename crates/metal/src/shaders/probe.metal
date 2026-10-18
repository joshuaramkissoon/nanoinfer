
// ---------------------------------------------------------------------------
// Microbenchmarks (examples/probe.rs): dispatch overhead and bandwidth.
// ---------------------------------------------------------------------------

kernel void empty_kernel(uint gid [[thread_position_in_grid]]) {}

// Read `len4` float4s and write one partial sum per thread so the compiler
// can't drop the loads. Grid-stride loop: measures achievable read bandwidth.
kernel void read_bw(device const float4* src [[buffer(0)]],
                    device float* sink [[buffer(1)]],
                    constant uint& len4 [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint grid [[threads_per_grid]]) {
    float4 acc = 0.0f;
    for (uint i = gid; i < len4; i += grid) acc += src[i];
    sink[gid] = acc.x + acc.y + acc.z + acc.w;
}

kernel void copy_bw(device const float4* src [[buffer(0)]],
                    device float4* dst [[buffer(1)]],
                    constant uint& len4 [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint grid [[threads_per_grid]]) {
    for (uint i = gid; i < len4; i += grid) dst[i] = src[i];
}

// Same, with 4 independent loads in flight per thread per iteration.
kernel void read_bw_x4(device const float4* src [[buffer(0)]],
                       device float* sink [[buffer(1)]],
                       constant uint& len4 [[buffer(2)]],
                       uint gid [[thread_position_in_grid]],
                       uint grid [[threads_per_grid]]) {
    float4 a = 0.0f, b = 0.0f, c = 0.0f, d = 0.0f;
    uint i = gid;
    for (; i + 3 * grid < len4; i += 4 * grid) {
        a += src[i]; b += src[i + grid]; c += src[i + 2 * grid]; d += src[i + 3 * grid];
    }
    for (; i < len4; i += grid) a += src[i];
    float4 s = a + b + c + d;
    sink[gid] = s.x + s.y + s.z + s.w;
}
