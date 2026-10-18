
// ---------------------------------------------------------------------------
// Stage metal-simd: kernels shaped for the GPU.
//
// A SIMD-group is 32 threads that execute in lockstep and can reduce across
// lanes in registers (simd_sum, simd_max). The rule for a coalesced load: the
// 32 lanes read 32 *adjacent* float4s, i.e. one 512-byte contiguous span.
// ---------------------------------------------------------------------------

// Matrix-vector product for decode: out[t, o] = x[t] . W[o] + bias[o].
// Each SIMD-group computes R consecutive rows; lane l reads float4s l, l+32,
// l+64, ... of each row (coalesced), multiplies by the matching float4 of x
// (loaded once, reused for all R rows), and simd_sum reduces the 32 partial
// sums. A threadgroup holds NSG SIMD-groups = NSG*R rows. grid.y = token.
template <uint R>
kernel void matvec_simd(device float* out [[buffer(0)]],
                        device const float* x [[buffer(1)]],
                        device const float* w [[buffer(2)]],
                        device const float* bias [[buffer(3)]],
                        constant uint& n [[buffer(4)]],
                        constant uint& d_in [[buffer(5)]],
                        constant uint& d_out [[buffer(6)]],
                        constant uint& has_bias [[buffer(7)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint nsg [[simdgroups_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint t = tg.y;
    uint row0 = (tg.x * nsg + sg) * R;
    if (row0 >= d_out) return;
    uint k4 = d_in / 4;
    device const float4* x4 = (device const float4*)(x + t * d_in);
    float acc[R];
    for (uint r = 0; r < R; r++) acc[r] = 0.0f;
    for (uint i = lane; i < k4; i += 32) {
        float4 xv = x4[i];
        for (uint r = 0; r < R; r++) {
            uint row = min(row0 + r, d_out - 1);  // clamp: keep loads in bounds, discard below
            device const float4* w4 = (device const float4*)(w + row * d_in);
            acc[r] += dot(xv, w4[i]);
        }
    }
    for (uint r = 0; r < R; r++) {
        float s = simd_sum(acc[r]);
        uint row = row0 + r;
        if (lane == 0 && row < d_out) out[t * d_out + row] = s + (has_bias ? bias[row] : 0.0f);
    }
}

template [[host_name("matvec_simd_r1")]] kernel void matvec_simd<1>(device float*, device const float*, device const float*, device const float*, constant uint&, constant uint&, constant uint&, constant uint&, uint2, uint, uint, uint);
template [[host_name("matvec_simd_r2")]] kernel void matvec_simd<2>(device float*, device const float*, device const float*, device const float*, constant uint&, constant uint&, constant uint&, constant uint&, uint2, uint, uint, uint);
template [[host_name("matvec_simd_r4")]] kernel void matvec_simd<4>(device float*, device const float*, device const float*, device const float*, constant uint&, constant uint&, constant uint&, constant uint&, uint2, uint, uint, uint);
template [[host_name("matvec_simd_r8")]] kernel void matvec_simd<8>(device float*, device const float*, device const float*, device const float*, constant uint&, constant uint&, constant uint&, constant uint&, uint2, uint, uint, uint);

// Sum a value across the whole threadgroup (<= 32 SIMD-groups). `scratch`
// holds 32 floats of threadgroup memory. Every thread gets the result.
inline float tg_sum(float v, threadgroup float* scratch, uint sg, uint lane, uint nsg) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = lane < nsg ? scratch[lane] : 0.0f;
    return simd_sum(s);
}

inline float tg_max(float v, threadgroup float* scratch, uint sg, uint lane, uint nsg) {
    v = simd_max(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = lane < nsg ? scratch[lane] : -INFINITY;
    return simd_max(s);
}

// LayerNorm, one threadgroup per row: each thread handles C/threads elements,
// reductions via simd_sum + threadgroup memory. Same two-pass maths as the
// reference (mean, then mean of squared deviations).
kernel void layernorm_simd(device float* out [[buffer(0)]],
                           device const float* x [[buffer(1)]],
                           device const float* w [[buffer(2)]],
                           device const float* b [[buffer(3)]],
                           constant uint& n [[buffer(4)]],
                           constant uint& C [[buffer(5)]],
                           threadgroup float* scratch [[threadgroup(0)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint ntg [[threads_per_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint nsg [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    device const float* xr = x + row * C;
    float s = 0.0f;
    for (uint i = tid; i < C; i += ntg) s += xr[i];
    float mean = tg_sum(s, scratch, sg, lane, nsg) / float(C);
    float v = 0.0f;
    for (uint i = tid; i < C; i += ntg) { float d = xr[i] - mean; v += d * d; }
    float var = tg_sum(v, scratch, sg, lane, nsg) / float(C);
    float rstd = 1.0f / sqrt(var + 1e-5f);
    for (uint i = tid; i < C; i += ntg) out[row * C + i] = (xr[i] - mean) * rstd * w[i] + b[i];
}

// Causal attention, one threadgroup per (query, head).
// 1. scores: SIMD-group g takes keys g, g+nsg, ...; each lane multiplies 2 of
//    the 64 dims (float2), simd_sum gives the dot product. Scores live in
//    threadgroup memory (n_kv <= 1024 floats).
// 2. softmax: threadgroup max and sum.
// 3. out: thread (e, part) accumulates dim e over keys part, part+P, ...;
//    the P partial sums are reduced through threadgroup memory.
// Requires head_dim == 64 and 256 threads per threadgroup.
kernel void attention_simd(device float* out [[buffer(0)]],
                           device const float* qkv [[buffer(1)]],
                           device const float* kc [[buffer(2)]],
                           device const float* vc [[buffer(3)]],
                           constant uint& n [[buffer(4)]],
                           constant uint& pos0 [[buffer(5)]],
                           constant uint& n_head [[buffer(6)]],
                           threadgroup float* tgm [[threadgroup(0)]],
                           uint tgid [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint nsg [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const uint HD = 64;
    uint C = n_head * HD;
    uint t = tgid / n_head, h = tgid % n_head;
    uint n_kv = pos0 + t + 1;
    threadgroup float* scores = tgm;            // [1024]
    threadgroup float* scratch = tgm + 1024;    // [32]
    threadgroup float* partial = tgm + 1056;    // [4 * 64]

    device const float2* q2 = (device const float2*)(qkv + t * 3 * C + h * HD);
    float2 qv = q2[lane];
    for (uint j = sg; j < n_kv; j += nsg) {
        device const float2* k2 = (device const float2*)(kc + j * C + h * HD);
        float d = simd_sum(dot(qv, k2[lane]));
        if (lane == 0) scores[j] = d * 0.125f;  // 1/sqrt(64)
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -INFINITY;
    for (uint j = tid; j < n_kv; j += 256) mx = max(mx, scores[j]);
    mx = tg_max(mx, scratch, sg, lane, nsg);
    float sum = 0.0f;
    for (uint j = tid; j < n_kv; j += 256) { float e = exp(scores[j] - mx); scores[j] = e; sum += e; }
    sum = tg_sum(sum, scratch, sg, lane, nsg);
    float inv = 1.0f / sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint e = tid % HD, part = tid / HD;  // 4 parts of 64 threads
    float acc = 0.0f;
    for (uint j = part; j < n_kv; j += 4) acc += scores[j] * vc[j * C + h * HD + e];
    partial[part * HD + e] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0) {
        float o = partial[e] + partial[HD + e] + partial[2 * HD + e] + partial[3 * HD + e];
        out[t * C + h * HD + e] = o * inv;
    }
}
