
// ---------------------------------------------------------------------------
// Stage metal-kv8: Q8 KV cache.
//
// At long context the f16 KV cache is ~30% of a Q4 decode token's bytes. Here
// each (position, head) slice of 64 values is stored as int8 with one f32
// scale (amax / 127): 68 bytes per 64 values instead of 128. The qkv matvec
// writes f32 K/V into the qkv buffer as in the unfused path, kv_quant8
// quantises them into the cache, and the attention kernels dequantise on load
// (copies of the f16-KV kernels with a scale per key).
// Cache layout per layer: int8 [n_ctx][C]; scales f32 [n_ctx][n_head].
// ---------------------------------------------------------------------------

// grid (n * n_head, 2) threadgroups of 64 threads: y = 0 K, 1 V.
kernel void kv_quant8(device char* kc [[buffer(0)]],
                      device char* vc [[buffer(1)]],
                      device float* ks [[buffer(2)]],
                      device float* vs [[buffer(3)]],
                      device const float* qkv [[buffer(4)]],
                      constant uint& pos0 [[buffer(5)]],
                      constant uint& n_head [[buffer(6)]],
                      threadgroup float* tgm [[threadgroup(0)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint d [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint HD = 64;
    uint C = n_head * HD;
    uint t = tg.x / n_head, h = tg.x % n_head, which = tg.y;
    float v = qkv[t * 3 * C + (1 + which) * C + h * HD + d];
    float m = simd_max(fabs(v));
    if (lane == 0) tgm[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = max(tgm[0], tgm[1]);
    float scale = amax / 127.0f;
    float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    uint p = pos0 + t;
    char q = char(clamp(rint(v * inv), -127.0f, 127.0f));
    if (which == 0) {
        kc[p * C + h * HD + d] = q;
        if (d == 0) ks[p * n_head + h] = scale;
    } else {
        vc[p * C + h * HD + d] = q;
        if (d == 0) vs[p * n_head + h] = scale;
    }
}

// Flash-decoding over the Q8 cache; same structure as attn_decode_split.
kernel void attn_decode_split_q8(device const float* qkv [[buffer(0)]],
                                 device const char* kc [[buffer(1)]],
                                 device const char* vc [[buffer(2)]],
                                 device float* part [[buffer(3)]],
                                 constant uint& n_kv [[buffer(4)]],
                                 constant uint& n_head [[buffer(5)]],
                                 constant uint& max_chunks [[buffer(6)]],
                                 device const float* ks [[buffer(7)]],
                                 device const float* vs [[buffer(8)]],
                                 threadgroup float* tgm [[threadgroup(0)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint nsg [[simdgroups_per_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint HD = 64;
    uint C = n_head * HD;
    uint h = tg.x, chunk = tg.y;
    uint j0 = chunk * ATT_CHUNK, j1 = min(j0 + ATT_CHUNK, n_kv);
    threadgroup float* scores = tgm;
    threadgroup float* scratch = tgm + 64;
    threadgroup float* partial = tgm + 96;
    float2 qv = ((device const float2*)(qkv + h * HD))[lane];
    for (uint j = j0 + sg; j < j1; j += nsg) {
        float d = simd_sum(dot(qv, float2(((device const char2*)(kc + j * C + h * HD))[lane])));
        if (lane == 0) scores[j - j0] = d * ks[j * n_head + h] * 0.125f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint len = j1 - j0;
    float mx = tid < len ? scores[tid] : -INFINITY;
    mx = tg_max(mx, scratch, sg, lane, nsg);
    float e = 0.0f;
    if (tid < len) { e = exp(scores[tid] - mx); scores[tid] = e; }
    float sum = tg_sum(e, scratch, sg, lane, nsg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint d = tid % HD, p = tid / HD;
    float acc = 0.0f;
    for (uint j = p; j < len; j += 4) {
        uint key = j0 + j;
        acc += scores[j] * vs[key * n_head + h] * float(vc[key * C + h * HD + d]);
    }
    partial[p * HD + d] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* out = part + (h * max_chunks + chunk) * (HD + 2);
    if (p == 0) out[2 + d] = partial[d] + partial[HD + d] + partial[2 * HD + d] + partial[3 * HD + d];
    if (tid == 0) { out[0] = mx; out[1] = sum; }
}

// Flash-attention prefill over the Q8 cache: attn_prefill_flash with a
// dequantising K/V block load. grid (ceil(n / 32), n_head), 128 threads.
kernel void attn_prefill_flash_q8(device float* out [[buffer(0)]],
                                  device const float* qkv [[buffer(1)]],
                                  device const char* kc [[buffer(2)]],
                                  device const char* vc [[buffer(3)]],
                                  constant uint& n [[buffer(4)]],
                                  constant uint& pos0 [[buffer(5)]],
                                  constant uint& n_head [[buffer(6)]],
                                  device const float* ksc [[buffer(7)]],
                                  device const float* vsc [[buffer(8)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    const uint HD = 64, BQ = 32, BK = 32;
    uint C = n_head * HD;
    uint qb = tg.x, h = tg.y;
    threadgroup float ks[BK * HD];
    threadgroup float vs[BK * HD];
    threadgroup float st[4][8 * BK];
    threadgroup float dg[4][64];
    threadgroup float lsum[4][8];
    uint q0 = qb * BQ + sg * 8;
    simdgroup_float8x8 qf[8];
    {
        uint qrow = min(q0, n - 1);
        for (uint d = 0; d < 8; d++) simdgroup_load(qf[d], qkv + qrow * 3 * C + h * HD + d * 8, 3 * C);
    }
    simdgroup_float8x8 of[8];
    for (uint d = 0; d < 8; d++) of[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint r = lane / 4, c0 = (lane % 4) * 8;
    uint qpos = pos0 + q0 + r;
    float m = -INFINITY, l = 0.0f;
    for (uint i = tid; i < 64; i += 128) for (uint s = 0; s < 4; s++) dg[s][i] = 0.0f;
    uint n_kv = pos0 + n;
    uint last_key = min(pos0 + qb * BQ + BQ - 1, n_kv - 1);
    for (uint k0 = 0; k0 <= last_key; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < BK * HD / 4; e += 128) {
            uint j = e / (HD / 4), d4 = e % (HD / 4);
            uint key = k0 + j;
            float4 kv = 0.0f, vv = 0.0f;
            if (key < n_kv) {
                kv = float4(((device const char4*)(kc + key * C + h * HD))[d4]) * ksc[key * n_head + h];
                vv = float4(((device const char4*)(vc + key * C + h * HD))[d4]) * vsc[key * n_head + h];
            }
            ((threadgroup float4*)ks)[e] = kv;
            ((threadgroup float4*)vs)[e] = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0; t < 4; t++) {
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint d = 0; d < 8; d++) {
                simdgroup_float8x8 kt;
                simdgroup_load(kt, ks + t * 8 * HD + d * 8, HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qf[d], kt, acc);
            }
            simdgroup_store(acc, st[sg] + t * 8, BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float sv[8];
        float mx = -INFINITY;
        for (uint c = 0; c < 8; c++) {
            uint key = k0 + c0 + c;
            float v = st[sg][r * BK + c0 + c] * 0.125f;
            v = (key <= qpos && key < n_kv) ? v : -INFINITY;
            sv[c] = v;
            mx = max(mx, v);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        float m_new = max(m, mx);
        float alpha = exp(m - m_new);
        float rs = 0.0f;
        for (uint c = 0; c < 8; c++) {
            float p = exp(sv[c] - m_new);
            st[sg][r * BK + c0 + c] = p;
            rs += p;
        }
        rs += simd_shuffle_xor(rs, 1);
        rs += simd_shuffle_xor(rs, 2);
        l = l * alpha + rs;
        m = m_new;
        if (lane % 4 == 0) dg[sg][r * 8 + r] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 a;
        simdgroup_load(a, dg[sg], 8);
        simdgroup_float8x8 pf[4];
        for (uint t = 0; t < 4; t++) simdgroup_load(pf[t], st[sg] + t * 8, BK);
        for (uint d = 0; d < 8; d++) {
            simdgroup_float8x8 o;
            simdgroup_multiply(o, a, of[d]);
            for (uint t = 0; t < 4; t++) {
                simdgroup_float8x8 vf;
                simdgroup_load(vf, vs + t * 8 * HD + d * 8, HD);
                simdgroup_multiply_accumulate(o, pf[t], vf, o);
            }
            of[d] = o;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane % 4 == 0) lsum[sg][r] = l;
    threadgroup float* ot = ks + sg * 8 * HD;
    for (uint d = 0; d < 8; d++) simdgroup_store(of[d], ot + d * 8, HD);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 8 * HD; e += 32) {
        uint rr = e / HD, d = e % HD;
        uint q = q0 + rr;
        if (q < n) out[q * C + h * HD + d] = ot[e] / lsum[sg][rr];
    }
}
