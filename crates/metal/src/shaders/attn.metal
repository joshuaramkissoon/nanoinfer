
// ---------------------------------------------------------------------------
// Stage metal-flashdec / metal-flash: attention that scales with context.
//
// Decode (flash-decoding): attention_simd gives each head ONE threadgroup, so
// at context 1000 just 12 threadgroups stream the 70 MB KV cache. Here the
// keys are split into chunks of ATT_CHUNK; threadgroup (head, chunk) computes
// a partial softmax (local max m, local sum l, unnormalised output o[64]),
// and a second kernel merges the chunks of each head:
//   M = max_s m_s,  L = sum_s l_s e^(m_s - M),  out = sum_s o_s e^(m_s - M) / L
// ---------------------------------------------------------------------------

constant uint ATT_CHUNK = 64;

// grid (n_head, n_chunks), 256 threads. part: [n_head, max_chunks, 66] floats (m, l, o[64]).
kernel void attn_decode_split(device const float* qkv [[buffer(0)]],
                              device const float* kc [[buffer(1)]],
                              device const float* vc [[buffer(2)]],
                              device float* part [[buffer(3)]],
                              constant uint& n_kv [[buffer(4)]],
                              constant uint& n_head [[buffer(5)]],
                              constant uint& max_chunks [[buffer(6)]],
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
    threadgroup float* scores = tgm;           // [ATT_CHUNK]
    threadgroup float* scratch = tgm + 64;     // [32]
    threadgroup float* partial = tgm + 96;     // [4 * 64]
    float2 qv = ((device const float2*)(qkv + h * HD))[lane];
    for (uint j = j0 + sg; j < j1; j += nsg) {
        float d = simd_sum(dot(qv, ((device const float2*)(kc + j * C + h * HD))[lane]));
        if (lane == 0) scores[j - j0] = d * 0.125f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint len = j1 - j0;
    float mx = tid < len ? scores[tid] : -INFINITY;
    mx = tg_max(mx, scratch, sg, lane, nsg);
    float e = 0.0f;
    if (tid < len) { e = exp(scores[tid] - mx); scores[tid] = e; }
    float sum = tg_sum(e, scratch, sg, lane, nsg);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint d = tid % HD, p = tid / HD;  // 4 parts x 64 dims
    float acc = 0.0f;
    for (uint j = p; j < len; j += 4) acc += scores[j] * vc[(j0 + j) * C + h * HD + d];
    partial[p * HD + d] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* out = part + (h * max_chunks + chunk) * (HD + 2);
    if (p == 0) out[2 + d] = partial[d] + partial[HD + d] + partial[2 * HD + d] + partial[3 * HD + d];
    if (tid == 0) { out[0] = mx; out[1] = sum; }
}

// grid n_head threadgroups of 64 threads: merge the chunks of one head.
kernel void attn_decode_merge(device const float* part [[buffer(0)]],
                              device float* att [[buffer(1)]],
                              constant uint& n_chunks [[buffer(2)]],
                              constant uint& n_head [[buffer(3)]],
                              constant uint& max_chunks [[buffer(4)]],
                              uint h [[threadgroup_position_in_grid]],
                              uint d [[thread_index_in_threadgroup]]) {
    const uint HD = 64;
    device const float* p = part + h * max_chunks * (HD + 2);
    float M = -INFINITY;
    for (uint s = 0; s < n_chunks; s++) M = max(M, p[s * (HD + 2)]);
    float L = 0.0f, o = 0.0f;
    for (uint s = 0; s < n_chunks; s++) {
        float w = exp(p[s * (HD + 2)] - M);
        L += p[s * (HD + 2) + 1] * w;
        o += p[s * (HD + 2) + 2 + d] * w;
    }
    att[h * HD + d] = o / L;
}

// ---------------------------------------------------------------------------
// Prefill: flash attention with simdgroup matrices.
//
// attention_simd gives every (query, head) its own threadgroup, so each query
// re-reads all previous keys from memory: O(n^2) traffic. Here a threadgroup
// owns 32 queries of one head and walks the keys in blocks of 32, staging each
// K/V block in threadgroup memory once for all 32 queries. Per block, each
// SIMD-group (8 queries) computes S = Q K^T with 8x8 MMAs, does the online
// softmax update on S in threadgroup memory (running max m, running sum l,
// rescale factor alpha = e^(m_old - m_new)), rescales its O accumulators by
// diag(alpha) (one more 8x8 MMA per tile) and adds P V. O is divided by l at
// the end. Scores never touch device memory.
// grid (ceil(n / 32), n_head), 128 threads; head_dim must be 64.
// ---------------------------------------------------------------------------
kernel void attn_prefill_flash(device float* out [[buffer(0)]],
                               device const float* qkv [[buffer(1)]],
                               device const float* kc [[buffer(2)]],
                               device const float* vc [[buffer(3)]],
                               constant uint& n [[buffer(4)]],
                               constant uint& pos0 [[buffer(5)]],
                               constant uint& n_head [[buffer(6)]],
                               uint2 tg [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    const uint HD = 64, BQ = 32, BK = 32;
    uint C = n_head * HD;
    uint qb = tg.x, h = tg.y;
    threadgroup float ks[BK * HD];
    threadgroup float vs[BK * HD];
    threadgroup float st[4][8 * BK];   // per-SIMD-group S / P tile
    threadgroup float dg[4][64];       // per-SIMD-group diag(alpha)
    threadgroup float lsum[4][8];

    uint q0 = qb * BQ + sg * 8;  // first query (relative) of this SIMD-group
    // Q fragments: 8 queries x 64 dims, rows clamped to n - 1 (padding rows are
    // computed but never written).
    simdgroup_float8x8 qf[8];
    {
        uint qrow = min(q0, n - 1);
        // If fewer than 8 valid rows remain, rows past n read neighbouring
        // memory of the qkv buffer (allocated for n_ctx rows), harmless.
        for (uint d = 0; d < 8; d++) simdgroup_load(qf[d], qkv + qrow * 3 * C + h * HD + d * 8, 3 * C);
    }
    simdgroup_float8x8 of[8];
    for (uint d = 0; d < 8; d++) of[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint r = lane / 4, c0 = (lane % 4) * 8;  // this lane's row and 8 columns of S
    uint qpos = pos0 + q0 + r;                // absolute position of that query
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
                kv = ((device const float4*)(kc + key * C + h * HD))[d4];
                vv = ((device const float4*)(vc + key * C + h * HD))[d4];
            }
            ((threadgroup float4*)ks)[e] = kv;
            ((threadgroup float4*)vs)[e] = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S = Q K^T for 4 sub-blocks of 8 keys.
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
        // Online softmax on this lane's 8 scores of row r.
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
        float alpha = exp(m - m_new);  // m = -inf on the first block: alpha = 0
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
        // O = diag(alpha) O + P V
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
    // Normalise and write: stage O (8 x 64) through threadgroup memory (ks is free).
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
