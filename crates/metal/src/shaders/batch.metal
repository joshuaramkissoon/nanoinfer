
// ---------------------------------------------------------------------------
// Round 2: batched, ragged, paged forward passes (crates/metal/src/batch.rs).
//
// One forward processes n "rows". A row is one token of one sequence:
// rows[i] = (seq, pos, -, -), and every kernel adds a `step` constant to pos
// (so a GPU-resident loop can reuse the same row list for many steps). Rows
// of different sequences sit side by side (batched decode), rows of the same
// sequence are consecutive positions (prefill chunks, speculative
// verification), and both can mix in one forward (continuous batching).
//
// KV cache: paged (vLLM's PagedAttention layout). Each layer has a pool of
// blocks of KV_BLOCK positions x C halfs; sequence `seq` maps its logical
// block b to a physical block through table[seq * max_blocks + b]. A
// contiguous cache is just the identity table.
//
// Token ids live in toks[seq * n_ctx + pos]: the CPU writes prompt tokens
// there, and the greedy argmax writes its prediction for (seq, pos) into
// (seq, pos + 1), which is exactly where the next step's embedding reads.
// ---------------------------------------------------------------------------

constant uint KV_BLOCK = 16;

inline uint kv_slot(device const uint* table, uint seq, uint pos, uint max_blocks) {
    return table[seq * max_blocks + pos / KV_BLOCK] * KV_BLOCK + pos % KV_BLOCK;
}

// x[i] = wte[toks[seq, pos]] + wpe[pos], one thread per element of [n, C].
kernel void b_embed(device float* x [[buffer(0)]],
                    device const float* wte [[buffer(1)]],
                    device const float* wpe [[buffer(2)]],
                    device const uint* toks [[buffer(3)]],
                    device const uint4* rows [[buffer(4)]],
                    constant uint& n [[buffer(5)]],
                    constant uint& C [[buffer(6)]],
                    constant uint& step [[buffer(7)]],
                    constant uint& n_ctx [[buffer(8)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= n * C) return;
    uint t = gid / C, i = gid % C;
    uint seq = rows[t].x, pos = rows[t].y + step;
    x[gid] = wte[toks[seq * n_ctx + pos] * C + i] + wpe[pos * C + i];
}

// ----- batched matvec: weights read once, dotted with NB token rows ---------
//
// Same lane layout as matvec_lanes (8 lanes per weight row, 4 rows per
// SIMD-group). Each lane dequantises one "unit" of its row into registers
// (UF4 float4s) and multiplies it with the matching slice of NB activation
// rows. grid.y walks chunks of NB rows.
//
// Batch invariance: each token's dot product is accumulated in exactly the
// same order whatever NB is (per unit: sum of UF4 float4 dots, then added to
// the token's accumulator; then the same 8-lane shuffle tree). So a sequence
// gets bit-identical logits whether it runs alone or in a batch of 16 — which
// is what makes speculative decoding's output provably equal to plain greedy.

struct BQ4 {  // unit = one block of 32 weights = 8 float4
    static void load(device const uchar* w, uint row, uint u, uint d_in, uint d_out, thread float4* wv) {
        uint nb = d_in / 32;
        device const half* scales = (device const half*)(w + d_out * d_in / 2);
        uint4 q = ((device const uint4*)(w + row * d_in / 2))[u];
        float d = float(scales[row * nb + u]);
        for (uint j = 0; j < 4; j++) {
            uint v = q[j];
            wv[j] = (float4(float(v & 15u), float((v >> 8) & 15u), float((v >> 16) & 15u), float((v >> 24) & 15u)) - 8.0f) * d;
            wv[4 + j] = (float4(float((v >> 4) & 15u), float((v >> 12) & 15u), float((v >> 20) & 15u), float(v >> 28)) - 8.0f) * d;
        }
    }
};
struct BQ8 {  // unit = half a block, 16 int8 = 4 float4
    static void load(device const uchar* w, uint row, uint u, uint d_in, uint d_out, thread float4* wv) {
        device const half* scales = (device const half*)(w + d_out * d_in);
        device const char4* q = (device const char4*)(w + row * d_in) + 4 * u;
        float d = float(scales[row * (d_in / 32) + u / 2]);
        for (uint j = 0; j < 4; j++) wv[j] = float4(q[j]) * d;
    }
};
struct BF16 {  // unit = 8 halfs = 2 float4
    static void load(device const uchar* w, uint row, uint u, uint d_in, uint d_out, thread float4* wv) {
        device const half4* w4 = (device const half4*)((device const half*)w + row * d_in);
        wv[0] = float4(w4[2 * u]);
        wv[1] = float4(w4[2 * u + 1]);
    }
};
struct BF32 {  // unit = 8 floats = 2 float4
    static void load(device const uchar* w, uint row, uint u, uint d_in, uint d_out, thread float4* wv) {
        device const float4* w4 = (device const float4*)((device const float*)w + row * d_in);
        wv[0] = w4[2 * u];
        wv[1] = w4[2 * u + 1];
    }
};

constant uint BEPI_PLAIN = 0;  // out[t, row] = s (+ bias)
constant uint BEPI_GELU = 1;   // out[t, row] = gelu(s + bias)
constant uint BEPI_RESID = 2;  // out[t, row] += s + bias
constant uint BEPI_QKV = 3;    // q -> out[t], k/v -> paged f16 cache at (seq, pos)

// Buffers: 0 out, 1 x [n, d_in], 2 w, 3 bias, 4 k pool, 5 v pool, 6 rows,
// 7 table, 8 d_in, 9 d_out, 10 has_bias, 11 n, 12 step, 13 max_blocks.
// grid (ceil(d_out / (4 * nsg)), ceil(n / NB)), 32 * nsg threads.
template <typename W, uint UF4, uint NB, uint EPI>
kernel void matvec_batch(device float* out [[buffer(0)]],
                         device const float* x [[buffer(1)]],
                         device const uchar* w [[buffer(2)]],
                         device const float* bias [[buffer(3)]],
                         device half* kp [[buffer(4)]],
                         device half* vp [[buffer(5)]],
                         device const uint4* rows [[buffer(6)]],
                         device const uint* table [[buffer(7)]],
                         constant uint& d_in [[buffer(8)]],
                         constant uint& d_out [[buffer(9)]],
                         constant uint& has_bias [[buffer(10)]],
                         constant uint& n [[buffer(11)]],
                         constant uint& step [[buffer(12)]],
                         constant uint& max_blocks [[buffer(13)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint nsg [[simdgroups_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint sub = lane / LPR, li = lane % LPR;
    uint row_u = (tg.x * nsg + sg) * 4 + sub;
    uint row = min(row_u, d_out - 1);
    uint t0 = tg.y * NB;
    uint nt = min(NB, n - t0);
    uint nu = d_in / (4 * UF4);
    float acc[NB];
    for (uint t = 0; t < NB; t++) acc[t] = 0.0f;
    for (uint u = li; u < nu; u += LPR) {
        float4 wv[UF4];
        W::load(w, row, u, d_in, d_out, wv);
        for (uint t = 0; t < NB; t++) {
            uint tt = t0 + min(t, nt - 1);  // clamp: padding tokens recompute the last one
            device const float4* x4 = (device const float4*)(x + tt * d_in) + u * UF4;
            float s = 0.0f;
            for (uint j = 0; j < UF4; j++) s += dot(wv[j], x4[j]);
            acc[t] += s;
        }
    }
    for (uint t = 0; t < NB; t++) {
        float s = acc[t];
        s += simd_shuffle_xor(s, 4);
        s += simd_shuffle_xor(s, 2);
        s += simd_shuffle_xor(s, 1);
        acc[t] = s;
    }
    if (li != 0 || row_u >= d_out) return;
    float b = has_bias ? bias[row] : 0.0f;
    for (uint t = 0; t < nt; t++) {
        uint tok = t0 + t;
        float s = acc[t] + b;
        if (EPI == BEPI_PLAIN) out[tok * d_out + row] = s;
        else if (EPI == BEPI_GELU) out[tok * d_out + row] = gelu_tanh(s);
        else if (EPI == BEPI_RESID) out[tok * d_out + row] += s;
        else {
            uint C = d_out / 3;
            if (row < C) {
                out[tok * d_out + row] = s;
            } else {
                uint4 ri = rows[tok];
                uint slot = kv_slot(table, ri.x, ri.y + step, max_blocks);
                if (row < 2 * C) kp[slot * C + row - C] = half(s);
                else vp[slot * C + row - 2 * C] = half(s);
            }
        }
    }
}

#define MVB_ARGS device float*, device const float*, device const uchar*, device const float*, device half*, device half*, device const uint4*, device const uint*, constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, constant uint&, uint2, uint, uint, uint
#define MVB_EPIS(NAME, W, UF4, NB) \
template [[host_name("mvb_" NAME "_b" #NB)]] kernel void matvec_batch<W, UF4, NB, BEPI_PLAIN>(MVB_ARGS); \
template [[host_name("mvb_gelu_" NAME "_b" #NB)]] kernel void matvec_batch<W, UF4, NB, BEPI_GELU>(MVB_ARGS); \
template [[host_name("mvb_resid_" NAME "_b" #NB)]] kernel void matvec_batch<W, UF4, NB, BEPI_RESID>(MVB_ARGS); \
template [[host_name("mvb_qkv_" NAME "_b" #NB)]] kernel void matvec_batch<W, UF4, NB, BEPI_QKV>(MVB_ARGS);
#define MVB_SET(NAME, W, UF4) \
MVB_EPIS(NAME, W, UF4, 1) MVB_EPIS(NAME, W, UF4, 2) MVB_EPIS(NAME, W, UF4, 4) \
MVB_EPIS(NAME, W, UF4, 8) MVB_EPIS(NAME, W, UF4, 12) MVB_EPIS(NAME, W, UF4, 16)
MVB_SET("q4_0", BQ4, 8)
MVB_SET("q8_0", BQ8, 4)
MVB_SET("f16", BF16, 2)
MVB_SET("f32", BF32, 2)

// ----- matvec_batch_r: R weight rows per lane, so each x load serves R rows --
//
// matvec_batch re-reads the activation slice once per weight row: at B = 8
// that is 32 bytes of x (from L2) per 0.56 bytes of Q4 weight, and the kernel
// tops out at ~250 GFLOP/s, cache-bound. Here a lane owns R rows of the same
// row-set (row = row0 + sub * R + r) and walks the same units for all of
// them: per float4 of x it does R dots. Each row's units still go to lanes
// li, li + 8, ... and reduce through the same shuffle tree, and each unit's
// sum is formed in the same order, so the results are bit-identical to
// matvec_batch for any R and NB.
struct RQ4 {
    struct raw_t { uint4 q; float d; };
    static raw_t load(device const uchar* w, uint row, uint u, uint d_in, uint d_out) {
        device const half* scales = (device const half*)(w + d_out * d_in / 2);
        return raw_t{((device const uint4*)(w + row * d_in / 2))[u], float(scales[row * (d_in / 32) + u])};
    }
    static float4 get(thread const raw_t& r, uint j) {
        uint v = r.q[j & 3];
        uint sh = j < 4 ? 0 : 4;
        return (float4(float((v >> sh) & 15u), float((v >> (8 + sh)) & 15u), float((v >> (16 + sh)) & 15u),
                       float((v >> (24 + sh)) & 15u)) - 8.0f) * r.d;
    }
};
struct RQ8 {
    struct raw_t { uint4 q; float d; };
    static raw_t load(device const uchar* w, uint row, uint u, uint d_in, uint d_out) {
        device const half* scales = (device const half*)(w + d_out * d_in);
        return raw_t{((device const uint4*)(w + row * d_in))[u], float(scales[row * (d_in / 32) + u / 2])};
    }
    static float4 get(thread const raw_t& r, uint j) { return float4(as_type<char4>(r.q[j])) * r.d; }
};
struct RF16 {
    struct raw_t { uint4 q; };
    static raw_t load(device const uchar* w, uint row, uint u, uint d_in, uint d_out) {
        return raw_t{((device const uint4*)((device const half*)w + row * d_in))[u]};
    }
    static float4 get(thread const raw_t& r, uint j) { return float4(as_type<half4>(uint2(r.q[2 * j], r.q[2 * j + 1]))); }
};
struct RF32 {
    struct raw_t { float4 a, b; };
    static raw_t load(device const uchar* w, uint row, uint u, uint d_in, uint d_out) {
        device const float4* w4 = (device const float4*)((device const float*)w + row * d_in);
        return raw_t{w4[2 * u], w4[2 * u + 1]};
    }
    static float4 get(thread const raw_t& r, uint j) { return j == 0 ? r.a : r.b; }
};

// Same buffers as matvec_batch. grid (ceil(d_out / (4 * R * nsg)), ceil(n / NB)).
template <typename W, uint UF4, uint NB, uint R, uint EPI>
kernel void matvec_batch_r(device float* out [[buffer(0)]],
                           device const float* x [[buffer(1)]],
                           device const uchar* w [[buffer(2)]],
                           device const float* bias [[buffer(3)]],
                           device half* kp [[buffer(4)]],
                           device half* vp [[buffer(5)]],
                           device const uint4* rows [[buffer(6)]],
                           device const uint* table [[buffer(7)]],
                           constant uint& d_in [[buffer(8)]],
                           constant uint& d_out [[buffer(9)]],
                           constant uint& has_bias [[buffer(10)]],
                           constant uint& n [[buffer(11)]],
                           constant uint& step [[buffer(12)]],
                           constant uint& max_blocks [[buffer(13)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint nsg [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    uint sub = lane / LPR, li = lane % LPR;
    uint rbase = ((tg.x * nsg + sg) * 4 + sub) * R;
    uint t0 = tg.y * NB;
    uint nt = min(NB, n - t0);
    uint nu = d_in / (4 * UF4);
    uint rr[R];
    for (uint r = 0; r < R; r++) rr[r] = min(rbase + r, d_out - 1);
    float acc[R][NB];
    for (uint r = 0; r < R; r++)
        for (uint t = 0; t < NB; t++) acc[r][t] = 0.0f;
    for (uint u = li; u < nu; u += LPR) {
        typename W::raw_t raw[R];
        for (uint r = 0; r < R; r++) raw[r] = W::load(w, rr[r], u, d_in, d_out);
        float s[R][NB];
        for (uint r = 0; r < R; r++)
            for (uint t = 0; t < NB; t++) s[r][t] = 0.0f;
        for (uint j = 0; j < UF4; j++) {
            float4 wj[R];
            for (uint r = 0; r < R; r++) wj[r] = W::get(raw[r], j);
            for (uint t = 0; t < NB; t++) {
                float4 xv = ((device const float4*)(x + (t0 + min(t, nt - 1)) * d_in))[u * UF4 + j];
                for (uint r = 0; r < R; r++) s[r][t] += dot(wj[r], xv);
            }
        }
        for (uint r = 0; r < R; r++)
            for (uint t = 0; t < NB; t++) acc[r][t] += s[r][t];
    }
    for (uint r = 0; r < R; r++)
        for (uint t = 0; t < NB; t++) {
            float v = acc[r][t];
            v += simd_shuffle_xor(v, 4);
            v += simd_shuffle_xor(v, 2);
            v += simd_shuffle_xor(v, 1);
            acc[r][t] = v;
        }
    if (li != 0) return;
    for (uint r = 0; r < R; r++) {
        uint row = rbase + r;
        if (row >= d_out) break;
        float b = has_bias ? bias[row] : 0.0f;
        for (uint t = 0; t < nt; t++) {
            uint tok = t0 + t;
            float v = acc[r][t] + b;
            if (EPI == BEPI_PLAIN) out[tok * d_out + row] = v;
            else if (EPI == BEPI_GELU) out[tok * d_out + row] = gelu_tanh(v);
            else if (EPI == BEPI_RESID) out[tok * d_out + row] += v;
            else {
                uint C = d_out / 3;
                if (row < C) {
                    out[tok * d_out + row] = v;
                } else {
                    uint4 ri = rows[tok];
                    uint slot = kv_slot(table, ri.x, ri.y + step, max_blocks);
                    if (row < 2 * C) kp[slot * C + row - C] = half(v);
                    else vp[slot * C + row - 2 * C] = half(v);
                }
            }
        }
    }
}

#define MVR_EPIS(NAME, W, UF4, NB, R) \
template [[host_name("mvr_" NAME "_b" #NB "_r" #R)]] kernel void matvec_batch_r<W, UF4, NB, R, BEPI_PLAIN>(MVB_ARGS); \
template [[host_name("mvr_gelu_" NAME "_b" #NB "_r" #R)]] kernel void matvec_batch_r<W, UF4, NB, R, BEPI_GELU>(MVB_ARGS); \
template [[host_name("mvr_resid_" NAME "_b" #NB "_r" #R)]] kernel void matvec_batch_r<W, UF4, NB, R, BEPI_RESID>(MVB_ARGS); \
template [[host_name("mvr_qkv_" NAME "_b" #NB "_r" #R)]] kernel void matvec_batch_r<W, UF4, NB, R, BEPI_QKV>(MVB_ARGS);
#define MVR_SET(NAME, W, UF4) \
MVR_EPIS(NAME, W, UF4, 1, 1) MVR_EPIS(NAME, W, UF4, 1, 2) MVR_EPIS(NAME, W, UF4, 2, 2) MVR_EPIS(NAME, W, UF4, 2, 4) \
MVR_EPIS(NAME, W, UF4, 4, 2) MVR_EPIS(NAME, W, UF4, 4, 4) MVR_EPIS(NAME, W, UF4, 8, 2) MVR_EPIS(NAME, W, UF4, 8, 4) \
MVR_EPIS(NAME, W, UF4, 12, 1) MVR_EPIS(NAME, W, UF4, 12, 2) MVR_EPIS(NAME, W, UF4, 12, 4) \
MVR_EPIS(NAME, W, UF4, 16, 1) MVR_EPIS(NAME, W, UF4, 16, 2) MVR_EPIS(NAME, W, UF4, 16, 4)
MVR_SET("q4_0", RQ4, 8)
MVR_SET("q8_0", RQ8, 4)
MVR_SET("f16", RF16, 2)
MVR_SET("f32", RF32, 2)


// GEMM path (many rows): matmul_mma_* writes qkv [n, 3C]; this copies K/V
// into the paged cache.
kernel void b_kv_write(device half* kp [[buffer(0)]],
                       device half* vp [[buffer(1)]],
                       device const float* qkv [[buffer(2)]],
                       device const uint4* rows [[buffer(3)]],
                       device const uint* table [[buffer(4)]],
                       constant uint& n [[buffer(5)]],
                       constant uint& C [[buffer(6)]],
                       constant uint& step [[buffer(7)]],
                       constant uint& max_blocks [[buffer(8)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= n * C) return;
    uint t = gid / C, i = gid % C;
    uint slot = kv_slot(table, rows[t].x, rows[t].y + step, max_blocks);
    kp[slot * C + i] = half(qkv[t * 3 * C + C + i]);
    vp[slot * C + i] = half(qkv[t * 3 * C + 2 * C + i]);
}

// ----- attention: flash-decoding per row over its own paged keys ------------
// grid (n_head, max chunks over rows, n), 256 threads. Same arithmetic as
// attn_decode_split<half>; rows whose context is shorter skip extra chunks.
kernel void b_attn_split(device const float* qkv [[buffer(0)]],
                         device const half* kp [[buffer(1)]],
                         device const half* vp [[buffer(2)]],
                         device float* part [[buffer(3)]],
                         device const uint4* rows [[buffer(4)]],
                         device const uint* table [[buffer(5)]],
                         constant uint& n_head [[buffer(6)]],
                         constant uint& max_chunks [[buffer(7)]],
                         constant uint& step [[buffer(8)]],
                         constant uint& max_blocks [[buffer(9)]],
                         threadgroup float* tgm [[threadgroup(0)]],
                         uint3 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint nsg [[simdgroups_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const uint HD = 64;
    uint C = n_head * HD;
    uint h = tg.x, chunk = tg.y, t = tg.z;
    uint seq = rows[t].x, n_kv = rows[t].y + step + 1;
    uint j0 = chunk * ATT_CHUNK;
    if (j0 >= n_kv) return;
    uint j1 = min(j0 + ATT_CHUNK, n_kv);
    threadgroup float* scores = tgm;
    threadgroup float* scratch = tgm + 64;
    threadgroup float* partial = tgm + 96;
    float2 qv = ((device const float2*)(qkv + t * 3 * C + h * HD))[lane];
    for (uint j = j0 + sg; j < j1; j += nsg) {
        uint slot = kv_slot(table, seq, j, max_blocks);
        float d = simd_sum(dot(qv, float2(((device const half2*)(kp + slot * C + h * HD))[lane])));
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
    uint d = tid % HD, p = tid / HD;
    float acc = 0.0f;
    for (uint j = p; j < len; j += 4) {
        uint slot = kv_slot(table, seq, j0 + j, max_blocks);
        acc += scores[j] * float(vp[slot * C + h * HD + d]);
    }
    partial[p * HD + d] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* o = part + ((t * n_head + h) * max_chunks + chunk) * (HD + 2);
    if (p == 0) o[2 + d] = partial[d] + partial[HD + d] + partial[2 * HD + d] + partial[3 * HD + d];
    if (tid == 0) { o[0] = mx; o[1] = sum; }
}

// grid (n_head, n), 64 threads.
kernel void b_attn_merge(device const float* part [[buffer(0)]],
                         device float* att [[buffer(1)]],
                         device const uint4* rows [[buffer(2)]],
                         constant uint& n_head [[buffer(3)]],
                         constant uint& max_chunks [[buffer(4)]],
                         constant uint& step [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint d [[thread_index_in_threadgroup]]) {
    const uint HD = 64;
    uint h = tg.x, t = tg.y;
    uint n_chunks = (rows[t].y + step + 1 + ATT_CHUNK - 1) / ATT_CHUNK;
    device const float* p = part + (t * n_head + h) * max_chunks * (HD + 2);
    float M = -INFINITY;
    for (uint s = 0; s < n_chunks; s++) M = max(M, p[s * (HD + 2)]);
    float L = 0.0f, o = 0.0f;
    for (uint s = 0; s < n_chunks; s++) {
        float w = exp(p[s * (HD + 2)] - M);
        L += p[s * (HD + 2) + 1] * w;
        o += p[s * (HD + 2) + 2 + d] * w;
    }
    att[t * n_head * HD + h * HD + d] = o / L;
}

// ----- LM head rows: gather + LayerNorm, then argmax per row ----------------

// out[i] = LayerNorm(x[lrows[i]]); one 256-thread threadgroup per output row
// (same maths as layernorm_simd).
kernel void b_ln_gather(device float* out [[buffer(0)]],
                        device const float* x [[buffer(1)]],
                        device const float* w [[buffer(2)]],
                        device const float* b [[buffer(3)]],
                        device const uint* lrows [[buffer(4)]],
                        constant uint& C [[buffer(5)]],
                        threadgroup float* scratch [[threadgroup(0)]],
                        uint i [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint ntg [[threads_per_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint nsg [[simdgroups_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    device const float* xr = x + lrows[i] * C;
    float s = 0.0f;
    for (uint k = tid; k < C; k += ntg) s += xr[k];
    float mean = tg_sum(s, scratch, sg, lane, nsg) / float(C);
    float v = 0.0f;
    for (uint k = tid; k < C; k += ntg) { float d = xr[k] - mean; v += d * d; }
    float var = tg_sum(v, scratch, sg, lane, nsg) / float(C);
    float rstd = 1.0f / sqrt(var + 1e-5f);
    for (uint k = tid; k < C; k += ntg) out[i * C + k] = (xr[k] - mean) * rstd * w[k] + b[k];
}

// One 1024-thread threadgroup per logit row: argmax (ties -> lowest index),
// written to out[i] and to toks[seq, pos + 1] (the next step's input).
kernel void b_argmax_rows(device const float* logits [[buffer(0)]],
                          device uint* out [[buffer(1)]],
                          device uint* toks [[buffer(2)]],
                          device const uint4* rows [[buffer(3)]],
                          device const uint* lrows [[buffer(4)]],
                          constant uint& V [[buffer(5)]],
                          constant uint& step [[buffer(6)]],
                          constant uint& n_ctx [[buffer(7)]],
                          threadgroup float* tgm [[threadgroup(0)]],
                          uint i [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint ntg [[threads_per_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    device const float* l = logits + i * V;
    float v = -INFINITY;
    uint idx = 0xffffffffu;
    for (uint j = tid; j < V; j += ntg) better(v, idx, l[j], j);
    for (uint off = 16; off > 0; off /= 2) better(v, idx, simd_shuffle_down(v, off), simd_shuffle_down(idx, off));
    threadgroup uint* ti = (threadgroup uint*)(tgm + 32);
    if (lane == 0) { tgm[sg] = v; ti[sg] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint nsg = ntg / 32;
        v = lane < nsg ? tgm[lane] : -INFINITY;
        idx = lane < nsg ? ti[lane] : 0xffffffffu;
        for (uint off = 16; off > 0; off /= 2) better(v, idx, simd_shuffle_down(v, off), simd_shuffle_down(idx, off));
        if (lane == 0) {
            out[i] = idx;
            uint4 ri = rows[lrows[i]];
            uint next = ri.y + step + 1;
            if (next < n_ctx) toks[ri.x * n_ctx + next] = idx;
        }
    }
}

// ----- GEMM v2: 64 x 64 tiles, weights staged as half -----------------------
//
// matmul_mma_q stages 32 tokens x 64 outputs per threadgroup with every
// operand in f32: each K step moves 8 KB of weights through threadgroup
// memory for 32 x 64 x 32 MACs. Here a threadgroup owns 64 tokens x 64
// outputs (4 SIMD-groups as 2 x 2, each 32 x 32 = 4 x 4 accumulators), and
// the dequantised weight tile is stored as half (exact for Q4_0/Q8_0 values
// times an f16 scale up to rounding; f16 weights are exact). Activations stay
// f32, accumulation f32 (mixed half x float MMA, as llama.cpp does).
template <typename L>
kernel void matmul_mma2(device float* out [[buffer(0)]],
                        device const float* x [[buffer(1)]],
                        device const uchar* w [[buffer(2)]],
                        device const float* bias [[buffer(3)]],
                        constant uint& n [[buffer(4)]],
                        constant uint& d_in [[buffer(5)]],
                        constant uint& d_out [[buffer(6)]],
                        constant uint& has_bias [[buffer(7)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint BM = 64, BN = 64, BK = 32;
    threadgroup float xs[BM * BK];   // 8 KB
    threadgroup half ws[BN * BK];    // 4 KB
    threadgroup float* cs = xs;      // output staging reuses xs + more: see below
    threadgroup float cs_extra[BM * BN - BM * BK];
    uint row0 = tg.y * BM, col0 = tg.x * BN;
    uint sm = sg / 2, sn = sg % 2;   // this SIMD-group's 32 x 32 quadrant
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < d_in; k0 += BK) {
        for (uint e = tid; e < BM * BK / 4; e += 128) {
            uint r = e / (BK / 4), k4 = e % (BK / 4);
            uint gr = row0 + r;
            float4 v = gr < n ? *((device const float4*)(x + gr * d_in + k0) + k4) : float4(0.0f);
            *((threadgroup float4*)(xs + r * BK) + k4) = v;
        }
        for (uint e = tid; e < BN * BK / 4; e += 128) {
            uint r = e / (BK / 4), k4 = e % (BK / 4);
            uint gc = col0 + r;
            float4 v = gc < d_out ? L::load(w, gc, k0 / 4 + k4, d_in, d_out) : float4(0.0f);
            *((threadgroup half4*)(ws + r * BK) + k4) = half4(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[4];
            simdgroup_half8x8 b[4];
            for (uint i = 0; i < 4; i++) simdgroup_load(a[i], xs + (sm * 32 + i * 8) * BK + kk, BK);
            for (uint j = 0; j < 4; j++)
                simdgroup_load(b[j], ws + (sn * 32 + j * 8) * BK + kk, BK, ulong2(0, 0), true);
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Stage the 64 x 64 result: rows 0..31 in xs (as [32][64]), 32..63 in cs_extra.
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++) {
            uint r = sm * 32 + i * 8, c = sn * 32 + j * 8;
            threadgroup float* base = r < 32 ? cs + r * BN : cs_extra + (r - 32) * BN;
            simdgroup_store(acc[i][j], base + c, BN);
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < BM * BN; e += 128) {
        uint r = e / BN, c = e % BN;
        uint gr = row0 + r, gc = col0 + c;
        float v = r < 32 ? cs[e] : cs_extra[e - 32 * BN];
        if (gr < n && gc < d_out) out[gr * d_out + gc] = v + (has_bias ? bias[gc] : 0.0f);
    }
}

struct LoadF32 {
    static float4 load(device const uchar* w, uint row, uint k4, uint d_in, uint rows) {
        return ((device const float4*)((device const float*)w + row * d_in))[k4];
    }
};

template [[host_name("matmul_mma2_f32")]] kernel void matmul_mma2<LoadF32>(MMA_ARGS);
template [[host_name("matmul_mma2_f16")]] kernel void matmul_mma2<LoadF16>(MMA_ARGS);
template [[host_name("matmul_mma2_q8_0")]] kernel void matmul_mma2<LoadQ8>(MMA_ARGS);
template [[host_name("matmul_mma2_q4_0")]] kernel void matmul_mma2<LoadQ4>(MMA_ARGS);
