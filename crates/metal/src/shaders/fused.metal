
// ---------------------------------------------------------------------------
// Stage metal-fused: fewer dispatches per token.
//
// At ~2 ms per token the small ops and their dispatch gaps are a large share
// of the token, so decode (n = 1) folds them into the matvecs:
//   ln_1 + qkv + kv_write    -> matvec_ln_qkv   (LayerNorm in the prologue,
//                                                K/V rows written straight
//                                                into the cache)
//   attn_proj + residual     -> matvec_resid    (x += W.a + b in the epilogue)
//   ln_2 + fc + gelu         -> matvec_ln_gelu
//   fc_proj + residual       -> matvec_resid
//   ln_f + lm_head           -> matvec_ln
// 11 dispatches per layer become 5 (+ attention). Every threadgroup stages its
// input row in threadgroup memory (normalising it on the way in when LN is
// fused; each threadgroup recomputes the row's mean/variance, 768 floats from
// L2, which is cheaper than a dispatch).
//
// Weight formats are functors returning one lane's partial dot product of a
// weight row with the staged x (threadgroup float4s).
// ---------------------------------------------------------------------------

struct WF32 {
    template <typename XP>
    static float dot_row(device const uchar* w, uint row, uint d_in, uint d_out, XP xs, uint lane) {
        device const float4* w4 = (device const float4*)((device const float*)w + row * d_in);
        float acc = 0.0f;
        for (uint i = lane; i < d_in / 4; i += 32) acc += dot(xs[i], w4[i]);
        return acc;
    }
};
struct WF16 {
    template <typename XP>
    static float dot_row(device const uchar* w, uint row, uint d_in, uint d_out, XP xs, uint lane) {
        device const half4* w4 = (device const half4*)((device const half*)w + row * d_in);
        float acc = 0.0f;
        for (uint i = lane; i < d_in / 8; i += 32)
            acc += dot(xs[2 * i], float4(w4[2 * i])) + dot(xs[2 * i + 1], float4(w4[2 * i + 1]));
        return acc;
    }
};
struct WQ8 {
    template <typename XP>
    static float dot_row(device const uchar* w, uint row, uint d_in, uint d_out, XP xs, uint lane) {
        device const half* scales = (device const half*)(w + d_out * d_in);
        device const char4* q = (device const char4*)(w + row * d_in);
        uint nb = d_in / 32;
        float acc = 0.0f;
        for (uint i = lane; i < d_in / 16; i += 32) {
            float s = dot(xs[4 * i], float4(q[4 * i])) + dot(xs[4 * i + 1], float4(q[4 * i + 1])) +
                      dot(xs[4 * i + 2], float4(q[4 * i + 2])) + dot(xs[4 * i + 3], float4(q[4 * i + 3]));
            acc += float(scales[row * nb + i / 2]) * s;
        }
        return acc;
    }
};
struct WQ4 {
    template <typename XP>
    static float dot_row(device const uchar* w, uint row, uint d_in, uint d_out, XP xs, uint lane) {
        uint nb = d_in / 32;
        device const half* scales = (device const half*)(w + d_out * d_in / 2);
        device const uint4* q4 = (device const uint4*)(w + row * d_in / 2);
        float acc = 0.0f;
        for (uint b = lane; b < nb; b += 32) {
            uint4 q = q4[b];
            XP xb = xs + 8 * b;
            float s = 0.0f;
            for (uint j = 0; j < 4; j++) {
                uint v = q[j];
                float4 lo = float4(float(v & 15u), float((v >> 8) & 15u), float((v >> 16) & 15u), float((v >> 24) & 15u));
                float4 hi = float4(float((v >> 4) & 15u), float((v >> 12) & 15u), float((v >> 20) & 15u), float(v >> 28));
                s += dot(lo - 8.0f, xb[j]) + dot(hi - 8.0f, xb[4 + j]);
            }
            acc += float(scales[row * nb + b]) * s;
        }
        return acc;
    }
};

// Stage the input row into threadgroup memory, optionally LayerNorm'd.
inline void stage_x(threadgroup float* xs, device const float* x, uint d_in, bool ln, device const float* lw,
                    device const float* lb, threadgroup float* scratch, uint tid, uint ntg, uint sg, uint lane, uint nsg) {
    if (!ln) {
        for (uint i = tid; i < d_in; i += ntg) xs[i] = x[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return;
    }
    float s = 0.0f;
    for (uint i = tid; i < d_in; i += ntg) { float v = x[i]; xs[i] = v; s += v; }
    float mean = tg_sum(s, scratch, sg, lane, nsg) / float(d_in);
    float v = 0.0f;
    for (uint i = tid; i < d_in; i += ntg) { float d = xs[i] - mean; v += d * d; }
    float var = tg_sum(v, scratch, sg, lane, nsg) / float(d_in);
    float rstd = 1.0f / sqrt(var + 1e-5f);
    for (uint i = tid; i < d_in; i += ntg) xs[i] = (xs[i] - mean) * rstd * lw[i] + lb[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

constant uint EPI_PLAIN = 0;  // out[row] = s (+ bias)
constant uint EPI_GELU = 1;   // out[row] = gelu(s + bias)
constant uint EPI_RESID = 2;  // out[row] += s + bias
constant uint EPI_QKV = 3;    // q -> out, k/v -> cache rows at pos

// One fused decode matvec. R rows per SIMD-group; threadgroup = nsg SIMD-groups.
// Buffers: 0 out, 1 x, 2 w, 3 bias, 4 ln_w, 5 ln_b, 6 k_cache, 7 v_cache,
// 8 d_in, 9 d_out, 10 has_bias, 11 pos (QKV only).
template <typename W, uint R, bool LN, uint EPI>
kernel void matvec_fused(device float* out [[buffer(0)]],
                         device const float* x [[buffer(1)]],
                         device const uchar* w [[buffer(2)]],
                         device const float* bias [[buffer(3)]],
                         device const float* lw [[buffer(4)]],
                         device const float* lb [[buffer(5)]],
                         device float* kc [[buffer(6)]],
                         device float* vc [[buffer(7)]],
                         constant uint& d_in [[buffer(8)]],
                         constant uint& d_out [[buffer(9)]],
                         constant uint& has_bias [[buffer(10)]],
                         constant uint& pos [[buffer(11)]],
                         threadgroup float* tgm [[threadgroup(0)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint ntg [[threads_per_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint nsg [[simdgroups_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float* scratch = tgm;     // [32]
    threadgroup float* xs = tgm + 32;     // [d_in]
    stage_x(xs, x, d_in, LN, lw, lb, scratch, tid, ntg, sg, lane, nsg);
    uint row0 = (tg * nsg + sg) * R;
    for (uint r = 0; r < R; r++) {
        uint row = row0 + r;
        if (row >= d_out) return;
        float s = simd_sum(W::dot_row(w, row, d_in, d_out, (threadgroup const float4*)xs, lane));
        if (lane != 0) continue;
        s += has_bias ? bias[row] : 0.0f;
        if (EPI == EPI_PLAIN) out[row] = s;
        else if (EPI == EPI_GELU) out[row] = gelu_tanh(s);
        else if (EPI == EPI_RESID) out[row] += s;
        else {  // EPI_QKV: d_out = 3C
            uint C = d_out / 3;
            if (row < C) out[row] = s;
            else if (row < 2 * C) kc[pos * C + row - C] = s;
            else vc[pos * C + row - 2 * C] = s;
        }
    }
}

#define FUSED_ARGS device float*, device const float*, device const uchar*, device const float*, device const float*, device const float*, device float*, device float*, constant uint&, constant uint&, constant uint&, constant uint&, threadgroup float*, uint, uint, uint, uint, uint, uint
#define FUSED_SET(NAME, W, R) \
template [[host_name("mvf_ln_qkv_" NAME)]] kernel void matvec_fused<W, R, true, EPI_QKV>(FUSED_ARGS); \
template [[host_name("mvf_ln_gelu_" NAME)]] kernel void matvec_fused<W, R, true, EPI_GELU>(FUSED_ARGS); \
template [[host_name("mvf_resid_" NAME)]] kernel void matvec_fused<W, R, false, EPI_RESID>(FUSED_ARGS); \
template [[host_name("mvf_ln_" NAME)]] kernel void matvec_fused<W, R, true, EPI_PLAIN>(FUSED_ARGS);
FUSED_SET("f32", WF32, 4)
FUSED_SET("f16", WF16, 4)
FUSED_SET("q8_0", WQ8, 4)
FUSED_SET("q4_0", WQ4, 4)

// Epilogue-only fusion (the version metal-fused keeps): x is read straight
// from device memory like the unfused matvec (no staging, no LN prologue), and
// the epilogue does bias + GELU / residual / KV-cache write. Same buffer
// layout as matvec_fused (4 and 5 unused).
template <typename W, uint R, uint EPI>
kernel void matvec_epi(device float* out [[buffer(0)]],
                       device const float* x [[buffer(1)]],
                       device const uchar* w [[buffer(2)]],
                       device const float* bias [[buffer(3)]],
                       device float* kc [[buffer(6)]],
                       device float* vc [[buffer(7)]],
                       constant uint& d_in [[buffer(8)]],
                       constant uint& d_out [[buffer(9)]],
                       constant uint& has_bias [[buffer(10)]],
                       constant uint& pos [[buffer(11)]],
                       uint tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint nsg [[simdgroups_per_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (tg * nsg + sg) * R;
    device const float4* x4 = (device const float4*)x;
    for (uint r = 0; r < R; r++) {
        uint row = row0 + r;
        if (row >= d_out) return;
        float s = simd_sum(W::dot_row(w, row, d_in, d_out, x4, lane));
        if (lane != 0) continue;
        s += has_bias ? bias[row] : 0.0f;
        if (EPI == EPI_PLAIN) out[row] = s;
        else if (EPI == EPI_GELU) out[row] = gelu_tanh(s);
        else if (EPI == EPI_RESID) out[row] += s;
        else {
            uint C = d_out / 3;
            if (row < C) out[row] = s;
            else if (row < 2 * C) kc[pos * C + row - C] = s;
            else vc[pos * C + row - 2 * C] = s;
        }
    }
}

#define EPI_ARGS device float*, device const float*, device const uchar*, device const float*, device float*, device float*, constant uint&, constant uint&, constant uint&, constant uint&, uint, uint, uint, uint
#define EPI_SET(NAME, W, R) \
template [[host_name("mve_qkv_" NAME)]] kernel void matvec_epi<W, R, EPI_QKV>(EPI_ARGS); \
template [[host_name("mve_gelu_" NAME)]] kernel void matvec_epi<W, R, EPI_GELU>(EPI_ARGS); \
template [[host_name("mve_resid_" NAME)]] kernel void matvec_epi<W, R, EPI_RESID>(EPI_ARGS); \
template [[host_name("mve_" NAME)]] kernel void matvec_epi<W, R, EPI_PLAIN>(EPI_ARGS);
EPI_SET("f32", WF32, 4)
EPI_SET("f16", WF16, 4)
EPI_SET("q8_0", WQ8, 2)
EPI_SET("q4_0", WQ4, 2)

// ----- GPU argmax: 50257 logits -> 1 token, without the CPU reading them ----

// Phase 1: each threadgroup reduces a strided slice to (max, index).
// Ties go to the lowest index (like torch.argmax and sampler::argmax).
inline void better(thread float& v, thread uint& i, float v2, uint i2) {
    if (v2 > v || (v2 == v && i2 < i)) { v = v2; i = i2; }
}

kernel void argmax_partial(device const float* logits [[buffer(0)]],
                           device float* pv [[buffer(1)]],
                           device uint* pi [[buffer(2)]],
                           constant uint& len [[buffer(3)]],
                           threadgroup float* tgm [[threadgroup(0)]],
                           uint tg [[threadgroup_position_in_grid]],
                           uint ntgs [[threadgroups_per_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint ntg [[threads_per_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    float v = -INFINITY;
    uint idx = 0xffffffffu;
    for (uint j = tg * ntg + tid; j < len; j += ntgs * ntg) better(v, idx, logits[j], j);
    for (uint off = 16; off > 0; off /= 2) better(v, idx, simd_shuffle_down(v, off), simd_shuffle_down(idx, off));
    threadgroup uint* ti = (threadgroup uint*)(tgm + 32);
    if (lane == 0) { tgm[sg] = v; ti[sg] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint nsg = ntg / 32;
        v = lane < nsg ? tgm[lane] : -INFINITY;
        idx = lane < nsg ? ti[lane] : 0xffffffffu;
        for (uint off = 16; off > 0; off /= 2) better(v, idx, simd_shuffle_down(v, off), simd_shuffle_down(idx, off));
        if (lane == 0) { pv[tg] = v; pi[tg] = idx; }
    }
}

// Phase 2: one SIMD-group reduces the partials; writes the token to out[0]
// and to gen[slot] (the next step's input when decoding without the CPU).
kernel void argmax_final(device const float* pv [[buffer(0)]],
                         device const uint* pi [[buffer(1)]],
                         constant uint& n_partial [[buffer(2)]],
                         device uint* out [[buffer(3)]],
                         device uint* gen [[buffer(4)]],
                         constant uint& slot [[buffer(5)]],
                         uint lane [[thread_index_in_simdgroup]]) {
    float v = -INFINITY;
    uint idx = 0xffffffffu;
    for (uint j = lane; j < n_partial; j += 32) better(v, idx, pv[j], pi[j]);
    for (uint off = 16; off > 0; off /= 2) better(v, idx, simd_shuffle_down(v, off), simd_shuffle_down(idx, off));
    if (lane == 0) { out[0] = idx; gen[slot] = idx; }
}

// Embedding of a single token whose id lives in a GPU buffer (gen[slot]).
kernel void embed_gen(device float* x [[buffer(0)]],
                      device const float* wte [[buffer(1)]],
                      device const float* wpe [[buffer(2)]],
                      device const uint* gen [[buffer(3)]],
                      constant uint& slot [[buffer(4)]],
                      constant uint& pos [[buffer(5)]],
                      constant uint& C [[buffer(6)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= C) return;
    x[i] = wte[gen[slot] * C + i] + wpe[pos * C + i];
}
