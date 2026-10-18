
// ---------------------------------------------------------------------------
// Stage metal-lanes: more loads in flight per lane for narrow rows.
//
// matvec_epi gives every row a whole SIMD-group and walks the rows one after
// another. A 768-wide Q4_0 row is 24 blocks of 16 bytes, so 8 of the 32 lanes
// idle and each busy lane has exactly ONE 16-byte load in flight before it
// waits on a simd_sum: the kernel is latency-bound, not bandwidth-bound.
//
// Here a SIMD-group works on 4 rows at once, 8 lanes per row (LPR). Lane li of
// a row reads blocks li, li + 8, li + 16, ... (each row's 8 lanes read 128
// contiguous bytes per step), and the R row-sets of a SIMD-group are walked in
// the same loop so each lane has R x (blocks / 8) independent loads to issue.
// The 8-lane partial sums are reduced with simd_shuffle_xor(4, 2, 1).
//
// Weight formats are "unit" functors: `n_units(d_in)` units per row, `unit()`
// returns one unit's dot product with x (Q4_0 unit = one 32-weight block,
// Q8_0 unit = half a block, 16 int8).
// ---------------------------------------------------------------------------

constant uint LPR = 8;  // lanes per row

struct UQ4 {
    static uint n_units(uint d_in) { return d_in / 32; }
    static float unit(device const uchar* w, uint row, uint u, uint d_in, uint d_out, device const float4* x4) {
        uint nb = d_in / 32;
        device const half* scales = (device const half*)(w + d_out * d_in / 2);
        uint4 q = ((device const uint4*)(w + row * d_in / 2))[u];
        return float(scales[row * nb + u]) * q4_block_dot(q, x4 + 8 * u);
    }
};

struct UQ8 {
    static uint n_units(uint d_in) { return d_in / 16; }
    static float unit(device const uchar* w, uint row, uint u, uint d_in, uint d_out, device const float4* x4) {
        device const half* scales = (device const half*)(w + d_out * d_in);
        device const char4* q = (device const char4*)(w + row * d_in) + 4 * u;
        float s = dot(x4[4 * u], float4(q[0])) + dot(x4[4 * u + 1], float4(q[1])) +
                  dot(x4[4 * u + 2], float4(q[2])) + dot(x4[4 * u + 3], float4(q[3]));
        return float(scales[row * (d_in / 32) + u / 2]) * s;
    }
};

struct UF16 {
    static uint n_units(uint d_in) { return d_in / 8; }
    static float unit(device const uchar* w, uint row, uint u, uint d_in, uint d_out, device const float4* x4) {
        device const half4* w4 = (device const half4*)((device const half*)w + row * d_in);
        return dot(x4[2 * u], float4(w4[2 * u])) + dot(x4[2 * u + 1], float4(w4[2 * u + 1]));
    }
};

// Same buffer layout as matvec_epi. A SIMD-group owns R * 4 rows:
// row = row0 + r * 4 + lane / LPR.
template <typename W, uint R, uint EPI>
kernel void matvec_lanes(device float* out [[buffer(0)]],
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
    const uint RPS = 32 / LPR;  // rows per step of a SIMD-group
    uint sub = lane / LPR, li = lane % LPR;
    uint row0 = (tg * nsg + sg) * R * RPS;
    device const float4* x4 = (device const float4*)x;
    uint nu = W::n_units(d_in);
    float acc[R];
    uint rows[R];
    for (uint r = 0; r < R; r++) {
        acc[r] = 0.0f;
        rows[r] = min(row0 + r * RPS + sub, d_out - 1);  // clamp: compute, then discard
    }
    for (uint u = li; u < nu; u += LPR)
        for (uint r = 0; r < R; r++) acc[r] += W::unit(w, rows[r], u, d_in, d_out, x4);
    for (uint r = 0; r < R; r++) {
        float s = acc[r];
        s += simd_shuffle_xor(s, 4);
        s += simd_shuffle_xor(s, 2);
        s += simd_shuffle_xor(s, 1);
        uint row = row0 + r * RPS + sub;
        if (li != 0 || row >= d_out) continue;
        s += has_bias ? bias[row] : 0.0f;
        if (EPI == EPI_PLAIN) out[row] = s;
        else if (EPI == EPI_GELU) out[row] = gelu_tanh(s);
        else if (EPI == EPI_RESID) out[row] += s;
        else if (EPI == EPI_QKV) {
            uint C = d_out / 3;
            if (row < C) out[row] = s;
            else if (row < 2 * C) kc[pos * C + row - C] = s;
            else vc[pos * C + row - 2 * C] = s;
        } else {
            uint C = d_out / 3;
            if (row < C) out[row] = s;
            else if (row < 2 * C) ((device half*)kc)[pos * C + row - C] = half(s);
            else ((device half*)vc)[pos * C + row - 2 * C] = half(s);
        }
    }
}

#define LANES_SET(NAME, W, R) \
template [[host_name("mvl_qkv_h_" NAME)]] kernel void matvec_lanes<W, R, EPI_QKV_H>(EPI_ARGS); \
template [[host_name("mvl_gelu_" NAME)]] kernel void matvec_lanes<W, R, EPI_GELU>(EPI_ARGS); \
template [[host_name("mvl_resid_" NAME)]] kernel void matvec_lanes<W, R, EPI_RESID>(EPI_ARGS); \
template [[host_name("mvl_" NAME)]] kernel void matvec_lanes<W, R, EPI_PLAIN>(EPI_ARGS);
LANES_SET("q4_0_r1", UQ4, 1)
LANES_SET("q4_0_r2", UQ4, 2)
LANES_SET("q8_0_r1", UQ8, 1)
LANES_SET("q8_0_r2", UQ8, 2)
LANES_SET("f16_r1", UF16, 1)
LANES_SET("f16_r2", UF16, 2)
