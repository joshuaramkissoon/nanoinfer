
// ---------------------------------------------------------------------------
// Stage metal-f16 / metal-q8 / metal-q4: fewer bytes per weight.
//
// Decode reads every weight once per token, so bytes per weight *is* the
// speed. Weights are dequantised in registers; activations and accumulation
// stay f32.
//
// Quantised layouts are structure-of-arrays (not GGML's interleaved
// {half d; qs[...]} blocks, whose 34/18-byte strides break 4-byte alignment):
//   [ qs: rows * d_in bytes (Q8_0) or rows * d_in / 2 bytes (Q4_0) ]
//   [ scales: rows * d_in / 32 halfs, one per block of 32 weights ]
// Same total bytes as GGML. Q4_0 packs a block of 32 as 16 bytes: byte i holds
// element i in its low nibble and element i + 16 in its high nibble; value =
// (nibble - 8) * scale. Q8_0: value = int8 * scale.
// ---------------------------------------------------------------------------

// f16 weights. Lane l reads 8 halfs (two half4, 16 bytes) per step, so a
// SIMD-group reads 512 contiguous bytes per row per step.
template <uint R>
kernel void matvec_f16(device float* out [[buffer(0)]],
                       device const float* x [[buffer(1)]],
                       device const half* w [[buffer(2)]],
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
    uint k8 = d_in / 8;
    device const float4* x4 = (device const float4*)(x + t * d_in);
    float acc[R];
    for (uint r = 0; r < R; r++) acc[r] = 0.0f;
    for (uint i = lane; i < k8; i += 32) {
        float4 xa = x4[2 * i], xb = x4[2 * i + 1];
        for (uint r = 0; r < R; r++) {
            uint row = min(row0 + r, d_out - 1);
            device const half4* w4 = (device const half4*)(w + row * d_in);
            acc[r] += dot(xa, float4(w4[2 * i])) + dot(xb, float4(w4[2 * i + 1]));
        }
    }
    for (uint r = 0; r < R; r++) {
        float s = simd_sum(acc[r]);
        uint row = row0 + r;
        if (lane == 0 && row < d_out) out[t * d_out + row] = s + (has_bias ? bias[row] : 0.0f);
    }
}

#define MATVEC_ARGS(T) device float*, device const float*, device const T*, device const float*, constant uint&, constant uint&, constant uint&, constant uint&, uint2, uint, uint, uint
template [[host_name("matvec_f16_r1")]] kernel void matvec_f16<1>(MATVEC_ARGS(half));
template [[host_name("matvec_f16_r2")]] kernel void matvec_f16<2>(MATVEC_ARGS(half));
template [[host_name("matvec_f16_r4")]] kernel void matvec_f16<4>(MATVEC_ARGS(half));
template [[host_name("matvec_f16_r8")]] kernel void matvec_f16<8>(MATVEC_ARGS(half));

// Q8_0: lane l reads 16 int8 (4 x char4) = half a block per step.
template <uint R>
kernel void matvec_q8_0(device float* out [[buffer(0)]],
                        device const float* x [[buffer(1)]],
                        device const uchar* w [[buffer(2)]],
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
    device const half* scales = (device const half*)(w + d_out * d_in);
    uint k16 = d_in / 16, nb = d_in / 32;
    device const float4* x4 = (device const float4*)(x + t * d_in);
    float acc[R];
    for (uint r = 0; r < R; r++) acc[r] = 0.0f;
    for (uint i = lane; i < k16; i += 32) {
        float4 xv[4] = {x4[4 * i], x4[4 * i + 1], x4[4 * i + 2], x4[4 * i + 3]};
        for (uint r = 0; r < R; r++) {
            uint row = min(row0 + r, d_out - 1);
            device const char4* q = (device const char4*)(w + row * d_in) + 4 * i;
            float d = float(scales[row * nb + i / 2]);
            float s = dot(xv[0], float4(q[0])) + dot(xv[1], float4(q[1])) + dot(xv[2], float4(q[2])) +
                      dot(xv[3], float4(q[3]));
            acc[r] += d * s;
        }
    }
    for (uint r = 0; r < R; r++) {
        float s = simd_sum(acc[r]);
        uint row = row0 + r;
        if (lane == 0 && row < d_out) out[t * d_out + row] = s + (has_bias ? bias[row] : 0.0f);
    }
}

template [[host_name("matvec_q8_0_r1")]] kernel void matvec_q8_0<1>(MATVEC_ARGS(uchar));
template [[host_name("matvec_q8_0_r2")]] kernel void matvec_q8_0<2>(MATVEC_ARGS(uchar));
template [[host_name("matvec_q8_0_r4")]] kernel void matvec_q8_0<4>(MATVEC_ARGS(uchar));

// Q4_0: lane l reads one whole block (16 bytes = uint4) per step.
inline float q4_block_dot(uint4 q, device const float4* xb) {
    // xb points at the 32 floats of this block: elements 0..15 pair with the
    // low nibbles of bytes 0..15, elements 16..31 with the high nibbles.
    float s = 0.0f;
    for (uint j = 0; j < 4; j++) {
        uint v = q[j];  // bytes 4j .. 4j+3
        float4 lo = float4(float(v & 15u), float((v >> 8) & 15u), float((v >> 16) & 15u), float((v >> 24) & 15u));
        float4 hi = float4(float((v >> 4) & 15u), float((v >> 12) & 15u), float((v >> 20) & 15u), float(v >> 28));
        s += dot(lo - 8.0f, xb[j]) + dot(hi - 8.0f, xb[4 + j]);
    }
    return s;
}

template <uint R>
kernel void matvec_q4_0(device float* out [[buffer(0)]],
                        device const float* x [[buffer(1)]],
                        device const uchar* w [[buffer(2)]],
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
    uint nb = d_in / 32;
    device const half* scales = (device const half*)(w + d_out * d_in / 2);
    device const float4* x4 = (device const float4*)(x + t * d_in);
    float acc[R];
    for (uint r = 0; r < R; r++) acc[r] = 0.0f;
    for (uint b = lane; b < nb; b += 32) {
        for (uint r = 0; r < R; r++) {
            uint row = min(row0 + r, d_out - 1);
            uint4 q = ((device const uint4*)(w + row * d_in / 2))[b];
            acc[r] += float(scales[row * nb + b]) * q4_block_dot(q, x4 + 8 * b);
        }
    }
    for (uint r = 0; r < R; r++) {
        float s = simd_sum(acc[r]);
        uint row = row0 + r;
        if (lane == 0 && row < d_out) out[t * d_out + row] = s + (has_bias ? bias[row] : 0.0f);
    }
}

template [[host_name("matvec_q4_0_r1")]] kernel void matvec_q4_0<1>(MATVEC_ARGS(uchar));
template [[host_name("matvec_q4_0_r2")]] kernel void matvec_q4_0<2>(MATVEC_ARGS(uchar));
template [[host_name("matvec_q4_0_r4")]] kernel void matvec_q4_0<4>(MATVEC_ARGS(uchar));

// ----- prefill: the metal-mma GEMM with a dequantising weight-tile loader ----

// Load weights (row, 4*k4 .. 4*k4+3) as float4.
struct LoadF16 {
    static float4 load(device const uchar* w, uint row, uint k4, uint d_in, uint rows) {
        return float4(((device const half4*)((device const half*)w + row * d_in))[k4]);
    }
};
struct LoadQ8 {
    static float4 load(device const uchar* w, uint row, uint k4, uint d_in, uint rows) {
        device const half* scales = (device const half*)(w + rows * d_in);
        char4 q = ((device const char4*)(w + row * d_in))[k4];
        return float4(q) * float(scales[row * (d_in / 32) + k4 / 8]);
    }
};
struct LoadQ4 {
    static float4 load(device const uchar* w, uint row, uint k4, uint d_in, uint rows) {
        device const half* scales = (device const half*)(w + rows * d_in / 2);
        uint b = k4 / 8, p = (k4 % 8) * 4;  // block, position of first element within it
        uint v = *((device const uint*)(w + row * d_in / 2 + b * 16 + (p % 16)));
        uint sh = p < 16 ? 0 : 4;
        float4 q = float4(float((v >> sh) & 15u), float((v >> (8 + sh)) & 15u), float((v >> (16 + sh)) & 15u),
                          float((v >> (24 + sh)) & 15u));
        return (q - 8.0f) * float(scales[row * (d_in / 32) + b]);
    }
};

template <typename L>
kernel void matmul_mma_q(device float* out [[buffer(0)]],
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
    const uint BM = 32, BN = 64, BK = 32;
    threadgroup float xs[BM * BK];
    threadgroup float ws[BN * BK];
    threadgroup float cs[BM * BN];
    uint row0 = tg.y * BM, col0 = tg.x * BN;
    simdgroup_float8x8 acc[4][2];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 2; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
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
            *((threadgroup float4*)(ws + r * BK) + k4) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[4], b[2];
            for (uint i = 0; i < 4; i++) simdgroup_load(a[i], xs + (i * 8) * BK + kk, BK);
            for (uint j = 0; j < 2; j++)
                simdgroup_load(b[j], ws + (sg * 16 + j * 8) * BK + kk, BK, ulong2(0, 0), true);
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 2; j++) simdgroup_store(acc[i][j], cs + (i * 8) * BN + sg * 16 + j * 8, BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < BM * BN; e += 128) {
        uint r = e / BN, c = e % BN;
        uint gr = row0 + r, gc = col0 + c;
        if (gr < n && gc < d_out) out[gr * d_out + gc] = cs[e] + (has_bias ? bias[gc] : 0.0f);
    }
}

#define MMA_ARGS device float*, device const float*, device const uchar*, device const float*, constant uint&, constant uint&, constant uint&, constant uint&, uint2, uint, uint
template [[host_name("matmul_mma_f16")]] kernel void matmul_mma_q<LoadF16>(MMA_ARGS);
template [[host_name("matmul_mma_q8_0")]] kernel void matmul_mma_q<LoadQ8>(MMA_ARGS);
template [[host_name("matmul_mma_q4_0")]] kernel void matmul_mma_q<LoadQ4>(MMA_ARGS);
