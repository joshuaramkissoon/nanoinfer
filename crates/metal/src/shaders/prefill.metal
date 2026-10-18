
// ---------------------------------------------------------------------------
// Stage metal-tiled / metal-mma: prefill matmuls.
//
// out[n, d_out] = X[n, d_in] . W[d_out, d_in]^T + bias. Both operands are
// contiguous along K = d_in, so a tile of X and a tile of W are loaded the
// same way. Prefill reuses every weight n times; the point of tiling is to
// load each weight from DRAM once per *tile* of tokens instead of once per
// token, and to do the reuse out of threadgroup memory / registers.
// ---------------------------------------------------------------------------

// Classic shared-memory tiling: a 256-thread threadgroup computes a 32x32
// output tile (32 tokens x 32 outputs); each thread owns a 1x4 strip. K is
// walked in steps of 32: each step stages X[32x32] and W[32x32] in
// threadgroup memory (+1 padding column against bank conflicts).
kernel void matmul_tiled(device float* out [[buffer(0)]],
                         device const float* x [[buffer(1)]],
                         device const float* w [[buffer(2)]],
                         device const float* bias [[buffer(3)]],
                         constant uint& n [[buffer(4)]],
                         constant uint& d_in [[buffer(5)]],
                         constant uint& d_out [[buffer(6)]],
                         constant uint& has_bias [[buffer(7)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    const uint T = 32;
    threadgroup float xs[T][T + 1];
    threadgroup float ws[T][T + 1];
    uint row0 = tg.y * T;     // first token
    uint col0 = tg.x * T;     // first output feature
    uint ty = tid / 8;        // token within tile, 0..31
    uint tx = (tid % 8) * 4;  // first of 4 outputs within tile
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k0 = 0; k0 < d_in; k0 += T) {
        // 1024 elements per tile, 4 per thread; consecutive threads read
        // consecutive k (coalesced).
        for (uint e = tid; e < T * T; e += 256) {
            uint r = e / T, k = e % T;
            uint gr = row0 + r, gc = col0 + r;
            xs[r][k] = gr < n ? x[gr * d_in + k0 + k] : 0.0f;
            ws[r][k] = gc < d_out ? w[gc * d_in + k0 + k] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < T; k++) {
            float a = xs[ty][k];
            for (uint j = 0; j < 4; j++) acc[j] += a * ws[tx + j][k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint r = row0 + ty;
    if (r >= n) return;
    for (uint j = 0; j < 4; j++) {
        uint c = col0 + tx + j;
        if (c < d_out) out[r * d_out + c] = acc[j] + (has_bias ? bias[c] : 0.0f);
    }
}

// simdgroup_matrix GEMM. A 128-thread threadgroup (4 SIMD-groups) computes a
// 32 (tokens) x 64 (outputs) tile; SIMD-group s owns a 32x16 slab = 4x2
// simdgroup_float8x8 accumulators. Per K step of 32, X[32x32] and W[64x32] are
// staged in threadgroup memory; W is loaded *transposed* by simdgroup_load so
// the hardware 8x8 multiply-accumulate sees X . W^T directly.
kernel void matmul_mma(device float* out [[buffer(0)]],
                       device const float* x [[buffer(1)]],
                       device const float* w [[buffer(2)]],
                       device const float* bias [[buffer(3)]],
                       constant uint& n [[buffer(4)]],
                       constant uint& d_in [[buffer(5)]],
                       constant uint& d_out [[buffer(6)]],
                       constant uint& has_bias [[buffer(7)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint BM = 32, BN = 64, BK = 32;
    threadgroup float xs[BM * BK];   // [32 tokens][32 k]
    threadgroup float ws[BN * BK];   // [64 outs][32 k]
    threadgroup float cs[BM * BN];   // output staging [32][64]
    uint row0 = tg.y * BM, col0 = tg.x * BN;
    simdgroup_float8x8 acc[4][2];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 2; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < d_in; k0 += BK) {
        // float4 loads: X tile = 256 float4s, W tile = 512 float4s, 128 threads.
        for (uint e = tid; e < BM * BK / 4; e += 128) {
            uint r = e / (BK / 4), k4 = e % (BK / 4);
            uint gr = row0 + r;
            float4 v = gr < n ? *((device const float4*)(x + gr * d_in + k0) + k4) : float4(0.0f);
            *((threadgroup float4*)(xs + r * BK) + k4) = v;
        }
        for (uint e = tid; e < BN * BK / 4; e += 128) {
            uint r = e / (BK / 4), k4 = e % (BK / 4);
            uint gc = col0 + r;
            float4 v = gc < d_out ? *((device const float4*)(w + gc * d_in + k0) + k4) : float4(0.0f);
            *((threadgroup float4*)(ws + r * BK) + k4) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[4], b[2];
            for (uint i = 0; i < 4; i++) simdgroup_load(a[i], xs + (i * 8) * BK + kk, BK);
            // B = W^T: element (k, o) lives at ws[o * BK + k]; transposed load.
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
