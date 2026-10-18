
// ---------------------------------------------------------------------------
// Stage metal-naive: a literal port of the CPU reference kernels. One thread per
// output element, serial loops, no threadgroup cooperation. Correct and easy
// to check; it shows what a GPU does with CPU-shaped code.
// ---------------------------------------------------------------------------

// x[t] = wte[tokens[t]] + wpe[pos0 + t]. One thread per element of [n, C].
kernel void embed(device float* x [[buffer(0)]],
                  device const float* wte [[buffer(1)]],
                  device const float* wpe [[buffer(2)]],
                  device const uint* tokens [[buffer(3)]],
                  constant uint& pos0 [[buffer(4)]],
                  constant uint& C [[buffer(5)]],
                  constant uint& n [[buffer(6)]],
                  uint gid [[thread_position_in_grid]]) {
    if (gid >= n * C) return;
    uint t = gid / C, i = gid % C;
    x[gid] = wte[tokens[t] * C + i] + wpe[(pos0 + t) * C + i];
}

// LayerNorm, one thread per row (two-pass mean/variance, like the CPU reference).
kernel void layernorm_naive(device float* out [[buffer(0)]],
                            device const float* x [[buffer(1)]],
                            device const float* w [[buffer(2)]],
                            device const float* b [[buffer(3)]],
                            constant uint& n [[buffer(4)]],
                            constant uint& C [[buffer(5)]],
                            uint row [[thread_position_in_grid]]) {
    if (row >= n) return;
    device const float* xr = x + row * C;
    float mean = 0.0f;
    for (uint i = 0; i < C; i++) mean += xr[i];
    mean /= float(C);
    float var = 0.0f;
    for (uint i = 0; i < C; i++) { float d = xr[i] - mean; var += d * d; }
    var /= float(C);
    float rstd = 1.0f / sqrt(var + 1e-5f);
    for (uint i = 0; i < C; i++) out[row * C + i] = (xr[i] - mean) * rstd * w[i] + b[i];
}

// out[t, o] = x[t] . W[o] + bias[o], one thread per output element.
// Neighbouring threads read neighbouring *rows* of W, i.e. addresses d_in*4
// bytes apart: every load in a SIMD-group touches a different cache line.
kernel void matmul_naive(device float* out [[buffer(0)]],
                         device const float* x [[buffer(1)]],
                         device const float* w [[buffer(2)]],
                         device const float* bias [[buffer(3)]],
                         constant uint& n [[buffer(4)]],
                         constant uint& d_in [[buffer(5)]],
                         constant uint& d_out [[buffer(6)]],
                         constant uint& has_bias [[buffer(7)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= n * d_out) return;
    uint t = gid / d_out, o = gid % d_out;
    device const float* xr = x + t * d_in;
    device const float* wr = w + o * d_in;
    float acc = has_bias ? bias[o] : 0.0f;
    for (uint i = 0; i < d_in; i++) acc += xr[i] * wr[i];
    out[gid] = acc;
}

kernel void gelu_naive(device float* x [[buffer(0)]],
                       constant uint& len [[buffer(1)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= len) return;
    x[gid] = gelu_tanh(x[gid]);
}

kernel void residual_add(device float* x [[buffer(0)]],
                         device const float* y [[buffer(1)]],
                         constant uint& len [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= len) return;
    x[gid] += y[gid];
}

// Copy the K and V thirds of qkv [n, 3C] into this layer's cache at pos0.
kernel void kv_write(device float* kc [[buffer(0)]],
                     device float* vc [[buffer(1)]],
                     device const float* qkv [[buffer(2)]],
                     constant uint& pos0 [[buffer(3)]],
                     constant uint& C [[buffer(4)]],
                     constant uint& n [[buffer(5)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= n * C) return;
    uint t = gid / C, i = gid % C;
    kc[(pos0 + t) * C + i] = qkv[t * 3 * C + C + i];
    vc[(pos0 + t) * C + i] = qkv[t * 3 * C + 2 * C + i];
}

// Causal attention, one thread per (query, head): scores into scratch,
// softmax, weighted sum of V. Exactly the CPU reference's loop nest.
kernel void attention_naive(device float* out [[buffer(0)]],
                            device const float* qkv [[buffer(1)]],
                            device const float* kc [[buffer(2)]],
                            device const float* vc [[buffer(3)]],
                            device float* scores [[buffer(4)]],
                            constant uint& n [[buffer(5)]],
                            constant uint& pos0 [[buffer(6)]],
                            constant uint& n_head [[buffer(7)]],
                            constant uint& hd [[buffer(8)]],
                            constant uint& max_kv [[buffer(9)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= n * n_head) return;
    uint C = n_head * hd;
    uint t = gid / n_head, h = gid % n_head;
    uint n_kv = pos0 + t + 1;
    device const float* q = qkv + t * 3 * C + h * hd;
    device float* s = scores + gid * max_kv;
    float scale = 1.0f / sqrt(float(hd));
    float mx = -INFINITY;
    for (uint j = 0; j < n_kv; j++) {
        device const float* k = kc + j * C + h * hd;
        float d = 0.0f;
        for (uint e = 0; e < hd; e++) d += q[e] * k[e];
        s[j] = d * scale;
        mx = max(mx, s[j]);
    }
    float sum = 0.0f;
    for (uint j = 0; j < n_kv; j++) { s[j] = exp(s[j] - mx); sum += s[j]; }
    float inv = 1.0f / sum;
    for (uint j = 0; j < n_kv; j++) s[j] *= inv;
    device float* o = out + t * C + h * hd;
    for (uint e = 0; e < hd; e++) {
        float acc = 0.0f;
        for (uint j = 0; j < n_kv; j++) acc += s[j] * vc[j * C + h * hd + e];
        o[e] = acc;
    }
}
