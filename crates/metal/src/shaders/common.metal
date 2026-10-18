// nanoinfer Metal kernels. Every file in this directory is concatenated into
// one library and compiled at startup (see gpu.rs). Kernels are only ever
// *added*: each stage of the hillclimb keeps running the kernels it was
// measured with.
//
// Conventions (same as the CPU engine):
// - row-major; weight matrices are [out, in]
// - activations are f32; weights are f32, half, or block-quantised
// - scalar parameters arrive via setBytes as `constant uint&` / `constant float&`
#include <metal_stdlib>
using namespace metal;

constant float SQRT_2_OVER_PI = 0.7978846f;

inline float gelu_tanh(float u) {
    return 0.5f * u * (1.0f + precise::tanh(SQRT_2_OVER_PI * (u + 0.044715f * u * u * u)));
}
