// Per-stage one-liners for tooltips. What a stage *actually runs* in this wasm
// build comes from the engine itself (StageInfo.wasmNote); unknown stages need
// no entry: the stage list always comes from the wasm build (Stage::ALL plus
// the browser-only rungs).

export const STAGE_NOTES: Record<string, { what: string }> = {
  naive: { what: "recompute the whole context for every token" },
  "kv-cache": { what: "cache keys and values; one new row per token" },
  unrolled: { what: "16 independent accumulators; LLVM vectorises them (simd128 here)" },
  neon: { what: "hand-written aarch64 NEON matmul" },
  tiled: { what: "4×4 register-blocked matmul for prefill" },
  threads: { what: "matmuls split across cores (rayon)" },
  "spin-pool": { what: "threads on a persistent spinning pool" },
  "wasm-simd": { what: "explicit wasm simd128 matmul, 4×4 register tile for prefill (browser only)" },
  "wasm-fma": { what: "the same with relaxed-SIMD fused multiply-add (browser only)" },
  "webgpu-naive": { what: "WebGPU: WGSL port of the CPU kernels, submit + wait after every op" },
  "webgpu-onesubmit": { what: "WebGPU: same kernels, one command encoder and one submit per token" },
  "webgpu-matvec": { what: "WebGPU: 64 threads per weight row, vec4 loads, workgroup reduction" },
  "webgpu-par": { what: "WebGPU: 256-thread workgroups for LayerNorm and attention" },
  "webgpu-subgroup": { what: "WebGPU: matvec reduced with subgroupAdd (native wgpu only)" },
  "webgpu-fused": { what: "WebGPU: bias+GELU, residuals and KV writes fused into the matmuls" },
  "webgpu-f16": { what: "WebGPU: f16 weights packed two per u32" },
  "webgpu-q8": { what: "WebGPU: int8 weights (Q8_0) dequantised in the kernel, f16 LM head" },
  "webgpu-argmax": { what: "WebGPU: argmax on the GPU, 4 bytes read back per token" },
  "webgpu-resident": { what: "WebGPU: 16 tokens per submit, the GPU feeds itself" },
  "webgpu-tiled": { what: "WebGPU: 64x64 tiled matmul for the prompt" },
};
