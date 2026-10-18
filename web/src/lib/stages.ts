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
};
