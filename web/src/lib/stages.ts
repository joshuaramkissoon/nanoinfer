// Per-stage notes the engine can't (yet) tell us. Unknown stages need no entry:
// the stage list itself always comes from the wasm build (Stage::ALL).

export const STAGE_NOTES: Record<string, { what: string; wasm?: string }> = {
  naive: { what: "recompute the whole context for every token" },
  "kv-cache": { what: "cache keys and values; one new row per token" },
  unrolled: { what: "16 independent accumulators; LLVM vectorises them (simd128 here)" },
  neon: { what: "hand-written aarch64 NEON matmul", wasm: "no NEON in wasm: runs the unrolled kernels here" },
  tiled: { what: "4×4 register-blocked matmul for prefill; decode unchanged (in wasm: scalar tiles, unrolled decode)" },
};
