// What each engine stage is, in plain words, for the stage picker and tooltips.
// The stage list itself always comes from the wasm build (core's Stage::ALL, the
// browser-only rungs and the WebGPU rungs), so a stage missing here still shows
// up: it gets a family from its name and its kernel string as the description.
// What a stage *actually runs* in this wasm build comes from the engine itself
// (StageInfo.wasmNote).

import type { StageInfo } from "../engine/types";

export type Family = "start" | "native" | "wasm" | "gpu";

export const FAMILIES: { key: Family; title: string; blurb: string }[] = [
  { key: "start", title: "Where we started", blurb: "The first correct engine, then the two ideas that made it usable." },
  {
    key: "native",
    title: "CPU tricks (written for the Mac)",
    blurb: "Hand-tuned for the M4's CPU: NEON instructions, OS threads, smaller numbers. A browser can't run most of that code, so these mostly fall back to simpler kernels here.",
  },
  { key: "wasm", title: "Browser CPU (WebAssembly)", blurb: "Rewritten for what a browser tab's CPU can do: WebAssembly SIMD, and threads where the page allows them." },
  { key: "gpu", title: "Browser GPU (WebGPU)", blurb: "The model on your graphics chip through WebGPU, one bottleneck at a time." },
];

/** A short plain-language name and one line on what changed. */
export const STAGE_NOTES: Record<string, { label: string; what: string }> = {
  naive: { label: "Starting point", what: "re-reads the whole text for every new token" },
  "kv-cache": { label: "KV cache", what: "remembers past tokens' keys and values: one new row per token" },
  unrolled: { label: "16 running totals", what: "sixteen independent sums, so the CPU can overlap the work (SIMD)" },
  neon: { label: "Hand-written SIMD", what: "Arm NEON multiply-adds, one weight row at a time" },
  tiled: { label: "Tiled prompt", what: "4×4 register tiles, so the prompt reuses each weight 4 times" },
  threads: { label: "All the cores", what: "matrix multiplies split across the CPU cores" },
  "spin-pool": { label: "Threads that never sleep", what: "a spinning thread pool: no wake-up cost per multiply" },
  "fast-ops": { label: "Everything vectorised", what: "GELU, softmax, attention and LayerNorm made parallel too" },
  f16: { label: "16-bit weights", what: "half the bytes per weight; the maths stays 32-bit" },
  q8: { label: "8-bit weights", what: "int8 blocks of 32 weights, int8 dot products" },
  q4: { label: "4-bit weights", what: "4-bit blocks, unpacked to int8 for the dot products" },
  "q8-fast": { label: "8-bit, low latency", what: "8-bit weights on a low-latency thread pool" },
  "q4-fast": { label: "4-bit, low latency", what: "4-bit weights on a low-latency thread pool" },
  "q8-rescore": { label: "8-bit, rescored head", what: "cheap 8-bit scores for every token, exact ones for the top few" },
  "q4-rescore": { label: "4-bit, rescored head", what: "4-bit weights with the rescored vocabulary step" },
  "q8-attn": { label: "8-bit, register attention", what: "attention kept in NEON registers while decoding" },
  "q4-attn": { label: "4-bit, register attention", what: "the 4-bit version with NEON attention" },
  "q8-kv16": { label: "8-bit, 16-bit KV cache", what: "keys and values stored as f16, laid out per head" },
  "q4-kv16": { label: "4-bit, 16-bit KV cache", what: "the 4-bit version with the f16 KV cache" },
  "q8-prefill": { label: "8-bit prompt tiles", what: "int8 dot-product tiles for reading the prompt too" },
  "wasm-simd": { label: "WebAssembly SIMD", what: "explicit 128-bit SIMD, 4 weight rows per pass" },
  "wasm-fma": { label: "SIMD + fused multiply-add", what: "the same with relaxed-SIMD FMA, where the browser has it" },
  "wasm-threads": { label: "SIMD on all cores", what: "the SIMD kernels on Web Workers sharing one memory" },
  "webgpu-naive": { label: "First GPU port", what: "the CPU kernels in WGSL; waits after every operation" },
  "webgpu-onesubmit": { label: "Wait once per token", what: "the same kernels, one submission per token" },
  "webgpu-matvec": { label: "GPU-shaped multiply", what: "64 threads per weight row, 4-wide loads" },
  "webgpu-par": { label: "Parallel norms + attention", what: "256-thread workgroups for LayerNorm and attention" },
  "webgpu-subgroup": { label: "Subgroup sums", what: "matvec reduced with subgroupAdd (native wgpu only)" },
  "webgpu-fused": { label: "Fused kernels", what: "bias, GELU, residuals and KV writes folded into the multiplies" },
  "webgpu-f16": { label: "GPU, 16-bit weights", what: "f16 weights packed two per 32-bit word" },
  "webgpu-q8": { label: "GPU, 8-bit weights", what: "int8 blocks unpacked inside the kernel; f16 vocabulary head" },
  "webgpu-argmax": { label: "Pick on the GPU", what: "the next token chosen on the GPU: 4 bytes read back" },
  "webgpu-resident": { label: "GPU feeds itself", what: "16 tokens per submission, no round trip through JavaScript" },
  "webgpu-tiled": { label: "Tiled prompt (GPU)", what: "64×64 tiled multiply for reading the prompt" },
  "webgpu-ln64": { label: "Register LayerNorm", what: "LayerNorm on 64 threads, the row held in registers" },
  "webgpu-q8head": { label: "8-bit vocabulary head", what: "the big 50,257-row output matrix in int8 too" },
  "webgpu-lnfold": { label: "Folded LayerNorm", what: "LayerNorm folded into the next weights: fewer kernels" },
  "webgpu-attn4": { label: "4-wide attention", what: "attention with 4-wide loads, the query in registers" },
  "webgpu-flashdec": { label: "Flash decoding", what: "attention split into chunks of 64 keys, then merged" },
  "webgpu-kv16": { label: "16-bit KV cache (GPU)", what: "keys and values stored as f16" },
};

export function familyOf(s: Pick<StageInfo, "name" | "backend" | "browserOnly">): Family {
  if (s.backend === "webgpu" || s.name.startsWith("webgpu")) return "gpu";
  if (s.browserOnly || s.name.startsWith("wasm")) return "wasm";
  if (s.name === "naive" || s.name === "kv-cache" || s.name === "unrolled") return "start";
  return "native";
}

export const stageLabel = (name: string) => STAGE_NOTES[name]?.label ?? name;
export const stageWhat = (s: Pick<StageInfo, "name" | "kernels">) => STAGE_NOTES[s.name]?.what ?? s.kernels;

/** A core stage whose wasm build runs some other stage's (or slower) code. */
export const fallback = (s: { wasmNote?: string; browserOnly?: boolean; backend?: string }) =>
  !!s.wasmNote && !s.browserOnly && s.backend !== "webgpu" && !/threads \(/.test(s.wasmNote);
