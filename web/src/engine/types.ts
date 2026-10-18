// Shared types between the engine worker and the page.

export interface StageInfo {
  name: string;
  kernels: string;
  index: number;
  /** "f32" | "f16" | "q8_0" | "q4_0" (from core's WeightFormat) */
  weightFormat: string;
  bitsPerWeight: number;
  /** a browser-only rung (crates/wasm kernels), not in core's Stage::ALL */
  browserOnly?: boolean;
  /** what this stage actually runs when compiled to wasm, if different from native */
  wasmNote?: string;
}

/** Which wasm build is running: simd128, relaxed (relaxed-SIMD FMA) or threads. */
export type BuildName = "simd128" | "relaxed" | "threads";

export interface ModelConfig {
  nVocab: number;
  nCtx: number;
  nEmbd: number;
  nHead: number;
  nLayer: number;
  nParams: number;
  /** Bytes of f32 weights streamed per decoded token (excludes wpe). */
  decodeWeightBytes: number;
}

export interface OpRec {
  kind: string;
  name: string;
  /** -1 = outside the transformer blocks (embed, ln_f, lm_head). */
  layer: number;
  dims: [number, number, number];
  flops: number;
  bytes: number;
  /** ms since the forward started */
  start: number;
  /** ms */
  dur: number;
}

export interface ForwardTrace {
  phase: "prefill" | "decode";
  nNew: number;
  nComputed: number;
  contextLen: number;
  /** -1 when the stage has no KV cache. */
  kvLen: number;
  kvCap: number;
  /** wall time of the whole forward, ms */
  ms: number;
  entropy: number;
  top: { id: number; p: number }[];
  attn: Float32Array | null;
  attnLayer: number;
  attnHead: number;
  ops: OpRec[];
}

export interface LoadProgress {
  file: string;
  loaded: number;
  total: number;
  cached: boolean;
  phase: "download" | "parse";
}

export type Req =
  | { t: "init"; build: BuildName | "auto"; threads?: number }
  | { t: "load"; stage: string; base: string; dtype: "f32" | "f16" }
  | { t: "open"; stage: string; maxSeq: number }
  | { t: "close"; slot: number }
  | { t: "setStage"; slot: number; stage: string }
  | { t: "reset"; slot: number }
  | { t: "forward"; slot: number; tokens: number[] }
  | { t: "sample"; slot: number; temperature: number; topK: number }
  | { t: "encode"; text: string }
  | { t: "capture"; topK: number; layer: number; head: number }
  | { t: "attention"; tokens: number[] }
  | { t: "lens"; tokens: number[]; positions: number[]; topK: number }
  | { t: "residual"; tokens: number[]; pos: number }
  | { t: "logits"; slot: number }
  | { t: "benchBandwidth"; mb: number; iters: number }
  | { t: "benchPeak"; millions: number }
  | { t: "seed"; seed: number }
  | { t: "clearCache" };

export interface InitResult {
  stages: StageInfo[];
  isolated: boolean;
  timerResUs: number;
  simd: boolean;
  build: BuildName;
  /** builds this browser can run */
  builds: BuildName[];
  threads: number;
  /** what the compute-roof probe measures: "mul+add" or "fma" */
  peakKind: string;
}

export interface LoadResult {
  config: ModelConfig;
  vocabBlob: Uint8Array;
  vocabOffsets: Uint32Array;
  loadMs: number;
  fromCache: boolean;
  weightBytes: number;
}
