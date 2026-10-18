// Shared types between the engine worker and the page.

export interface StageInfo {
  name: string;
  kernels: string;
  index: number;
  /** "f32" | "f16" | "q8_0" | "q4_0" (from core's WeightFormat) */
  weightFormat: string;
  bitsPerWeight: number;
}

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
  | { t: "init" }
  | { t: "load"; stage: string; base: string }
  | { t: "open"; stage: string; maxSeq: number }
  | { t: "close"; slot: number }
  | { t: "setStage"; slot: number; stage: string }
  | { t: "reset"; slot: number }
  | { t: "forward"; slot: number; tokens: number[] }
  | { t: "sample"; slot: number; temperature: number; topK: number }
  | { t: "encode"; text: string }
  | { t: "capture"; topK: number; layer: number; head: number }
  | { t: "attention"; tokens: number[] }
  | { t: "benchBandwidth"; mb: number; iters: number }
  | { t: "benchPeak"; millions: number }
  | { t: "seed"; seed: number }
  | { t: "clearCache" };

export interface InitResult {
  stages: StageInfo[];
  isolated: boolean;
  timerResUs: number;
  simd: boolean;
}

export interface LoadResult {
  config: ModelConfig;
  vocabBlob: Uint8Array;
  vocabOffsets: Uint32Array;
  loadMs: number;
  fromCache: boolean;
  weightBytes: number;
}
