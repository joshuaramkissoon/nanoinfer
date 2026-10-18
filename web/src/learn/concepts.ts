// The constraint map: the handful of physical limits that make inference hard, and the techniques that work around
// each. Shared by chapter 0.2 (fig 0.2.4) and the field guide's entry view, so the two always agree.

/** colour role from web/DESIGN.md */
export type Role = "memory" | "compute" | "comm" | "time";

export interface Technique {
  label: string;
  /** a course chapter slug, or "#<field-guide id>" */
  to: string;
}

export interface Constraint {
  key: string;
  role: Role;
  name: string;
  /** the limit, in one line */
  why: string;
  techniques: Technique[];
}

export const CONSTRAINTS: Constraint[] = [
  {
    key: "bandwidth",
    role: "memory",
    name: "Memory bandwidth",
    why: "Every token reads every weight. Decode waits on bytes, not maths.",
    techniques: [
      { label: "quantisation", to: "smaller-numbers" },
      { label: "batching", to: "batching" },
      { label: "speculative decoding", to: "speculative-decoding" },
      { label: "HBM / GPUs", to: "the-hardware" },
      { label: "kernel fusion", to: "#fusion" },
    ],
  },
  {
    key: "capacity",
    role: "memory",
    name: "Memory capacity",
    why: "Weights plus every user's KV cache must fit in the accelerator's memory.",
    techniques: [
      { label: "quantisation", to: "#quantisation" },
      { label: "GQA / MQA", to: "#gqa" },
      { label: "paged attention", to: "#paged-attention" },
      { label: "KV-cache quantisation", to: "#kv-quant" },
      { label: "tensor parallelism", to: "#tensor-parallel" },
    ],
  },
  {
    key: "sequential",
    role: "time",
    name: "Sequential decode",
    why: "Token n+1 needs token n. One answer can't be parallelised across its tokens.",
    techniques: [
      { label: "KV cache", to: "kv-cache" },
      { label: "speculative decoding", to: "#speculative" },
      { label: "batching across users", to: "#batching" },
    ],
  },
  {
    key: "compute",
    role: "compute",
    name: "Compute",
    why: "Prefill and big batches do enough maths per byte to wait on arithmetic instead.",
    techniques: [
      { label: "tensor cores", to: "#tensor-cores" },
      { label: "FlashAttention", to: "#flashattention" },
      { label: "SIMD + threads", to: "faster-arithmetic" },
      { label: "MoE (fewer FLOPs/token)", to: "#moe" },
    ],
  },
  {
    key: "comm",
    role: "comm",
    name: "Communication",
    why: "Models bigger than one chip must be split, and the pieces must talk every layer.",
    techniques: [
      { label: "tensor parallelism", to: "#tensor-parallel" },
      { label: "pipeline parallelism", to: "#pipeline-parallel" },
      { label: "expert parallelism", to: "#expert-parallel" },
      { label: "NVLink", to: "#nvlink" },
      { label: "all-reduce", to: "many-gpus" },
    ],
  },
  {
    key: "traffic",
    role: "time",
    name: "Unpredictable traffic",
    why: "Requests arrive at random, prompts and answers vary 100× in length, and users want low latency.",
    techniques: [
      { label: "continuous batching", to: "#continuous-batching" },
      { label: "prefix caching", to: "#prefix-caching" },
      { label: "disaggregated serving", to: "#disaggregated" },
      { label: "chunked prefill", to: "#chunked-prefill" },
    ],
  },
];

export const techHref = (t: Technique) => (t.to.startsWith("#") ? `/learn/field-guide${t.to}` : `/learn/${t.to}`);
