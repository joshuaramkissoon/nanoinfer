// Numbers shared by the part-0 chapters. "measured" ones are ours (docs/LOG.md, public/learn/data.json);
// "external" ones are spec-sheet figures and say where they come from.

export interface Machine {
  key: string;
  name: string;
  /** read bandwidth, GB/s */
  bw: number;
  /** memory capacity, GB */
  cap: number;
  /** peak dense FLOP/s at the format inference uses (TFLOP/s) */
  tflops: number;
  src: "measured" | "external";
  note: string;
}

export const MACHINES: Machine[] = [
  { key: "m4cpu", name: "M4 CPU (10 cores)", bw: 98, cap: 24, tflops: 0.6, src: "measured", note: "98 GB/s measured with 10 threads (120 GB/s spec); 24 GB shared with the GPU; ~0.6 TFLOP/s f32 estimated from 4 P-cores × 4 FMA pipes × 4 lanes" },
  { key: "m4gpu", name: "M4 GPU (10 cores)", bw: 105, cap: 24, tflops: 4.3, src: "measured", note: "105 GB/s measured (Metal read probe, quiet machine); same 24 GB pool as the CPU; ~4.3 TFLOP/s f32" },
  { key: "h100", name: "H100 SXM", bw: 3350, cap: 80, tflops: 989, src: "external", note: "NVIDIA H100 SXM datasheet: 80 GB HBM3 at 3.35 TB/s, ~989 TFLOP/s dense bf16 on tensor cores" },
  { key: "h100x8", name: "8 × H100 (one server)", bw: 8 * 3350, cap: 640, tflops: 8 * 989, src: "external", note: "an HGX/DGX H100 server: eight H100s joined by NVLink; bandwidth and capacity add up only if the model is split perfectly" },
];

export interface Model {
  key: string;
  name: string;
  /** parameters */
  params: number;
  blurb: string;
}

export const MODELS: Model[] = [
  { key: "gpt2", name: "GPT-2 small", params: 124e6, blurb: "the model on this site (2019)" },
  { key: "7b", name: "a 7B model", params: 7e9, blurb: "Llama-2-7B / Mistral-7B class: runs on a laptop when quantised" },
  { key: "70b", name: "a 70B model", params: 70e9, blurb: "Llama-3-70B class: a serious open model" },
  { key: "405b", name: "a 405B model", params: 405e9, blurb: "Llama-3.1-405B: frontier-scale open weights" },
];

export interface Format {
  key: string;
  name: string;
  bytes: number;
}

export const FORMATS: Format[] = [
  { key: "fp32", name: "fp32", bytes: 4 },
  { key: "bf16", name: "bf16", bytes: 2 },
  { key: "int8", name: "int8", bytes: 1 },
  { key: "int4", name: "int4", bytes: 0.5 },
];

export const SOURCES =
  "M4 numbers: our measurements (docs/LOG.md). H100: NVIDIA H100 SXM datasheet (HBM3 3.35 TB/s, 80 GB, 989 TFLOP/s dense bf16). NVLink 4: 900 GB/s total per GPU (450 each way). PCIe 5 ×16: ~64 GB/s each way. InfiniBand NDR: 400 Gb/s ≈ 50 GB/s per port.";

/** "1.2 GB", "495 MB", "140 GB" */
export function fmtBytes(b: number): string {
  if (b >= 1e12) return `${+(b / 1e12).toPrecision(3)} TB`;
  if (b >= 1e9) return `${+(b / 1e9).toPrecision(3)} GB`;
  return `${+(b / 1e6).toPrecision(3)} MB`;
}

/** two significant figures, grouped: 0.37, 48, 240, 2,400 */
export function fmtRate(x: number): string {
  if (x < 1) return x.toFixed(2);
  if (x < 10) return x.toFixed(1);
  const p = 10 ** Math.max(0, Math.floor(Math.log10(x)) - 1);
  return (Math.round(x / p) * p).toLocaleString("en-GB");
}
