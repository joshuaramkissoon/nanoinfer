// The roofs a run should be judged against. A WebGPU run reads its weights
// through the GPU, so its speed limit is the GPU's read bandwidth (the WebGPU
// probe), not the wasm CPU probe; and a stage's bytes per token follow its own
// weight and LM-head formats. Lab figs 5 and 6 both use these.

import { isGpu, useEngine } from "../engine/client";
import type { ModelConfig, ProbeRow, StageInfo } from "../engine/types";

/** Bytes per weight for core's WeightFormat names. */
export const BYTES_PER_WEIGHT: Record<string, number> = {
  f32: 4,
  f16: 2,
  q8_0: 1.0625, // 32 int8 + one f16 scale per block of 32
  q4_0: 0.5625, // 32 nibbles + one f16 scale per block of 32
};
export const FORMAT_LABEL: Record<string, string> = { f32: "f32", f16: "f16", q8_0: "Q8_0", q4_0: "Q4_0" };

const bpw = (fmt: string | undefined) => BYTES_PER_WEIGHT[fmt ?? "f32"] ?? 4;

/** Does this stage keep its KV cache in f16? (the kv16 stages and the GPU rungs built on them) */
export function kvBytesPerElem(stage: StageInfo | undefined, stages: StageInfo[]): number {
  if (!stage) return 4;
  if (stage.name.includes("kv16")) return 2;
  if (stage.backend === "webgpu") {
    const kv16 = stages.find((s) => s.name === "webgpu-kv16");
    if (kv16 && stage.index > kv16.index) return 2;
  }
  return 4;
}

/**
 * Bytes one decode step streams: the transformer blocks' weights in the
 * stage's weight format, the tied embedding / LM head in its head format,
 * plus the KV cache read by attention. Biases and LayerNorm (<0.1%) are
 * counted at the weight format too, which is close enough.
 */
export function decodeBytes(
  cfg: ModelConfig,
  fmt: { weightFormat?: string; headFormat?: string },
  kvLen: number,
  kvElem = 4,
): { weights: number; head: number; kv: number; total: number } {
  const params = cfg.decodeWeightBytes / 4; // decodeWeightBytes is f32 bytes
  const headParams = cfg.nVocab * cfg.nEmbd;
  const head = headParams * bpw(fmt.headFormat ?? fmt.weightFormat);
  const weights = Math.max(0, params - headParams) * bpw(fmt.weightFormat);
  const kv = Math.max(0, kvLen) * 2 * cfg.nLayer * cfg.nEmbd * kvElem;
  return { weights, head, kv, total: weights + head + kv };
}

/** The GPU read bandwidth from a WebGPU probe's rows (best, else median). */
export function gpuBandwidthOf(rows: ProbeRow[] | null | undefined): number | null {
  if (!rows) return null;
  const best = rows.find((r) => r.unit === "GB/s" && /best/.test(r.label));
  const any = best ?? rows.find((r) => r.unit === "GB/s");
  return any ? any.value : null;
}

export interface Roofs {
  backend: "cpu" | "gpu";
  /** read bandwidth, GB/s, for this backend */
  bw: number | null;
  /** where bw came from */
  bwFrom: "this tab" | "recorded" | null;
  /** compute roof, GFLOP/s (only measured for the CPU) */
  peak: number | null;
  stage: StageInfo | undefined;
}

/** Roofs for the shared run's stage. */
export function useRoofs(stageName: string | undefined): Roofs {
  const stages = useEngine((s) => s.stages);
  const cpuBw = useEngine((s) => s.bandwidthGBs);
  const peak = useEngine((s) => s.peakGflops);
  const gpuLive = useEngine((s) => s.gpuBandwidthGBs);
  const rec = useEngine((s) => s.recorded);
  const source = useEngine((s) => s.source);
  const stage = stages.find((s) => s.name === stageName) ?? rec?.meta.stages.find((s) => s.name === stageName);
  const gpu = !!stageName && (isGpu(stageName) || stage?.backend === "webgpu");
  if (!gpu) return { backend: "cpu", bw: cpuBw, bwFrom: cpuBw == null ? null : source === "live" ? "this tab" : "recorded", peak, stage };
  const recBw = gpuBandwidthOf(rec?.webgpu?.probe);
  return {
    backend: "gpu",
    bw: gpuLive ?? recBw,
    bwFrom: gpuLive != null ? "this tab" : recBw != null ? "recorded" : null,
    peak: null,
    stage,
  };
}
