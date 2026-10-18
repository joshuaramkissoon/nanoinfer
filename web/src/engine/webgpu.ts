// WebGPU recording: Node has no WebGPU, so the WebGPU rungs' recorded numbers
// come from a real browser. `recordWebgpu()` runs in the page (headless
// Chrome via web/scripts/record-webgpu.sh, or any tab from the console),
// and returns the JSON saved as public/recorded/webgpu.json.

import { engine, getState } from "./client";
import { benchStage } from "./generate";
import type { ProbeRow, StageInfo } from "./types";

export interface WebgpuRecording {
  recordedAt: string;
  ua: string;
  adapter: string;
  features: string[];
  /** dispatch / submit / round-trip probe in this browser */
  probe: ProbeRow[];
  stages: StageInfo[];
  ladder: Record<string, { tokS: number; msPerTok: number; prefillTokS: number; promptLen: number; genTokens: number; dispatches?: number; submits?: number }>;
  /** greedy continuation of the race prompt on the fastest WebGPU rung */
  sample: { stage: string; text: string; tokens: number[]; ms: number } | null;
}

const idle = async () => {
  for (let i = 0; i < 2400 && getState().busy; i++) await new Promise((r) => setTimeout(r, 250));
};

export async function recordWebgpu(log: (s: string) => void = console.log): Promise<WebgpuRecording> {
  const st = getState();
  if (!st.webgpu) throw new Error(st.webgpuError ?? "no WebGPU");
  if (st.status !== "ready") throw new Error("load the model first");
  const stages = st.stages.filter((s) => s.backend === "webgpu");
  const ladder: WebgpuRecording["ladder"] = {};
  for (const s of stages) {
    await idle();
    log(`bench ${s.name}`);
    const b = await benchStage(s.name);
    if (b) ladder[s.name] = { tokS: b.tokS, msPerTok: b.msPerTok, prefillTokS: b.prefillTokS ?? 0, promptLen: b.promptLen, genTokens: b.genTokens, dispatches: b.dispatches, submits: b.submits };
  }
  await idle();
  log("probe");
  const probe = await engine.gpuProbe(7);
  let sample: WebgpuRecording["sample"] = null;
  const best = Object.entries(ladder).sort((a, b) => b[1].tokS - a[1].tokS)[0]?.[0];
  if (best) {
    const prompt = await engine.encode("The fastest way to read a book is");
    const slot = await engine.open(best, prompt.length + 41);
    try {
      await engine.forward(slot, prompt);
      const first = (await engine.sample(slot, 0, 0)).id;
      const g = await engine.gpuGreedy(slot, first, 39);
      const tokens = [first, ...g.tokens];
      const { decodeTokens } = await import("./client");
      sample = { stage: best, tokens, text: decodeTokens(tokens), ms: g.ms };
    } finally {
      await engine.close(slot);
    }
  }
  return {
    recordedAt: new Date().toISOString().slice(0, 10),
    ua: navigator.userAgent,
    adapter: st.webgpu.adapter,
    features: st.webgpu.features,
    probe,
    stages,
    ladder,
    sample,
  };
}
