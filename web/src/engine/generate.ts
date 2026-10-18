// The generation loop. Lives on the main thread (each forward is one RPC
// round trip, ~0.1 ms against 100+ ms of compute), which makes it trivially
// cancellable and lets every step update the shared run immediately.

import { engine, getState, setState, type Run, type Tok, claim, release, median, type BrowserBench, benchKey, isGpu } from "./client";
import type { ForwardTrace } from "./types";

export interface GenOpts {
  stage: string;
  /** context to start from: prompt tokens, or a branched prefix */
  prefix: Tok[];
  n: number;
  temperature: number;
  topK: number;
  /** force this token first (branching from the probability bars) */
  forced?: { id: number; p: number; top: Tok["top"]; entropy: number | null } | null;
  signal: { aborted: boolean };
  attn?: [number, number] | null;
  /** keep the forwards of the prefix run (when branching) */
  keepForwards?: ForwardTrace[];
}

const HERO_SLOT = 0;

export async function generate(o: GenOpts): Promise<Run | null> {
  if (!claim("hero")) return null;
  try {
    await engine.setStage(HERO_SLOT, o.stage);
    await engine.seed(1 + Math.floor(Math.random() * 2 ** 31));
    await engine.capture(10, o.attn?.[0] ?? -1, o.attn?.[1] ?? -1);
    const tokens = [...o.prefix];
    if (o.forced) tokens.push({ id: o.forced.id, prompt: false, p: o.forced.p, top: o.forced.top, entropy: o.forced.entropy });
    let run: Run = {
      stage: o.stage,
      tokens,
      forwards: o.keepForwards ? [...o.keepForwards] : [],
      done: false,
      t0: performance.now(),
      t1: performance.now(),
    };
    setState({ run });
    let tr = await engine.forward(HERO_SLOT, tokens.map((t) => t.id));
    run = { ...run, forwards: [...run.forwards, tr], t0: performance.now() };
    setState({ run });
    const ctxMax = getState().config?.nCtx ?? 1024;
    for (let i = 0; i < o.n && !o.signal.aborted && run.tokens.length < ctxMax; i++) {
      const { id, p } = await engine.sample(HERO_SLOT, o.temperature, o.topK);
      const tok: Tok = { id, prompt: false, p, top: tr.top, entropy: tr.entropy };
      run = { ...run, tokens: [...run.tokens, tok], t1: performance.now() };
      setState({ run });
      if (i === o.n - 1 || o.signal.aborted) break;
      tr = await engine.forward(HERO_SLOT, [id]);
      run = { ...run, forwards: [...run.forwards, tr], t1: performance.now() };
      setState({ run });
    }
    run = { ...run, done: true, t1: performance.now() };
    setState({ run });
    return run;
  } finally {
    release("hero");
  }
}

/** A fixed-shape benchmark for the Ladder: prompt P, then G greedy decode steps. */
export async function benchStage(stage: string, promptLen = 64, gen = 8): Promise<BrowserBench | null> {
  const who = `bench:${stage}`;
  if (!claim(who)) return null;
  const slot = await engine.open(stage, promptLen + (isGpu(stage) ? Math.max(gen, 32) : gen) + 1);
  try {
    await engine.capture(0);
    const text =
      "The history of computing is a story of making machines do arithmetic faster. Early computers filled rooms and performed a few thousand operations per second. ";
    let ids = await engine.encode(text);
    while (ids.length < promptLen) ids = ids.concat(ids);
    ids = ids.slice(0, promptLen);
    if (isGpu(stage)) {
      // WebGPU rungs: the whole benchmark runs in the worker on the stage's
      // fastest greedy path (GPU argmax / resident loop where it has one),
      // best of 3 after a warm-up run that compiles pipelines and binds buffers.
      await engine.gpuBench(slot, ids, gen);
      const runs = [];
      for (let i = 0; i < 3; i++) runs.push(await engine.gpuBench(slot, ids, Math.max(gen, 32)));
      const best = runs.reduce((a, b) => (b.decodeMs < a.decodeMs ? b : a));
      const g = Math.max(gen, 32);
      const build = getState().build ?? undefined;
      const b: BrowserBench = {
        stage,
        build,
        prefillTokS: (promptLen / Math.min(...runs.map((r) => r.prefillMs))) * 1000,
        tokS: (g / best.decodeMs) * 1000,
        msPerTok: best.decodeMs / g,
        promptLen,
        genTokens: g,
        at: Date.now(),
        ua: navigator.userAgent,
        dispatches: best.dispatches,
        submits: best.submits,
        adapter: getState().webgpu?.adapter,
      };
      setState((s) => ({ bench: { ...s.bench, [benchKey(build ?? null, stage)]: b } }));
      return b;
    }
    // warm-up (wasm tier-up, first touch of buffers): a few single-token forwards
    for (let i = 0; i < 3; i++) await engine.forward(slot, [ids[i]]);
    await engine.reset(slot);
    let tr = await engine.forward(slot, ids);
    const prefillMs = tr.ms;
    const ms: number[] = [];
    for (let i = 0; i < gen; i++) {
      const { id } = await engine.sample(slot, 0, 0);
      tr = await engine.forward(slot, [id]);
      ms.push(tr.ms);
    }
    const m = median(ms);
    const build = getState().build ?? undefined;
    const b: BrowserBench = {
      stage,
      build,
      prefillTokS: (promptLen / prefillMs) * 1000,
      tokS: 1000 / m,
      msPerTok: m,
      promptLen,
      genTokens: gen,
      at: Date.now(),
      ua: navigator.userAgent,
    };
    setState((s) => ({ bench: { ...s.bench, [benchKey(build ?? null, stage)]: b } }));
    return b;
  } finally {
    await engine.close(slot);
    await engine.capture(10);
    release(who);
  }
}
