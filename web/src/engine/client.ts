// Typed RPC to the engine worker plus a tiny shared store every panel reads.
// One engine, one store: the run you trigger in the hero drives every figure.

import { useSyncExternalStore } from "react";
import type { ForwardTrace, InitResult, LoadProgress, LoadResult, ModelConfig, Req, StageInfo } from "./types";

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void; onEvent?: (m: any) => void };

class Rpc {
  private worker: Worker;
  private id = 0;
  private pending = new Map<number, Pending>();
  constructor() {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e) => {
      const m = e.data;
      const p = this.pending.get(m.id);
      if (!p) return;
      if (m.t === "ok") {
        this.pending.delete(m.id);
        p.resolve(m.result);
      } else if (m.t === "err") {
        this.pending.delete(m.id);
        p.reject(new Error(m.error));
      } else p.onEvent?.(m);
    };
    this.worker.onerror = (e) => {
      const err = new Error(e.message || "engine worker crashed");
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
      setState({ status: "error", error: err.message });
    };
  }
  call<T>(req: Req, onEvent?: (m: any) => void): Promise<T> {
    const id = ++this.id;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onEvent });
      this.worker.postMessage({ id, req });
    });
  }
}

// ---- store ------------------------------------------------------------------------

export interface Tok {
  id: number;
  prompt: boolean;
  /** probability the model gave this token when it was chosen (generated only) */
  p: number | null;
  /** next-token distribution the token was sampled from */
  top: { id: number; p: number }[] | null;
  entropy: number | null;
}

export interface Run {
  stage: string;
  tokens: Tok[];
  /** every forward call of the run, in order (prefill first) */
  forwards: ForwardTrace[];
  done: boolean;
  /** wall-clock start of decoding, for an honest end-to-end tok/s */
  t0: number;
  t1: number;
}

export interface BrowserBench {
  stage: string;
  tokS: number;
  msPerTok: number;
  promptLen: number;
  genTokens: number;
  at: number;
  ua: string;
}

export interface State {
  status: "boot" | "idle" | "loading" | "ready" | "error";
  error: string | null;
  stages: StageInfo[];
  isolated: boolean;
  timerResUs: number;
  progress: LoadProgress | null;
  config: ModelConfig | null;
  load: { ms: number; fromCache: boolean; bytes: number } | null;
  bandwidthGBs: number | null;
  peakGflops: number | null;
  run: Run | null;
  /** which panel currently drives the engine (others wait) */
  busy: string | null;
  bench: Record<string, BrowserBench>;
}

const BENCH_KEY = "nanoinfer-browser-bench-v1";
function loadBench(): Record<string, BrowserBench> {
  try {
    return JSON.parse(localStorage.getItem(BENCH_KEY) ?? "{}");
  } catch {
    return {};
  }
}

let state: State = {
  status: "boot",
  error: null,
  stages: [],
  isolated: false,
  timerResUs: 0,
  progress: null,
  config: null,
  load: null,
  bandwidthGBs: null,
  peakGflops: null,
  run: null,
  busy: null,
  bench: loadBench(),
};
const listeners = new Set<() => void>();

export function setState(patch: Partial<State> | ((s: State) => Partial<State>)) {
  const p = typeof patch === "function" ? patch(state) : patch;
  state = { ...state, ...p };
  if (p.bench) {
    try {
      localStorage.setItem(BENCH_KEY, JSON.stringify(state.bench));
    } catch {
      /* storage blocked */
    }
  }
  for (const l of listeners) l();
}
export const getState = () => state;

export function useEngine<T>(sel: (s: State) => T): T {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => sel(state),
  );
}

// ---- vocab ------------------------------------------------------------------------

let vocabBlob = new Uint8Array();
let vocabOffs = new Uint32Array();
const utf8 = new TextDecoder("utf-8", { fatal: false });

export function tokenBytes(id: number): Uint8Array {
  return vocabBlob.subarray(vocabOffs[id], vocabOffs[id + 1]);
}
/** One token as text (a lone partial UTF-8 byte shows as U+FFFD). */
export function tokenText(id: number): string {
  if (!vocabOffs.length) return `#${id}`;
  return utf8.decode(tokenBytes(id));
}
/** Several tokens decoded together (so multi-token characters come out right). */
export function decodeTokens(ids: number[]): string {
  let n = 0;
  for (const id of ids) n += vocabOffs[id + 1] - vocabOffs[id];
  const out = new Uint8Array(n);
  let o = 0;
  for (const id of ids) {
    const b = tokenBytes(id);
    out.set(b, o);
    o += b.length;
  }
  return utf8.decode(out);
}
/** Make whitespace visible for token chips. */
export function showToken(id: number): string {
  return tokenText(id).replace(/ /g, "·").replace(/\n/g, "⏎").replace(/\t/g, "⇥");
}

// ---- engine API ---------------------------------------------------------------------

let rpc: Rpc | null = null;
const r = () => (rpc ??= new Rpc());

export const WEIGHTS_BASE = (import.meta.env.VITE_WEIGHTS_BASE as string | undefined) ?? "/weights/gpt2";

export async function boot() {
  if (state.status !== "boot") return;
  try {
    const res = await r().call<InitResult>({ t: "init" });
    setState({ status: "idle", stages: res.stages, isolated: res.isolated, timerResUs: res.timerResUs });
    // Never auto-download 500 MB, but if this browser already has the weights
    // cached, loading costs nothing: go straight to live.
    try {
      const abs = new URL(`${WEIGHTS_BASE}/model.safetensors`, location.href).href;
      const name = abs.replace(/^https?:\/\//, "").replace(/[^a-zA-Z0-9._-]/g, "_");
      const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle("nanoinfer-weights-v1");
      await dir.getFileHandle(name + ".ok");
      void loadModel();
    } catch {
      /* not cached (or no OPFS) */
    }
  } catch (e) {
    setState({ status: "error", error: (e as Error).message });
  }
}

export async function loadModel() {
  if (state.status === "loading" || state.status === "ready") return;
  setState({ status: "loading", error: null });
  try {
    const stage = state.stages.find((s) => s.name === "kv-cache")?.name ?? state.stages[0].name;
    const res = await r().call<LoadResult>({ t: "load", stage, base: WEIGHTS_BASE }, (m) => {
      if (m.t === "progress") setState({ progress: m.p });
    });
    vocabBlob = res.vocabBlob;
    vocabOffs = res.vocabOffsets;
    setState({
      status: "ready",
      config: res.config,
      progress: null,
      load: { ms: res.loadMs, fromCache: res.fromCache, bytes: res.weightBytes },
    });
    // Measure the two roofs once, right away (each takes well under a second).
    measureRoofs();
  } catch (e) {
    setState({ status: "error", error: (e as Error).message, progress: null });
  }
}

export async function measureRoofs() {
  // Best of two probes: the first one right after the 1 GB weight load reads
  // ~40% low (cold pages / clocks), so measure, wait a beat, measure again.
  const probe = async () => {
    const bw = await r().call<number>({ t: "benchBandwidth", mb: 256, iters: 8 });
    const pk = await r().call<number>({ t: "benchPeak", millions: 400 });
    setState((s) => ({ bandwidthGBs: Math.max(bw, s.bandwidthGBs ?? 0), peakGflops: Math.max(pk, s.peakGflops ?? 0) }));
  };
  await probe();
  await new Promise((res) => setTimeout(res, 1500));
  await probe();
}

export const engine = {
  open: (stage: string, maxSeq = 0) => r().call<number>({ t: "open", stage, maxSeq }),
  close: (slot: number) => r().call<null>({ t: "close", slot }),
  setStage: (slot: number, stage: string) => r().call<null>({ t: "setStage", slot, stage }),
  reset: (slot: number) => r().call<null>({ t: "reset", slot }),
  forward: (slot: number, tokens: number[]) => r().call<ForwardTrace>({ t: "forward", slot, tokens }),
  sample: (slot: number, temperature: number, topK: number) =>
    r().call<{ id: number; p: number }>({ t: "sample", slot, temperature, topK }),
  encode: (text: string) => r().call<number[]>({ t: "encode", text }),
  capture: (topK: number, layer = -1, head = -1) => r().call<null>({ t: "capture", topK, layer, head }),
  attention: (tokens: number[]) => r().call<Float32Array>({ t: "attention", tokens }),
  clearCache: () => r().call<boolean>({ t: "clearCache" }),
  seed: (seed: number) => r().call<null>({ t: "seed", seed }),
};

/** Claim the engine for a panel; returns false if another panel holds it. */
export function claim(who: string): boolean {
  if (state.busy && state.busy !== who) return false;
  setState({ busy: who });
  return true;
}
export function release(who: string) {
  if (state.busy === who) setState({ busy: null });
}

export function median(xs: number[]): number {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Decode forwards of a run (excludes prefill). */
export const decodes = (run: Run | null) => (run ? run.forwards.filter((f) => f.phase === "decode") : []);

// Dev-only handle for poking the live engine from the console / test harness.
if (import.meta.env.DEV) (globalThis as any).__nanoinfer = { getState, setState, engine, measureRoofs, loadModel, bench: () => import("./generate") };
