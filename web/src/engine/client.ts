// Typed RPC to the engine worker plus a tiny shared store every panel reads.
// One engine, one store: the run you trigger in the hero drives every figure.

import { useSyncExternalStore } from "react";
import type { BuildName, ForwardTrace, GpuBench, GpuGreedy, GpuInfo, InitResult, LoadProgress, LoadResult, ModelConfig, ProbeRow, Req, StageInfo } from "./types";
import { loadRecorded, type Recorded } from "../lib/recorded";

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
      // An uncaught error with nothing in flight (seen once at startup from
      // the threads build's worker pool) leaves the engine usable: log it.
      if (!this.pending.size) {
        console.warn("engine worker:", e.message);
        e.preventDefault();
        return;
      }
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
  /** WebGPU rungs: per decode token */
  dispatches?: number;
  submits?: number;
  adapter?: string;
  /** wasm build it was measured on */
  build?: BuildName;
  prefillTokS?: number;
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
  /** browser benchmarks, keyed `${build}/${stage}` */
  bench: Record<string, BrowserBench>;
  /** where the shared figures' data comes from right now */
  source: "none" | "recorded" | "live";
  recorded: Recorded | null;
  build: BuildName | null;
  builds: BuildName[];
  threads: number;
  peakKind: string;
  /** which checkpoint to download: f16 (half the bytes, widened to f32 on load) or f32 */
  dtype: "f16" | "f32";
  f16Available: boolean;
  /** the WebGPU adapter (null: none, see webgpuError) */
  webgpu: GpuInfo | null;
  webgpuError: string | null;
  /** WebGPU dispatch / submit / round-trip probe, measured in this tab */
  gpuProbe: ProbeRow[] | null;
}

const BENCH_KEY = "nanoinfer-browser-bench-v2";
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
  source: "none",
  recorded: null,
  build: null,
  builds: [],
  threads: 1,
  peakKind: "mul+add",
  dtype: "f16",
  f16Available: false,
  webgpu: null,
  webgpuError: null,
  gpuProbe: null,
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

// Until the live vocab arrives, tokens decode from the recorded subset.
const EMPTY = new Uint8Array();
export function tokenBytes(id: number): Uint8Array {
  if (!vocabOffs.length) return state.recorded?.vocab.get(id) ?? EMPTY;
  return vocabBlob.subarray(vocabOffs[id], vocabOffs[id + 1]);
}
/** One token as text (a lone partial UTF-8 byte shows as U+FFFD). */
export function tokenText(id: number): string {
  const b = tokenBytes(id);
  if (!b.length && !vocabOffs.length) return `#${id}`;
  return utf8.decode(b);
}
/** Several tokens decoded together (so multi-token characters come out right). */
export function decodeTokens(ids: number[]): string {
  let n = 0;
  for (const id of ids) n += tokenBytes(id).length;
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

const params = new URLSearchParams(location.search);
/** `?build=simd128|relaxed|threads` forces a wasm build (default: best supported);
 *  `?threads=N` sizes the threads build's worker pool (default: all cores). */
export const BUILD_PREF = (params.get("build") as BuildName | null) ?? "auto";

const weightFile = () => (state.dtype === "f16" ? "model-f16.safetensors" : "model.safetensors");
export const weightLabel = () => (state.dtype === "f16" ? "f16 · 274 MB" : "f32 · 548 MB");

/** Show the recorded run in every figure (only while nothing live exists). */
async function applyRecorded() {
  const rec = await loadRecorded();
  if (!rec) return;
  setState((s) =>
    s.source === "live"
      ? { recorded: rec }
      : {
          recorded: rec,
          source: "recorded",
          run: rec.run,
          config: rec.meta.config,
          bandwidthGBs: rec.meta.bandwidthGBs,
          peakGflops: rec.meta.peakGflops,
          stages: s.stages.length ? s.stages : rec.meta.stages,
        },
  );
}

let booted = false;
export async function boot() {
  // once per page: React StrictMode runs effects twice in dev, and a second
  // init would try to start a second rayon pool in the threads build
  if (booted || state.status !== "boot") return;
  booted = true;
  void applyRecorded();
  try {
    const head = await fetch(`${WEIGHTS_BASE}/model-f16.safetensors`, { method: "HEAD" }).catch(() => null);
    const f16Available = !!head?.ok;
    setState({ f16Available, dtype: params.get("weights") === "f32" || !f16Available ? "f32" : "f16" });
    const res = await r().call<InitResult>({ t: "init", build: BUILD_PREF, threads: Number(params.get("threads")) || undefined });
    setState({
      status: "idle",
      stages: res.stages,
      isolated: res.isolated,
      timerResUs: res.timerResUs,
      build: res.build,
      builds: res.builds,
      threads: res.threads,
      peakKind: res.peakKind,
      webgpu: res.webgpu,
      webgpuError: res.webgpuError,
    });
    // Never auto-download, but if this browser already has the weights
    // cached, loading costs nothing: go straight to live.
    try {
      const abs = new URL(`${WEIGHTS_BASE}/${weightFile()}`, location.href).href;
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
    const res = await r().call<LoadResult>({ t: "load", stage, base: WEIGHTS_BASE, dtype: state.dtype }, (m) => {
      if (m.t === "progress") setState({ progress: m.p });
    });
    vocabBlob = res.vocabBlob;
    vocabOffs = res.vocabOffsets;
    setState({
      status: "ready",
      source: "live",
      config: res.config,
      progress: null,
      load: { ms: res.loadMs, fromCache: res.fromCache, bytes: res.weightBytes },
      // the recorded roofs are someone else's machine: re-measure here
      bandwidthGBs: null,
      peakGflops: null,
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
  // V8 first runs wasm in its baseline compiler (Liftoff) and tiers up to
  // TurboFan once code is hot, so the first peak probe reads ~2x low: warm up.
  await r().call<number>({ t: "benchPeak", millions: 100 });
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
  lens: (tokens: number[], positions: number[], topK: number) => r().call<Float32Array>({ t: "lens", tokens, positions, topK }),
  residual: (tokens: number[], pos: number) => r().call<Float32Array>({ t: "residual", tokens, pos }),
  logits: (slot: number) => r().call<Float32Array>({ t: "logits", slot }),
  clearCache: () => r().call<boolean>({ t: "clearCache" }),
  seed: (seed: number) => r().call<null>({ t: "seed", seed }),
  /** WebGPU sessions: greedy-decode n tokens on the stage's fastest path */
  gpuGreedy: (slot: number, first: number, n: number) => r().call<GpuGreedy>({ t: "gpuGreedy", slot, first, n }),
  gpuBench: (slot: number, prompt: number[], gen: number) => r().call<GpuBench>({ t: "gpuBench", slot, prompt, gen }),
  gpuProfile: (slot: number, tokens: number[], reps = 20) =>
    r().call<{ name: string; layer: number; ms: number }[]>({ t: "gpuProfile", slot, tokens, reps }),
  gpuProbe: (reps = 5) => r().call<ProbeRow[]>({ t: "gpuProbe", reps }),
};

/** Is this a WebGPU rung (runs on the GPU through crates/webgpu)? */
export const isGpu = (stage: string) => stage.startsWith("webgpu-");

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

/** Key for a browser benchmark result. */
export const benchKey = (build: string | null, stage: string) => `${build ?? "?"}/${stage}`;

/** Is the shared data live (from this tab) or recorded? */
export const useLive = () => useEngine((s) => s.source === "live");

/** Decode forwards of a run (excludes prefill). */
export const decodes = (run: Run | null) => (run ? run.forwards.filter((f) => f.phase === "decode") : []);

// Dev-only handle for poking the live engine from the console / test harness.
if (import.meta.env.DEV)
  (globalThis as any).__nanoinfer = { getState, setState, engine, measureRoofs, loadModel, bench: () => import("./generate"), webgpu: () => import("./webgpu") };

/** Where the shared run currently comes from: the recording, or this tab. */
export function useRunSource(): "live" | "recorded" | "none" {
  return useEngine((s) => (!s.run ? "none" : s.recorded && s.run === s.recorded.run ? "recorded" : "live"));
}
