// Building blocks shared by the Foundations chapters: figure frame, controls,
// callouts, and loaders for the two real data sources (the engine recording in
// public/recorded/ and the precomputed public/learn/data.json).

import { useEffect, useRef, useState, type ReactNode } from "react";
import { linkTo } from "./router";
import { loadRecorded, type Recorded } from "../lib/recorded";
import { claim, engine, getState, release } from "../engine/client";
import type { ForwardTrace } from "../engine/types";

// ---- the live engine ------------------------------------------------------------------

/** The fastest CPU stage this tab's wasm build has. */
export function bestStage(): string {
  const names = getState().stages.map((s) => s.name);
  for (const s of ["wasm-fma", "wasm-simd", "unrolled", "kv-cache"]) if (names.includes(s)) return s;
  return names[0] ?? "kv-cache";
}

/** One forward pass over `tokens` on a fresh session; returns the trace (with
 *  top-k next-token probabilities). Null if another figure holds the engine. */
export async function liveForward(who: string, tokens: number[], topK = 10): Promise<ForwardTrace | null> {
  if (getState().status !== "ready" || !claim(who)) return null;
  let slot = -1;
  try {
    slot = await engine.open(bestStage(), Math.max(64, tokens.length + 8));
    await engine.capture(topK);
    return await engine.forward(slot, tokens);
  } finally {
    if (slot >= 0) await engine.close(slot);
    release(who);
  }
}

// ---- data -------------------------------------------------------------------------

export interface LearnData {
  source: string;
  emb: {
    words: number[];
    group: string[];
    xy: [number, number][];
    near: [number, number][][];
    vecs: { s: number; b64: string }[];
    text: Record<string, string>;
    norm_mean: number;
  };
  quant: {
    matrix: string;
    n: number;
    hist: { lo: number; hi: number; counts: number[] };
    std: number;
    absmax: number;
    block: { row: number; col: number; w: number[] };
    block2: { row: number; col: number; w: number[] };
    err: { bits: number; rmse: number; rel: number; bytesPerWeight: number }[];
  };
  corner: number[][];
  batching: { machine: string; series: { label: string; points: [number, number, number, number][] }[] };
  serving: Record<string, ServingRate>;
  headline: {
    naive: BenchBest;
    gpu: BenchBest;
    /** GPU read bandwidth, GB/s (Metal probe, quiet machine) */
    gpuReadGBs: number;
    gpuSteps: { label: string; stage: string; tokS: number; work: number; idle: number; note: string }[];
    conditions: string;
  };
  webgpu: { what: string; stages: { stage: string; decode: number; prefill: number; metal: string | null; metalDecode: number | null }[] };
  spec: Record<string, SpecPair>;
}

export interface BenchBest {
  stage: string;
  decode: number;
  prefill: number;
  note: string;
  when: string;
}

export interface ServingRun {
  throughput: number;
  ttft_p50_ms: number;
  ttft_p99_ms: number;
  makespan: number;
  /** [arrival, admitted, first token, finished] in seconds, then output tokens */
  gantt: [number, number, number, number, number][];
}
export interface ServingRate {
  static: ServingRun;
  continuous: ServingRun;
  note: string;
  requests: number;
  maxBatch: number;
  kv: {
    pages: number;
    pageTokens: number;
    stats: { avg_live_seqs: number; avg_pages_per_seq: number; internal_fragmentation: number; snapshots: number };
    staticPeak: number;
    contPeak: number;
    /** each live request: [id, length in tokens, page ids in position order] */
    snapshots: { iter: number; seqs: [number, number, number[]][] }[];
  };
}

export interface SpecPrompt {
  prompt: string;
  text: string;
  target: number;
  draft: number;
  c: number;
  alpha: number;
  bestK: number;
  best: number;
  bestTokS: number;
  /** [k, speedup, alpha] */
  perK: [number, number, number][];
  /** k = 4: [token text, 1 = a draft guess the big model accepted, 0 = written by the big model] */
  playback: [string, number][];
  k4: { speedup: number; tokS: number; perForward: number };
}
export interface SpecPair {
  target: string;
  targetDtype: string;
  draft: string;
  draftDtype: string;
  note: string;
  prompts: SpecPrompt[];
}

let dataP: Promise<LearnData | null> | null = null;
function loadData() {
  dataP ??= fetch(`${import.meta.env.BASE_URL}learn/data.json`)
    .then((r) => (r.ok ? (r.json() as Promise<LearnData>) : null))
    .catch(() => null);
  return dataP;
}

export function useLearnData(): LearnData | null {
  const [d, setD] = useState<LearnData | null>(null);
  useEffect(() => {
    let live = true;
    loadData().then((x) => live && setD(x));
    return () => void (live = false);
  }, []);
  return d;
}

export function useRecorded(): Recorded | null {
  const [d, setD] = useState<Recorded | null>(null);
  useEffect(() => {
    let live = true;
    loadRecorded().then((x) => live && setD(x));
    return () => void (live = false);
  }, []);
  return d;
}

/** int8 base64 vector -> Float32Array */
export function unpackVec(v: { s: number; b64: string }): Float32Array {
  const bin = atob(v.b64);
  const out = new Float32Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = ((bin.charCodeAt(i) << 24) >> 24) * v.s;
  return out;
}

// ---- frame --------------------------------------------------------------------------

/** True while the element is within ~100px of the viewport. Unlike a one-shot "seen" latch it turns false again, so
 *  looping figures pass it to useFrame and stop costing frames the moment they scroll away (and resume on return).
 *  Usage: `const [onScreen, vis] = useOnScreen<HTMLElement>(); useFrame(vis && playing, …); <Fig figRef={onScreen}>` */
export function useOnScreen<T extends Element>(): [(el: T | null) => void, boolean] {
  const [v, setV] = useState(false);
  const io = useRef<IntersectionObserver | null>(null);
  const ref = (el: T | null) => {
    io.current?.disconnect();
    io.current = null;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") return setV(true);
    io.current = new IntersectionObserver(([e]) => setV(e.isIntersecting), { rootMargin: "100px 0px" });
    io.current.observe(el);
  };
  // a stable callback: React calls it with null on unmount, which disconnects (no effect cleanup, so StrictMode's
  // double-invoked effects can't silently kill the observer)
  const stable = useRef(ref);
  return [stable.current, v];
}

export type Prov = "toy" | "diagram" | "recorded" | "live" | "measured" | "weights" | "external" | "estimate";
const PROV: Record<Prov, string> = {
  toy: "toy example",
  diagram: "diagram",
  recorded: "real engine · recorded on an M4",
  live: "live · the engine in this tab",
  measured: "measured · our M4 benchmarks",
  weights: "real GPT-2 weights",
  external: "external · spec sheets",
  estimate: "estimate · arithmetic shown",
};

export function Fig({
  n,
  title,
  prov,
  children,
  caption,
  tryit,
  wide,
  aside,
  source,
  figRef,
}: {
  n: string;
  title: string;
  prov: Prov;
  children: ReactNode;
  /** "what you're looking at": axes, colours, what to notice */
  caption?: ReactNode;
  /** "try": one concrete thing to do with the figure */
  tryit?: ReactNode;
  wide?: boolean;
  /** extra element in the header (e.g. a "run live" button) */
  aside?: ReactNode;
  /** where external numbers come from (shown under the caption) */
  source?: ReactNode;
  /** attach a useOnScreen ref to the figure, to pause its animation while it's off screen */
  figRef?: (el: HTMLElement | null) => void;
}) {
  return (
    <figure ref={figRef} className={`lr-fig${wide ? " wide" : ""}`} data-ask-section>
      <header className="lr-fig-head">
        <span className="lr-fig-title">
          <span className="n">fig {n}</span>
          {title}
        </span>
        <span className="lr-fig-prov">
          {aside}
          <span className={`lr-prov ${prov}`}>{PROV[prov]}</span>
        </span>
      </header>
      <div className="lr-fig-body">{children}</div>
      {(caption || tryit || source) && (
        <figcaption className="lr-cap">
          {caption && (
            <p>
              <span className="lr-cap-k">what you're looking at</span>
              {caption}
            </p>
          )}
          {tryit && (
            <p>
              <span className="lr-cap-k">try</span>
              {tryit}
            </p>
          )}
          {source && (
            <p className="lr-src">
              <span className="lr-cap-k">sources</span>
              {source}
            </p>
          )}
        </figcaption>
      )}
    </figure>
  );
}

/** The one sentence to remember from a section. */
export function Idea({ children, label = "the idea" }: { children: ReactNode; label?: string }) {
  return (
    <aside className="lr-idea">
      <span className="lr-idea-k">{label}</span>
      <div>{children}</div>
    </aside>
  );
}

/** The disclosure mark: a + drawn on a 12px grid, centred in its circle, that turns to × when open. */
export function PlusIcon() {
  return (
    <span className="plus" aria-hidden>
      <svg viewBox="0 0 12 12" width="10" height="10">
        <path d="M6 1.5v9M1.5 6h9" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" fill="none" />
      </svg>
    </span>
  );
}

/** Progressive disclosure: the extra depth, folded. */
export function Deeper({ title, children, id }: { title: string; children: ReactNode; id?: string }) {
  return (
    <details className="lr-deeper" id={id}>
      <summary>
        <PlusIcon />
        {title}
      </summary>
      <div className="lr-deeper-body">{children}</div>
    </details>
  );
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  fmt,
  width,
}: {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  fmt?: (v: number) => ReactNode;
  width?: string;
}) {
  return (
    <label className="lr-slider">
      <span className="k">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} style={width ? { width } : undefined} />
      <span className="v">{fmt ? fmt(value) : value}</span>
    </label>
  );
}

export function Seg<T extends string | number>({
  options,
  value,
  onChange,
  label,
}: {
  options: { v: T; label: ReactNode }[];
  value: T;
  onChange: (v: T) => void;
  label?: string;
}) {
  return (
    <span className="lr-seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.v)} aria-pressed={o.v === value} onClick={() => onChange(o.v)}>
          {o.label}
        </button>
      ))}
    </span>
  );
}

export function Stat({ k, v, unit, accent }: { k: ReactNode; v: ReactNode; unit?: ReactNode; accent?: boolean }) {
  return (
    <div className="lr-stat">
      <div className="k">{k}</div>
      <div className={`v${accent ? " accent" : ""}`}>
        {v}
        {unit && <small> {unit}</small>}
      </div>
    </div>
  );
}

export function Controls({ children }: { children: ReactNode }) {
  return <div className="lr-controls">{children}</div>;
}

/** A token chip. */
export function Chip({ children, tone, title, onClick, on }: { children: ReactNode; tone?: number; title?: string; onClick?: () => void; on?: boolean }) {
  const cls = `lr-chip${tone != null ? ` t${tone % 6}` : ""}${on ? " on" : ""}${onClick ? " btn" : ""}`;
  return onClick ? (
    <button className={cls} title={title} onClick={onClick}>
      {children}
    </button>
  ) : (
    <span className={cls} title={title}>
      {children}
    </span>
  );
}

// ---- motion --------------------------------------------------------------------------

/** Calls fn(dtSeconds) every frame while `on`. Looping figures pass `onScreen && …` (see useOnScreen). */
export function useFrame(on: boolean, fn: (dt: number) => void) {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    if (!on) return;
    let raf = 0, last = performance.now();
    const tick = (t: number) => {
      ref.current(Math.min(0.1, (t - last) / 1000));
      last = t;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [on]);
}

/** A step counter that advances every `ms` while playing. */
export function useStepper(n: number, ms: number, playing: boolean): [number, (i: number) => void] {
  const [i, setI] = useState(0);
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setI((x) => (x + 1) % n), ms);
    return () => clearInterval(t);
  }, [n, ms, playing]);
  return [i, setI];
}

export const nf = (x: number, d = 0) => x.toLocaleString("en-GB", { maximumFractionDigits: d, minimumFractionDigits: d });

// ---- chart colours ------------------------------------------------------------------------

/** The chart palette (validated with the dataviz skill's checker against both surfaces):
 *  c1 ember = the thing the chart is about, c2 blue = memory / the comparison, c3 olive = a third
 *  series. Everything else is grey (pal.muted). */
export function chartColors(dark: boolean) {
  return dark ? { c1: "#c96f38", c2: "#5b97d0", c3: "#8a9d3c" } : { c1: "#b8541f", c2: "#2f6fa8", c3: "#7a8f2a" };
}

// ---- glossary -------------------------------------------------------------------------------

export const GLOSSARY: Record<string, string> = {
  token: "A piece of text the model reads or writes: a word, part of a word, or a single character. GPT-2 knows 50,257 of them.",
  "language model": "A program that, given some text, gives a probability for every possible next token.",
  probability: "How likely something is, from 0% (never) to 100% (certain). All the options together add up to 100%.",
  vector: "A list of numbers. GPT-2 represents every token as a list of 768 numbers.",
  embedding: "The list of 768 numbers a token is turned into before the model works on it.",
  weights: "The numbers the model learned in training. GPT-2 small has 124 million of them; using the model means doing arithmetic with all of them.",
  "dot product": "Multiply two equal-length lists number by number, then add up the results.",
  matrix: "A grid of numbers. Multiplying it by a vector does one dot product per row.",
  layer: "One stage of the model. GPT-2 has 12, each reading the token's vector and adding an edit to it.",
  attention: "The step where a token looks back at earlier tokens and blends in information from them.",
  softmax: "Turns any list of scores into percentages that add up to 100%, keeping the biggest score the biggest.",
  logit: "The model's raw score for one possible next token, before softmax turns the scores into probabilities.",
  "forward pass": "One full run of the model: the tokens go in at the top, through all 12 layers, and scores for the next token come out.",
  prefill: "The first forward pass, which reads the whole prompt at once.",
  decode: "Every forward pass after the first: each one produces exactly one new token.",
  "tok/s": "Tokens per second: how fast the model writes. Roughly, a token is three quarters of a word.",
  bandwidth: "How many bytes per second can be read from memory. Our M4 Mac mini's spec: 120 GB/s.",
  FLOP: "One arithmetic operation: a single multiply or a single add. FLOP/s = how many per second.",
  "memory-bound": "Waiting on memory: the arithmetic is quick, but the numbers can't arrive any faster.",
  "compute-bound": "Waiting on arithmetic: the numbers are there, but the processor can't do the maths any faster.",
  "KV cache": "Each past token's keys and values, saved so they never have to be recomputed.",
  quantization: "Storing each weight in fewer bits (say 8 or 4 instead of 32), so there are fewer bytes to read.",
  SIMD: "Single instruction, multiple data: one instruction that does the same operation on 4, 8 or more numbers at once.",
  thread: "One independent stream of work. A CPU core runs one at a time; a GPU runs thousands.",
  GPU: "A chip with thousands of small, simple cores built to do the same arithmetic on lots of numbers at once.",
  batch: "Several requests processed together in one forward pass, so each weight read from memory is used for all of them.",
  "speculative decoding": "A small, fast model guesses the next few tokens; the big model checks all the guesses in one pass and keeps the ones it agrees with.",
  "time to first token": "How long a user waits, after sending a request, before the first word of the answer appears.",
  page: "A fixed-size block of KV-cache memory (here 16 tokens' worth). A request gets pages as it grows and hands them back when it's done.",
  GB: "Gigabyte: a billion bytes. A byte is 8 bits; one 32-bit number takes 4 bytes.",
  inference: "Running a trained model to get answers. The weights are frozen; only the input changes.",
  training: "Finding the weights: adjusting billions of numbers until the model predicts text well. Done once, at great expense.",
  parameters: "Another word for weights. “7B” means seven billion of them.",
  HBM: "High-bandwidth memory: stacks of DRAM packaged right next to a GPU die. An H100 has 80 GB of it at ~3.35 TB/s.",
  "unified memory": "One pool of memory shared by the CPU and GPU (Apple silicon), so nothing is copied between them.",
  MoE: "Mixture of experts: each layer has many parallel feed-forward blocks and a router sends each token to only a few of them.",
  "all-reduce": "Every device contributes a partial result and every device ends up with the total.",
  TTFT: "Time to first token: how long until the first word of the answer appears. Dominated by prefill and queueing.",
  TPOT: "Time per output token: the gap between words once the answer is streaming. Dominated by decode.",
  throughput: "Total tokens per second across all users: what the operator pays for.",
  latency: "How long one user waits: for the first token, and between tokens.",
  bf16: "A 16-bit number format with fp32's range and less precision: the default for modern model weights.",
  NVLink: "NVIDIA's GPU-to-GPU link inside a server: ~900 GB/s total per H100, about 7× PCIe 5.",
  kernel: "One function that runs on the GPU across thousands of threads, such as one matrix multiply.",
  "context window": "The most tokens a model can look at at once (its prompt plus its answer so far).",
  "tensor parallel": "Splitting every weight matrix across GPUs so each does a slice of every multiply; needs an all-reduce per layer.",
  "pipeline parallel": "Splitting the model by layers across GPUs, like stages of an assembly line.",
  "continuous batching": "Re-forming the batch at every decode step: finished requests leave, waiting ones join.",
  "paged attention": "Storing the KV cache in fixed-size pages with a per-request page table, like virtual memory.",
  "prefix caching": "Keeping the KV cache of a shared prompt prefix (a system prompt) so the next request skips its prefill.",
  goodput: "Throughput counting only requests that met their latency targets.",
};

/** Glossary terms that have a fuller entry in the field guide (/learn/field-guide#id). */
export const GUIDE_LINK: Record<string, string> = {
  token: "token",
  weights: "parameters",
  embedding: "embeddings",
  layer: "layers",
  attention: "attention",
  softmax: "attention",
  prefill: "prefill",
  decode: "decode",
  "tok/s": "toks",
  bandwidth: "mbu",
  "memory-bound": "memory-bound",
  "compute-bound": "memory-bound",
  "KV cache": "kv-cache",
  quantization: "quantisation",
  GPU: "tensor-cores",
  batch: "batching",
  "speculative decoding": "speculative",
  "time to first token": "ttft",
  page: "paged-attention",
  inference: "inference",
  parameters: "parameters",
  HBM: "hbm",
  "unified memory": "unified-memory",
  MoE: "moe",
  "all-reduce": "all-reduce",
  TTFT: "ttft",
  TPOT: "tpot",
  throughput: "throughput-latency",
  latency: "throughput-latency",
  bf16: "formats",
  NVLink: "nvlink",
  kernel: "kernels",
  "context window": "context-window",
  "tensor parallel": "tensor-parallel",
  "pipeline parallel": "pipeline-parallel",
  "continuous batching": "continuous-batching",
  "paged attention": "paged-attention",
  "prefix caching": "prefix-caching",
  goodput: "goodput",
  "forward pass": "forward-pass",
};

/** Small colour key for captions: <Swatch c={cc.c2}>blue</Swatch>: bytes read. */
export function Swatch({ c, children }: { c: string; children: ReactNode }) {
  return (
    <b className="lr-sw">
      <i style={{ background: c }} />
      {children}
    </b>
  );
}

/** A glossary term: dotted underline, one-line definition on hover or focus, and a link to the field guide entry. */
export function G({ k, children }: { k: string; children?: ReactNode }) {
  const def = GLOSSARY[k];
  if (!def) return <>{children ?? k}</>;
  const gid = GUIDE_LINK[k];
  const href = gid ? `/learn/field-guide#${gid}` : null;
  return (
    <span className="lr-g" tabIndex={0}>
      {children ?? k}
      <span className="lr-g-tip" role="tooltip">
        <b>{k}</b> {def}
        {href && (
          <a className="lr-g-more" href={href} onClick={linkTo(href)}>
            field guide →
          </a>
        )}
      </span>
    </span>
  );
}
