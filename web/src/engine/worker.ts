/// <reference lib="webworker" />
// The engine lives here. The main thread never touches wasm, so a 150 ms
// forward pass never janks the page.

import type * as Wasm from "./pkg/nanoinfer.js";
import type { BuildName, ForwardTrace, GpuInfo, InitResult, LoadResult, OpRec, Req, StageInfo } from "./types";

declare const self: DedicatedWorkerGlobalScope;

// Three builds of the same crate (crates/wasm/build.sh); we load the best one
// this browser can run. Globs, so a missing build (no nightly for threads) just
// isn't offered.
const MODS = import.meta.glob("./pkg*/nanoinfer.js") as Record<string, () => Promise<typeof Wasm>>;
const URLS = import.meta.glob("./pkg*/nanoinfer_bg.wasm", { query: "?url", import: "default", eager: true }) as Record<string, string>;
const DIRS: Record<BuildName, string> = { simd128: "./pkg", relaxed: "./pkg-relaxed", threads: "./pkg-threads" };

let wasm: typeof Wasm;
type Engine = Wasm.Engine;

const CACHE = "nanoinfer-weights-v1";
let engine: Engine | null = null;
let names: string[] = [];
let kinds: string[] = [];
let layout = { header: 16, stride: 10 };

// ---- WebGPU rungs (crates/webgpu through wgpu) ----------------------------------
// WebGPU sessions live in the wasm module's registry; the page addresses them
// like any other slot: ids >= GPU_BASE, or an engine slot (the hero's slot 0)
// redirected to a WebGPU session by setStage.
const GPU_BASE = 1_000_000;
let gpu: GpuInfo | null = null;
let gpuError: string | null = null;
let gpuStageNames = new Set<string>();
const redirect = new Map<number, number>();
let gNames: string[] = [];
let gKinds: string[] = [];
let topK = 10;
const isGpuStage = (s: string) => gpuStageNames.has(s);
const gslot = (slot: number): number | undefined => (slot >= GPU_BASE ? slot - GPU_BASE : redirect.get(slot));

async function initGpu(): Promise<StageInfo[]> {
  if (typeof (wasm as { webgpu_init?: unknown }).webgpu_init !== "function") {
    gpuError = "this wasm build has no WebGPU rungs";
    return [];
  }
  if (!("gpu" in navigator)) {
    gpuError = "this browser has no WebGPU (navigator.gpu is missing)";
    return [];
  }
  try {
    const info = await Promise.race([
      wasm.webgpu_init(),
      new Promise((_, rej) => setTimeout(() => rej(new Error("WebGPU adapter request timed out")), 8000)),
    ]);
    gpu = info as GpuInfo;
    const st = (wasm.webgpu_stages() as StageInfo[]).filter((s) => s.supported !== false);
    gpuStageNames = new Set(st.map((s) => s.name));
    return st;
  } catch (e) {
    gpuError = e instanceof Error ? e.message : String(e);
    return [];
  }
}

function timerResolutionUs(): number {
  // Smallest non-zero step performance.now() can make.
  let best = Infinity;
  for (let i = 0; i < 40; i++) {
    const a = performance.now();
    let b = a;
    while (b === a) b = performance.now();
    best = Math.min(best, b - a);
  }
  return best * 1000;
}

/** OPFS file name for a URL (flat, filesystem-safe). */
export const opfsName = (url: string) => url.replace(/^https?:\/\//, "").replace(/[^a-zA-Z0-9._-]/g, "_");

async function opfsDir(): Promise<FileSystemDirectoryHandle | null> {
  try {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(CACHE, { create: true });
  } catch {
    return null;
  }
}

/**
 * Stream a URL chunk by chunk. Files are cached in OPFS (origin private file
 * system): written as they download, so nothing is held twice, and big files
 * work where the Cache API gives up. A ".ok" marker records a complete write.
 */
async function fetchBytes(
  url: string,
  onChunk: (c: Uint8Array, total: number) => void,
  progress: (loaded: number, total: number, cached: boolean) => void,
): Promise<{ total: number; cached: boolean }> {
  const abs = new URL(url, self.location.href).href;
  const name = opfsName(abs);
  const dir = await opfsDir();
  let body: ReadableStream<Uint8Array> | null = null;
  let total = 0;
  let cached = false;
  let sink: FileSystemWritableFileStream | null = null;
  if (dir) {
    try {
      const ok = await (await (await dir.getFileHandle(name + ".ok")).getFile()).text();
      const file = await (await dir.getFileHandle(name)).getFile();
      if (String(file.size) === ok.trim()) {
        body = file.stream();
        total = file.size;
        cached = true;
      }
    } catch {
      /* not cached */
    }
  }
  if (!body) {
    const res = await fetch(abs);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status} (weights missing? run scripts/fetch_weights.sh)`);
    total = Number(res.headers.get("Content-Length") ?? 0);
    body = res.body!;
    if (dir) {
      try {
        await dir.removeEntry(name + ".ok").catch(() => {});
        sink = await (await dir.getFileHandle(name, { create: true })).createWritable();
      } catch {
        sink = null;
      }
    }
  }
  const reader = body.getReader();
  let loaded = 0;
  let lastPost = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    onChunk(value, total);
    if (sink) {
      try {
        await sink.write(value);
      } catch {
        await sink.abort().catch(() => {});
        sink = null;
      }
    }
    loaded += value.byteLength;
    const now = performance.now();
    if (now - lastPost > 60) {
      progress(loaded, total || loaded, cached);
      lastPost = now;
    }
  }
  if (sink && dir) {
    try {
      await sink.close();
      const w = await (await dir.getFileHandle(name + ".ok", { create: true })).createWritable();
      await w.write(String(loaded));
      await w.close();
    } catch {
      /* quota: fine, just not cached */
    }
  }
  progress(loaded, loaded, cached);
  return { total: loaded, cached };
}

function decode(buf: Float64Array, onGpu = false): ForwardTrace {
  const H = layout.header;
  const nOps = buf[1], nTop = buf[2], nAttn = buf[3];
  if (onGpu) {
    if (buf[12] !== gNames.length) gNames = wasm.webgpu_op_names() as string[];
    if (buf[13] !== gKinds.length) gKinds = wasm.webgpu_op_kinds() as string[];
  } else {
    if (buf[12] !== names.length) names = engine!.op_names() as string[];
    if (buf[13] !== kinds.length) kinds = engine!.op_kinds() as string[];
  }
  const nm = onGpu ? gNames : names;
  const kd = onGpu ? gKinds : kinds;
  const top: { id: number; p: number }[] = [];
  let o = H;
  for (let i = 0; i < nTop; i++, o += 2) top.push({ id: buf[o], p: buf[o + 1] });
  const attn = nAttn > 0 ? Float32Array.from(buf.subarray(o, o + nAttn)) : null;
  o += nAttn;
  const ops: OpRec[] = new Array(nOps);
  const S = layout.stride;
  for (let i = 0; i < nOps; i++, o += S) {
    ops[i] = {
      kind: kd[buf[o]] ?? "?",
      name: nm[buf[o + 1]] ?? "?",
      layer: buf[o + 2],
      dims: [buf[o + 3], buf[o + 4], buf[o + 5]],
      flops: buf[o + 6],
      bytes: buf[o + 7],
      start: buf[o + 8],
      dur: buf[o + 9],
    };
  }
  return {
    phase: buf[4] === 0 ? "prefill" : "decode",
    nNew: buf[5],
    nComputed: buf[6],
    contextLen: buf[7],
    kvLen: buf[8],
    kvCap: buf[9],
    ms: buf[10],
    entropy: buf[11],
    top,
    attn,
    attnLayer: buf[14],
    attnHead: buf[15],
    ops,
  };
}

/** Which builds exist and validate here, best first. */
async function usableBuilds(): Promise<{ name: BuildName; bytes: ArrayBuffer }[]> {
  const out: { name: BuildName; bytes: ArrayBuffer }[] = [];
  for (const name of ["threads", "relaxed", "simd128"] as BuildName[]) {
    const url = URLS[`${DIRS[name]}/nanoinfer_bg.wasm`];
    if (!url || !MODS[`${DIRS[name]}/nanoinfer.js`]) continue;
    if (name === "threads" && !(self.crossOriginIsolated && typeof SharedArrayBuffer !== "undefined")) continue;
    try {
      const bytes = await (await fetch(url)).arrayBuffer();
      // validate() is the real feature test: relaxed-SIMD opcodes, shared memory.
      if (WebAssembly.validate(bytes)) out.push({ name, bytes });
    } catch {
      /* not served */
    }
  }
  return out;
}

function need(): Engine {
  if (!engine) throw new Error("engine not loaded");
  return engine;
}

async function handle(req: Req, post: (m: unknown) => void): Promise<[unknown, Transferable[]?]> {
  switch (req.t) {
    case "init": {
      const builds = await usableBuilds();
      if (!builds.length) throw new Error("this browser can't run WebAssembly SIMD");
      const pick = builds.find((b) => b.name === req.build) ?? builds[0];
      wasm = await MODS[`${DIRS[pick.name]}/nanoinfer.js`]();
      await wasm.default({ module_or_path: pick.bytes });
      let threads = 1;
      if (pick.name === "threads") {
        // Spawns the rayon pool: one Web Worker per core, all sharing this heap.
        await (wasm as unknown as { initThreadPool(n: number): Promise<void> }).initThreadPool(req.threads || navigator.hardwareConcurrency || 4);
        threads = wasm.thread_count();
      }
      const l = wasm.trace_layout();
      layout = { header: l[1], stride: l[2] };
      const gpuStages = await initGpu();
      const r: InitResult = {
        stages: [...(wasm.stages() as StageInfo[]), ...gpuStages],
        webgpu: gpu,
        webgpuError: gpuError,
        isolated: self.crossOriginIsolated,
        timerResUs: timerResolutionUs(),
        simd: true,
        build: wasm.build_name() as BuildName,
        builds: builds.map((b) => b.name),
        threads,
        peakKind: wasm.peak_kind(),
      };
      return [r];
    }
    case "load": {
      const t0 = performance.now();
      const text = async (f: string) => {
        const chunks: Uint8Array[] = [];
        await fetchBytes(`${req.base}/${f}`, (c) => chunks.push(c), () => {});
        const n = chunks.reduce((a, c) => a + c.byteLength, 0);
        const out = new Uint8Array(n);
        let o = 0;
        for (const c of chunks) {
          out.set(c, o);
          o += c.byteLength;
        }
        return out;
      };
      const [vocab, merges] = await Promise.all([text("vocab.json"), text("merges.txt")]);
      const hold: { bytes: Wasm.Bytes | null } = { bytes: null };
      const f16 = req.dtype === "f16";
      const file = f16 ? "model-f16.safetensors" : "model.safetensors";
      const { total, cached } = await fetchBytes(
        `${req.base}/${file}`,
        (c, size) => {
          // Reserve the whole (f32) file up front: growing a 500 MB Vec by
          // doubling would briefly need ~1.5x that. The f16 buffer widens to
          // f32 as it streams in.
          hold.bytes ??= f16 ? wasm.Bytes.from_f16(size) : new wasm.Bytes(size);
          hold.bytes.push(c);
        },
        (loaded, tot, cached) => post({ t: "progress", p: { file, loaded, total: tot, cached, phase: "download" } }),
      );
      post({ t: "progress", p: { file, loaded: total, total, cached, phase: "parse" } });
      engine?.free();
      const bytes = hold.bytes;
      if (!bytes) throw new Error("model.safetensors was empty");
      // Yield so the "parse" progress message goes out before we block.
      await new Promise((r) => setTimeout(r, 0));
      engine = new wasm.Engine(bytes, vocab, merges, req.stage);
      names = [];
      kinds = [];
      const c = engine.config();
      const blob = engine.vocab_blob();
      const offs = engine.vocab_offsets();
      const r: LoadResult = {
        config: {
          nVocab: c[0], nCtx: c[1], nEmbd: c[2], nHead: c[3], nLayer: c[4], nParams: c[5], decodeWeightBytes: c[6],
        },
        vocabBlob: blob,
        vocabOffsets: offs,
        loadMs: performance.now() - t0,
        fromCache: cached,
        weightBytes: total,
      };
      return [r, [blob.buffer, offs.buffer]];
    }
    case "open":
      if (isGpuStage(req.stage)) return [GPU_BASE + wasm.webgpu_open(need(), req.stage, req.maxSeq)];
      return [need().open(req.stage, req.maxSeq)];
    case "close": {
      const g = gslot(req.slot);
      if (g !== undefined) {
        wasm.webgpu_close(g);
        redirect.delete(req.slot);
        wasm.webgpu_trim();
      }
      if (req.slot < GPU_BASE) need().close(req.slot);
      return [null];
    }
    case "setStage": {
      const g = gslot(req.slot);
      if (isGpuStage(req.stage)) {
        if (req.slot >= GPU_BASE) throw new Error("can't change a WebGPU session's stage");
        if (g !== undefined && wasm.webgpu_stage(g) === req.stage) {
          wasm.webgpu_reset(g);
          return [null];
        }
        if (g !== undefined) wasm.webgpu_close(g);
        redirect.set(req.slot, wasm.webgpu_open(need(), req.stage, 0));
        wasm.webgpu_trim();
        return [null];
      }
      if (g !== undefined) {
        wasm.webgpu_close(g);
        redirect.delete(req.slot);
        wasm.webgpu_trim();
      }
      need().set_stage(req.slot, req.stage);
      return [null];
    }
    case "reset": {
      const g = gslot(req.slot);
      if (g !== undefined) wasm.webgpu_reset(g);
      else need().reset(req.slot);
      return [null];
    }
    case "forward": {
      const g = gslot(req.slot);
      if (g !== undefined) {
        const tr = decode(await wasm.webgpu_forward(g, Uint32Array.from(req.tokens), topK), true);
        return [tr];
      }
      const buf = need().forward(req.slot, Uint32Array.from(req.tokens));
      const tr = decode(buf);
      return [tr, tr.attn ? [tr.attn.buffer] : []];
    }
    case "sample": {
      const e = need();
      const g = gslot(req.slot);
      if (g !== undefined) {
        const id = wasm.webgpu_sample(e, g, req.temperature, req.topK);
        return [{ id, p: wasm.webgpu_prob_of(g, id) }];
      }
      const id = e.sample(req.slot, req.temperature, req.topK);
      return [{ id, p: e.prob_of(req.slot, id) }];
    }
    case "gpuGreedy": {
      const g = gslot(req.slot);
      if (g === undefined) throw new Error("gpuGreedy: not a WebGPU session");
      const r = await wasm.webgpu_greedy(g, req.first, req.n);
      return [{ ms: r[0], dispatches: r[1], submits: r[2], gpuMs: isNaN(r[3]) ? null : r[3], tokens: Array.from(r.subarray(4)) }];
    }
    case "gpuBench": {
      const g = gslot(req.slot);
      if (g === undefined) throw new Error("gpuBench: not a WebGPU session");
      const r = await wasm.webgpu_bench(g, Uint32Array.from(req.prompt), req.gen);
      return [{ prefillMs: r[0], decodeMs: r[1], dispatches: r[2], submits: r[3], gpuMs: isNaN(r[4]) ? null : r[4] }];
    }
    case "gpuProfile": {
      const g = gslot(req.slot);
      if (g === undefined) throw new Error("gpuProfile: not a WebGPU session");
      return [await wasm.webgpu_profile(g, Uint32Array.from(req.tokens), req.reps)];
    }
    case "gpuProbe":
      if (!gpu) throw new Error(gpuError ?? "no WebGPU");
      return [await wasm.webgpu_probe(req.reps)];
    case "encode":
      return [Array.from(need().encode(req.text))];
    case "capture":
      topK = req.topK;
      need().set_capture(req.topK, req.layer, req.head);
      return [null];
    case "attention": {
      const a = need().attention_maps(Uint32Array.from(req.tokens));
      return [a, [a.buffer]];
    }
    case "lens": {
      const a = need().logit_lens(Uint32Array.from(req.tokens), Uint32Array.from(req.positions), req.topK);
      return [a, [a.buffer]];
    }
    case "residual": {
      const a = need().residual_stream(Uint32Array.from(req.tokens), req.pos);
      return [a, [a.buffer]];
    }
    case "logits": {
      const g = gslot(req.slot);
      const a = g !== undefined ? wasm.webgpu_logits(g) : need().logits(req.slot);
      return [a, [a.buffer]];
    }
    case "benchBandwidth":
      return [wasm.bench_bandwidth(req.mb, req.iters)];
    case "benchPeak":
      return [wasm.bench_peak_gflops(req.millions)];
    case "seed":
      need().set_seed(BigInt(req.seed));
      return [null];
    case "clearCache": {
      try {
        const root = await navigator.storage.getDirectory();
        await caches.delete(CACHE).catch(() => false); // older builds used the Cache API
        await root.removeEntry(CACHE, { recursive: true });
        return [true];
      } catch {
        return [false];
      }
    }
  }
}

// Requests are handled strictly one at a time, in order.
let queue: Promise<void> = Promise.resolve();
self.onmessage = (e: MessageEvent<{ id: number; req: Req }>) => {
  const { id, req } = e.data;
  queue = queue.then(async () => {
    try {
      const [result, transfer] = await handle(req, (m) => self.postMessage({ id, ...(m as object) }));
      self.postMessage({ id, t: "ok", result }, transfer ?? []);
    } catch (err) {
      self.postMessage({ id, t: "err", error: err instanceof Error ? err.message : String(err) });
    }
  });
};
