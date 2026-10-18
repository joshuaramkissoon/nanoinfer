/// <reference lib="webworker" />
// The engine lives here. The main thread never touches wasm, so a 150 ms
// forward pass never janks the page.

import init, * as wasm from "./pkg/nanoinfer.js";
import wasmUrl from "./pkg/nanoinfer_bg.wasm?url";
import type { ForwardTrace, InitResult, LoadResult, OpRec, Req, StageInfo } from "./types";

declare const self: DedicatedWorkerGlobalScope;

const CACHE = "nanoinfer-weights-v1";
let engine: wasm.Engine | null = null;
let names: string[] = [];
let kinds: string[] = [];
let layout = { header: 16, stride: 10 };

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

function decode(buf: Float64Array): ForwardTrace {
  const H = layout.header;
  const nOps = buf[1], nTop = buf[2], nAttn = buf[3];
  if (buf[12] !== names.length) names = engine!.op_names() as string[];
  if (buf[13] !== kinds.length) kinds = engine!.op_kinds() as string[];
  const top: { id: number; p: number }[] = [];
  let o = H;
  for (let i = 0; i < nTop; i++, o += 2) top.push({ id: buf[o], p: buf[o + 1] });
  const attn = nAttn > 0 ? Float32Array.from(buf.subarray(o, o + nAttn)) : null;
  o += nAttn;
  const ops: OpRec[] = new Array(nOps);
  const S = layout.stride;
  for (let i = 0; i < nOps; i++, o += S) {
    ops[i] = {
      kind: kinds[buf[o]] ?? "?",
      name: names[buf[o + 1]] ?? "?",
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

function need(): wasm.Engine {
  if (!engine) throw new Error("engine not loaded");
  return engine;
}

async function handle(req: Req, post: (m: unknown) => void): Promise<[unknown, Transferable[]?]> {
  switch (req.t) {
    case "init": {
      await init({ module_or_path: wasmUrl });
      const l = wasm.trace_layout();
      layout = { header: l[1], stride: l[2] };
      const r: InitResult = {
        stages: wasm.stages() as StageInfo[],
        isolated: self.crossOriginIsolated,
        timerResUs: timerResolutionUs(),
        simd: true,
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
      const hold: { bytes: wasm.Bytes | null } = { bytes: null };
      const { total, cached } = await fetchBytes(
        `${req.base}/model.safetensors`,
        (c, size) => {
          // Reserve the whole file up front: growing a 500 MB Vec by doubling
          // would briefly need ~1.5x that.
          hold.bytes ??= new wasm.Bytes(size);
          hold.bytes.push(c);
        },
        (loaded, tot, cached) =>
          post({ t: "progress", p: { file: "model.safetensors", loaded, total: tot, cached, phase: "download" } }),
      );
      post({ t: "progress", p: { file: "model.safetensors", loaded: total, total, cached, phase: "parse" } });
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
      return [need().open(req.stage, req.maxSeq)];
    case "close":
      need().close(req.slot);
      return [null];
    case "setStage":
      need().set_stage(req.slot, req.stage);
      return [null];
    case "reset":
      need().reset(req.slot);
      return [null];
    case "forward": {
      const buf = need().forward(req.slot, Uint32Array.from(req.tokens));
      const tr = decode(buf);
      return [tr, tr.attn ? [tr.attn.buffer] : []];
    }
    case "sample": {
      const e = need();
      const id = e.sample(req.slot, req.temperature, req.topK);
      return [{ id, p: e.prob_of(req.slot, id) }];
    }
    case "encode":
      return [Array.from(need().encode(req.text))];
    case "capture":
      need().set_capture(req.topK, req.layer, req.head);
      return [null];
    case "attention": {
      const a = need().attention_maps(Uint32Array.from(req.tokens));
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
