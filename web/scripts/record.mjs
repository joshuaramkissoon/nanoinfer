#!/usr/bin/env node
// Record real engine data for the page's no-download fallback.
//
// Runs the *same wasm* the page runs (web/src/engine/pkg-relaxed, built by
// `npm run wasm`) in Node, on V8 (Chrome's engine), and writes a few small
// files to web/public/recorded/. Every figure renders from these until the
// reader presses "run live", which swaps in data from their own tab.
//
//   cd web && npm run wasm && node scripts/record.mjs
//
// Output (all KB-sized, committed):
//   meta.json       machine, runtime, build, stages, config, measured roofs
//   run.json        the hero run: tokens, top-10s, every forward's op trace
//                   (packed), a vocab subset, the walkthrough token's residual
//                   stream and logits, and the logit lens
//   attention.bin   144 heads × T×T causal attention (u8, log-quantised)
//   bench.json      per-stage tok/s (the Ladder) and the race lanes
//   native.json     latest native result per stage, from bench/results.jsonl
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const here = import.meta.dirname;
const root = path.resolve(here, "../..");
const OUT = path.join(here, "../public/recorded");
const WEIGHTS = process.env.NANOINFER_WEIGHTS ?? path.join(root, "weights/gpt2");
const BUILD = process.env.BUILD ?? "pkg-relaxed";
const QUICK = !!process.env.QUICK; // skip the slow ladder/race (for iterating)

const pkg = path.join(here, "../src/engine", BUILD);
const wasm = await import(path.join(pkg, "nanoinfer.js"));
await wasm.default({ module_or_path: fs.readFileSync(path.join(pkg, "nanoinfer_bg.wasm")) });

// ---- load -------------------------------------------------------------------
const t0 = performance.now();
const file = path.join(WEIGHTS, "model.safetensors");
const size = fs.statSync(file).size;
const bytes = new wasm.Bytes(size);
{
  const fd = fs.openSync(file, "r");
  const chunk = Buffer.alloc(8 << 20);
  for (let n; (n = fs.readSync(fd, chunk, 0, chunk.length, null)) > 0; ) bytes.push(chunk.subarray(0, n));
  fs.closeSync(fd);
}
const engine = new wasm.Engine(bytes, fs.readFileSync(path.join(WEIGHTS, "vocab.json")), fs.readFileSync(path.join(WEIGHTS, "merges.txt")), "kv-cache");
const loadMs = performance.now() - t0;
const c = engine.config();
const config = { nVocab: c[0], nCtx: c[1], nEmbd: c[2], nHead: c[3], nLayer: c[4], nParams: c[5], decodeWeightBytes: c[6] };
const stages = wasm.stages();
const blob = engine.vocab_blob();
const offs = engine.vocab_offsets();
const tokBytes = (id) => blob.subarray(offs[id], offs[id + 1]);
const utf8 = new TextDecoder();
const text = (ids) => utf8.decode(Buffer.concat(ids.map((i) => Buffer.from(tokBytes(i)))));
console.log(`loaded in ${loadMs.toFixed(0)} ms · build ${wasm.build_name()} · ${stages.map((s) => s.name).join(" ")}`);

// ---- packed trace decode (same as web/src/engine/worker.ts) --------------------
const [, H, S] = wasm.trace_layout();
let names = [], kinds = [];
function decode(buf) {
  if (buf[12] !== names.length) names = engine.op_names();
  if (buf[13] !== kinds.length) kinds = engine.op_kinds();
  const nOps = buf[1], nTop = buf[2], nAttn = buf[3];
  let o = H;
  const top = [];
  for (let i = 0; i < nTop; i++, o += 2) top.push({ id: buf[o], p: buf[o + 1] });
  o += nAttn;
  const ops = [];
  for (let i = 0; i < nOps; i++, o += S)
    ops.push({ kind: kinds[buf[o]], name: names[buf[o + 1]], layer: buf[o + 2], dims: [buf[o + 3], buf[o + 4], buf[o + 5]], flops: buf[o + 6], bytes: buf[o + 7], start: buf[o + 8], dur: buf[o + 9] });
  return { phase: buf[4] === 0 ? "prefill" : "decode", nNew: buf[5], nComputed: buf[6], contextLen: buf[7], kvLen: buf[8], kvCap: buf[9], ms: buf[10], entropy: buf[11], top, ops };
}
const fwd = (slot, ids) => decode(engine.forward(slot, Uint32Array.from(ids)));
const median = (xs) => { const s = [...xs].sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

// ---- roofs ------------------------------------------------------------------------
let bw = 0, peak = 0;
for (let i = 0; i < 3; i++) {
  bw = Math.max(bw, wasm.bench_bandwidth(256, 8));
  peak = Math.max(peak, wasm.bench_peak_gflops(400));
}
console.log(`roofs: ${bw.toFixed(1)} GB/s · ${peak.toFixed(1)} GFLOP/s`);

// ---- hero run ---------------------------------------------------------------------
const PROMPT = "In a shocking finding, scientists discovered a herd of unicorns living in";
const HERO_STAGE = stages.some((s) => s.name === "wasm-fma") ? "wasm-fma" : "unrolled";
const GEN = 32;
engine.set_stage(0, HERO_STAGE);
engine.set_seed(BigInt(process.env.SEED ?? 11));
engine.set_capture(10, -1, -1);
const prompt = Array.from(engine.encode(PROMPT));
const tokens = prompt.map((id) => ({ id, prompt: true, p: null, top: null, entropy: null }));
const forwards = [];
let tr = fwd(0, prompt);
forwards.push(tr);
for (let i = 0; i < GEN; i++) {
  const id = engine.sample(0, 0.8, 40);
  tokens.push({ id, prompt: false, p: engine.prob_of(0, id), top: tr.top, entropy: tr.entropy });
  if (i === GEN - 1) break;
  tr = fwd(0, [id]);
  forwards.push(tr);
}
const ids = tokens.map((t) => t.id);
console.log(`hero (${HERO_STAGE}): ${JSON.stringify(text(ids))}`);

// walkthrough token: the input of the last decode forward
const last = forwards.at(-1);
const wPos = last.contextLen - 1;
const logits = engine.logits(0); // the distribution that produced ids[wPos + 1]
const resid = engine.residual_stream(Uint32Array.from(ids.slice(0, wPos + 1)), wPos);

// logits summary: top 256 exactly, the rest as a histogram (enough to compute
// the softmax normaliser at any temperature)
const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
const TOPN = 256;
const topLogits = order.slice(0, TOPN).map((i) => [i, +logits[i].toFixed(3)]);
const tailVals = order.slice(TOPN).map((i) => logits[i]);
const tMin = Math.min(...tailVals), tMax = Math.max(...tailVals), BINS = 96;
const hist = new Array(BINS).fill(0);
for (const v of tailVals) hist[Math.min(BINS - 1, Math.floor(((v - tMin) / (tMax - tMin)) * BINS))]++;
const logitSummary = { top: topLogits, tail: { min: +tMin.toFixed(3), max: +tMax.toFixed(3), counts: hist } };

// residual stream, int8 per depth with its own scale
const C = config.nEmbd, D = config.nLayer + 1;
const residQ = [];
for (let s = 0; s < D; s++) {
  const row = resid.subarray(s * C, (s + 1) * C);
  const m = Math.max(...row.map(Math.abs));
  const q = Int8Array.from(row, (v) => Math.round((v / m) * 127));
  residQ.push({ scale: m / 127, b64: Buffer.from(q.buffer).toString("base64") });
}

// logit lens over the last 24 positions of the run (includes wPos)
const LK = 5;
const lensPos = Array.from({ length: Math.min(24, ids.length) }, (_, i) => ids.length - Math.min(24, ids.length) + i);
const tl = performance.now();
const lensRaw = engine.logit_lens(Uint32Array.from(ids), Uint32Array.from(lensPos), LK);
const lensMs = performance.now() - tl;
const LS = 2 * LK + 4;
const lens = { positions: lensPos, k: LK, ms: lensMs, rows: [] };
for (let s = 0; s < D; s++)
  for (let p = 0; p < lensPos.length; p++) {
    const o = (s * lensPos.length + p) * LS;
    const r = lensRaw.subarray(o, o + LS);
    lens.rows.push([...Array.from(r.subarray(0, LK)), ...Array.from(r.subarray(LK)).map((v) => +v.toPrecision(4))]);
  }
console.log(`lens: ${lensPos.length} positions in ${lensMs.toFixed(0)} ms`);

// attention maps over the first 40 tokens, u8 on a log scale (0 = below 1e-4)
const AT = Math.min(40, ids.length);
const ta = performance.now();
const maps = engine.attention_maps(Uint32Array.from(ids.slice(0, AT)));
const attnMs = performance.now() - ta;
const heads = config.nLayer * config.nHead;
const tri = (AT * (AT + 1)) / 2;
const att = new Uint8Array(heads * tri);
for (let h = 0, o = 0; h < heads; h++)
  for (let i = 0; i < AT; i++)
    for (let j = 0; j <= i; j++, o++) {
      const p = maps[h * AT * AT + i * AT + j];
      att[o] = p < 1e-4 ? 0 : 1 + Math.round(((Math.log10(p) + 4) / 4) * 254);
    }

// ---- ladder + race -------------------------------------------------------------------
const bench = { ladder: {}, race: null };
if (!QUICK) {
  engine.set_capture(0, -1, -1);
  const benchText = "The history of computing is a story of making machines do arithmetic faster. Early computers filled rooms and performed a few thousand operations per second. ";
  let bIds = Array.from(engine.encode(benchText));
  while (bIds.length < 64) bIds = bIds.concat(bIds);
  bIds = bIds.slice(0, 64);
  // Same shape as the page's Ladder (64-token prompt, greedy, median of the
  // decode steps). Stages are measured round-robin, ROUNDS times, keeping each
  // stage's best round: on a shared machine a busy neighbour only ever makes a
  // round slower, and interleaving spreads that over every stage instead of
  // whichever one happened to run during it.
  const ROUNDS = Number(process.env.ROUNDS ?? 4);
  const best = {};
  for (let round = 0; round < ROUNDS; round++) {
    for (const st of stages) {
      if (st.name === "naive" && round > 0) continue;
      const gen = st.name === "naive" ? 4 : 16;
      const slot = engine.open(st.name, 64 + gen + 1);
      for (let i = 0; i < 3; i++) fwd(slot, [bIds[i]]);
      engine.reset(slot);
      const pre = fwd(slot, bIds);
      const ms = [];
      for (let i = 0; i < gen; i++) ms.push(fwd(slot, [engine.sample(slot, 0, 0)]).ms);
      engine.close(slot);
      const b = (best[st.name] ??= { m: Infinity, pre: Infinity, gen });
      b.m = Math.min(b.m, median(ms));
      b.pre = Math.min(b.pre, pre.ms);
    }
  }
  for (const st of stages) {
    const { m, pre, gen } = best[st.name];
    bench.ladder[st.name] = { tokS: 1000 / m, msPerTok: m, prefillTokS: (64 / pre) * 1000, promptLen: 64, genTokens: gen, rounds: ROUNDS };
    console.log(`  ${st.name.padEnd(10)} ${(1000 / m).toFixed(1)} tok/s decode · ${((64 / pre) * 1000).toFixed(0)} tok/s prefill`);
  }
  // race: every stage on the same prompt, greedy, up to 40 tokens or 6 s of its own engine time
  const RACE_PROMPT = "The fastest way to read a book is";
  const rp = Array.from(engine.encode(RACE_PROMPT));
  const lanes = [];
  // each lane raced twice; keep the faster run (same tokens, less contention)
  for (const st of stages) {
    let keep = null;
    for (let tryN = 0; tryN < 2; tryN++) {
      const slot = engine.open(st.name, rp.length + 41);
      fwd(slot, rp);
      const lane = { stage: st.name, tokens: [], ms: [] };
      let used = 0;
      while (lane.tokens.length < 40 && used < 6000) {
        const id = engine.sample(slot, 0, 0);
        const f = fwd(slot, [id]);
        lane.tokens.push(id);
        lane.ms.push(+f.ms.toFixed(3));
        used += f.ms;
      }
      engine.close(slot);
      lane.used = used;
      if (!keep || lane.tokens.length > keep.tokens.length || (lane.tokens.length === keep.tokens.length && used < keep.used)) keep = lane;
      if (st.name === "naive") break;
    }
    delete keep.used;
    lanes.push(keep);
    console.log(`  race ${st.name.padEnd(10)} ${keep.tokens.length} tok in ${keep.ms.reduce((a, b) => a + b, 0).toFixed(0)} ms`);
  }
  bench.race = { prompt: rp, text: RACE_PROMPT, lanes };
} else if (fs.existsSync(path.join(OUT, "bench.json"))) {
  Object.assign(bench, JSON.parse(fs.readFileSync(path.join(OUT, "bench.json"), "utf8")));
}

// ---- pack forwards ---------------------------------------------------------------------
// Op identity (kind, name, layer) is interned in `defs` along with the dims,
// flops and bytes of its first appearance; per forward we store the def
// sequence (deduplicated), the gap before each op and its duration (both in
// 0.1 µs), plus [index, d0, d1, d2, flops, bytes] wherever an op's shape
// differs from its def (attention grows with context; prefill has n rows).
const defs = [], defKey = new Map(), seqs = [], seqKey = new Map();
// seed defs from the last (decode-shaped) forward so decode ops need no overrides
for (const o of forwards.at(-1).ops) {
  const k = `${o.kind}|${o.name}|${o.layer}`;
  if (!defKey.has(k)) {
    defKey.set(k, defs.length);
    defs.push([o.kind, o.name, o.layer, ...o.dims, o.flops, o.bytes]);
  }
}
const packed = forwards.map((f) => {
  const seq = [], g = [], d = [], fb = [];
  let t = 0;
  f.ops.forEach((o, i) => {
    const k = `${o.kind}|${o.name}|${o.layer}`;
    let di = defKey.get(k);
    if (di == null) {
      di = defs.length;
      defKey.set(k, di);
      defs.push([o.kind, o.name, o.layer, ...o.dims, o.flops, o.bytes]);
    }
    seq.push(di);
    const df = defs[di];
    if (df[3] !== o.dims[0] || df[4] !== o.dims[1] || df[5] !== o.dims[2] || df[6] !== o.flops || df[7] !== o.bytes) fb.push(i, ...o.dims, o.flops, o.bytes);
    g.push(Math.max(0, Math.round((o.start - t) * 1e4)));
    d.push(Math.round(o.dur * 1e4));
    t = o.start + o.dur;
  });
  const sk = seq.join(",");
  let si = seqKey.get(sk);
  if (si == null) {
    si = seqs.length;
    seqKey.set(sk, si);
    seqs.push(seq);
  }
  return { ph: f.phase === "prefill" ? 0 : 1, n: [f.nNew, f.nComputed, f.contextLen, f.kvLen, f.kvCap], ms: +f.ms.toFixed(4), h: +f.entropy.toFixed(4), top: f.top.map((x) => [x.id, +x.p.toPrecision(4)]), s: si, g, d, fb };
});

// ---- vocab subset ---------------------------------------------------------------------------
const used = new Set(ids);
for (const t of tokens) t.top?.forEach((x) => used.add(x.id));
for (const f of forwards) f.top.forEach((x) => used.add(x.id));
for (const r of lens.rows) r.slice(0, LK).forEach((id) => used.add(id));
topLogits.forEach(([id]) => used.add(id));
bench.race?.lanes.forEach((l) => l.tokens.forEach((id) => used.add(id)));
bench.race?.prompt.forEach((id) => used.add(id));
const vocab = {};
for (const id of [...used].sort((a, b) => a - b)) vocab[id] = Buffer.from(tokBytes(id)).toString("hex");

// ---- write ------------------------------------------------------------------------------------
fs.mkdirSync(OUT, { recursive: true });
const meta = {
  recordedAt: new Date().toISOString().slice(0, 10),
  machine: os.cpus()[0]?.model ?? "unknown",
  runtime: `Node ${process.versions.node} · V8 ${process.versions.v8}`,
  build: wasm.build_name(),
  stages,
  config,
  bandwidthGBs: bw,
  peakGflops: peak,
  loadMs,
};
const run = {
  stage: HERO_STAGE,
  prompt: PROMPT,
  temperature: 0.8,
  tokens: tokens.map((t) => [t.id, t.prompt ? 1 : 0, t.p == null ? null : +t.p.toPrecision(4), t.top?.map((x) => [x.id, +x.p.toPrecision(4)]) ?? null, t.entropy == null ? null : +t.entropy.toFixed(4)]),
  defs,
  seqs,
  forwards: packed,
  walk: { pos: wPos, forward: forwards.length - 1, logits: logitSummary, resid: residQ },
  lens,
  attention: { T: AT, ms: attnMs, file: "attention.bin" },
  vocab,
};
const w = (f, s) => {
  fs.writeFileSync(path.join(OUT, f), s);
  console.log(`  ${f.padEnd(14)} ${(fs.statSync(path.join(OUT, f)).size / 1024).toFixed(1)} KB`);
};
w("meta.json", JSON.stringify(meta));
w("run.json", JSON.stringify(run));
w("attention.bin", att);
w("bench.json", JSON.stringify(bench));

// native: latest headline run per stage
const nativeFile = path.join(root, "bench/results.jsonl");
if (fs.existsSync(nativeFile)) {
  const rows = fs.readFileSync(nativeFile, "utf8").split("\n").filter(Boolean).flatMap((l) => { try { return [JSON.parse(l)]; } catch { return []; } });
  w("native.jsonl", rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
}
