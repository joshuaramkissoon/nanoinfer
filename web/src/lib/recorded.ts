// The recorded fallback: real engine output captured by web/scripts/record.mjs
// (the same wasm, run in Node/V8 on the M4) and committed under
// public/recorded/. Every figure can render from it before the reader opts in
// to the 274 MB download; going live replaces it piece by piece.

import type { ForwardTrace, ModelConfig, OpRec, StageInfo } from "../engine/types";
import type { Run, Tok } from "../engine/client";

export interface LensData {
  /** positions (indices into the token list) the lens was computed for */
  positions: number[];
  tokens: number[];
  k: number;
  ms: number;
  /** rows[depth][pi] */
  rows: LensCell[][];
}
export interface LensCell {
  top: { id: number; p: number }[];
  entropy: number;
  pFinal: number;
  /** probability of the token that actually came next (-1 for the last position) */
  pNext: number;
  norm: number;
}

/** Logits as the walkthrough's sampling explorer needs them: the top N exactly,
 *  the rest as a histogram (enough for the softmax normaliser at any temperature). */
export interface LogitSummary {
  top: [number, number][];
  tail: { min: number; max: number; counts: number[] };
}

export interface WalkData {
  /** position of the walkthrough token (the input of the forward) */
  pos: number;
  forward: ForwardTrace;
  logits: LogitSummary;
  /** residual stream [depth][C] */
  resid: Float32Array[];
  /** logit lens for this one position, per depth */
  lens: LensCell[];
}

export interface RaceLane {
  stage: string;
  tokens: number[];
  ms: number[];
}

export interface Recorded {
  meta: {
    recordedAt: string;
    machine: string;
    runtime: string;
    build: string;
    stages: StageInfo[];
    config: ModelConfig;
    bandwidthGBs: number;
    peakGflops: number;
  };
  run: Run;
  attention: { tokens: number[]; maps: Float32Array; ms: number };
  lens: LensData;
  walk: WalkData;
  ladder: Record<string, { tokS: number; msPerTok: number; prefillTokS: number; promptLen: number; genTokens: number }>;
  race: { prompt: number[]; text: string; lanes: RaceLane[] } | null;
  vocab: Map<number, Uint8Array>;
}

const BASE = `${import.meta.env.BASE_URL}recorded/`;

const hex = (s: string) => {
  const out = new Uint8Array(s.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(s.slice(i * 2, i * 2 + 2), 16);
  return out;
};

function lensCell(r: number[], k: number): LensCell {
  const top = [];
  for (let j = 0; j < k; j++) top.push({ id: r[j], p: r[k + j] });
  return { top, entropy: r[2 * k], pFinal: r[2 * k + 1], pNext: r[2 * k + 2], norm: r[2 * k + 3] };
}

/** Unpack the flat lens array that `Engine::logit_lens` returns. */
export function unpackLens(flat: Float32Array, tokens: number[], positions: number[], k: number, depth: number, ms: number): LensData {
  const S = 2 * k + 4;
  const rows: LensCell[][] = [];
  for (let s = 0; s < depth; s++) {
    const row: LensCell[] = [];
    for (let p = 0; p < positions.length; p++) row.push(lensCell(Array.from(flat.subarray((s * positions.length + p) * S, (s * positions.length + p + 1) * S)), k));
    rows.push(row);
  }
  return { positions, tokens, k, ms, rows };
}

/** Summarise full logits the same way the recorder does. */
export function summariseLogits(logits: Float32Array, topN = 256, bins = 96): LogitSummary {
  const idx = Array.from(logits.keys());
  idx.sort((a, b) => logits[b] - logits[a]);
  const top = idx.slice(0, topN).map((i) => [i, logits[i]] as [number, number]);
  let min = Infinity, max = -Infinity;
  for (const i of idx.slice(topN)) {
    min = Math.min(min, logits[i]);
    max = Math.max(max, logits[i]);
  }
  const counts = new Array(bins).fill(0);
  for (const i of idx.slice(topN)) counts[Math.min(bins - 1, Math.floor(((logits[i] - min) / (max - min || 1)) * bins))]++;
  return { top, tail: { min, max, counts } };
}

/** Softmax over a LogitSummary at temperature t (t = 0 means argmax). Returns
 *  the top tokens' probabilities and the total mass left in the tail. */
export function softmaxAt(s: LogitSummary, t: number): { probs: { id: number; p: number; logit: number }[]; tailMass: number } {
  if (t <= 0) {
    return { probs: s.top.map(([id, l], i) => ({ id, p: i === 0 ? 1 : 0, logit: l })), tailMass: 0 };
  }
  const max = s.top[0][1];
  let z = 0;
  const ex = s.top.map(([, l]) => Math.exp((l - max) / t));
  for (const e of ex) z += e;
  const { min, max: tmax, counts } = s.tail;
  const w = (tmax - min) / counts.length;
  let tail = 0;
  counts.forEach((n, i) => (tail += n * Math.exp((min + (i + 0.5) * w - max) / t)));
  z += tail;
  return { probs: s.top.map(([id, l], i) => ({ id, p: ex[i] / z, logit: l })), tailMass: tail / z };
}

type PackedFwd = { ph: number; n: number[]; ms: number; h: number; top: [number, number][]; s: number; g: number[]; d: number[]; fb: number[] };

function unpackForwards(defs: (string | number)[][], seqs: number[][], fwds: PackedFwd[]): ForwardTrace[] {
  return fwds.map((f) => {
    const seq = seqs[f.s];
    const ops: OpRec[] = [];
    let t = 0;
    seq.forEach((di, i) => {
      const d = defs[di];
      const start = t + f.g[i] / 1e4;
      const dur = f.d[i] / 1e4;
      ops.push({
        kind: d[0] as string,
        name: d[1] as string,
        layer: d[2] as number,
        dims: [d[3] as number, d[4] as number, d[5] as number],
        flops: d[6] as number,
        bytes: d[7] as number,
        start,
        dur,
      });
      t = start + dur;
    });
    for (let j = 0; j < f.fb.length; j += 6) {
      const o = ops[f.fb[j]];
      o.dims = [f.fb[j + 1], f.fb[j + 2], f.fb[j + 3]];
      o.flops = f.fb[j + 4];
      o.bytes = f.fb[j + 5];
    }
    return {
      phase: f.ph === 0 ? "prefill" : "decode",
      nNew: f.n[0],
      nComputed: f.n[1],
      contextLen: f.n[2],
      kvLen: f.n[3],
      kvCap: f.n[4],
      ms: f.ms,
      entropy: f.h,
      top: f.top.map(([id, p]) => ({ id, p })),
      attn: null,
      attnLayer: -1,
      attnHead: -1,
      ops,
    };
  });
}

let pending: Promise<Recorded | null> | null = null;

export function loadRecorded(): Promise<Recorded | null> {
  pending ??= (async () => {
    try {
      const [meta, run, bench, attBuf] = await Promise.all([
        fetch(`${BASE}meta.json`).then((r) => r.json()),
        fetch(`${BASE}run.json`).then((r) => r.json()),
        fetch(`${BASE}bench.json`).then((r) => (r.ok ? r.json() : { ladder: {}, race: null })),
        fetch(`${BASE}attention.bin`).then((r) => r.arrayBuffer()),
      ]);
      const cfg = meta.config as ModelConfig;
      const tokens: Tok[] = run.tokens.map((t: [number, number, number | null, [number, number][] | null, number | null]) => ({
        id: t[0],
        prompt: !!t[1],
        p: t[2],
        top: t[3]?.map(([id, p]) => ({ id, p })) ?? null,
        entropy: t[4],
      }));
      const forwards = unpackForwards(run.defs, run.seqs, run.forwards);
      const ids = tokens.map((t) => t.id);

      // attention: u8 log-quantised lower triangles -> dense [heads, T, T]
      const T = run.attention.T as number;
      const heads = cfg.nLayer * cfg.nHead;
      const q = new Uint8Array(attBuf);
      const maps = new Float32Array(heads * T * T);
      for (let h = 0, o = 0; h < heads; h++)
        for (let i = 0; i < T; i++)
          for (let j = 0; j <= i; j++, o++) maps[h * T * T + i * T + j] = q[o] === 0 ? 0 : 10 ** (((q[o] - 1) / 254) * 4 - 4);

      const L = run.lens;
      const depth = cfg.nLayer + 1;
      const lensRows: LensCell[][] = [];
      for (let s = 0; s < depth; s++) lensRows.push(L.positions.map((_: number, p: number) => lensCell(L.rows[s * L.positions.length + p], L.k)));
      const lens: LensData = { positions: L.positions, tokens: ids, k: L.k, ms: L.ms, rows: lensRows };

      const wp = L.positions.indexOf(run.walk.pos);
      const resid = (run.walk.resid as { scale: number; b64: string }[]).map(({ scale, b64 }) => {
        const bin = atob(b64);
        const out = new Float32Array(bin.length);
        for (let i = 0; i < bin.length; i++) out[i] = ((bin.charCodeAt(i) << 24) >> 24) * scale;
        return out;
      });
      const walk: WalkData = {
        pos: run.walk.pos,
        forward: forwards[run.walk.forward],
        logits: run.walk.logits,
        resid,
        lens: wp >= 0 ? lensRows.map((r) => r[wp]) : [],
      };

      const vocab = new Map<number, Uint8Array>();
      for (const [id, h] of Object.entries(run.vocab as Record<string, string>)) vocab.set(+id, hex(h));

      const r: Recorded = {
        meta,
        run: { stage: run.stage, tokens, forwards, done: true, t0: 0, t1: 0 },
        attention: { tokens: ids.slice(0, T), maps, ms: run.attention.ms },
        lens,
        walk,
        ladder: bench.ladder ?? {},
        race: bench.race ?? null,
        vocab,
      };
      return r;
    } catch (e) {
      console.warn("no recorded data", e);
      return null;
    }
  })();
  return pending;
}
