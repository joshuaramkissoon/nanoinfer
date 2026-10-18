// GPT-2's byte-pair-encoding tokenizer, in ~80 lines of TypeScript, reading the
// same vocab.json + merges.txt the engine loads. It runs without the 274 MB
// weights, so the tokens chapter is live from the first second. It also keeps
// every merge step, which the engine's tokenizer has no reason to expose.

import { WEIGHTS_BASE } from "../engine/client";

/** GPT-2 maps each of the 256 byte values to a printable character so the
 *  vocabulary can be stored as text. A leading space becomes "Ġ". */
function byteTable(): { enc: string[]; dec: Map<string, number> } {
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++)
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n++);
    }
  const enc: string[] = new Array(256);
  const dec = new Map<string, number>();
  bs.forEach((b, i) => {
    enc[b] = String.fromCharCode(cs[i]);
    dec.set(enc[b], b);
  });
  return { enc, dec };
}

// The split GPT-2 does before BPE: words with their leading space, numbers,
// punctuation runs, whitespace.
const PRE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export interface Piece {
  id: number;
  /** the piece as text (spaces kept) */
  text: string;
  /** index of the pre-tokenizer chunk ("word") it came from */
  word: number;
}

export interface Tokenizer {
  vocabSize: number;
  encode(text: string): Piece[];
  /** BPE on one chunk, every intermediate state (first = single bytes, last = tokens) */
  steps(chunk: string): string[][];
  decode(id: number): string;
  /** merge rank of a pair (lower = learned earlier = more common) */
  rank(a: string, b: string): number | undefined;
}

const utf8 = new TextDecoder("utf-8", { fatal: false });
const enc8 = new TextEncoder();

let pending: Promise<Tokenizer> | null = null;

export function loadTokenizer(): Promise<Tokenizer> {
  pending ??= (async () => {
    const [vocab, merges] = await Promise.all([
      fetch(`${WEIGHTS_BASE}/vocab.json`).then((r) => {
        if (!r.ok) throw new Error(`vocab.json: ${r.status}`);
        return r.json() as Promise<Record<string, number>>;
      }),
      fetch(`${WEIGHTS_BASE}/merges.txt`).then((r) => {
        if (!r.ok) throw new Error(`merges.txt: ${r.status}`);
        return r.text();
      }),
    ]);
    const { enc, dec } = byteTable();
    const inv: string[] = [];
    for (const [k, v] of Object.entries(vocab)) inv[v] = k;
    const ranks = new Map<string, number>();
    merges.split("\n").forEach((line, i) => {
      if (!line || line.startsWith("#")) return;
      ranks.set(line, i);
    });
    const toText = (s: string) => utf8.decode(Uint8Array.from([...s].map((c) => dec.get(c) ?? 63)));
    const mapChunk = (chunk: string) => [...enc8.encode(chunk)].map((b) => enc[b]);

    const steps = (chunk: string): string[][] => {
      let parts = mapChunk(chunk);
      const out = [parts];
      while (parts.length > 1) {
        let best = Infinity, at = -1;
        for (let i = 0; i < parts.length - 1; i++) {
          const r = ranks.get(parts[i] + " " + parts[i + 1]);
          if (r != null && r < best) [best, at] = [r, i];
        }
        if (at < 0) break;
        const a = parts[at], b = parts[at + 1];
        const next: string[] = [];
        for (let i = 0; i < parts.length; i++) {
          if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) {
            next.push(a + b);
            i++;
          } else next.push(parts[i]);
        }
        parts = next;
        out.push(parts);
      }
      return out;
    };

    const cache = new Map<string, string[]>();
    return {
      vocabSize: inv.length,
      encode(text: string) {
        const out: Piece[] = [];
        let w = 0;
        for (const m of text.matchAll(PRE)) {
          const chunk = m[0];
          let parts = cache.get(chunk);
          if (!parts) {
            const s = steps(chunk);
            parts = s[s.length - 1];
            if (cache.size < 5000) cache.set(chunk, parts);
          }
          for (const p of parts) out.push({ id: vocab[p] ?? -1, text: toText(p), word: w });
          w++;
        }
        return out;
      },
      steps: (chunk: string) => steps(chunk).map((s) => s.map(toText)),
      decode: (id: number) => (inv[id] != null ? toText(inv[id]) : `#${id}`),
      rank: (a: string, b: string) => ranks.get(mapChunk(a).join("") + " " + mapChunk(b).join("")),
    };
  })();
  pending.catch(() => (pending = null));
  return pending;
}

/** Show whitespace in a token chip. */
export const visible = (s: string) => s.replace(/ /g, "·").replace(/\n/g, "⏎").replace(/\t/g, "⇥");
