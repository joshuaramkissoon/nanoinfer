// Native results from the hillclimb (bench/results.jsonl, one JSON object per
// line, appended by `nanoinfer bench`). Tolerant of missing fields so the
// format can evolve.

import { useEffect, useState } from "react";

export interface NativeResult {
  stage: string;
  kernels?: string;
  decode_tok_s: number;
  decode_tok_s_std?: number;
  prefill_tok_s?: number;
  decode_ceiling_tok_s?: number;
  pct_of_ceiling?: number;
  bandwidth_gbs?: number;
  dtype?: string;
  threads?: number;
  prompt_len?: number;
  gen_tokens?: number;
  machine?: string;
  git_sha?: string;
  timestamp?: string | number;
  note?: string;
  traced?: boolean;
}

export const NATIVE_URL = (import.meta.env.VITE_BENCH_URL as string | undefined) ?? "/bench/results.jsonl";

let cache: Promise<NativeResult[]> | null = null;

export function loadNative(): Promise<NativeResult[]> {
  cache ??= fetch(NATIVE_URL)
    .then((r) => (r.ok ? r.text() : ""))
    .then((t) =>
      t
        .split("\n")
        .map((l) => l.trim())
        .filter(Boolean)
        .flatMap((l) => {
          try {
            const o = JSON.parse(l);
            const tok = Number(o.decode_tok_s ?? o.tok_s);
            return o.stage && isFinite(tok) ? [{ ...o, decode_tok_s: tok } as NativeResult] : [];
          } catch {
            return [];
          }
        }),
    )
    .catch(() => []);
  return cache;
}

export function useNative(): NativeResult[] | null {
  const [r, setR] = useState<NativeResult[] | null>(null);
  useEffect(() => {
    let live = true;
    loadNative().then((x) => live && setR(x));
    return () => {
      live = false;
    };
  }, []);
  return r;
}

/** The headline result per stage: latest untraced run at the default bench
 *  shape (64-token prompt), falling back to the latest run of any shape. */
export function latestByStage(rs: NativeResult[]): Map<string, NativeResult> {
  const m = new Map<string, NativeResult>();
  const rank = (r: NativeResult) => (r.traced ? 0 : 2) + ((r.prompt_len ?? 64) === 64 ? 1 : 0);
  for (const r of rs) {
    const cur = m.get(r.stage);
    if (!cur || rank(r) >= rank(cur)) m.set(r.stage, r);
  }
  return m;
}

/** Native M4 reference bandwidth, used when results.jsonl doesn't say. */
export const M4_GBS = 120;
