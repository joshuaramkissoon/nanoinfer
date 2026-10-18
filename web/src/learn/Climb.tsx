// The climb: one chart that gains a bar per optimisation chapter (part 3), plus
// the "work per token" grid shared by the starting-line and KV-cache chapters.

import { useState } from "react";
import { useWidth } from "../ui/Figure";
import { usePalette } from "../ui/theme";
import { log } from "../lib/fmt";
import { chartColors, Fig, Seg, Stat, Swatch, useFrame, useOnScreen, useLearnData, type LearnData } from "./kit";

export interface Milestone {
  key: string;
  label: string;
  tokS: number;
  what: string;
  /** a different measure (total across many users) */
  total?: boolean;
}

// Decode tokens per second, GPT-2 124M on the M4 Mac mini. The CPU steps are from
// docs/LOG.md; the GPU and batching steps come from the bench files via
// public/learn/data.json (re-measured on a quiet machine), so they never go stale.
const CPU_STEPS: Milestone[] = [
  { key: "naive", label: "first version", tokS: 0.37, what: "recomputes the whole text for every token · 1 core" },
  { key: "kv", label: "+ KV cache", tokS: 18.95, what: "remembers past tokens' keys and values · 1 core" },
  { key: "arith", label: "+ faster arithmetic", tokS: 100.5, what: "16 running totals, SIMD · 1 core" },
  { key: "threads", label: "+ many cores", tokS: 146, what: "6 threads that don't sleep, vectorised small ops" },
  { key: "quant", label: "+ smaller numbers", tokS: 497, what: "4-bit weights, 16-bit KV cache · CPU, 4 threads" },
];

/** The best batched point for 4-bit weights: [users, total tok/s, per-user tok/s]. */
export function bestBatch(d: LearnData): [number, number, number] {
  const q4 = d.batching.series.find((s) => s.label === "q4_0") ?? d.batching.series[0];
  const p = q4.points.reduce((a, b) => (b[1] > a[1] ? b : a));
  return [p[0], p[1], p[2]];
}

export function milestones(d: LearnData | null): Milestone[] {
  const gpu = d?.headline.gpu.decode ?? NaN;
  const [B, total, per] = d ? bestBatch(d) : [256, NaN, NaN];
  return [
    { ...CPU_STEPS[0], tokS: d?.headline.naive.decode ?? CPU_STEPS[0].tokS },
    ...CPU_STEPS.slice(1),
    { key: "gpu", label: "+ the GPU", tokS: gpu, what: "Metal kernels shaped for the GPU, never waiting, 4-bit · M4 GPU · quiet machine" },
    { key: "batch", label: "+ batching", tokS: total, what: `${B} users at once: ${Math.round(per)} tok/s each · M4 GPU`, total: true },
  ];
}

/** "N× faster": best single-user decode over the first version, rounded to two figures. */
export function climbFactor(d: LearnData | null): string | null {
  if (!d) return null;
  const x = d.headline.gpu.decode / d.headline.naive.decode;
  const p = 10 ** Math.max(0, Math.floor(Math.log10(x)) - 1);
  return (Math.round(x / p) * p).toLocaleString("en-GB");
}

const LIMIT_F32 = 243; // 120 GB/s ÷ 494.6 MB of f32 weights per token

/** fig: decode tok/s per milestone, up to and including `upto`. */
export function Climb({ upto, n }: { upto: number; n: string }) {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const d = useLearnData();
  const MILESTONES = milestones(d);
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const W = Math.max(300, w);
  const labelW = Math.min(150, W * 0.36);
  const row = 30, bar = 16, top = 8;
  const H = top + MILESTONES.length * row + 30;
  const x = log(0.1, 10000, labelW + 4, W - (upto >= 6 ? 84 : 44));
  const cur = MILESTONES[upto];
  const prev = upto > 0 ? MILESTONES[upto - 1] : null;
  const shown = hover ?? upto;
  return (
    <Fig
      n={n}
      title="The climb so far"
      prov="measured"
      caption={
        <>
          How fast the engine writes, after each optimisation: tokens per second for one user, on our M4 Mac mini. The scale is
          logarithmic: each gridline is 10× the last, so equal steps mean equal multiples. Grey bars are earlier steps, the
          ember bar is this chapter's. The dashed line is the speed limit for 32-bit weights (chapter 10).
          {upto >= 5 &&
            " The GPU and batching bars were measured on a quiet machine (64-token prompt, then 64 tokens). An earlier GPU run, with other benchmarks competing for memory, gave about three quarters as much."}
          {MILESTONES[upto].total && " The batching bar counts every user's tokens together, so it's allowed past the line: it's a different measure."}
        </>
      }
    >
      <div ref={ref} onMouseLeave={() => setHover(null)}>
        {w > 0 && (upto < 5 || d) && (
          <svg width={W} height={H} role="img" aria-label="decode speed after each optimisation">
            {[0.1, 1, 10, 100, 1000, 10000].map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={top} y2={H - 26} stroke={pal.rule} />
                <text x={x(t)} y={H - 12} textAnchor="middle" fontSize={10} fill={pal.muted}>
                  {t >= 1000 ? `${t / 1000}k` : t}
                </text>
              </g>
            ))}
            <text x={W - 4} y={H - 1} textAnchor="end" fontSize={10} fill={pal.muted}>
              tokens per second →
            </text>
            {MILESTONES.map((m, i) => {
              const y0 = top + i * row + (row - bar) / 2;
              const on = i <= upto;
              const x1 = x(m.tokS);
              const x0 = x(0.1);
              const fill = i === upto ? cc.c1 : pal.muted;
              return (
                <g key={m.key} onMouseEnter={() => on && setHover(i)} style={{ cursor: on ? "default" : undefined }}>
                  <rect x={0} y={top + i * row} width={W} height={row} fill="transparent" />
                  <text x={labelW} y={y0 + bar / 2 + 4} textAnchor="end" fontSize={11.5} fill={on ? (i === upto ? pal.fg : pal.sub) : pal.rule} fontWeight={i === upto ? 700 : 400}>
                    {m.label}
                  </text>
                  {on ? (
                    <>
                      <path
                        d={`M${x0},${y0} H${x1 - 4} a4,4 0 0 1 4,4 V${y0 + bar - 4} a4,4 0 0 1 -4,4 H${x0} Z`}
                        fill={fill}
                        opacity={i === upto || hover === i ? 1 : 0.55}
                      />
                      <text x={x1 + 6} y={y0 + bar / 2 + 4} fontSize={11} fill={i === upto ? pal.fg : pal.sub} fontWeight={i === upto ? 700 : 400}>
                        {m.tokS < 1 ? m.tokS.toFixed(2) : m.tokS < 100 ? m.tokS.toFixed(1) : Math.round(m.tokS).toLocaleString()}
                        {m.total ? " total" : ""}
                      </text>
                    </>
                  ) : (
                    <line x1={x0} x2={x0 + 40} y1={y0 + bar / 2} y2={y0 + bar / 2} stroke={pal.rule} />
                  )}
                </g>
              );
            })}
            <line x1={x(LIMIT_F32)} x2={x(LIMIT_F32)} y1={top - 4} y2={H - 26} stroke={pal.fg} strokeWidth={1} strokeDasharray="4 3" opacity={0.7} />
            <text x={x(LIMIT_F32) - 4} y={top + 4} textAnchor="end" fontSize={10} fill={pal.sub}>
              limit, 32-bit: 243
            </text>
          </svg>
        )}
      </div>
      <div className="lr-note">
        <b>{MILESTONES[shown].label.replace(/^\+ /, "")}</b>: {MILESTONES[shown].what}
        {shown === upto && prev && (
          <>
            {" "}
            · <span className="nv-accent">{(cur.tokS / prev.tokS).toFixed(cur.tokS / prev.tokS < 10 ? 1 : 0)}× the step before</span>
          </>
        )}
      </div>
    </Fig>
  );
}

/** Work per token: which rows of the text get pushed through the model at each step, and (with a cache) which are
 *  read back from the KV cache instead. Shared by chapter 11 (no cache) and chapter 12 (toggle). */
export function WorkGrid({ n, mode: fixed }: { n: string; mode?: "naive" | "cache" }) {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [mode, setMode] = useState<"naive" | "cache">(fixed ?? "cache");
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const P = 6, N = 14; // prompt rows, generated tokens
  const [onScreen, vis] = useOnScreen<HTMLElement>();
  useFrame(vis && playing, (dt) => setT((v) => (v + dt * 1.6) % (N + 4)));
  const step = Math.min(N + 1, Math.floor(t) + 1); // passes shown (1 = the prefill pass)
  const [ref, w] = useWidth<HTMLDivElement>();
  const W = Math.max(300, w);
  const cols = N + 1; // prefill + N decode steps
  const left = W < 480 ? 58 : 76;
  const cs = Math.max(8, Math.min(26, Math.floor((W - left - 8) / cols) - 3));
  const rows = P + N;
  const rh = Math.max(7, Math.min(11, Math.round(cs * 0.5)));
  const gy = 4;
  const H = gy + rows * (rh + 2) + 22;
  const cache = mode === "cache";
  // per pass s: rows that exist = P + s; computed = all of them (no cache) or just the new one(s)
  const computed = (s: number) => (s === 0 ? P : cache ? 1 : P + s);
  let total = 0, naiveTotal = 0;
  for (let s = 0; s < step; s++) {
    total += computed(s);
    naiveTotal += s === 0 ? P : P + s;
  }
  const last = step - 1;
  const cached = cache ? P + last : 0; // tokens whose keys and values are in the cache after this pass
  const perTok = 2 * 12 * 768 * 4; // K and V, 12 layers, 768 numbers, f32 = 73.7 KB
  const kb = (cached * perTok) / 1000;
  const xOf = (s: number) => left + s * (cs + 3);
  const yOf = (r: number) => gy + r * (rh + 2);
  return (
    <Fig
      figRef={onScreen}
      n={n}
      title={cache ? "With a KV cache, each pass computes one new token" : "Without a cache, every pass redoes the whole text"}
      prov="toy"
      caption={
        <>
          Columns are passes through the model, left to right in time; rows are the tokens of the text, the first {P} typed by a
          person. <Swatch c={cc.c1}>Ember</Swatch>: pushed through all 12 layers in that pass.{" "}
          {cache ? (
            <>
              <Swatch c={cc.c2}>Blue</Swatch>: read back from the KV cache instead (each token's saved keys and values). The
              strip underneath is the cache itself: one page per token, growing by one every pass.
            </>
          ) : (
            <>Nothing is remembered between passes, so every column is taller than the last.</>
          )}
        </>
      }
      tryit={
        fixed ? (
          <>Watch the total under the grid: the work grows with the square of the text.</>
        ) : (
          <>Switch to no cache and compare the totals; switch back and watch the cache grow by one page per pass.</>
        )
      }
    >
      <div className="lr-controls">
        {!fixed && (
          <Seg
            options={[
              { v: "naive", label: "no cache" },
              { v: "cache", label: "KV cache" },
            ]}
            value={mode}
            onChange={setMode}
            label="mode"
          />
        )}
        <button className="lr-btn" onClick={() => setPlaying((p) => !p)}>
          {playing ? "❚❚ pause" : "▶ play"}
        </button>
      </div>
      <div className="p0-legend lr-small">
        <span>
          <i style={{ background: cc.c1 }} />
          computed this pass
        </span>
        {cache && (
          <span>
            <i style={{ background: cc.c2, opacity: 0.55 }} />
            read from the KV cache
          </span>
        )}
      </div>
      <div ref={ref}>
        {w > 0 && (
          <svg width={W} height={H} role="img" aria-label={cache ? "each pass computes one token and reads the rest from the cache" : "each pass recomputes every token"}>
            {/* row labels: a bracket for the prompt, then the generated tokens */}
            <path d={`M${left - 10},${yOf(0)} H${left - 6} M${left - 10},${yOf(0)} V${yOf(P) - 2} H${left - 6}`} stroke={pal.muted} fill="none" />
            <text x={left - 14} y={(yOf(0) + yOf(P)) / 2 + 3} textAnchor="end" fontSize={11} fill={pal.sub}>
              prompt
            </text>
            <text x={left - 14} y={(yOf(P) + yOf(P + Math.min(last, N) + 1)) / 2 + 3} textAnchor="end" fontSize={11} fill={pal.sub} opacity={last > 1 ? 1 : 0}>
              new
            </text>
            {Array.from({ length: step }, (_, s) => {
              const first = s === 0 || !cache ? 0 : P + s - 1;
              const now = s === last;
              return (
                <g key={s} opacity={now ? 1 : 0.62}>
                  {Array.from({ length: P + s }, (_, r) => {
                    const lit = r >= first;
                    return <rect key={r} x={xOf(s)} y={yOf(r)} width={cs} height={rh} rx={2} fill={lit ? cc.c1 : cc.c2} fillOpacity={lit ? 1 : 0.42} />;
                  })}
                </g>
              );
            })}
            <text x={xOf(0)} y={H - 6} fontSize={11} fill={pal.muted}>
              passes, in time →
            </text>
          </svg>
        )}
      </div>
      {cache && (
        <div className="lr-kvstrip" aria-label={`KV cache: ${cached} tokens`}>
          <span className="k">KV cache</span>
          <span className="pages">
            {Array.from({ length: P + N }, (_, i) => (
              <i key={i} style={{ background: i < cached ? cc.c2 : "transparent", borderColor: i < cached ? cc.c2 : pal.rule }} />
            ))}
          </span>
          <span className="v">
            {cached} tokens · {kb >= 1000 ? `${(kb / 1000).toFixed(2)} MB` : `${Math.round(kb)} KB`}
          </span>
        </div>
      )}
      <div className="lr-stats">
        <Stat k="this pass computed" v={computed(last)} unit={computed(last) === 1 ? "token" : "tokens"} accent />
        <Stat k="total so far" v={total} unit={cache ? `(no cache: ${naiveTotal})` : "tokens"} />
        <Stat k={cache ? "each token adds" : "growth"} v={cache ? "73.7" : "n²"} unit={cache ? "KB to the cache" : "work for n tokens"} />
      </div>
    </Fig>
  );
}
