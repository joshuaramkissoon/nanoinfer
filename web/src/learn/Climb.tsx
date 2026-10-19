// The climb: one chart that gains a bar per optimisation chapter (part 3), plus
// the "work per token" grid shared by the starting-line and KV-cache chapters.

import { useState } from "react";
import { useWidth } from "../ui/Figure";
import { usePalette } from "../ui/theme";
import { log } from "../lib/fmt";
import { chartColors, Fig, Seg, Stat, Swatch, useFrame, useOnScreen, useLearnData, type LearnData, T, PlayPause, usePlaying } from "./kit";

export interface Milestone {
  key: string;
  label: string;
  tokS: number;
  what: string;
  /** a different measure (total across many users) */
  total?: boolean;
  /** MB of weights read per decoded token by this step's format */
  mb?: number;
}

/** Apple's spec read bandwidth for the M4 (GB/s). Every step's limit uses it, so the percentages compare like with like. */
const SPEC_GBS = 120;
/** This step's own speed limit: spec bandwidth ÷ its bytes per token (chapter 10). */
export const limitOf = (m: Milestone) => (m.mb ? (SPEC_GBS * 1000) / m.mb : null);

// Decode tokens per second, GPT-2 124M on the M4 Mac mini. The CPU steps are from
// docs/LOG.md; the GPU and batching steps come from the bench files via
// public/learn/data.json (re-measured on a quiet machine), so they never go stale.
// Bytes per token (docs/LOG.md, "every stage back to back" and the Metal Q4 table): f32 494.6 MB; CPU q4-kv16 97 MB
// (4-bit blocks with an f32 scale, 8-bit scoring matrix); GPU metal-lanes 89.3 MB (Q4_0 blocks, Q8_0 scoring matrix).
const CPU_STEPS: Milestone[] = [
  { key: "naive", label: "first version", tokS: 0.37, mb: 494.6, what: "recomputes the whole text for every token · 1 core · 32-bit" },
  { key: "kv", label: "+ KV cache", tokS: 18.95, mb: 494.6, what: "remembers past tokens' keys and values · 1 core · 32-bit" },
  { key: "arith", label: "+ faster arithmetic", tokS: 100.5, mb: 494.6, what: "16 running totals, SIMD · 1 core · 32-bit" },
  { key: "threads", label: "+ many cores", tokS: 146, mb: 494.6, what: "6 threads that don't sleep, vectorised small ops · 32-bit" },
  { key: "quant", label: "+ smaller numbers", tokS: 497, mb: 97, what: "4-bit weights, 16-bit KV cache · CPU, 4 threads" },
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
    { key: "gpu", label: "+ the GPU", tokS: gpu, mb: 89.3, what: "Metal kernels shaped for the GPU, never waiting, 4-bit · M4 GPU · quiet machine" },
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

const fmtTok = (v: number) => (v < 1 ? v.toFixed(2) : v < 100 ? v.toFixed(1) : Math.round(v).toLocaleString("en-GB"));

/** fig: decode tok/s per milestone, up to and including `upto`. A lollipop chart on a log axis: a grey dot per earlier
 *  step, an ember dot for this chapter's, a short fg tick at each step's own speed limit (spec bandwidth ÷ its bytes
 *  per token), and the batching total as a hollow ring because it is a different measure (all users together). */
export function Climb({ upto, n }: { upto: number; n: string }) {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const d = useLearnData();
  const MILESTONES = milestones(d);
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const W = Math.max(300, w);
  const narrow = W < 520;
  const labelW = narrow ? 116 : 150;
  const valW = narrow ? 92 : 112;
  const row = 30, top = 22;
  const H = top + MILESTONES.length * row + 34;
  const x = log(0.1, 10000, labelW + 10, W - valW - 10);
  const cur = MILESTONES[upto];
  const prev = upto > 0 ? MILESTONES[upto - 1] : null;
  const shown = hover ?? upto;
  const ticks = [0.1, 1, 10, 100, 1000, 10000];
  return (
    <Fig
      n={n}
      title="The climb so far"
      prov="measured"
      alt={`Decode speed after each optimisation, up to ${cur.label.replace(/^\+ /, "")}: ${fmtTok(cur.tokS)} tokens per second.`}
      caption={
        <>
          How fast the engine writes after each optimisation: tokens per second for one user on our M4 Mac mini, on a log
          scale (each gridline is 10×). Grey dots are earlier steps, the <Swatch c={cc.c1}>ember</Swatch> dot is this chapter's.
          The short tick on each row is that step's own speed limit (chapter 10): {SPEC_GBS} GB/s ÷ the bytes it reads per
          token; the percentage is how much of it the step reached.
          {upto >= 4 && " Smaller numbers mean fewer bytes, so from chapter 15 on the limit itself moves right."}
          {upto >= 5 &&
            " The GPU and batching steps were measured on a quiet machine (64-token prompt, then 64 tokens). An earlier GPU run, with other benchmarks competing for memory, gave about three quarters as much."}
          {MILESTONES[upto].total && " The batching ring counts every user's tokens together, so a one-user limit doesn't apply to it."}
        </>
      }
    >
      <div ref={ref} onMouseLeave={() => setHover(null)}>
        {w > 0 && (upto < 5 || d) && (
          <svg width={W} height={H}>
            <text x={labelW + 10} y={11} fontSize={T.xs} fill={pal.muted}>
              {narrow ? "log scale" : "tokens per second, one user · log scale"}
            </text>
            <text x={W} y={11} textAnchor="end" fontSize={T.xs} fill={pal.muted}>
              tok/s · % of limit
            </text>
            {ticks.map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={top - 4} y2={H - 28} stroke={pal.rule} />
                <text x={x(t)} y={H - 12} textAnchor="middle" fontSize={T.xs} fill={pal.muted}>
                  {t >= 1000 ? `${t / 1000}k` : t}
                </text>
              </g>
            ))}
            {MILESTONES.map((m, i) => {
              const cy = top + i * row + row / 2;
              const on = i <= upto;
              const now = i === upto;
              const x1 = x(m.tokS);
              const lim = limitOf(m);
              const pct = lim ? (m.tokS / lim) * 100 : null;
              const mark = now ? cc.c1 : pal.muted;
              const hot = now || hover === i;
              return (
                <g key={m.key} onMouseEnter={() => on && setHover(i)} onClick={() => on && setHover(i)}>
                  <rect x={0} y={cy - row / 2} width={W} height={row} fill="transparent" />
                  <text x={labelW} y={cy + 4} textAnchor="end" fontSize={narrow ? T.xs : T.sm} fill={on ? (now ? pal.fg : pal.sub) : pal.muted} fontWeight={now ? 700 : 400} opacity={on ? 1 : 0.6}>
                    {narrow ? m.label.replace(/^\+ /, "") : m.label}
                  </text>
                  {on ? (
                    <>
                      <line x1={x(0.1)} x2={x1} y1={cy} y2={cy} stroke={mark} strokeWidth={now ? 2 : 1.5} opacity={now ? 1 : hot ? 0.9 : 0.55} />
                      {lim && <line x1={x(lim)} x2={x(lim)} y1={cy - 8} y2={cy + 8} stroke={pal.fg} strokeWidth={1.5} opacity={0.8} />}
                      {m.total ? (
                        <circle cx={x1} cy={cy} r={5.5} fill={pal.bg} stroke={mark} strokeWidth={2} />
                      ) : (
                        <circle cx={x1} cy={cy} r={5} fill={mark} stroke={pal.bg} strokeWidth={2} />
                      )}
                      <text x={W} y={cy + 4} textAnchor="end" fontSize={narrow ? T.xs : T.sm} fill={now ? pal.fg : pal.sub} fontWeight={now ? 700 : 400}>
                        {fmtTok(m.tokS)}
                        {m.total ? " total" : ""}
                        {pct != null && (
                          <tspan fontWeight={400} fill={pal.muted}>
                            {` · ${pct < 1 ? pct.toFixed(1) : pct.toFixed(0)}%`}
                          </tspan>
                        )}
                      </text>
                    </>
                  ) : (
                    <line x1={x(0.1)} x2={x(0.1) + 32} y1={cy} y2={cy} stroke={pal.rule} />
                  )}
                </g>
              );
            })}
          </svg>
        )}
      </div>
      <div className="lr-note">
        <b>{MILESTONES[shown].label.replace(/^\+ /, "")}</b>: {fmtTok(MILESTONES[shown].tokS)} tok/s{MILESTONES[shown].total ? " in total" : ""} · {MILESTONES[shown].what}
        {limitOf(MILESTONES[shown]) && ` · limit ${Math.round(limitOf(MILESTONES[shown])!).toLocaleString("en-GB")} tok/s (${MILESTONES[shown].mb} MB per token)`}
        {shown === upto && prev && (
          <>
            {" "}
            · {(cur.tokS / prev.tokS).toFixed(cur.tokS / prev.tokS < 10 ? 1 : 0)}× the step before
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
  const [playing, setPlaying] = usePlaying();
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
        <PlayPause playing={playing} onToggle={() => setPlaying((p) => !p)} />
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
            <text x={left - 14} y={(yOf(0) + yOf(P)) / 2 + 3} textAnchor="end" fontSize={T.xs} fill={pal.sub}>
              prompt
            </text>
            <text x={left - 14} y={(yOf(P) + yOf(P + Math.min(last, N) + 1)) / 2 + 3} textAnchor="end" fontSize={T.xs} fill={pal.sub} opacity={last > 1 ? 1 : 0}>
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
            <text x={xOf(0)} y={H - 6} fontSize={T.xs} fill={pal.muted}>
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
