// The climb: one chart that gains a bar per optimisation chapter (part 3), plus
// the "work per token" grid shared by the starting-line and KV-cache chapters.

import { useState } from "react";
import { useWidth } from "../ui/Figure";
import { usePalette } from "../ui/theme";
import { log } from "../lib/fmt";
import { chartColors, Fig, Seg, useFrame } from "./kit";

export interface Milestone {
  key: string;
  label: string;
  tokS: number;
  what: string;
  /** a different measure (total across many users) */
  total?: boolean;
}

// Decode tokens per second, GPT-2 124M on the M4 Mac mini (docs/LOG.md, docs/WRITEUP.md, bench/batching.json).
export const MILESTONES: Milestone[] = [
  { key: "naive", label: "first version", tokS: 0.37, what: "recomputes the whole text for every token · 1 core" },
  { key: "kv", label: "+ KV cache", tokS: 18.95, what: "remembers past tokens' keys and values · 1 core" },
  { key: "arith", label: "+ faster arithmetic", tokS: 100.5, what: "16 running totals, SIMD · 1 core" },
  { key: "threads", label: "+ many cores", tokS: 146, what: "6 threads that don't sleep, vectorised small ops" },
  { key: "quant", label: "+ smaller numbers", tokS: 497, what: "4-bit weights, 16-bit KV cache · CPU, 4 threads" },
  { key: "gpu", label: "+ the GPU", tokS: 641, what: "Metal kernels, no waiting, 4-bit · M4 GPU" },
  { key: "batch", label: "+ batching", tokS: 6453, what: "256 users at once: 25 tok/s each · M4 GPU", total: true },
];

const LIMIT_F32 = 243; // 120 GB/s ÷ 494.6 MB of f32 weights per token

/** fig: decode tok/s per milestone, up to and including `upto`. */
export function Climb({ upto, n }: { upto: number; n: string }) {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const W = Math.max(300, w);
  const labelW = Math.min(150, W * 0.36);
  const row = 30, bar = 16, top = 8;
  const H = top + MILESTONES.length * row + 30;
  const x = log(0.1, 10000, labelW + 4, W - 44);
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
          {MILESTONES[upto].total && " The batching bar counts every user's tokens together, so it's allowed past the line: it's a different measure."}
        </>
      }
    >
      <div ref={ref} onMouseLeave={() => setHover(null)}>
        {w > 0 && (
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

/** Work per token: which rows of the text get pushed through the model at each step. */
export function WorkGrid({ n, mode: fixed }: { n: string; mode?: "naive" | "cache" }) {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [mode, setMode] = useState<"naive" | "cache">(fixed ?? "naive");
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const P = 6, N = 18; // prompt rows, generated tokens
  useFrame(playing, (dt) => setT((v) => (v + dt * 2.2) % (N + 4)));
  const step = Math.min(N, Math.floor(t)); // steps done
  const [ref, w] = useWidth<HTMLDivElement>();
  const W = Math.max(300, w);
  const cols = N + 1; // prefill + N decode steps
  const cs = Math.min(18, Math.floor((W - 70) / cols) - 2);
  const rows = P + N;
  const rs = Math.min(10, cs * 0.6);
  let rowsDone = 0;
  for (let s = 0; s < step; s++) rowsDone += s === 0 ? P : mode === "naive" ? P + s : 1;
  let cacheRows = 0, naiveRows = 0;
  for (let s = 0; s < step; s++) {
    cacheRows += s === 0 ? P : 1;
    naiveRows += s === 0 ? P : P + s;
  }
  const H = rows * (rs + 2) + 34;
  return (
    <Fig
      n={n}
      title={mode === "naive" ? "Work per token, without a cache" : "Work per token, with a KV cache"}
      prov="toy"
      caption={
        <>
          Each column is one pass through the model, left to right in time; the first column is the prompt ({P} tokens). Each
          row is one token of the text. A filled cell means that token was pushed through all 12 layers in that pass. The
          number underneath is how many rows that pass computed.{" "}
          {mode === "naive"
            ? "Without a cache every pass recomputes every token, so each column is taller than the last."
            : "With a cache, each pass computes only the newest token: the old tokens' keys and values are already saved."}
        </>
      }
      tryit={fixed ? <>Watch the total. Every new token costs more than the one before it.</> : <>Switch between the two and compare the totals.</>}
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
        <span className="lr-small">
          rows computed: <b style={{ color: pal.fg }}>{rowsDone}</b>
          {!fixed && <> (no cache: {naiveRows} · cache: {cacheRows})</>}
        </span>
      </div>
      <div ref={ref}>
        {w > 0 && (
          <svg width={W} height={H}>
            {Array.from({ length: cols }, (_, s) => {
              const n = s === 0 ? P : mode === "naive" ? P + s : 1;
              const first = s === 0 ? 0 : mode === "naive" ? 0 : P + s - 1;
              const done = s < step;
              const xx = 40 + s * (cs + 2);
              return (
                <g key={s} opacity={done ? 1 : 0.18}>
                  {Array.from({ length: P + s }, (_, r) => {
                    const lit = r >= first && r < first + n;
                    return (
                      <rect
                        key={r}
                        x={xx}
                        y={4 + r * (rs + 2)}
                        width={cs}
                        height={rs}
                        rx={2}
                        fill={lit ? cc.c1 : pal.rule}
                        opacity={lit ? (r === P + s - 1 || s === 0 ? 1 : 0.55) : 0.6}
                      />
                    );
                  })}
                  {done && (
                    <text x={xx + cs / 2} y={H - 14} textAnchor="middle" fontSize={9} fill={pal.muted}>
                      {n}
                    </text>
                  )}
                </g>
              );
            })}
            <text x={34} y={10} textAnchor="end" fontSize={9.5} fill={pal.muted}>
              text
            </text>
            <text x={34} y={4 + P * (rs + 2) - 2} textAnchor="end" fontSize={9} fill={pal.muted}>
              prompt
            </text>
            <text x={W - 2} y={H - 1} textAnchor="end" fontSize={9.5} fill={pal.muted}>
              passes, in time →
            </text>
          </svg>
        )}
      </div>
    </Fig>
  );
}
