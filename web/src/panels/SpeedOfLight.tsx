import { useState } from "react";
import { decodes, median, useEngine, useRunSource } from "../engine/client";
import { useFigureSource, NeedsEngine } from "../ui/Figure";
import { usePalette } from "../ui/theme";
import { fmtBytes, log, pct, sig } from "../lib/fmt";
import { M4_GBS, latestByStage, useNative } from "../lib/native";

// bytes per weight; `fmt` is core's WeightFormat name, so a dtype goes "live"
// (no longer a projection) as soon as a stage with that format exists.
const DTYPES = [
  { k: "f32", fmt: "f32", b: 4 },
  { k: "f16", fmt: "f16", b: 2 },
  { k: "int8", fmt: "q8_0", b: 1.0625 }, // Q8_0: 32 int8 + one scale per block
  { k: "int4", fmt: "q4_0", b: 0.5625 }, // Q4_0
];

const LO = 1, HI = 3000;
const A0 = Math.PI * 1.08, A1 = -Math.PI * 0.08; // gauge sweep (radians, math orientation)

/** Speed-of-light gauge: tok/s achieved vs the bandwidth ceiling. */
export default function SpeedOfLight() {
  const src = useRunSource();
  useFigureSource(src);
  const tab = src === "recorded" ? "the recorded tab's" : "this tab's";
  const pal = usePalette();
  const cfg = useEngine((s) => s.config);
  const bw = useEngine((s) => s.bandwidthGBs);
  const run = useEngine((s) => s.run);
  const bench = useEngine((s) => s.bench);
  const native = useNative();
  const stages = useEngine((s) => s.stages);
  const [dt, setDt] = useState(DTYPES[0]);
  const liveFormats = new Set(stages.map((s) => s.weightFormat ?? "f32"));

  if (!cfg) return <NeedsEngine>The fastest any engine could ever decode on this machine, and how close this one gets.</NeedsEngine>;

  const params = cfg.decodeWeightBytes / 4;
  const bytes = params * dt.b; // (biases/LayerNorm stay f32 in practice: <0.1% of bytes)
  const kvRead = run ? (run.forwards.at(-1)?.kvLen ?? 0) * 2 * cfg.nLayer * cfg.nEmbd * 4 : 0;
  const ceilBrowser = bw ? (bw * 1e9) / (bytes + kvRead) : null;
  const ceilNative = (M4_GBS * 1e9) / (bytes + kvRead);
  const dec = decodes(run);
  const liveTok = dec.length ? 1000 / median(dec.slice(-16).map((f) => f.ms)) : (bench[run?.stage ?? "kv-cache"]?.tokS ?? null);
  const nat = native ? latestByStage(native) : null;
  const bestNative = nat ? Math.max(0, ...[...nat.values()].filter((r) => (r.dtype ?? "f32") === "f32").map((r) => r.decode_tok_s)) : 0;

  // geometry
  const W = 520, H = 300, cx = W / 2, cy = 236, R = 190;
  const ang = log(LO, HI, A0, A1);
  const pt = (v: number, r: number) => {
    const a = ang(Math.min(HI, Math.max(LO, v)));
    return [cx + r * Math.cos(a), cy - r * Math.sin(a)];
  };
  const arc = (v0: number, v1: number, r: number) => {
    const [x0, y0] = pt(v0, r), [x1, y1] = pt(v1, r);
    const large = Math.abs(ang(v1) - ang(v0)) > Math.PI ? 1 : 0;
    return `M${x0},${y0} A${r},${r} 0 ${large} 1 ${x1},${y1}`;
  };
  const ticks = [1, 3, 10, 30, 100, 300, 1000, 3000];
  const needle = liveTok ?? LO;
  const [nx, ny] = pt(needle, R - 30);
  const ceil = ceilBrowser ?? ceilNative;

  return (
    <div>
      <div className="nv-controls" style={{ justifyContent: "space-between" }}>
        <div className="nv-seg" role="group" aria-label="weight dtype">
          {DTYPES.map((d) => (
            <button key={d.k} aria-pressed={d.k === dt.k} onClick={() => setDt(d)}>
              {d.k}
            </button>
          ))}
        </div>
        <span className="nv-muted" style={{ fontSize: "0.75rem" }}>
          {fmtBytes(bytes)} weights {kvRead ? `+ ${fmtBytes(kvRead)} KV ` : ""}read per token
          {!liveFormats.has(dt.fmt) && " · projected: no stage uses this format yet"}
        </span>
      </div>
      <svg className="nv-svg" viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: 620, margin: "0.6rem auto 0" }} role="img"
        aria-label={`gauge: ${liveTok ? sig(liveTok, 3) : "no"} tok/s achieved; ceiling ${sig(ceil, 3)} tok/s`}>
        <path d={arc(LO, HI, R)} stroke={pal.rule} strokeWidth={14} fill="none" strokeLinecap="butt" />
        {/* the forbidden zone: faster than physics allows */}
        <path d={arc(ceil, HI, R)} stroke={pal.heat[1]} strokeWidth={14} fill="none" opacity={0.7} />
        {liveTok && <path d={arc(LO, Math.min(liveTok, HI), R)} stroke={pal.accent} strokeWidth={14} fill="none" />}
        {ticks.map((t) => {
          const [x0, y0] = pt(t, R + 10), [x1, y1] = pt(t, R + 16), [tx, ty] = pt(t, R + 30);
          return (
            <g key={t}>
              <line x1={x0} y1={y0} x2={x1} y2={y1} stroke={pal.muted} />
              <text x={tx} y={ty} textAnchor="middle" dominantBaseline="middle">{t}</text>
            </g>
          );
        })}
        {[
          ceilBrowser && { v: ceilBrowser, l: `${tab} ceiling`, c: pal.fg },
          { v: ceilNative, l: "M4 ceiling", c: pal.sub },
          bestNative > 0 && dt.k === "f32" && { v: bestNative, l: "best native", c: pal.cool },
        ]
          .filter(Boolean)
          .map((m) => {
            const mm = m as { v: number; l: string; c: string };
            const [x0, y0] = pt(mm.v, R - 12), [x1, y1] = pt(mm.v, R + 9);
            return <line key={mm.l} x1={x0} y1={y0} x2={x1} y2={y1} stroke={mm.c} strokeWidth={2.5} />;
          })}
        <line x1={cx} y1={cy} x2={nx} y2={ny} stroke={pal.fg} strokeWidth={2.5} strokeLinecap="round"
          style={{ transition: "all 0.6s cubic-bezier(0.16,1,0.3,1)" }} />
        <circle cx={cx} cy={cy} r={6} fill={pal.fg} />
        <text x={cx} y={cy + 40} textAnchor="middle" style={{ fill: pal.fg, fontSize: 30, fontWeight: 700, letterSpacing: "-0.04em" }}>
          {liveTok ? sig(liveTok, 3) : "–"}
        </text>
        <text x={cx} y={cy + 58} textAnchor="middle">
          tok/s achieved {liveTok && ceilBrowser ? `· ${pct(liveTok / ceilBrowser, 0)} of ${tab} ceiling` : ""}
        </text>
      </svg>
      <div className="nv-legend" style={{ justifyContent: "center", marginTop: 0 }}>
        {ceilBrowser && <span><i style={{ background: pal.fg }} />{tab} ceiling {sig(ceilBrowser, 3)}</span>}
        <span><i style={{ background: pal.sub }} />M4 ceiling (120 GB/s) {sig(ceilNative, 3)}</span>
        {bestNative > 0 && dt.k === "f32" && <span><i style={{ background: pal.cool }} />best native run {sig(bestNative, 3)}</span>}
        <span><i style={{ background: pal.accent }} />achieved here</span>
      </div>
      <div className="nv-stats" style={{ justifyContent: "center", marginTop: "1rem" }}>
        <div className="nv-stat"><span className="k">measured read bandwidth ({src === "recorded" ? "recorded" : "this tab"})</span><span className="v">{bw ? sig(bw, 3) : "…"}<small>GB/s</small></span></div>
        <div className="nv-stat"><span className="k">÷ bytes per token</span><span className="v">{fmtBytes(bytes + kvRead)}</span></div>
        <div className="nv-stat"><span className="k">= ceiling</span><span className="v nv-accent">{ceilBrowser ? sig(ceilBrowser, 3) : "…"}<small>tok/s</small></span></div>
      </div>
    </div>
  );
}
