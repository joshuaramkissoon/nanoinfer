import { useMemo, useState } from "react";
import { decodes, useEngine } from "../engine/client";
import { NeedsEngine, Tip, useWidth } from "../ui/Figure";
import { KIND_LABEL, kindColor, usePalette } from "../ui/theme";
import { aggregateByName, type OpAgg } from "../lib/agg";
import { log, logTicks, sig, short } from "../lib/fmt";

const X0 = 0.03, X1 = 300; // FLOP per byte
const Y0 = 0.01, Y1 = 200; // GFLOP/s

/** Roofline: every op as a dot at (arithmetic intensity, achieved GFLOP/s). */
export default function Roofline() {
  const run = useEngine((s) => s.run);
  const bw = useEngine((s) => s.bandwidthGBs);
  const peak = useEngine((s) => s.peakGflops);
  const timer = useEngine((s) => s.timerResUs);
  const pal = usePalette();
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ a: OpAgg; x: number; y: number; prefill: boolean } | null>(null);
  const [showPrefill, setShowPrefill] = useState(true);

  const dec = decodes(run).slice(-16);
  const pre = run?.forwards.filter((f) => f.phase === "prefill" && f.nComputed > 1) ?? [];
  const decAgg = useMemo(() => aggregateByName(dec).rows, [dec]);
  const preAgg = useMemo(() => aggregateByName(pre).rows, [pre]);

  if (!run || !dec.length || !bw || !peak)
    return <NeedsEngine>Every op of a real forward pass, placed by how much math it does per byte it reads.</NeedsEngine>;

  const H = Math.min(420, Math.max(280, w * 0.55));
  const m = { l: 46, r: 14, t: 12, b: 34 };
  const x = log(X0, X1, m.l, w - m.r);
  const y = log(Y0, Y1, H - m.b, m.t);
  const ridge = peak / bw;
  const roof = (i: number) => Math.min(peak, bw * i);
  const total = dec.reduce((a, f) => a + f.ms, 0);
  // ops shorter than ~3 timer ticks are too noisy to place honestly
  const minMs = (timer * 3) / 1000;
  const dots = (rows: OpAgg[], prefill: boolean) =>
    rows
      .filter((a) => a.ms / a.calls > minMs && a.flops > 0 && a.bytes > 0)
      .map((a) => ({ a, prefill, I: a.flops / a.bytes, G: a.flops / a.ms / 1e6 }));
  const all = [...dots(decAgg, false), ...(showPrefill ? dots(preAgg, true) : [])];
  // Label the big decode dots; nudge labels down until they stop overlapping.
  const labels: { key: string; x: number; y: number; text: string }[] = [];
  for (const d of [...all].filter((d) => !d.prefill && (d.a.ms / total > 0.04 || d.a.kind === "lm_head")).sort((p, q) => q.a.ms - p.a.ms)) {
    const r = 3 + Math.sqrt(d.a.ms / total) * 22;
    let lx = x(d.I) + r + 4, ly = y(d.G) + 3;
    const wTxt = d.a.name.length * 6.2;
    while (labels.some((l) => Math.abs(l.y - ly) < 11 && lx < l.x + l.text.length * 6.2 && l.x < lx + wTxt)) ly += 11;
    labels.push({ key: d.a.name, x: lx, y: ly, text: d.a.name });
  }

  return (
    <div>
      <div className="nv-controls" style={{ justifyContent: "space-between", marginBottom: "0.6rem" }}>
        <span className="nv-muted" style={{ fontSize: "0.75rem" }}>
          dots = ops summed over all 12 layers and the last {dec.length} tokens · size = share of time
        </span>
        {pre.length > 0 && (
          <label className="nv-field">
            <input type="checkbox" checked={showPrefill} onChange={(e) => setShowPrefill(e.target.checked)} /> show prompt
            (prefill, {pre[0].nComputed} rows) as rings
          </label>
        )}
      </div>
      <div ref={ref} style={{ position: "relative" }} onPointerLeave={() => setHover(null)}>
        <svg className="nv-svg" viewBox={`0 0 ${w || 1} ${H}`} height={H} role="img" aria-label="roofline plot">
          {logTicks(X0, X1).map((t) => (
            <g key={`x${t}`}>
              <line x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} stroke={pal.rule} strokeDasharray="2 3" />
              <text x={x(t)} y={H - m.b + 14} textAnchor="middle">{sig(t, 1)}</text>
            </g>
          ))}
          {logTicks(Y0, Y1).map((t) => (
            <g key={`y${t}`}>
              <line x1={m.l} x2={w - m.r} y1={y(t)} y2={y(t)} stroke={pal.rule} strokeDasharray="2 3" />
              <text x={m.l - 6} y={y(t)} textAnchor="end" dominantBaseline="middle">{sig(t, 1)}</text>
            </g>
          ))}
          <text x={(w + m.l) / 2} y={H - 4} textAnchor="middle">arithmetic intensity (FLOP per byte) →</text>
          <text transform={`translate(11 ${(H - m.b) / 2}) rotate(-90)`} textAnchor="middle">GFLOP/s →</text>
          {/* memory-bound region shading */}
          <path
            d={`M${x(X0)},${y(roof(X0))} L${x(ridge)},${y(peak)} L${x(X1)},${y(peak)} L${x(X1)},${y(Y0)} L${x(X0)},${y(Y0)} Z`}
            fill={pal.code}
            opacity={0.55}
          />
          <path d={`M${x(X0)},${y(roof(X0))} L${x(ridge)},${y(peak)} L${x(X1)},${y(peak)}`} stroke={pal.fg} strokeWidth={1.8} fill="none" />
          <text x={x(X0 * 2.2)} y={y(roof(X0 * 2.2)) - 8} style={{ fill: pal.fg }} transform={`rotate(${-Math.atan2(y(roof(X0 * 2.2)) - y(roof(X0 * 22)), x(X0 * 22) - x(X0 * 2.2)) * 180 / Math.PI} ${x(X0 * 2.2)} ${y(roof(X0 * 2.2)) - 8})`}>
            memory roof · {sig(bw, 3)} GB/s
          </text>
          <text x={w - m.r - 4} y={y(peak) - 6} textAnchor="end" style={{ fill: pal.fg }}>
            compute roof · {sig(peak, 3)} GFLOP/s (simd128, 1 core)
          </text>
          <line x1={x(ridge)} x2={x(ridge)} y1={y(peak)} y2={H - m.b} stroke={pal.muted} strokeDasharray="4 3" />
          <text x={x(ridge) + 4} y={H - m.b - 6} style={{ fontSize: 9 }}>ridge {sig(ridge, 2)}</text>
          {all.map((d) => {
            const r = 3 + Math.sqrt(d.a.ms / total) * 22;
            const c = kindColor(pal, d.a.kind);
            const px = x(d.I), py = y(d.G);
            return (
              <g key={`${d.prefill}-${d.a.name}`} onPointerEnter={() => setHover({ a: d.a, x: px, y: py, prefill: d.prefill })}>
                <circle cx={px} cy={py} r={Math.max(r, 8)} fill="transparent" />
                <circle
                  cx={px}
                  cy={py}
                  r={r}
                  fill={d.prefill ? "none" : c}
                  stroke={c}
                  strokeWidth={d.prefill ? 1.6 : 1}
                  fillOpacity={0.85}
                  style={{ transition: "cx 0.5s, cy 0.5s" }}
                />
              </g>
            );
          })}
          {labels.map((l) => (
            <text key={l.key} x={l.x} y={l.y} style={{ fill: pal.sub, pointerEvents: "none" }}>{l.text}</text>
          ))}
        </svg>
        {hover && (
          <Tip x={Math.min(Math.max(hover.x, 120), w - 120)} y={hover.y - 6}>
            <b>{hover.a.name}</b> {hover.prefill ? "(prefill)" : "(decode)"} · {short(hover.a.flops / hover.a.calls)} FLOP ·{" "}
            {short(hover.a.bytes / hover.a.calls)} B per call
            <br />
            {sig(hover.a.flops / hover.a.bytes, 2)} FLOP/B · {sig(hover.a.flops / hover.a.ms / 1e6, 3)} GFLOP/s ·{" "}
            {sig(hover.a.bytes / hover.a.ms / 1e6, 3)} GB/s · {sig((100 * hover.a.flops) / hover.a.ms / 1e6 / roof(hover.a.flops / hover.a.bytes), 2)}% of roof
          </Tip>
        )}
      </div>
      <div className="nv-legend">
        {[...new Set(all.map((d) => d.a.kind))].map((k) => (
          <span key={k}>
            <i style={{ background: kindColor(pal, k) }} />
            {KIND_LABEL[k] ?? k}
          </span>
        ))}
      </div>
    </div>
  );
}
