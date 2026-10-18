import { useEffect, useMemo, useState } from "react";
import { decodes, measureGpuRoof, useEngine, useRunSource } from "../engine/client";
import { useRoofs } from "../lib/roofs";
import { NeedsEngine, Tip, useFigureSource, useWidth } from "../ui/Figure";
import { KIND_LABEL, kindColor, usePalette } from "../ui/theme";
import { aggregateByName, type OpAgg } from "../lib/agg";
import { log, logTicks, sig, short } from "../lib/fmt";

const X0 = 0.03, X1 = 300; // FLOP per byte
const Y0 = 0.01, Y1 = 200; // GFLOP/s

/** Roofline: every op as a dot at (arithmetic intensity, achieved GFLOP/s). */
export default function Roofline() {
  const run = useEngine((s) => s.run);
  const roofs = useRoofs(run?.stage);
  const gpu = roofs.backend === "gpu";
  const bw = roofs.bw;
  const peak = roofs.peak;
  const busy = useEngine((s) => s.busy);
  const status = useEngine((s) => s.status);
  const timer = useEngine((s) => s.timerResUs || 5);
  const src = useRunSource();
  useFigureSource(src);
  const peakKind = useEngine((s) => (src === "recorded" ? (s.recorded?.meta.build === "simd128" ? "mul+add" : "fma") : s.peakKind));
  const pal = usePalette();
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ a: OpAgg; x: number; y: number; prefill: boolean } | null>(null);
  const [showPrefill, setShowPrefill] = useState(true);

  const dec = decodes(run).slice(-16);
  const pre = run?.forwards.filter((f) => f.phase === "prefill" && f.nComputed > 1) ?? [];
  const decAgg = useMemo(() => aggregateByName(dec).rows, [dec]);
  const preAgg = useMemo(() => aggregateByName(pre).rows, [pre]);

  // A live WebGPU run goes on the GPU's own memory roof: measure it once.
  useEffect(() => {
    if (gpu && src === "live" && run?.done && !busy && status === "ready") void measureGpuRoof();
  }, [gpu, src, run?.done, busy, status]);

  if (!run || !dec.length || !bw || (!gpu && !peak))
    return <NeedsEngine>Every op of a real forward pass, placed by how much math it does per byte it reads.</NeedsEngine>;

  const H = Math.min(420, Math.max(280, w * 0.55));
  const m = { l: 46, r: 14, t: 12, b: 34 };
  const x = log(X0, X1, m.l, w - m.r);
  // GPU runs: no measured compute roof, so give the slanted roof room to climb
  const YT = gpu ? 3000 : Y1;
  const y = log(Y0, YT, H - m.b, m.t);
  // The GPU has no measured compute roof here (the M4 GPU's f32 peak is far
  // above this chart), so a WebGPU run gets the slanted memory roof only.
  const top = peak ?? YT;
  const ridge = top / bw;
  const roof = (i: number) => Math.min(top, bw * i);
  const total = dec.reduce((a, f) => a + f.ms, 0);
  // ops shorter than ~3 timer ticks are too noisy to place honestly
  const minMs = (timer * 3) / 1000;
  const dots = (rows: OpAgg[], prefill: boolean) =>
    rows
      .filter((a) => a.ms / a.calls > minMs && a.flops > 0 && a.bytes > 0)
      .map((a) => ({ a, prefill, I: a.flops / a.bytes, G: a.flops / a.ms / 1e6 }));
  const all = [...dots(decAgg, false), ...(showPrefill ? dots(preAgg, true) : [])];
  // Label the big decode dots. Each label tries eight spots around its dot
  // (right first) and takes the first that collides with no other label, no
  // dot and nothing outside the plot.
  type Box = { x0: number; y0: number; x1: number; y1: number };
  const hit = (a: Box, b: Box) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
  const dotBoxes: Box[] = all.map((d) => {
    const r = 3 + Math.sqrt(d.a.ms / total) * 22;
    return { x0: x(d.I) - r, y0: y(d.G) - r, x1: x(d.I) + r, y1: y(d.G) + r };
  });
  const labels: { key: string; x: number; y: number; text: string; anchor: "start" | "end" | "middle" }[] = [];
  const placed: Box[] = [];
  const CW = 6.2, LH = 11;
  for (const d of [...all].filter((d) => !d.prefill && (d.a.ms / total > 0.03 || d.a.kind === "lm_head")).sort((p, q) => q.a.ms - p.a.ms)) {
    const r = 3 + Math.sqrt(d.a.ms / total) * 22;
    const cx = x(d.I), cy = y(d.G), tw = d.a.name.length * CW;
    const spots: [number, number, "start" | "end" | "middle"][] = [
      [cx + r + 4, cy + 3, "start"],
      [cx - r - 4, cy + 3, "end"],
      [cx, cy - r - 5, "middle"],
      [cx, cy + r + 11, "middle"],
      [cx + r + 3, cy - r - 2, "start"],
      [cx + r + 3, cy + r + 9, "start"],
      [cx - r - 3, cy - r - 2, "end"],
      [cx - r - 3, cy + r + 9, "end"],
    ];
    const box = ([lx, ly, a]: (typeof spots)[number]): Box => {
      const x0 = a === "start" ? lx : a === "end" ? lx - tw : lx - tw / 2;
      return { x0: x0 - 1, y0: ly - LH + 2, x1: x0 + tw + 1, y1: ly + 2 };
    };
    const ok = (s: (typeof spots)[number]) => {
      const b = box(s);
      return b.x0 > m.l && b.x1 < w - m.r && b.y0 > m.t && b.y1 < H - m.b && !placed.some((p) => hit(p, b)) && !dotBoxes.some((p, i) => all[i] !== d && hit(p, b));
    };
    const spot = spots.find(ok);
    if (!spot) continue; // unlabeled beats overlapping; the tooltip still names it
    placed.push(box(spot));
    labels.push({ key: d.a.name, x: spot[0], y: spot[1], text: d.a.name, anchor: spot[2] });
  }

  return (
    <div>
      <div className="nv-controls" style={{ justifyContent: "space-between", marginBottom: "0.6rem" }}>
        <span className="nv-muted" style={{ fontSize: "0.75rem" }}>
          {gpu
            ? `${run.stage} runs the whole token on the GPU, so it reports one op per token · last ${dec.length} tokens · roof = this ${roofs.bwFrom === "recorded" ? "recorded" : "tab's"} WebGPU bandwidth probe · no GPU compute roof is measured here (the M4 GPU's is above this chart)`
            : `dots = ops summed over all 12 layers and the last ${dec.length} tokens · size = share of time · roofs = this ${roofs.bwFrom === "recorded" ? "recorded" : "tab's"} wasm CPU probes`}
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
          {logTicks(Y0, YT).map((t) => (
            <g key={`y${t}`}>
              <line x1={m.l} x2={w - m.r} y1={y(t)} y2={y(t)} stroke={pal.rule} strokeDasharray="2 3" />
              <text x={m.l - 6} y={y(t)} textAnchor="end" dominantBaseline="middle">{sig(t, 1)}</text>
            </g>
          ))}
          <text x={(w + m.l) / 2} y={H - 4} textAnchor="middle">arithmetic intensity (FLOP per byte) →</text>
          <text transform={`translate(11 ${(H - m.b) / 2}) rotate(-90)`} textAnchor="middle">GFLOP/s →</text>
          {/* memory-bound region shading */}
          <path
            d={`M${x(X0)},${y(roof(X0))} L${x(ridge)},${y(top)} L${x(X1)},${y(top)} L${x(X1)},${y(Y0)} L${x(X0)},${y(Y0)} Z`}
            fill={pal.code}
            opacity={0.55}
          />
          {gpu ? (
            <path d={`M${x(X0)},${y(roof(X0))} L${x(ridge)},${y(top)}`} stroke={pal.fg} strokeWidth={1.8} fill="none" />
          ) : (
            <path d={`M${x(X0)},${y(roof(X0))} L${x(ridge)},${y(top)} L${x(X1)},${y(top)}`} stroke={pal.fg} strokeWidth={1.8} fill="none" />
          )}
          <text x={x(X0 * 2.2)} y={y(roof(X0 * 2.2)) - 8} style={{ fill: pal.fg }} transform={`rotate(${-Math.atan2(y(roof(X0 * 2.2)) - y(roof(X0 * 22)), x(X0 * 22) - x(X0 * 2.2)) * 180 / Math.PI} ${x(X0 * 2.2)} ${y(roof(X0 * 2.2)) - 8})`}>
            {gpu ? "GPU" : "CPU"} memory roof · {sig(bw, 3)} GB/s
          </text>
          {gpu ? null : (
            <>
              <text x={w - m.r - 4} y={y(top) - 6} textAnchor="end" style={{ fill: pal.fg }}>
                CPU compute roof · {sig(top, 3)} GFLOP/s ({peakKind === "fma" ? "relaxed-SIMD FMA" : "simd128 mul+add"}, 1 core)
              </text>
              <line x1={x(ridge)} x2={x(ridge)} y1={y(top)} y2={H - m.b} stroke={pal.muted} strokeDasharray="4 3" />
              <text x={x(ridge) + 4} y={H - m.b - 6} style={{ fontSize: 9 }}>ridge {sig(ridge, 2)}</text>
            </>
          )}
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
            <text key={l.key} x={l.x} y={l.y} textAnchor={l.anchor} style={{ fill: pal.sub, pointerEvents: "none" }}>{l.text}</text>
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
