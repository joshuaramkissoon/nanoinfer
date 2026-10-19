import { useEffect, useMemo, useRef, useState } from "react";
import { decodes, useEngine, useRunSource } from "../engine/client";
import type { ForwardTrace, OpRec } from "../engine/types";
import { useFigureSource, NeedsEngine, Tip, fitCanvas, useWidth } from "../ui/Figure";
import { KIND_LABEL, kindColor, usePalette } from "../ui/theme";
import { aggregateByName } from "../lib/agg";
import { fmtMs, pct, sig } from "../lib/fmt";

const ROWS = 24;
const ROW_H = 13;
const GAP = 3;
const LEFT = 34;

/** One horizontal strip per decoded token; every op is a block of its measured width. */
export default function TimeStrip() {
  useFigureSource(useRunSource());
  const run = useEngine((s) => s.run);
  const pal = usePalette();
  const [wrapRef, w] = useWidth<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const [mode, setMode] = useState<"abs" | "share">("abs");
  const [hover, setHover] = useState<{ x: number; y: number; op: OpRec; f: ForwardTrace; row: number } | null>(null);

  const all = decodes(run);
  const fwds = all.slice(-ROWS);
  const first = all.length - fwds.length;
  const maxMs = Math.max(1, ...fwds.map((f) => f.ms));
  const h = Math.max(1, fwds.length) * (ROW_H + GAP) + 18;
  const plotW = Math.max(10, w - LEFT);

  useEffect(() => {
    const c = canvas.current;
    if (!c || !w) return;
    const ctx = fitCanvas(c, w, h);
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px JetBrains Mono, monospace";
    ctx.textBaseline = "middle";
    fwds.forEach((f, r) => {
      const y = r * (ROW_H + GAP);
      const scale = mode === "abs" ? plotW / maxMs : plotW / f.ms;
      ctx.fillStyle = pal.muted;
      ctx.textAlign = "right";
      ctx.fillText(String(first + r + 1), LEFT - 8, y + ROW_H / 2);
      ctx.fillStyle = pal.code;
      ctx.fillRect(LEFT, y, f.ms * scale, ROW_H);
      for (const o of f.ops) {
        const x = LEFT + o.start * scale;
        const ow = Math.max(0.6, o.dur * scale);
        ctx.fillStyle = kindColor(pal, o.kind);
        ctx.globalAlpha = hover && hover.row === r && hover.op === o ? 1 : o.layer % 2 === 0 || o.layer < 0 ? 0.95 : 0.78;
        ctx.fillRect(x, y, ow, ROW_H);
      }
      ctx.globalAlpha = 1;
    });
    // time axis
    if (mode === "abs") {
      const y = fwds.length * (ROW_H + GAP) + 8;
      ctx.fillStyle = pal.muted;
      ctx.textAlign = "left";
      ctx.fillText("0", LEFT, y);
      ctx.textAlign = "right";
      ctx.fillText(fmtMs(maxMs), w, y);
    }
  }, [fwds, w, h, pal, mode, hover, maxMs, plotW, first]);

  const onMove = (e: React.PointerEvent) => {
    const rect = canvas.current!.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const r = Math.floor(y / (ROW_H + GAP));
    const f = fwds[r];
    if (!f) return setHover(null);
    const scale = mode === "abs" ? plotW / maxMs : plotW / f.ms;
    const t = (x - LEFT) / scale;
    const op = f.ops.find((o) => t >= o.start && t <= o.start + Math.max(o.dur, 0.6 / scale));
    setHover(op ? { x, y: r * (ROW_H + GAP), op, f, row: r } : null);
  };

  const { rows, totalMs } = useMemo(() => aggregateByName(all), [all]);
  const kinds = useMemo(() => [...new Set(fwds.flatMap((f) => f.ops.map((o) => o.kind)))], [fwds]);

  if (!all.length) return <NeedsEngine>Generate some text in fig 1 and every op of every token appears here.</NeedsEngine>;

  const lm = rows.find((r) => r.kind === "lm_head");
  return (
    <div>
      <div className="nv-controls" style={{ marginBottom: "0.9rem", justifyContent: "space-between" }}>
        <span className="nv-muted" style={{ fontSize: "var(--t-sm)" }}>
          last {fwds.length} decoded tokens · {run?.stage} · {fwds[0]?.ops.length} ops per token
        </span>
        <div className="nv-seg">
          <button aria-pressed={mode === "abs"} onClick={() => setMode("abs")}>
            milliseconds
          </button>
          <button aria-pressed={mode === "share"} onClick={() => setMode("share")}>
            share of token
          </button>
        </div>
      </div>
      <div ref={wrapRef} style={{ position: "relative" }}>
        <canvas ref={canvas} className="nv-canvas" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
        {hover && (
          <Tip x={Math.min(Math.max(hover.x, 120), w - 120)} y={hover.y}>
            <b>{hover.op.name}</b>
            {hover.op.layer >= 0 ? ` · layer ${hover.op.layer + 1}` : ""} · {fmtMs(hover.op.dur)} ({pct(hover.op.dur / hover.f.ms, 1)})
            <br />[{hover.op.dims.filter((d, i) => d || i < 2).join("×")}] · {sig(hover.op.flops / hover.op.dur / 1e6, 3)} GFLOP/s ·{" "}
            {sig(hover.op.bytes / hover.op.dur / 1e6, 3)} GB/s
          </Tip>
        )}
      </div>
      <div className="nv-legend">
        {kinds.map((k) => (
          <span key={k}>
            <i style={{ background: kindColor(pal, k) }} />
            {KIND_LABEL[k] ?? k}
          </span>
        ))}
      </div>

      <div style={{ marginTop: "1.6rem", display: "grid", gap: "0.3rem", fontSize: "var(--t-sm)" }}>
        <div className="nv-muted" style={{ display: "grid", gridTemplateColumns: "minmax(4.5rem, 8rem) 1fr 3.5rem 3.5rem", gap: "0.6rem" }}>
          <span>op ×12 layers</span>
          <span>share of time</span>
          <span style={{ textAlign: "right" }}>ms/tok</span>
          <span style={{ textAlign: "right" }}>GB/s</span>
        </div>
        {rows.slice(0, 9).map((r) => (
          <div key={r.name} style={{ display: "grid", gridTemplateColumns: "minmax(4.5rem, 8rem) 1fr 3.5rem 3.5rem", gap: "0.6rem", alignItems: "center" }}>
            <span style={{ fontWeight: r.kind === "lm_head" ? 700 : 400 }}>{r.name}</span>
            <span style={{ height: "0.65rem", background: pal.code, borderRadius: 2, position: "relative" }}>
              <i style={{ position: "absolute", inset: 0, width: pct(r.ms / totalMs, 2), background: kindColor(pal, r.kind), borderRadius: 2 }} />
              <span style={{ position: "absolute", left: `calc(${pct(r.ms / totalMs, 2)} + 6px)`, top: "-0.2rem", color: pal.sub }}>
                {pct(r.ms / totalMs, 1)}
              </span>
            </span>
            <span style={{ textAlign: "right" }}>{sig(r.ms / all.length, 3)}</span>
            <span style={{ textAlign: "right" }} className="nv-sub">
              {sig(r.bytes / r.ms / 1e6, 3)}
            </span>
          </div>
        ))}
        {lm && (
          <div className="nv-sub" style={{ marginTop: "0.6rem" }}>
            The LM head alone is one matmul, yet it takes <b className="nv-accent">{pct(lm.ms / totalMs, 0)}</b> of every token: it
            reads all {sig(lm.bytes / lm.calls / 1e6, 3)} MB of the embedding table to score 50,257 candidates.
          </div>
        )}
      </div>
    </div>
  );
}
