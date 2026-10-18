import { useMemo, useState } from "react";
import { decodes, useEngine, useRunSource } from "../engine/client";
import { useFigureSource, NeedsEngine, Tip, useWidth } from "../ui/Figure";
import { kindColor, usePalette } from "../ui/theme";
import { fmtBytes, pct } from "../lib/fmt";

interface Leaf {
  key: string;
  name: string;
  kind: string;
  layer: number;
  bytes: number;
}
interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Squarified treemap (Bruls et al.), enough for ~100 leaves. */
function squarify<T extends { v: number }>(items: T[], r: Rect): (T & Rect)[] {
  const out: (T & Rect)[] = [];
  const total = items.reduce((a, b) => a + b.v, 0);
  if (!total) return out;
  const scale = (r.w * r.h) / total;
  let rest = items.map((it) => ({ it, a: it.v * scale })).sort((a, b) => b.a - a.a);
  let { x, y, w, h } = r;
  while (rest.length) {
    const side = Math.min(w, h);
    let row: typeof rest = [];
    let worst = Infinity;
    for (const c of rest) {
      const next = [...row, c];
      const s = next.reduce((a, b) => a + b.a, 0);
      const mx = Math.max(...next.map((n) => n.a)), mn = Math.min(...next.map((n) => n.a));
      const wr = Math.max((side * side * mx) / (s * s), (s * s) / (side * side * mn));
      if (wr > worst) break;
      worst = wr;
      row = next;
    }
    const s = row.reduce((a, b) => a + b.a, 0);
    const thick = s / side;
    let off = 0;
    for (const c of row) {
      const len = c.a / thick;
      out.push(w >= h ? { ...c.it, x, y: y + off, w: thick, h: len } : { ...c.it, x: x + off, y, w: len, h: thick });
      off += len;
    }
    if (w >= h) {
      x += thick;
      w -= thick;
    } else {
      y += thick;
      h -= thick;
    }
    rest = rest.slice(row.length);
  }
  return out;
}

const DT = [
  { k: "f32", f: 1 },
  { k: "f16", f: 0.5 },
  { k: "int8", f: 1.0625 / 4 },
];

/** What one decode step reads from memory, op by op, from the trace's byte counts. */
export default function ByteBudget() {
  useFigureSource(useRunSource());
  const run = useEngine((s) => s.run);
  const pal = usePalette();
  const [ref, w] = useWidth<HTMLDivElement>();
  const [dt, setDt] = useState(DT[0]);
  const [hover, setHover] = useState<(Leaf & Rect & { v: number }) | null>(null);
  const f = decodes(run).at(-1);

  const leaves = useMemo<Leaf[]>(() => {
    if (!f) return [];
    return f.ops.map((o, i) => {
      // Weight-bearing ops shrink with dtype; activations and KV stay f32 here.
      const isWeight = o.kind === "matmul" || o.kind === "lm_head";
      const wb = isWeight ? o.dims[1] * o.dims[2] * 4 : 0;
      const bytes = isWeight ? wb * dt.f + (o.bytes - wb) : o.bytes;
      return { key: `${i}`, name: o.name, kind: o.kind, layer: o.layer, bytes };
    });
  }, [f, dt]);

  const total = leaves.reduce((a, b) => a + b.bytes, 0);
  const H = Math.max(240, Math.min(420, w * 0.5));
  const rects = useMemo(() => squarify(leaves.map((l) => ({ ...l, v: l.bytes })), { x: 0, y: 0, w: w || 1, h: H }), [leaves, w, H]);
  const groups = useMemo(() => {
    const m = new Map<string, { kind: string; bytes: number }>();
    for (const l of leaves) {
      const g = m.get(l.name) ?? { kind: l.kind, bytes: 0 };
      g.bytes += l.bytes;
      m.set(l.name, g);
    }
    return [...m.entries()].sort((a, b) => b[1].bytes - a[1].bytes);
  }, [leaves]);

  if (!f) return <NeedsEngine>Every byte one token reads, as area.</NeedsEngine>;

  return (
    <div>
      <div className="nv-controls" style={{ justifyContent: "space-between", marginBottom: "0.8rem" }}>
        <div className="nv-seg" role="group" aria-label="weight dtype">
          {DT.map((d) => (
            <button key={d.k} aria-pressed={d.k === dt.k} onClick={() => setDt(d)}>
              weights {d.k}
            </button>
          ))}
        </div>
        <span className="nv-muted" style={{ fontSize: "0.75rem" }}>
          {fmtBytes(total)} per token at context {f.contextLen}
          {dt.k !== "f32" ? " · projected" : ""}
        </span>
      </div>
      <div ref={ref} style={{ position: "relative" }} onPointerLeave={() => setHover(null)}>
        <svg className="nv-svg" viewBox={`0 0 ${w || 1} ${H}`} height={H} role="img" aria-label="treemap of bytes read per token">
          {rects.map((r) => (
            <g key={r.key} onPointerEnter={() => setHover(r)}>
              <rect
                x={r.x + 0.5}
                y={r.y + 0.5}
                width={Math.max(0, r.w - 1)}
                height={Math.max(0, r.h - 1)}
                fill={kindColor(pal, r.kind)}
                opacity={hover && hover.name !== r.name ? 0.35 : r.layer % 2 ? 0.82 : 1}
                style={{ transition: "opacity 0.15s" }}
              />
              {r.w > 64 && r.h > 30 && (
                <text x={r.x + 7} y={r.y + 16} style={{ fill: pal.bg, fontSize: 11, fontWeight: 700 }}>
                  {r.name}
                  {r.layer >= 0 ? ` · L${r.layer}` : ""}
                </text>
              )}
              {r.w > 64 && r.h > 46 && (
                <text x={r.x + 7} y={r.y + 31} style={{ fill: pal.bg, fontSize: 10, opacity: 0.85 }}>
                  {fmtBytes(r.bytes)} · {pct(r.bytes / total, r.bytes / total > 0.1 ? 0 : 1)}
                </text>
              )}
            </g>
          ))}
        </svg>
        {hover && (
          <Tip x={Math.min(Math.max(hover.x + hover.w / 2, 110), w - 110)} y={hover.y + 8}>
            <b>{hover.name}</b>
            {hover.layer >= 0 ? ` · layer ${hover.layer}` : ""} · {fmtBytes(hover.bytes)} ({pct(hover.bytes / total, 1)})
          </Tip>
        )}
      </div>
      <div className="nv-legend">
        {groups.slice(0, 8).map(([n, g]) => (
          <span key={n}>
            <i style={{ background: kindColor(pal, g.kind) }} />
            {n} {pct(g.bytes / total, 1)}
          </span>
        ))}
      </div>
    </div>
  );
}
