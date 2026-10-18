import { useEffect, useMemo, useRef, useState } from "react";
import { claim, engine, release, showToken, tokenText, useEngine } from "../engine/client";
import { NeedsEngine, Tip, fitCanvas, useFigureSource, useInView, useWidth } from "../ui/Figure";
import { css, heatAt, heatLut, rgb, usePalette, type Palette } from "../ui/theme";
import { fmtMs } from "../lib/fmt";

const MAX_T = 64;

/** Draw one head's T×T causal attention matrix into a canvas, pixel per cell. */
function paint(c: HTMLCanvasElement, maps: Float32Array, idx: number, T: number, pal: Palette, q: number | null) {
  c.width = T;
  c.height = T;
  const ctx = c.getContext("2d")!;
  const img = ctx.createImageData(T, T);
  const base = idx * T * T;
  const bg = rgb(pal.code);
  const lut = heatLut(pal);
  for (let i = 0; i < T; i++) {
    const dim = q != null && i !== q ? 0.55 : 1;
    for (let j = 0; j < T; j++) {
      const o = (i * T + j) * 4;
      let r = bg[0], g = bg[1], b = bg[2];
      if (j <= i) {
        // colour = attention relative to uniform (p·(i+1) = 1 means "spread evenly"),
        // on a log scale from 1/4 of uniform to 16x uniform
        const rel = maps[base + i * T + j] * (i + 1);
        const t = (Math.log2(Math.max(rel, 1e-6)) + 2) / 6;
        const k = Math.max(0, Math.min(255, (t * 255) | 0)) * 3;
        r = lut[k] * dim + bg[0] * (1 - dim);
        g = lut[k + 1] * dim + bg[1] * (1 - dim);
        b = lut[k + 2] * dim + bg[2] * (1 - dim);
      }
      img.data[o] = r;
      img.data[o + 1] = g;
      img.data[o + 2] = b;
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/** 12 layers × 12 heads of real attention on the text from fig. 1. */
export default function Attention() {
  const status = useEngine((s) => s.status);
  const run = useEngine((s) => s.run);
  const busy = useEngine((s) => s.busy);
  const cfg = useEngine((s) => s.config);
  const pal = usePalette();
  const [viewRef, inView] = useInView<HTMLDivElement>();
  const [data, setData] = useState<{ tokens: number[]; maps: Float32Array; ms: number } | null>(null);
  const [sel, setSel] = useState<[number, number]>([5, 1]);
  const [q, setQ] = useState<number | null>(null);
  const [computing, setComputing] = useState(false);
  const cells = useRef<(HTMLCanvasElement | null)[]>([]);
  const big = useRef<HTMLCanvasElement>(null);
  const [bigRef, bw] = useWidth<HTMLDivElement>();
  const [cell, setCell] = useState<{ i: number; j: number; x: number; y: number } | null>(null);
  const rec = useEngine((s) => s.recorded);
  const L = cfg?.nLayer ?? 12, Hh = cfg?.nHead ?? 12;
  const ready = status === "ready";

  const source = useMemo(() => (run?.done ? run.tokens.slice(0, MAX_T).map((t) => t.id) : null), [run]);

  const compute = async () => {
    if (!source || !claim("attention")) return;
    setComputing(true);
    try {
      const t0 = performance.now();
      const maps = await engine.attention(source);
      setData({ tokens: source, maps, ms: performance.now() - t0 });
      setQ(null);
      // Start on the clearest "previous token" head: a classic, easy to read.
      const n = source.length;
      let best = 0, bestScore = -1;
      for (let hIdx = 0; hIdx < maps.length / (n * n); hIdx++) {
        let sc = 0;
        for (let i = 1; i < n; i++) sc += maps[hIdx * n * n + i * n + i - 1];
        if (sc > bestScore) [best, bestScore] = [hIdx, sc];
      }
      setSel([Math.floor(best / Hh), best % Hh]);
    } finally {
      setComputing(false);
      release("attention");
    }
  };

  // Compute automatically once the figure is on screen and a run has finished.
  useEffect(() => {
    if (ready && inView && source && !busy && !computing && (!data || data.tokens.join() !== source.join())) void compute();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView, source, busy]);

  // Recorded maps until this tab has its own.
  const shown = data ?? (!ready && rec ? rec.attention : null);
  useFigureSource(data ? "live" : shown ? "recorded" : "none");
  // first time the recording shows, start on its clearest previous-token head
  useEffect(() => {
    if (data || !shown) return;
    const n = shown.tokens.length;
    let best = 0, bestScore = -1;
    for (let hIdx = 0; hIdx < shown.maps.length / (n * n); hIdx++) {
      let sc = 0;
      for (let i = 1; i < n; i++) sc += shown.maps[hIdx * n * n + i * n + i - 1];
      if (sc > bestScore) [best, bestScore] = [hIdx, sc];
    }
    setSel([Math.floor(best / Hh), best % Hh]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shown === rec?.attention]);

  const T = shown?.tokens.length ?? 0;
  useEffect(() => {
    if (!shown) return;
    cells.current.forEach((c, i) => c && paint(c, shown.maps, i, T, pal, null));
  }, [shown, pal, T]);

  // Big view: the selected head with token labels on both axes, and the
  // selected query's row outlined.
  const GL = bw > 420 ? 84 : 56; // left gutter (query labels)
  const GT = bw > 420 ? 64 : 44; // top gutter (key labels)
  const size = Math.max(0, bw - GL);
  useEffect(() => {
    const c = big.current;
    if (!c || !shown || !bw) return;
    const ctx = fitCanvas(c, bw, size + GT);
    const off = document.createElement("canvas");
    paint(off, shown.maps, sel[0] * Hh + sel[1], T, pal, q);
    ctx.clearRect(0, 0, bw, size + GT);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, GL, GT, size, size);
    const cellPx = size / T;
    const every = Math.max(1, Math.ceil(10 / cellPx));
    ctx.font = `${Math.min(11, Math.max(8, cellPx * 0.8))}px JetBrains Mono, monospace`;
    ctx.textBaseline = "middle";
    const lab = (id: number) => {
      const t = showToken(id);
      const max = GL > 60 ? 10 : 6;
      return t.length > max ? t.slice(0, max - 1) + "…" : t;
    };
    for (let i = 0; i < T; i += 1) {
      if (i % every && i !== q && i !== cell?.j) continue;
      ctx.fillStyle = i === q ? pal.accent : i === cell?.j ? pal.fg : pal.muted;
      ctx.textAlign = "right";
      ctx.fillText(lab(shown.tokens[i]), GL - 5, GT + (i + 0.5) * cellPx);
      ctx.save();
      ctx.translate(GL + (i + 0.5) * cellPx, GT - 5);
      ctx.rotate(-Math.PI / 3);
      ctx.textAlign = "left";
      ctx.fillStyle = i === cell?.j ? pal.fg : pal.muted;
      ctx.fillText(lab(shown.tokens[i]), 0, 0);
      ctx.restore();
    }
    if (q != null) {
      ctx.strokeStyle = pal.accent;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(GL + 0.75, GT + q * cellPx, size - 1.5, cellPx);
    }
    if (cell) {
      ctx.strokeStyle = pal.fg;
      ctx.lineWidth = 1.2;
      ctx.strokeRect(GL + cell.j * cellPx, GT + cell.i * cellPx, cellPx, cellPx);
    }
  }, [shown, sel, pal, q, bw, T, Hh, GL, GT, size, cell]);

  if (!shown && !ready) return <NeedsEngine>Where every head of every layer looks, on the text GPT-2 just wrote.</NeedsEngine>;

  const qi = q ?? T - 1;
  const head = sel[0] * Hh + sel[1];
  const rowOf = (i: number) => (shown ? shown.maps.subarray(head * T * T + i * T, head * T * T + i * T + T) : null);
  const row = rowOf(qi);
  const rmax = row ? Math.max(...row) : 1;
  const topOf = (i: number) => {
    const r = rowOf(i);
    if (!r) return [];
    return Array.from(r.subarray(0, i + 1), (p, j) => ({ j, p })).sort((a, b) => b.p - a.p).slice(0, 3);
  };

  return (
    <div ref={viewRef}>
      {!shown ? (
        <div className="nv-empty">{computing ? "running an analysis pass over the text…" : source ? "waiting for the engine…" : "Generate text in fig. 1 first; this figure analyses it."}</div>
      ) : (
        <div className="nv-attn">
          <div>
            <div className="nv-heads">
              <span />
              {Array.from({ length: Hh }, (_, h) => (
                <span key={h} className="lab">h{h}</span>
              ))}
              {Array.from({ length: L }, (_, l) => [
                <span key={`l${l}`} className="lab" style={{ paddingRight: 4 }}>L{l}</span>,
                ...Array.from({ length: Hh }, (_, h) => (
                  <canvas
                    key={`${l}-${h}`}
                    ref={(el) => (cells.current[l * Hh + h] = el)}
                    className={sel[0] === l && sel[1] === h ? "on" : ""}
                    onClick={() => setSel([l, h])}
                    title={`layer ${l}, head ${h}`}
                    aria-label={`layer ${l} head ${h}`}
                  />
                )),
              ])}
            </div>
            <div className="nv-muted" style={{ fontSize: "0.7rem", marginTop: "0.6rem" }}>
              {L * Hh} heads × {T}×{T} · one analysis pass in {fmtMs(shown.ms)} · rows = query token, columns = the tokens it reads · colour = attention vs. spreading evenly · click a head
            </div>
          </div>
          <div>
            <div style={{ fontSize: "0.78rem", marginBottom: "0.5rem" }}>
              <b>layer {sel[0]}, head {sel[1]}</b>
              <span className="nv-muted"> · query “{showToken(shown.tokens[qi])}” at {qi} looks most at </span>
              {topOf(qi).map((t, k) => (
                <span key={k} className="nv-chip" style={{ marginRight: 4 }}>
                  {showToken(shown.tokens[t.j])} <span className="nv-muted">{(t.p * 100).toFixed(0)}%</span>
                </span>
              ))}
            </div>
            <div ref={bigRef} style={{ position: "relative" }} onPointerLeave={() => setCell(null)}>
              <canvas
                ref={big}
                className="nv-canvas"
                style={{ cursor: "crosshair" }}
                onPointerMove={(e) => {
                  const r = e.currentTarget.getBoundingClientRect();
                  const px = e.clientX - r.left - GL, py = e.clientY - r.top - GT;
                  if (px < 0 || py < 0 || px > size || py > size) return setCell(null);
                  const i = Math.min(T - 1, Math.floor((py / size) * T));
                  const j = Math.min(T - 1, Math.floor((px / size) * T));
                  setQ(i);
                  setCell({ i, j: Math.min(i, j), x: e.clientX - r.left, y: e.clientY - r.top });
                }}
              />
              {cell && (
                <Tip x={Math.min(Math.max(cell.x, 120), bw - 120)} y={cell.y - 14}>
                  “{showToken(shown.tokens[cell.i])}” → “{showToken(shown.tokens[cell.j])}” ·{" "}
                  <b>{((rowOf(cell.i)?.[cell.j] ?? 0) * 100).toFixed(1)}%</b> of its attention
                </Tip>
              )}
            </div>
            <div className="nv-attn-tokens" style={{ marginTop: "0.8rem" }}>
              {shown.tokens.map((id, j) => {
                const v = row && j <= qi ? row[j] / (rmax || 1) : 0;
                return (
                  <span
                    key={j}
                    className={j === qi ? "q" : ""}
                    style={{ background: j <= qi && v > 0.02 ? css(heatAt(pal, 0.15 + 0.85 * v)) : undefined, color: v > 0.6 ? pal.bg : undefined }}
                    onPointerEnter={() => setQ(j)}
                  >
                    {tokenText(id)}
                  </span>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
