import { useEffect, useMemo, useRef, useState } from "react";
import { claim, engine, release, showToken, tokenText, useEngine } from "../engine/client";
import { NeedsEngine, fitCanvas, useInView, useWidth } from "../ui/Figure";
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
  const L = cfg?.nLayer ?? 12, Hh = cfg?.nHead ?? 12;

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
    if (inView && source && !busy && !computing && (!data || data.tokens.join() !== source.join())) void compute();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView, source, busy]);

  const T = data?.tokens.length ?? 0;
  useEffect(() => {
    if (!data) return;
    cells.current.forEach((c, i) => c && paint(c, data.maps, i, T, pal, null));
  }, [data, pal, T]);

  // Big view: the selected head, with the selected query's row highlighted.
  useEffect(() => {
    const c = big.current;
    if (!c || !data || !bw) return;
    const size = bw;
    const ctx = fitCanvas(c, size, size);
    const off = document.createElement("canvas");
    paint(off, data.maps, sel[0] * Hh + sel[1], T, pal, q);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, size, size);
    ctx.drawImage(off, 0, 0, size, size);
    if (q != null) {
      ctx.strokeStyle = pal.accent;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(0.75, (q * size) / T, size - 1.5, size / T);
    }
  }, [data, sel, pal, q, bw, T, Hh]);

  if (status !== "ready") return <NeedsEngine>Where every head of every layer looks, on the text GPT-2 just wrote.</NeedsEngine>;

  const qi = q ?? T - 1;
  const row = data ? data.maps.subarray((sel[0] * Hh + sel[1]) * T * T + qi * T, (sel[0] * Hh + sel[1]) * T * T + qi * T + T) : null;
  const rmax = row ? Math.max(...row) : 1;

  return (
    <div ref={viewRef}>
      {!data ? (
        <div className="nv-empty">{computing ? "running an analysis pass over the text…" : source ? "waiting for the engine…" : "Generate text in fig. 1 first; this figure analyses it."}</div>
      ) : (
        <div className="nv-attn">
          <div>
            <div className="nv-heads" onPointerLeave={() => {}}>
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
              {L * Hh} heads × {T}×{T} · one analysis pass in {fmtMs(data.ms)} · rows = query token, columns = the tokens it reads · colour = attention vs. spreading evenly
            </div>
          </div>
          <div>
            <div style={{ fontSize: "0.78rem", marginBottom: "0.5rem" }}>
              <b>layer {sel[0]}, head {sel[1]}</b>
              <span className="nv-muted"> · query “{showToken(data.tokens[qi])}” at {qi}</span>
            </div>
            <div ref={bigRef}>
              <canvas
                ref={big}
                className="nv-canvas"
                style={{ borderRadius: 4, cursor: "crosshair", imageRendering: "pixelated" }}
                onPointerMove={(e) => {
                  const r = e.currentTarget.getBoundingClientRect();
                  setQ(Math.min(T - 1, Math.max(0, Math.floor(((e.clientY - r.top) / r.height) * T))));
                }}
              />
            </div>
            <div className="nv-attn-tokens" style={{ marginTop: "0.8rem" }}>
              {data.tokens.map((id, j) => {
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
