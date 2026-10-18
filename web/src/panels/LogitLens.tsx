import { useEffect, useMemo, useState } from "react";
import { claim, engine, release, showToken, tokenText, useEngine, useRunSource } from "../engine/client";
import { NeedsEngine, Tip, useFigureSource, useInView, useWidth } from "../ui/Figure";
import { css, heatAt, usePalette } from "../ui/theme";
import { fmtMs, sig } from "../lib/fmt";
import { unpackLens, type LensData } from "../lib/recorded";

const MAX_T = 64; // tokens fed to the analysis pass
const N_POS = 24; // positions decoded (the last 24 of those)
const K = 5;

/**
 * The logit lens: stop the network after every layer, push the residual stream
 * through GPT-2's own final LayerNorm and LM head, and read off what it would
 * say. Rows are depth (embedding at the bottom, layer 12 on top), columns are
 * positions in the text from fig. 1.
 */
export default function LogitLens() {
  const status = useEngine((s) => s.status);
  const run = useEngine((s) => s.run);
  const busy = useEngine((s) => s.busy);
  const rec = useEngine((s) => s.recorded);
  const src = useRunSource();
  const pal = usePalette();
  const [viewRef, inView] = useInView<HTMLDivElement>();
  const [gridRef, gw] = useWidth<HTMLDivElement>();
  const [live, setLive] = useState<LensData | null>(null);
  const [computing, setComputing] = useState(false);
  const [col, setCol] = useState<number | null>(null);
  const [hover, setHover] = useState<{ s: number; p: number; x: number; y: number } | null>(null);
  const ready = status === "ready";

  // Live: analyse the hero run once it's done (and only when on screen).
  const source = useMemo(() => (src === "live" && run?.done ? run.tokens.slice(0, MAX_T).map((t) => t.id) : null), [run, src]);
  useEffect(() => {
    if (!ready || !inView || !source || busy || computing) return;
    if (live && live.tokens.join() === source.join()) return;
    if (!claim("lens")) return;
    setComputing(true);
    const n = source.length;
    const positions = Array.from({ length: Math.min(N_POS, n) }, (_, i) => n - Math.min(N_POS, n) + i);
    const t0 = performance.now();
    engine
      .lens(source, positions, K)
      .then((flat) => {
        const depth = flat.length / positions.length / (2 * K + 4);
        setLive(unpackLens(flat, source, positions, K, depth, performance.now() - t0));
        setCol(null);
      })
      .finally(() => {
        setComputing(false);
        release("lens");
      });
  }, [ready, inView, source, busy, computing, live]);

  const data = live ?? (src !== "live" && rec ? rec.lens : null);
  useFigureSource(live ? "live" : data ? "recorded" : "none");

  if (!data) {
    return (
      <div ref={viewRef}>
        {ready || src === "live" ? (
          <div className="nv-empty">{computing ? "decoding the residual stream after every layer…" : "waiting for fig. 1 to finish…"}</div>
        ) : (
          <NeedsEngine>What GPT-2 would say if you stopped it after each layer.</NeedsEngine>
        )}
      </div>
    );
  }

  const P = data.positions.length;
  const depth = data.rows.length;
  const sel = col ?? Math.max(0, P - 2);
  const cw = gw ? Math.max(30, (gw - 40) / P) : 40;
  const ch = 26;
  const lockIn = (p: number) => {
    // first depth from which the top-1 guess equals the final answer and stays
    const fin = data.rows[depth - 1][p].top[0].id;
    let d = depth - 1;
    while (d > 0 && data.rows[d - 1][p].top[0].id === fin) d--;
    return d;
  };
  const selPos = data.positions[sel];
  const next = selPos + 1 < data.tokens.length ? data.tokens[selPos + 1] : null;
  const cellOf = (s: number, p: number) => data.rows[s][p];

  return (
    <div ref={viewRef}>
      <div className="nv-controls" style={{ justifyContent: "space-between", marginBottom: "0.8rem" }}>
        <span className="nv-muted" style={{ fontSize: "0.75rem" }}>
          {P} positions × {depth} depths · {depth * P} LM-head evaluations in {fmtMs(data.ms)} · cell = the top guess, shade = its probability ·
          <span style={{ color: pal.fg }}> ▁</span> = matches the token that actually came next
        </span>
        {computing && <span className="nv-muted" style={{ fontSize: "0.75rem" }}>recomputing…</span>}
      </div>

      <div className="nv-lens-scroll">
        <div ref={gridRef} className="nv-lens-grid" style={{ gridTemplateColumns: `40px repeat(${P}, minmax(30px, 1fr))` }} onPointerLeave={() => setHover(null)}>
          {Array.from({ length: depth }, (_, k) => depth - 1 - k).map((s) => [
            <span key={`l${s}`} className="nv-lens-lab">{s === 0 ? "emb" : `L${s}`}</span>,
            ...Array.from({ length: P }, (_, p) => {
              const c = cellOf(s, p);
              const top = c.top[0];
              const nx = data.positions[p] + 1 < data.tokens.length ? data.tokens[data.positions[p] + 1] : null;
              const hitNext = nx != null && top.id === nx;
              const bg = css(heatAt(pal, 0.05 + 0.95 * top.p));
              return (
                <button
                  key={`${s}-${p}`}
                  className={`nv-lens-cell${p === sel ? " sel" : ""}${hitNext ? " hit" : ""}`}
                  style={{ background: bg, color: top.p > 0.55 ? pal.bg : pal.fg, height: ch }}
                  onClick={() => setCol(p)}
                  onPointerEnter={(e) => {
                    const r = (e.currentTarget.parentElement as HTMLElement).getBoundingClientRect();
                    const b = e.currentTarget.getBoundingClientRect();
                    setHover({ s, p, x: b.left - r.left + b.width / 2, y: b.top - r.top });
                  }}
                  aria-label={`depth ${s}, position ${data.positions[p]}: ${tokenText(top.id)} ${(top.p * 100).toFixed(0)}%`}
                >
                  {clip(showToken(top.id), cw)}
                </button>
              );
            }),
          ])}
          <span />
          {data.positions.map((pos, p) => (
            <button key={`t${p}`} className={`nv-lens-tok${p === sel ? " sel" : ""}`} onClick={() => setCol(p)} title={`position ${pos}`}>
              {clip(showToken(data.tokens[pos]), cw)}
            </button>
          ))}
          {hover && (
            <Tip x={Math.min(Math.max(hover.x, 110), (gw || 400) - 110)} y={hover.y}>
              <b>{hover.s === 0 ? "after the embedding" : `after layer ${hover.s}`}</b> · reading “{showToken(data.tokens[data.positions[hover.p]])}”
              <br />
              {cellOf(hover.s, hover.p).top.map((t) => `${showToken(t.id)} ${(t.p * 100).toFixed(1)}%`).join(" · ")}
            </Tip>
          )}
        </div>
      </div>

      <div className="nv-lens-detail">
        <div>
          <div style={{ fontSize: "0.78rem", marginBottom: "0.6rem" }}>
            <b>reading “{showToken(data.tokens[selPos])}”</b>
            <span className="nv-muted">
              {" "}
              at position {selPos}
              {next != null ? <> · the next token was “{showToken(next)}”</> : null}
            </span>
          </div>
          <div className="nv-lens-rows">
            {Array.from({ length: depth }, (_, k) => depth - 1 - k).map((s) => {
              const c = cellOf(s, sel);
              return (
                <div key={s} className="nv-lens-row">
                  <span className="nv-muted">{s === 0 ? "emb" : `L${s}`}</span>
                  <span className="chips">
                    {c.top.slice(0, 3).map((t, j) => (
                      <span key={j} className={`nv-chip${next === t.id ? " next" : ""}`} style={{ opacity: 0.45 + 0.55 * Math.min(1, t.p * 2.5) }}>
                        {showToken(t.id)} <span className="nv-muted">{(t.p * 100).toFixed(t.p < 0.1 ? 1 : 0)}%</span>
                      </span>
                    ))}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
        <DepthChart data={data} p={sel} lock={lockIn(sel)} />
      </div>
    </div>
  );
}

const clip = (t: string, w: number) => {
  const n = Math.max(2, Math.floor((w - 6) / 6.6));
  return t.length > n ? t.slice(0, n - 1) + "…" : t;
};

/** p(final answer), p(actual next token) and entropy vs depth for one position,
 *  plus the residual stream's size. */
function DepthChart({ data, p, lock }: { data: LensData; p: number; lock: number }) {
  const pal = usePalette();
  const [ref, w] = useWidth<HTMLDivElement>();
  const depth = data.rows.length;
  const H = 210;
  const m = { l: 34, r: 12, t: 14, b: 30 };
  const x = (s: number) => m.l + ((w - m.l - m.r) * s) / (depth - 1);
  const y = (v: number) => H - m.b - v * (H - m.t - m.b);
  const cells = data.rows.map((r) => r[p]);
  const maxNorm = Math.max(...cells.map((c) => c.norm));
  const hasNext = cells[0].pNext >= 0;
  const line = (f: (c: (typeof cells)[number]) => number) => cells.map((c, s) => `${x(s)},${y(f(c))}`).join(" ");
  return (
    <div ref={ref}>
      <div className="nv-muted" style={{ fontSize: "0.72rem", marginBottom: "0.4rem" }}>
        how sure it is, layer by layer · the guess locks in at {lock === 0 ? "the embedding" : `layer ${lock}`}
      </div>
      {w > 0 && (
        <svg className="nv-svg" viewBox={`0 0 ${w} ${H}`} height={H} role="img" aria-label="probability by depth">
          {[0, 0.5, 1].map((v) => (
            <g key={v}>
              <line x1={m.l} x2={w - m.r} y1={y(v)} y2={y(v)} stroke={pal.rule} strokeDasharray="2 3" />
              <text x={m.l - 5} y={y(v)} textAnchor="end" dominantBaseline="middle">{v * 100}%</text>
            </g>
          ))}
          {cells.map((c, s) => (
            <rect key={s} x={x(s) - 5} width={10} y={H - m.b - (c.norm / maxNorm) * 26} height={(c.norm / maxNorm) * 26} fill={pal.code} />
          ))}
          <line x1={x(lock)} x2={x(lock)} y1={m.t} y2={H - m.b} stroke={pal.accent} strokeDasharray="3 3" opacity={0.6} />
          <polyline fill="none" stroke={pal.accent} strokeWidth={2} points={line((c) => c.pFinal)} />
          {hasNext && <polyline fill="none" stroke={pal.cool} strokeWidth={1.6} strokeDasharray="4 3" points={line((c) => Math.max(0, c.pNext))} />}
          {cells.map((c, s) => (
            <circle key={s} cx={x(s)} cy={y(c.pFinal)} r={2.6} fill={pal.accent} />
          ))}
          {cells.map((_, s) => (s % 2 === 0 || s === depth - 1 ? (
            <text key={s} x={x(s)} y={H - m.b + 14} textAnchor="middle">{s === 0 ? "emb" : s}</text>
          ) : null))}
          <text x={(w + m.l) / 2} y={H - 3} textAnchor="middle">depth (layers run) →</text>
        </svg>
      )}
      <div className="nv-legend" style={{ marginTop: "0.3rem" }}>
        <span><i style={{ background: pal.accent }} />p(final answer “{showToken(cells[depth - 1].top[0].id)}”)</span>
        {hasNext && <span><i style={{ background: pal.cool }} />p(the token that came next)</span>}
        <span><i style={{ background: pal.code }} />size of the residual stream (‖x‖ {sig(cells[0].norm, 2)} → {sig(cells[depth - 1].norm, 3)})</span>
      </div>
    </div>
  );
}
