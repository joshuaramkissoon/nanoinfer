import { useEffect, useRef, useState } from "react";
import { SpaceKey, Tk } from "../../ui/Tok";
import { decodeTokens, tokenText } from "../../engine/client";
import { fitCanvas, useWidth } from "../../ui/Figure";
import { usePalette, type Palette } from "../../ui/theme";
import { Deeper, Fig, G, Idea, useFrame, useOnScreen, useRecorded } from "../kit";

/** fig 6.1: inside one block */
function BlockDiagram() {
  const pal = usePalette();
  const [t, setT] = useState(0);
  const [onScreen, vis] = useOnScreen<HTMLElement>();
  useFrame(vis, (dt) => setT((v) => (v + dt * 0.35) % 1));
  const W = 600, H = 250, sx = 150;
  const y = (f: number) => 20 + f * (H - 40);
  const pulse = y(t);
  const box = (x: number, yy: number, w: number, label: string, sub: string, c: string) => (
    <g>
      <rect x={x} y={yy - 18} width={w} height={36} rx={7} fill={pal.dark ? "#262421" : "#fff"} stroke={c} />
      <text x={x + w / 2} y={yy - 1} textAnchor="middle" fontSize={12} fontWeight={700} fill={pal.fg}>
        {label}
      </text>
      <text x={x + w / 2} y={yy + 12} textAnchor="middle" fontSize={9.5} fill={pal.muted}>
        {sub}
      </text>
    </g>
  );
  const branch = (yy0: number, yy1: number, label: string, sub: string, c: string) => (
    <g>
      <path d={`M${sx},${yy0} H${sx + 60} V${yy1 - 30}`} fill="none" stroke={pal.muted} strokeDasharray="3 3" />
      {box(sx + 60 - 70, (yy0 + yy1) / 2, 140, label, sub, c)}
      <path d={`M${sx + 60},${(yy0 + yy1) / 2 + 18} V${yy1} H${sx + 9}`} fill="none" stroke={c} strokeWidth={1.5} />
      <circle cx={sx} cy={yy1} r={9} fill={pal.bg} stroke={c} strokeWidth={1.5} />
      <text x={sx} y={yy1 + 4} textAnchor="middle" fontSize={13} fill={c} fontWeight={700}>
        +
      </text>
    </g>
  );
  return (
    <Fig
      figRef={onScreen}
      n="6.1"
      title="One transformer block"
      prov="diagram"
      caption={
        <>
          The thick ember line is the <b>residual stream</b>: the token's 768 numbers flowing from top to bottom. A block
          reads a copy, runs attention (gather from earlier tokens, chapter 5), and <b>adds</b> the result back on (⊕). Then
          it does the same with an MLP: two big matrices with a bend in between, working on this token alone. GPT-2 stacks
          12 of these blocks.
        </>
      }
    >
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: 560, display: "block", margin: "0 auto" }}>
        <line x1={sx} x2={sx} y1={y(0)} y2={y(1)} stroke={pal.heat[3]} strokeWidth={7} strokeLinecap="round" opacity={0.85} />
        <circle cx={sx} cy={pulse} r={5} fill={pal.accent} />
        <text x={sx - 14} y={y(0) + 4} textAnchor="end" fontSize={11} fill={pal.sub}>
          stream in
        </text>
        <text x={sx - 14} y={y(1) + 4} textAnchor="end" fontSize={11} fill={pal.sub}>
          stream out
        </text>
        {branch(y(0.08), y(0.47), "attention", "look at earlier tokens", pal.cool)}
        {branch(y(0.55), y(0.94), "MLP", "768 → 3072 → 768", pal.heat[3])}
        <text x={sx + 210} y={y(0.27) + 4} fontSize={10.5} fill={pal.muted}>
          4 matrices: query, key, value, output
        </text>
        <text x={sx + 210} y={y(0.745) + 4} fontSize={10.5} fill={pal.muted}>
          2 matrices: up, down (+ GELU bend)
        </text>
      </svg>
    </Fig>
  );
}

function Strip({ v, pal, hi }: { v: Float32Array; pal: Palette; hi: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [box, w] = useWidth<HTMLDivElement>();
  useEffect(() => {
    const c = ref.current;
    if (!c || !w) return;
    const H = hi ? 22 : 9;
    const ctx = fitCanvas(c, w, H);
    ctx.clearRect(0, 0, w, H);
    let m = 0;
    for (const x of v) m = Math.max(m, Math.abs(x));
    const bw = w / v.length;
    for (let i = 0; i < v.length; i++) {
      const t = v[i] / (m || 1);
      ctx.fillStyle = t >= 0 ? pal.accent : pal.cool;
      ctx.globalAlpha = Math.min(1, Math.abs(t) * 3) * (hi ? 1 : 0.6);
      ctx.fillRect(i * bw, 0, Math.max(1, bw), H);
    }
    ctx.globalAlpha = 1;
  }, [v, w, pal, hi]);
  return (
    <div ref={box} style={{ flex: 1 }}>
      <canvas ref={ref} style={{ width: "100%", display: "block", borderRadius: 2 }} />
    </div>
  );
}

/** fig 6.2: the real residual stream + logit lens */
function Stream() {
  const rec = useRecorded();
  const pal = usePalette();
  const [d, setD] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [onScreen2, vis2] = useOnScreen<HTMLElement>();
  useEffect(() => {
    if (!playing || !vis2) return;
    const t = setInterval(() => setD((x) => (x >= 12 ? (setPlaying(false), 12) : x + 1)), 700);
    return () => clearInterval(t);
  }, [playing, vis2]);
  if (!rec || !rec.walk.lens.length) return <Fig figRef={onScreen2} n="6.2" title="Watching the guess form" prov="recorded"><div className="lr-empty">loading…</div></Fig>;
  const { walk } = rec;
  const toks = rec.run.tokens.map((t) => t.id);
  const me = toks[walk.pos];
  const cell = walk.lens[d];
  const label = (i: number) => (i === 0 ? "embedding" : `after layer ${i}`);
  return (
    <Fig
      figRef={onScreen2}
      n="6.2"
      title="Watching the guess form"
      prov="recorded"
      caption={
        <>
          One real token, “<Tk s=" group" />”, travelling down GPT-2. Each stripe is the actual residual stream (768 numbers, ember
          positive, blue negative) at that depth: after the embedding, then after each of the 12 layers. Below: if the model
          stopped right there and turned the stream into a guess, these are the guesses it would make.
        </>
      }
      tryit={<>Press <b>send it down</b>. Early on it mostly guesses words that look like “group”; watch where “<Tk s=" of" />” takes over.</>}
    >
      <div className="lr-text" style={{ fontSize: "0.9rem", marginBottom: "0.8rem" }}>
        <span className="ctx">…{decodeTokens(toks.slice(Math.max(0, walk.pos - 12), walk.pos))}</span>
        <span style={{ outline: `2px solid ${pal.accent}`, borderRadius: 3, fontWeight: 700 }}>{tokenText(me)}</span>
        <span className="nv-muted"> → ?</span>
      </div>
      <div className="lr-controls">
        <button className="lr-btn" onClick={() => (d >= 12 && setD(0), setPlaying((p) => !p))}>
          {playing ? "❚❚ pause" : "▶ send it down"}
        </button>
        <input type="range" min={0} max={12} value={d} onChange={(e) => (setPlaying(false), setD(+e.target.value))} style={{ accentColor: pal.accent, width: "12rem" }} aria-label="depth" />
        <span className="lr-small">{label(d)}</span>
      </div>
      <div>
        <div style={{ display: "grid", gap: 3 }}>
          {walk.resid.map((v, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: "0.6rem", cursor: "pointer", opacity: i > d ? 0.25 : 1 }} onClick={() => (setPlaying(false), setD(i))}>
              <span className="lr-small" style={{ width: "4.6rem", textAlign: "right", color: i === d ? pal.accent : undefined, fontWeight: i === d ? 700 : 400 }}>
                {i === 0 ? "embed" : `L${i}`}
              </span>
              <Strip v={v} pal={pal} hi={i === d} />
            </div>
          ))}
        </div>
        <div style={{ marginTop: "1.2rem" }}>
          <div className="lr-small lr-guess-head">
            <span>if the model stopped {d === 0 ? "before any layer" : `after layer ${d}`}, it would say:</span>
            <SpaceKey />
          </div>
          <div className="lr-bars">
            {cell.top.map((t, i) => (
              <div key={t.id} className={`lr-bar${i === 0 && d === 12 ? " hi" : ""}`} style={{ gridTemplateColumns: "minmax(4rem,8rem) 1fr 3.4rem" }}>
                <span className="tk"><Tk s={tokenText(t.id)} /></span>
                <span className="b">
                  <i style={{ width: `${Math.max(1, t.p * 100)}%` }} />
                </span>
                <span className="pc">{(t.p * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
          <div className="lr-note">
            chance of the final answer “<Tk s={tokenText(walk.lens[12].top[0].id)} />”: <b>{(cell.pFinal * 100).toFixed(1)}%</b>
          </div>
        </div>
      </div>
    </Fig>
  );
}

export default function Block() {
  return (
    <>
      <p>
        We now have the pieces: embeddings (a token becomes 768 numbers), matrix multiplies (transform those numbers), and
        attention (mix in information from earlier tokens). A <b>transformer block</b> puts them together, and GPT-2 is
        just twelve blocks in a row: twelve <G k="layer">layers</G>.
      </p>
      <BlockDiagram />
      <p>
        The key design choice is the <b>add</b>. A block doesn't replace the token's vector, it adds a small edit to it. So
        the vector is like a shared notebook travelling down the model: each layer reads it, writes a note in the margin,
        and passes it on. That notebook is called the residual stream.
      </p>
      <Idea>The model is 12 identical-shaped blocks. Each reads the residual stream, computes an edit (attention, then an MLP), and adds it back. At the end, the stream is turned into a score for every token.</Idea>
      <p>Because each layer only adds, we can peek at the notebook halfway through and ask: what would you guess right now?</p>
      <Stream />
      <Deeper title="What turns the stream into a guess?">
        <p>
          After the last block, the stream is normalised and dot-producted with every row of the embedding table from chapter
          3: one score (a <b>logit</b>) per token, 50,257 of them. Softmax turns the scores into the probabilities you saw in
          chapter 1. The logit lens just does that same final step early.
        </p>
        <p>
          The engine runs 4 matrix multiplies per block: query, key and value fused into one, then attention's output, then
          the MLP's up and down. 12 blocks × 4 + the final scoring matrix = the 49 matrix multiplies in chapter 4's
          breakdown.
        </p>
      </Deeper>
    </>
  );
}
