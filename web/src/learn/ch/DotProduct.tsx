import { useMemo, useRef, useState } from "react";
import { Tk } from "../../ui/Tok";
import { useWidth } from "../../ui/Figure";
import { KIND_LABEL, css, heatAt, kindColor, usePalette, type Palette } from "../../ui/theme";
import { Deeper, Fig, G, Idea, Slider, nf, unpackVec, useLearnData, useRecorded, useFrame, useOnScreen, T, PlayPause, usePlaying, Swatch } from "../kit";

/** fig 4.1: drag two arrows, read their dot product */
function DotToy() {
  const pal = usePalette();
  const [a, setA] = useState<[number, number]>([2.2, 1.2]);
  const [b, setB] = useState<[number, number]>([1.4, 2.4]);
  const [drag, setDrag] = useState<"a" | "b" | null>(null);
  const svg = useRef<SVGSVGElement>(null);
  const S = 240, R = 3.2, k = S / 2 / R;
  const toPx = ([x, y]: [number, number]) => [S / 2 + x * k, S / 2 - y * k];
  const move = (e: React.PointerEvent) => {
    if (!drag || !svg.current) return;
    const r = svg.current.getBoundingClientRect();
    const x = Math.max(-R, Math.min(R, ((e.clientX - r.left) * (S / r.width) - S / 2) / k));
    const y = Math.max(-R, Math.min(R, (S / 2 - (e.clientY - r.top) * (S / r.height)) / k));
    (drag === "a" ? setA : setB)([+x.toFixed(1), +y.toFixed(1)]);
  };
  const dot = a[0] * b[0] + a[1] * b[1];
  const max = R * R * 2;
  const ang = (Math.acos(Math.max(-1, Math.min(1, dot / (Math.hypot(...a) * Math.hypot(...b) || 1)))) * 180) / Math.PI;
  const arrow = (v: [number, number], c: string, id: "a" | "b") => {
    const [x, y] = toPx(v);
    return (
      <g>
        <line x1={S / 2} y1={S / 2} x2={x} y2={y} stroke={c} strokeWidth={2.5} strokeLinecap="round" />
        <circle
          cx={x}
          cy={y}
          r={drag === id ? 10 : 8}
          fill={c}
          fillOpacity={0.25}
          stroke={c}
          strokeWidth={2}
          style={{ cursor: "grab", touchAction: "none" }}
          onPointerDown={(e) => {
            (e.target as Element).setPointerCapture(e.pointerId);
            setDrag(id);
          }}
        />
        <text x={x + 12} y={y - 8} fontSize={T.sm} fontWeight={700} fill={c}>
          {id}
        </text>
      </g>
    );
  };
  return (
    <Fig
      n="4.2"
      title="The dot product"
      prov="toy"
      caption={
        <>
          Two vectors of 2 numbers each, drawn as arrows. Their dot product is on the right: multiply matching numbers, then
          add. The neighbours in chapter 3 come from this, with 768 numbers instead of 2.
        </>
      }
      tryit={<>Drag the arrow heads. Make the dot product zero, then as negative as you can. Notice the angle each time.</>}
    >
      <div style={{ display: "flex", flexWrap: "wrap", gap: "1.2rem 2rem", alignItems: "center" }}>
        <svg ref={svg} viewBox={`0 0 ${S} ${S}`} style={{ width: "100%", maxWidth: 260, flex: "1 1 220px" }} onPointerMove={move} onPointerUp={() => setDrag(null)}>
          {[-3, -2, -1, 1, 2, 3].map((t) => (
            <g key={t} stroke={pal.rule}>
              <line x1={S / 2 + t * k} x2={S / 2 + t * k} y1={0} y2={S} />
              <line y1={S / 2 + t * k} y2={S / 2 + t * k} x1={0} x2={S} />
            </g>
          ))}
          <line x1={0} x2={S} y1={S / 2} y2={S / 2} stroke={pal.muted} />
          <line y1={0} y2={S} x1={S / 2} x2={S / 2} stroke={pal.muted} />
          {arrow(a, pal.accent, "a")}
          {arrow(b, pal.cool, "b")}
        </svg>
        <div style={{ fontSize: "var(--t-md)", lineHeight: 2, flex: "1 1 220px" }}>
          <div>
            <Swatch c={pal.accent}>a</Swatch> = [{a[0].toFixed(1)}, {a[1].toFixed(1)}]
          </div>
          <div>
            <Swatch c={pal.cool}>b</Swatch> = [{b[0].toFixed(1)}, {b[1].toFixed(1)}]
          </div>
          <div style={{ marginTop: "0.6rem" }}>
            a·b = {a[0].toFixed(1)}×{b[0].toFixed(1)} + {a[1].toFixed(1)}×{b[1].toFixed(1)}
          </div>
          <div style={{ fontSize: "var(--t-xl)", fontWeight: 700 }}>= {dot.toFixed(2)}</div>
          <svg width="100%" height={18} viewBox="0 0 200 18" preserveAspectRatio="none">
            <rect x={0} y={7} width={200} height={4} fill={pal.rule} rx={2} />
            <rect x={dot >= 0 ? 100 : 100 + (dot / max) * 100} y={5} width={Math.abs(dot / max) * 100} height={8} fill={dot >= 0 ? pal.accent : pal.cool} rx={2} />
            <line x1={100} x2={100} y1={0} y2={18} stroke={pal.muted} />
          </svg>
          <div className="lr-small">angle between them: {ang.toFixed(0)}°</div>
        </div>
      </div>
    </Fig>
  );
}

function diverge(p: Palette, t: number): string {
  // -1..1 -> cool .. bg .. accent
  const bg = p.dark ? [41, 39, 36] : [233, 229, 220];
  const pos = heatAt(p, 0.85), neg = p.dark ? [127, 176, 207] : [60, 110, 143];
  const c = t >= 0 ? pos : neg;
  const k = Math.min(1, Math.abs(t));
  return css([bg[0] + (c[0] - bg[0]) * k, bg[1] + (c[1] - bg[1]) * k, bg[2] + (c[2] - bg[2]) * k]);
}

/** fig 4.2: a real corner of GPT-2's first weight matrix times a real embedding */
function MatVec() {
  const d = useLearnData();
  const pal = usePalette();
  const [t, setT] = useState(0);
  const [speed, setSpeed] = useState(4);
  const [playing, setPlaying] = usePlaying();
  const N = 8;
  const W = d?.corner ?? null;
  const x = useMemo(() => {
    if (!d) return null;
    const i = d.emb.words.findIndex((id) => d.emb.text[id] === " dog");
    return Array.from(unpackVec(d.emb.vecs[Math.max(0, i)]).slice(0, N));
  }, [d]);
  const [onScreen, vis] = useOnScreen<HTMLElement>();
  useFrame(vis && playing && !!W, (dt) => setT((v) => (v + dt * speed) % (N * N + 6)));
  if (!W || !x) return <Fig figRef={onScreen} n="4.1" title="A matrix times a vector" prov="weights"><div className="lr-empty">loading…</div></Fig>;
  const step = Math.min(N * N, Math.floor(t));
  const row = Math.min(N - 1, Math.floor(step / N)), col = step % N;
  const y = W.map((r, i) => {
    const upto = i < row || step >= N * N ? N : i === row ? col : 0;
    let s = 0;
    for (let j = 0; j < upto; j++) s += r[j] * x[j];
    return { s, done: upto === N, started: upto > 0 };
  });
  const wmax = Math.max(...W.flat().map(Math.abs)), xmax = Math.max(...x.map(Math.abs));
  const ymax = Math.max(...W.map((r) => Math.abs(r.reduce((a, v, j) => a + v * x[j], 0))));
  const C = 38, G = 3, ox = 4, oy = 46;
  const fin = step >= N * N;
  const eqRow = fin || !Number.isFinite(row) ? N - 1 : row;
  const eqUpto = fin || !Number.isFinite(col) ? N : col + 1; // terms of the current row shown so far (the active one included)
  const posC = diverge(pal, 1), negC = diverge(pal, -1);
  return (
    <Fig
      figRef={onScreen}
      n="4.1"
      title="A matrix times a vector"
      prov="weights"
      caption={
        <>
          Top row: the first 8 of the 768 numbers for <Tk s=" dog" />. The grid: the top-left 8 × 8 corner of GPT-2's first
          weight matrix, coloured by sign (<Swatch c={posC}>ember</Swatch> positive, <Swatch c={negC}>blue</Swatch>
          negative). The highlighted row is multiplied with the vector one term at a time, and the sum lands in the output
          column.
        </>
      }
      tryit={
        <>
          Watch one row finish, then count the multiply-adds in this corner. Notice how that count scales to the full
          768 × 2304 matrix.
        </>
      }
    >
      <div className="lr-controls">
        <PlayPause playing={playing} onToggle={() => setPlaying((p) => !p)} />
        <Slider label="speed" value={speed} min={1} max={16} onChange={setSpeed} fmt={(v) => `${v}/s`} />
        <span className="lr-small">
          multiply-adds: <b style={{ color: pal.fg }}>{fin ? 64 : step}</b> / 64
        </span>
      </div>
      <div className="p0-legend lr-small">
        <span>
          <i style={{ background: posC }} />
          positive
        </span>
        <span>
          <i style={{ background: negC }} />
          negative
        </span>
        <span style={{ color: pal.muted }}>faint = near zero</span>
      </div>
      <div style={{ overflowX: "auto" }}>
        <svg width={ox + N * (C + G) + 130} height={oy + N * (C + G) + 6} style={{ maxWidth: "100%", display: "block", margin: "0 auto" }} viewBox={`0 0 ${ox + N * (C + G) + 130} ${oy + N * (C + G) + 6}`}>
          {/* x on top */}
          <text x={ox} y={11} fontSize={T.xs} fill={pal.sub}>
            vector x: “␣dog”, first 8 of its 768 numbers
          </text>
          {x.map((v, j) => (
            <g key={j}>
              <rect x={ox + j * (C + G)} y={18} width={C} height={22} rx={3} fill={diverge(pal, v / xmax)} stroke={!fin && j === col ? pal.fg : "none"} strokeWidth={1.5} />
              <text x={ox + j * (C + G) + C / 2} y={33} fontSize={T.xs} textAnchor="middle" fill={pal.fg}>
                {v.toFixed(2)}
              </text>
            </g>
          ))}
          {W.map((r, i) =>
            r.map((v, j) => {
              const active = !fin && i === row && j === col;
              const doneCell = fin || i < row || (i === row && j < col);
              return (
                <g key={`${i}-${j}`}>
                  <rect
                    x={ox + j * (C + G)}
                    y={oy + i * (C + G)}
                    width={C}
                    height={C}
                    rx={3}
                    fill={diverge(pal, v / wmax)}
                    opacity={!fin && i !== row ? 0.45 : 1}
                    stroke={active ? pal.fg : "none"}
                    strokeWidth={2}
                  />
                  {(active || (i === row && !fin)) && (
                    <text x={ox + j * (C + G) + C / 2} y={oy + i * (C + G) + C / 2 + 4} fontSize={T.xs} textAnchor="middle" fill={pal.fg} opacity={doneCell || active ? 1 : 0.6}>
                      {v.toFixed(2)}
                    </text>
                  )}
                </g>
              );
            }),
          )}
          {/* output column */}
          <text x={ox + N * (C + G) + 18} y={oy - 6} fontSize={T.xs} fill={pal.sub}>
            output
          </text>
          {y.map((o, i) => (
            <g key={i}>
              <text x={ox + N * (C + G) + 4} y={oy + i * (C + G) + C / 2 + 4} fontSize={T.xs} fill={pal.muted} opacity={o.started ? 1 : 0}>
                =
              </text>
              <rect
                x={ox + N * (C + G) + 18}
                y={oy + i * (C + G)}
                width={o.started ? Math.max(3, (Math.abs(o.s) / ymax) * 60) : 0}
                height={C}
                rx={3}
                fill={o.s >= 0 ? pal.accent : pal.cool}
                opacity={o.done ? 0.9 : 0.45}
              />
              {o.started && (
                <text x={ox + N * (C + G) + 84} y={oy + i * (C + G) + C / 2 + 4} fontSize={T.xs} fill={pal.fg}>
                  {o.s.toFixed(3)}
                </text>
              )}
            </g>
          ))}
        </svg>
      </div>
      <p className="lr-note lr-eq" aria-live="off">
        row {eqRow + 1} · x ={" "}
        {Array.from({ length: eqUpto }, (_, j) => (
          <span key={j} style={{ color: !fin && j === eqUpto - 1 ? pal.fg : undefined, fontWeight: !fin && j === eqUpto - 1 ? 700 : undefined }}>
            {j ? " + " : ""}
            {W[eqRow][j].toFixed(2)}×{x[j].toFixed(2)}
          </span>
        ))}
        {eqUpto < N ? " + …" : ""} = <b style={{ color: pal.fg }}>{W[eqRow].slice(0, eqUpto).reduce((a, v, j) => a + v * x[j], 0).toFixed(3)}</b>
      </p>
    </Fig>
  );
}

/** fig 4.3: where one real token's arithmetic goes */
function WorkShare() {
  const rec = useRecorded();
  const pal = usePalette();
  const [ref, w] = useWidth<HTMLDivElement>();
  const f = rec?.run.forwards.find((x) => x.phase === "decode") ?? null;
  const rows = useMemo(() => {
    if (!f) return [];
    const m = new Map<string, number>();
    for (const o of f.ops) m.set(o.kind, (m.get(o.kind) ?? 0) + o.flops);
    return [...m.entries()].filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]);
  }, [f]);
  const total = rows.reduce((a, [, v]) => a + v, 0);
  let acc = 0;
  return (
    <Fig
      n="4.3"
      title="Where one token's arithmetic goes"
      prov="recorded"
      caption={
        <>
          Every operation the engine ran to produce one real token, grouped by kind and sized by its arithmetic. The strip on
          top shows the same shares as one bar. “matmul” is fig 4.1's matrix-times-vector, 48 of them. “lm head” is the
          LM head, the final matrix that scores all 50,257 tokens.
        </>
      }
    >
      <div ref={ref}>
        {f && w > 0 && (
          <>
            <svg width={w} height={34}>
              {rows.map(([k, v]) => {
                const x0 = (acc / total) * w;
                acc += v;
                return <rect key={k} x={x0} y={4} width={Math.max(1, (v / total) * w - 1)} height={26} fill={kindColor(pal, k)} rx={2} />;
              })}
            </svg>
            <div className="lr-bars" style={{ marginTop: "0.6rem" }}>
              {rows.map(([k, v]) => (
                <div key={k} className="lr-bar" style={{ gridTemplateColumns: "7rem 1fr 7.5rem" }}>
                  <span className="tk">
                    <i style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: kindColor(pal, k), marginRight: 6 }} />
                    {KIND_LABEL[k] ?? k}
                  </span>
                  <span className="b">
                    <i style={{ width: `${Math.max(0.5, (v / total) * 100)}%`, background: kindColor(pal, k) }} />
                  </span>
                  <span className="pc">
                    {((v / total) * 100).toFixed(v / total < 0.01 ? 2 : 1)}% · {nf(v / 1e6, 1)}M
                  </span>
                </div>
              ))}
            </div>
            <div className="lr-note">
              total: <b>{nf(total / 1e6, 0)} million</b> operations for one token, about two per weight in the model (124M).
            </div>
          </>
        )}
      </div>
    </Fig>
  );
}

export default function DotProduct() {
  return (
    <>
      <p>
        One operation does nearly all of a model's work. Take two lists of numbers of the same length, multiply them pair by
        pair, and add up the results. That's a <G k="dot product" />:
      </p>
      <code className="lr-math">[1, 2, 3] · [4, 5, 6]  =  1×4 + 2×5 + 3×6  =  32</code>
      <p>
        Now stack many lists into a grid, a <G k="matrix" />, and take the dot product of every row with the same vector.
        You get a new vector, one number per row. That's a <b>matrix-vector multiply</b>. It's how the model transforms a
        token's 768 numbers into something new. Each row of the matrix asks one question of the input, and the answer is a
        single number. Training learned the rows; they are the model's <G k="weights" />. One 768 × 2304 matrix is{" "}
        1.77 million multiply-adds for one token, and GPT-2 has 49 matrices.
      </p>
      <MatVec />
      <p>
        A model is mostly this, with big matrices, many times over. The engine counted every operation it ran for one real
        token. Matrix-times-vector came to <span className="k">99%</span> of the arithmetic, about 250 million multiplies and
        adds, roughly two for every weight. Everything else (attention and the steps in between) is the last 1%.
      </p>
      <Idea>Running a model is overwhelmingly multiply-and-add: matrices times vectors. Part 2 shows that the cost is reading those matrices, not the arithmetic.</Idea>
      <Deeper title="What does a dot product measure? (drag some arrows)">
        <DotToy />
      </Deeper>
      <Deeper title="See the engine's breakdown of one token">
        <WorkShare />
      </Deeper>
      <Deeper title="Two operations per weight, once per token">
        <p>
          Each weight is used once per token. It is multiplied by one input number, and the result is added to a running
          total. That's 2 operations (<G k="FLOP">FLOPs</G>) per weight. So a 124M-weight model needs about 250 million FLOPs
          per token, and a 7-billion-weight model about 14 billion.
        </p>
        <p>
          “Once per token” is the important part. Each weight is fetched from memory, used for one multiply-add, and not
          needed again until the next token. Part 2 shows why that, and not the arithmetic, sets the speed.
        </p>
      </Deeper>
    </>
  );
}
