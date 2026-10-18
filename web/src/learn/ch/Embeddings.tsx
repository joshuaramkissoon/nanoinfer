import { useEffect, useMemo, useRef, useState } from "react";
import { fitCanvas, useWidth } from "../../ui/Figure";
import { usePalette, type Palette } from "../../ui/theme";
import { visible } from "../bpe";
import { chartColors, Chip, Deeper, Fig, G, Idea, unpackVec, useLearnData, type LearnData } from "../kit";

const GROUP_LABEL: Record<string, string> = {
  numbers: "numbers",
  days: "days",
  countries: "countries",
  animals: "animals",
  colours: "colours",
  food: "food",
};

function Barcode({ v, pal }: { v: Float32Array; pal: Palette }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [box, w] = useWidth<HTMLDivElement>();
  useEffect(() => {
    const c = ref.current;
    if (!c || !w) return;
    const H = 44;
    const ctx = fitCanvas(c, w, H);
    ctx.clearRect(0, 0, w, H);
    let m = 0;
    for (const x of v) m = Math.max(m, Math.abs(x));
    const bw = w / v.length;
    for (let i = 0; i < v.length; i++) {
      const t = v[i] / (m || 1);
      const h = Math.max(1, Math.abs(t) * (H / 2 - 1));
      ctx.fillStyle = t >= 0 ? pal.accent : pal.cool;
      ctx.globalAlpha = 0.35 + 0.65 * Math.min(1, Math.abs(t) * 2);
      ctx.fillRect(i * bw, t >= 0 ? H / 2 - h : H / 2, Math.max(1, bw), h);
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = pal.rule;
    ctx.fillRect(0, H / 2, w, 1);
  }, [v, w, pal]);
  return (
    <div ref={box}>
      <canvas ref={ref} style={{ width: "100%", display: "block" }} />
    </div>
  );
}

function Explorer({ d }: { d: LearnData }) {
  const e = d.emb;
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [sel, setSel] = useState(() => Math.max(0, e.words.findIndex((id) => e.text[id] === " dog")));
  const [hover, setHover] = useState<number | null>(null);
  const [mapRef, mw] = useWidth<HTMLDivElement>();
  const vec = useMemo(() => unpackVec(e.vecs[sel]), [e, sel]);
  const groups = useMemo(() => [...new Set(e.group)], [e]);
  const W = Math.max(260, mw), H = Math.min(340, Math.max(240, W * 0.42));
  const px = (x: number) => 14 + x * (W - 28);
  const py = (y: number) => 12 + (1 - y) * (H - 24);
  const show = hover ?? sel;
  return (
    <Fig
      n="3.1"
      title="Words as points"
      prov="weights"
      caption={
        <>
          {e.words.length} words, each placed by its real row of numbers in GPT-2's table. 768 numbers can't be drawn, so the
          map squashes them down to 2: it's a rough sketch, but the topics (our labels; the model was never told them)
          clearly clump together. The selected word is ringed; its group is ember. Below: its 768 numbers as a barcode
          (ember up = positive, blue down = negative), and its exact nearest tokens out of ~30,000 whole words, measured in
          all 768 dimensions.
        </>
      }
      tryit={<>Click “Monday”, then a country, then a colour. Look at who the neighbours are.</>}
    >
      <div>
        <div ref={mapRef}>
          {mw > 0 && (
            <svg width={W} height={H} onMouseLeave={() => setHover(null)} role="img" aria-label="2D map of word embeddings">
              {e.near[sel].map(([id]) => {
                const j = e.words.indexOf(id);
                if (j < 0) return null;
                return <line key={id} x1={px(e.xy[sel][0])} y1={py(e.xy[sel][1])} x2={px(e.xy[j][0])} y2={py(e.xy[j][1])} stroke={cc.c1} strokeOpacity={0.35} />;
              })}
              {e.words.map((id, i) => {
                const [x, y] = e.xy[i];
                const on = i === sel, mine = e.group[i] === e.group[sel];
                return (
                  <g key={id} onMouseEnter={() => setHover(i)} onClick={() => setSel(i)} style={{ cursor: "pointer" }}>
                    <circle cx={px(x)} cy={py(y)} r={12} fill="transparent" />
                    <circle
                      cx={px(x)}
                      cy={py(y)}
                      r={on ? 6 : 4}
                      fill={mine ? cc.c1 : pal.muted}
                      opacity={mine ? 1 : 0.55}
                      stroke={on ? pal.fg : pal.dark ? "#211f1d" : "#fbfaf7"}
                      strokeWidth={2}
                    />
                  </g>
                );
              })}
              {(() => {
                const placed: [number, number][] = [];
                return groups.map((g) => {
                  const pts = e.xy.filter((_, i) => e.group[i] === g);
                  const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length;
                  const x = Math.min(W - 50, Math.max(40, px(cx)));
                  let y = py(Math.max(...pts.map((p) => p[1]))) - 12;
                  if (placed.some(([a, b]) => Math.abs(a - x) < 80 && Math.abs(b - y) < 16)) y = py(Math.min(...pts.map((p) => p[1]))) + 22;
                  placed.push([x, y]);
                  return (
                    <text key={g} x={x} y={y} textAnchor="middle" fontSize={11} fill={g === e.group[sel] ? pal.fg : pal.sub} fontWeight={g === e.group[sel] ? 700 : 400} style={{ pointerEvents: "none" }}>
                      {GROUP_LABEL[g] ?? g}
                    </text>
                  );
                });
              })()}
              {[show].map((i) => (
                <text key="lab" x={px(e.xy[i][0]) + 10} y={py(e.xy[i][1]) + 18} fontSize={12} fontWeight={700} fill={pal.fg} style={{ pointerEvents: "none" }}>
                  {e.text[e.words[i]].trim()}
                </text>
              ))}
            </svg>
          )}
          <div style={{ display: "grid", gap: "0.35rem", marginTop: "0.6rem" }}>
            {groups.map((g) => (
              <div key={g} style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "0.15rem 0.2rem" }}>
                <span className="lr-small" style={{ width: "5.5rem" }}>
                  {GROUP_LABEL[g] ?? g}
                </span>
                {e.words.map((id, i) =>
                  e.group[i] !== g ? null : (
                    <button key={id} className={`ew${i === sel ? " on" : ""}`} onClick={() => setSel(i)} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
                      {e.text[id].trim()}
                    </button>
                  ),
                )}
              </div>
            ))}
          </div>
        </div>
        <div style={{ marginTop: "1.4rem", paddingTop: "1.1rem", borderTop: `1px solid ${pal.rule}` }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "0.6rem", flexWrap: "wrap" }}>
            <Chip tone={0}>
              <span style={{ fontSize: "1.1rem", fontWeight: 700 }}>{visible(e.text[e.words[show]])}</span>
            </Chip>
            <span className="lr-small">
              token {e.words[show]} → row {e.words[show].toLocaleString()} of the table
            </span>
          </div>
          <div className="lr-small" style={{ margin: "0.9rem 0 0.3rem" }}>
            its 768 numbers <span style={{ color: pal.accent }}>▮ positive</span> <span style={{ color: pal.cool }}>▮ negative</span>
          </div>
          <Barcode v={hover != null ? unpackVec(e.vecs[hover]) : vec} pal={pal} />
          <div className="lr-small" style={{ margin: "1rem 0 0.35rem" }}>
            nearest tokens to “{e.text[e.words[sel]].trim()}” (cosine similarity, 1 = same direction)
          </div>
          <div className="lr-bars">
            {e.near[sel].map(([id, s]) => (
              <div key={id} className="lr-bar" style={{ gridTemplateColumns: "minmax(4rem,7rem) 1fr 3rem" }}>
                <span className="tk">{visible(e.text[id])}</span>
                <span className="b">
                  <i style={{ width: `${Math.max(2, s * 100)}%` }} />
                </span>
                <span className="pc">{s.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Fig>
  );
}

export default function Embeddings() {
  const d = useLearnData();
  return (
    <>
      <p>
        A token ID like <code>3290</code> (that's “·dog”) is just a row number. The model's first step is to look up that
        row in a big table and fetch what's there: a list of <span className="k">768</span> numbers. That list is the
        token's <G k="embedding" />, and from here on the model only ever works with lists of numbers like it, called{" "}
        <G k="vector">vectors</G>.
      </p>
      <p>
        The table has one row for each of the 50,257 tokens: 50,257 × 768 ≈ 38.6 million numbers. Nobody wrote those
        numbers. They started random and were nudged, during training, until they helped the model guess well. What they
        ended up encoding is surprisingly sensible:
      </p>
      {d ? (
        <Explorer d={d} />
      ) : (
        <Fig n="3.1" title="Words as points" prov="weights">
          <div className="lr-empty">loading…</div>
        </Fig>
      )}
      <p>
        Days sit with days, countries with countries. " Monday"'s nearest neighbours are the other six days. " dog" is
        next to " dogs", " puppy" and " canine". Nobody told the model about weekdays or pets; tokens that show up in
        similar places in text simply ended up with similar numbers.
      </p>
      <Idea>An embedding turns a token into a point in a 768-dimensional space, where similar meanings land close together. "Close" is measured with a dot product, the subject of the next chapter.</Idea>
      <Deeper title="How big is this table, and why does it matter for speed?">
        <p>
          38.6 million numbers at 4 bytes each is <b>154 MB</b>, about a third of GPT-2 small. GPT-2 uses the same table a
          second time at the very end: to score all 50,257 possible next tokens it compares its final vector with every row.
          So every single token the model writes has to read this whole table from memory. Keep that in mind for chapter 10,
          where reading memory turns out to be the speed limit.
        </p>
        <p>
          The first lookup itself is nearly free: one row, 768 numbers, 3 KB. In the engine's timings it takes about a
          microsecond.
        </p>
      </Deeper>
    </>
  );
}
