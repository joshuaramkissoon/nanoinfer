import { useEffect, useMemo, useState } from "react";
import { Tk } from "../../ui/Tok";
import { claim, engine, release, tokenText, useEngine } from "../../engine/client";
import { LiveButton } from "../../ui/Figure";
import { css, rgb, usePalette, type Palette } from "../../ui/theme";
import { Deeper, Fig, G, Idea, Seg, Slider, Swatch, useRecorded, T } from "../kit";

// ---- toy -----------------------------------------------------------------------------

const WORDS = ["The", "cat", "sat", "on", "the", "mat", "because", "it", "was", "tired"];
const KEYS: [number, number][] = [
  [0.2, -0.9], [1, 0.3], [0.3, 0.9], [-0.6, 0.5], [0.2, -0.9], [0.75, -0.35], [-0.9, -0.2], [0.55, 0.1], [0.2, 0.8], [0.1, 0.7],
];
const QUERIES: Record<number, [number, number]> = {
  2: [0.95, 0.15], // sat: who sat?
  5: [-0.55, 0.65], // mat: on what?
  7: [1, 0.2], // it: which noun?
  9: [0.9, 0.4], // tired: who is?
};

function valueColor(p: Palette, i: number): [number, number, number] {
  if (i === 1) return rgb(p.accent);
  if (i === 5) return rgb(p.cool);
  if (i === 2 || i === 8) return rgb(p.down);
  return rgb(p.muted);
}

function ToyAttention() {
  const pal = usePalette();
  const [q, setQ] = useState(7);
  const [sharp, setSharp] = useState(3);
  const n = q + 1;
  const scores = KEYS.slice(0, n).map((k) => (k[0] * QUERIES[q][0] + k[1] * QUERIES[q][1]) * sharp);
  const mx = Math.max(...scores);
  const ex = scores.map((s) => Math.exp(s - mx));
  const z = ex.reduce((a, b) => a + b, 0);
  const wts = ex.map((e) => e / z);
  const mix = wts.reduce<[number, number, number]>((acc, w, i) => {
    const c = valueColor(pal, i);
    return [acc[0] + w * c[0], acc[1] + w * c[1], acc[2] + w * c[2]];
  }, [0, 0, 0]);
  const top = wts.indexOf(Math.max(...wts));
  const Wd = 560, x = (i: number) => 28 + i * ((Wd - 56) / (WORDS.length - 1));
  return (
    <Fig
      n="5.1"
      title="One query, looking back"
      prov="toy"
      caption={
        <>
          A toy sentence. The outlined token asks the question (its <b>query</b>). Each arc is an earlier token's share of
          the attention; thicker means more. The box at the bottom blends the tokens' <b>values</b>, drawn as colours. The
          vectors are hand-made with 2 numbers; GPT-2's are learned and 64 long.
        </>
      }
      tryit={
        <>
          Pick “it”, then drag <b>sharpness</b> (or use the arrow keys) from low to high. Notice how the arcs change at each
          end.
        </>
      }
    >
      <div className="lr-controls">
        <span className="lr-small">query token:</span>
        <Seg options={Object.keys(QUERIES).map((k) => ({ v: +k, label: WORDS[+k] }))} value={q} onChange={setQ} label="query token" />
        <Slider label="sharpness" value={sharp} min={0.2} max={8} step={0.1} onChange={setSharp} fmt={(v) => v.toFixed(1)} />
      </div>
      <div style={{ overflowX: "auto" }}>
        <svg viewBox={`0 0 ${Wd} 150`} style={{ width: "100%", minWidth: 460 }}>
          {wts.map((w, i) => {
            const x1 = x(q), x2 = x(i);
            const h = Math.min(90, 18 + Math.abs(x1 - x2) * 0.32);
            return i === q && n > 1 ? (
              <circle key={i} cx={x1} cy={84} r={14} fill="none" stroke={pal.accent} strokeOpacity={0.15 + w} strokeWidth={0.5 + w * 7} />
            ) : (
              <path
                key={i}
                d={`M${x1},96 C${x1},${96 - h} ${x2},${96 - h} ${x2},96`}
                fill="none"
                stroke={pal.accent}
                strokeOpacity={0.12 + w * 0.88}
                strokeWidth={0.5 + w * 9}
                strokeLinecap="round"
              />
            );
          })}
          {WORDS.map((wd, i) => (
            <g key={i} opacity={i > q ? 0.3 : 1}>
              <rect x={x(i) - 25} y={102} width={50} height={22} rx={5} fill={css(valueColor(pal, i))} fillOpacity={i <= q ? 0.25 : 0.1} stroke={i === q ? pal.fg : "none"} />
              <text x={x(i)} y={117} fontSize={wd.length > 5 ? T.xs : T.sm} textAnchor="middle" fill={pal.fg} fontWeight={i === q ? 700 : 400}>
                {wd}
              </text>
              {i <= q && (
                <text x={x(i)} y={140} fontSize={T.xs} textAnchor="middle" fill={i === top ? pal.fg : pal.muted} fontWeight={i === top ? 700 : 400}>
                  {(wts[i] * 100).toFixed(0)}%
                </text>
              )}
            </g>
          ))}
        </svg>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: "0.8rem", marginTop: "0.4rem", fontSize: "var(--t-sm)" }}>
        <span style={{ width: 44, height: 22, borderRadius: 5, background: css(mix), display: "inline-block", border: `1px solid ${pal.rule}` }} />
        <span>
          new vector for “{WORDS[q]}” = <b>{(wts[top] * 100).toFixed(0)}%</b> “{WORDS[top]}” + the rest
          {top === 1 && q === 7 && <span className="nv-muted"> · “it” found its cat</span>}
        </span>
      </div>
    </Fig>
  );
}

// ---- real ----------------------------------------------------------------------------

type Maps = { tokens: number[]; maps: Float32Array };

function presets(m: Maps, H: number) {
  const T = m.tokens.length, n = m.maps.length / (T * T);
  const score = (f: (h: number) => number) => {
    let best = 0, bs = -Infinity;
    for (let h = 0; h < n; h++) {
      const s = f(h);
      if (s > bs) [best, bs] = [h, s];
    }
    return [Math.floor(best / H), best % H] as [number, number];
  };
  const at = (h: number, i: number, j: number) => m.maps[h * T * T + i * T + j];
  return {
    prev: score((h) => {
      let s = 0;
      for (let i = 2; i < T; i++) s += at(h, i, i - 1);
      return s;
    }),
    first: score((h) => {
      let s = 0;
      for (let i = 4; i < T; i++) s += at(h, i, 0);
      return s;
    }),
    self: score((h) => {
      let s = 0;
      for (let i = 2; i < T; i++) s += at(h, i, i);
      return s;
    }),
    spread: score((h) => {
      let s = 0;
      for (let i = 8; i < T; i++) {
        let e = 0;
        for (let j = 0; j <= i; j++) {
          const p = at(h, i, j);
          if (p > 0) e -= p * Math.log(p);
        }
        s += e / Math.log(i + 1);
      }
      return s;
    }),
  };
}

function RealAttention() {
  const rec = useRecorded();
  const pal = usePalette();
  const ready = useEngine((s) => s.status === "ready");
  const H = 12;
  const [mine, setMine] = useState<Maps | null>(null);
  const [text, setText] = useState("The trophy didn't fit in the suitcase because it was too big.");
  const [busy, setBusy] = useState(false);
  const data: Maps | null = useMemo(() => mine ?? (rec ? { tokens: rec.attention.tokens, maps: rec.attention.maps } : null), [mine, rec]);
  const T = data?.tokens.length ?? 0;
  const pre = useMemo(() => (data ? presets(data, H) : null), [data]);
  const [lh, setLh] = useState<[number, number]>([4, 11]);
  const [mode, setMode] = useState<"prev" | "first" | "self" | "spread" | "custom">("prev");
  const [q, setQ] = useState<number | null>(null);
  useEffect(() => {
    if (pre && mode !== "custom") setLh(pre[mode]);
  }, [pre, mode]);
  useEffect(() => setQ(null), [data]);
  const qi = q ?? T - 1;
  const head = lh[0] * H + lh[1];
  const row = data ? data.maps.subarray(head * T * T + qi * T, head * T * T + qi * T + T) : null;
  let rmax = 0;
  if (row) for (let j = 0; j <= qi; j++) rmax = Math.max(rmax, row[j]);

  const runMine = async () => {
    if (!claim("learn-attn")) return;
    setBusy(true);
    try {
      const ids = (await engine.encode(text)).slice(0, 64);
      const maps = await engine.attention(ids);
      setMine({ tokens: ids, maps });
    } finally {
      release("learn-attn");
      setBusy(false);
    }
  };
  const accent = rgb(pal.accent);
  return (
    <Fig
      n="5.2"
      title="GPT-2's real attention"
      prov={mine ? "live" : "recorded"}
      aside={!ready ? <LiveButton label="try your own sentence" /> : undefined}
      caption={
        <>
          A real sentence and one of GPT-2's 144 attention heads (12 layers × 12 heads). The outlined token is the query.
          The <Swatch c={pal.accent}>ember</Swatch> shading on earlier tokens is how much attention this head gives each one. The
          presets jump to heads with easy-to-name patterns; most heads are harder to describe.
        </>
      }
      tryit={<>Select different tokens. Try “previous token”, then “first token”, then explore with the layer and head sliders. Notice how rare an easy-to-name pattern is.</>}
    >
      {ready && (
        <div className="lr-controls" style={{ flexWrap: "nowrap" }}>
          <input className="lr-input" value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && runMine()} aria-label="your sentence" />
          <button className="lr-btn solid" onClick={runMine} disabled={busy}>
            {busy ? "…" : "attend"}
          </button>
        </div>
      )}
      <div className="lr-controls">
        <Seg
          options={[
            { v: "prev", label: "previous token" },
            { v: "first", label: "first token" },
            { v: "self", label: "itself" },
            { v: "spread", label: "spreads out" },
          ]}
          value={mode as "prev"}
          onChange={setMode}
          label="head preset"
        />
        <Slider label="layer" value={lh[0]} min={0} max={11} onChange={(v) => (setMode("custom"), setLh([v, lh[1]]))} fmt={(v) => v + 1} width="5.5rem" />
        <Slider label="head" value={lh[1]} min={0} max={11} onChange={(v) => (setMode("custom"), setLh([lh[0], v]))} fmt={(v) => v + 1} width="5.5rem" />
      </div>
      <div style={{ lineHeight: 2.15, fontSize: "var(--t-md)" }}>
        {data?.tokens.map((id, j) => {
          const w = row && j <= qi ? row[j] / (rmax || 1) : 0;
          const isQ = j === qi;
          return (
            <span
              key={j}
              onClick={() => setQ(j)}
              title={row && j <= qi ? `${(row[j] * 100).toFixed(1)}%` : undefined}
              style={{
                cursor: "pointer",
                whiteSpace: "pre",
                borderRadius: 4,
                padding: "0.1rem 0",
                background: j <= qi && !isQ ? `rgba(${accent[0]},${accent[1]},${accent[2]},${(0.06 + 0.8 * Math.sqrt(w)).toFixed(3)})` : undefined,
                outline: isQ ? `2px solid ${pal.fg}` : undefined,
                color: j > qi ? pal.muted : pal.fg,
                fontWeight: isQ ? 700 : 400,
              }}
            >
              {tokenText(id)}
            </span>
          );
        })}
      </div>
      {row && (
        <div className="lr-note">
          layer {lh[0] + 1}, head {lh[1] + 1} · query “<Tk s={tokenText(data!.tokens[qi])} />” looks hardest at{" "}
          {(() => {
            let b = 0;
            for (let j = 0; j <= qi; j++) if (row[j] > row[b]) b = j;
            return (
              <b>
                “<Tk s={tokenText(data!.tokens[b])} />” ({(row[b] * 100).toFixed(0)}%)
              </b>
            );
          })()}
        </div>
      )}
    </Fig>
  );
}

export default function Attention() {
  return (
    <>
      <p>
        Each token's vector so far holds information about that token only. The word “it” needs the words before it.{" "}
        <G k="attention">Attention</G> is how a token gathers information from the tokens before it. It is built from the
        dot products of the last chapter.
      </p>
      <p>Each token makes three vectors from its embedding (by multiplying it with three learned matrices):</p>
      <ul className="lr-list">
        <li>
          a <b>query</b>: what am I looking for?
        </li>
        <li>
          a <b>key</b>: what do I contain, for anyone who's looking?
        </li>
        <li>
          a <b>value</b>: what I'll hand over if someone looks at me.
        </li>
      </ul>
      <p>
        A token takes the dot product of its query with every earlier token's key: a score for each. <G k="softmax" /> turns
        the scores into percentages. The token's new vector is the earlier tokens' values blended in those proportions.
      </p>
      <ToyAttention />
      <Idea>Attention is a soft lookup: score every earlier token with a dot product, turn the scores into percentages, blend the values.</Idea>
      <p>
        In the real model the vectors are learned. GPT-2 has 144 attention heads (12 per layer), and each one learns its
        own pattern.
      </p>
      <p>
        Now look at what “it” needed from the tokens before it: only their keys and values. Those don't change when a new
        token arrives, so there's no reason to compute them twice. Keeping them is the <b>KV cache</b>. It is per-request
        state that grows by one entry per token, and the first big speed-up in part 3 (chapter 12).
      </p>
      <Deeper title="See GPT-2's real attention heads">
        <RealAttention />
      </Deeper>
      <Deeper title="Does attention cost much?">
        <p>
          For one new token, attention is one dot product per earlier token and one blend. With a few dozen tokens of
          context that's tiny next to the big weight matrices. In the engine's measurements it was <b>0.7%</b> of a token's
          arithmetic. It grows with the length of the text, though, and at about 1,000 tokens it becomes a real cost.
        </p>
        <p>
          A token can only look backwards. Future tokens are hidden, because when the model is writing they don't exist yet.
          That one rule is what lets it write left to right.
        </p>
      </Deeper>
    </>
  );
}
