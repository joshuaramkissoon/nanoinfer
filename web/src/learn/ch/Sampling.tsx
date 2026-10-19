import { useMemo, useState } from "react";
import { SpaceKey, Tk } from "../../ui/Tok";
import { decodeTokens, tokenText } from "../../engine/client";
import { softmaxAt, type LogitSummary } from "../../lib/recorded";
import { usePalette } from "../../ui/theme";
import { Chip, Deeper, Fig, Idea, Seg, Slider, Stat, useRecorded } from "../kit";

/** Probabilities after temperature and top-k, plus entropy (in "effective choices"). */
function dist(s: LogitSummary, t: number, k: number) {
  const { probs, tailMass } = softmaxAt(s, t);
  const kk = k <= 0 ? probs.length : Math.min(k, probs.length);
  const keep = probs.slice(0, kk);
  const tail = k <= 0 ? tailMass + probs.slice(kk).reduce((a, b) => a + b.p, 0) : 0;
  const z = keep.reduce((a, b) => a + b.p, 0) + tail;
  const out = keep.map((p) => ({ ...p, p: p.p / z }));
  const tailP = tail / z;
  let H = 0;
  for (const p of out) if (p.p > 0) H -= p.p * Math.log(p.p);
  if (tailP > 0 && t > 0) {
    // the tail's tokens, spread as the histogram says
    const { counts, min, max } = s.tail;
    const w = (max - min) / counts.length;
    const top = s.top[0][1];
    let zt = 0;
    const ex = counts.map((n, i) => n * Math.exp((min + (i + 0.5) * w - top) / t));
    for (const e of ex) zt += e;
    counts.forEach((n, i) => {
      if (!n) return;
      const pi = (tailP * ex[i]) / zt / n;
      if (pi > 0) H -= n * pi * Math.log(pi);
    });
  }
  return { probs: out, tailP, eff: Math.exp(H) };
}

function Sampler() {
  const rec = useRecorded();
  const pal = usePalette();
  const [t, setT] = useState(0.8);
  const [k, setK] = useState(40);
  const [draws, setDraws] = useState<(number | "tail")[]>([]);
  const d = useMemo(() => (rec ? dist(rec.walk.logits, t, k) : null), [rec, t, k]);
  if (!rec || !d) return <Fig n="7.1" title="Temperature and top-k" prov="recorded"><div className="lr-empty">loading…</div></Fig>;
  const toks = rec.run.tokens.map((x) => x.id);
  const pos = rec.walk.pos;
  const draw = (n: number) => {
    const out: (number | "tail")[] = [];
    for (let i = 0; i < n; i++) {
      let r = Math.random(), pick: number | "tail" = "tail";
      for (const p of d.probs) {
        if ((r -= p.p) < 0) {
          pick = p.id;
          break;
        }
      }
      out.push(pick);
    }
    setDraws((x) => [...out, ...x].slice(0, 40));
  };
  const shown = d.probs.slice(0, 12);
  const pmax = Math.max(...shown.map((x) => x.p), 1e-9);
  return (
    <Fig
      n="7.1"
      title="Temperature and top-k"
      prov="recorded"
      caption={
        <>
          GPT-2's probabilities at one step of a real run, for the text shown. Bars: the 12 most likely next tokens, and the
          share left for everything else. The run used temperature 0.8 and top-k 40. “Effectively choosing among” is how many
          equally likely options this spread is worth.
        </>
      }
      tryit={
        <>
          Press <b>draw 20</b> at temperature 0.8, then at 2, then at 2 with top-k 5. Notice when rare tokens appear.
        </>
      }
    >
      <div className="lr-text" style={{ fontSize: "var(--t-md)", marginBottom: "0.6rem" }}>
        <span className="ctx">…{decodeTokens(toks.slice(Math.max(0, pos - 14), pos + 1))}</span>
        <span className="nv-accent"> ▍</span>
      </div>
      <div className="lr-controls">
        <Slider label="temperature" value={t} min={0} max={2} step={0.05} onChange={setT} fmt={(v) => (v === 0 ? "0 · greedy" : v.toFixed(2))} />
        <span className="lr-slider">
          <span className="k">top-k</span>
          <Seg
            options={[
              { v: 1, label: "1" },
              { v: 5, label: "5" },
              { v: 40, label: "40" },
              { v: 0, label: "all" },
            ]}
            value={k}
            onChange={setK}
            label="top-k"
          />
        </span>
      </div>
      <div>
        <div className="lr-keyrow">
          <SpaceKey />
        </div>
        <div className="lr-bars">
          {shown.map((p, i) => (
            <div key={p.id} className={`lr-bar${i === 0 ? " hi" : ""}`} style={{ opacity: p.p === 0 ? 0.35 : 1 }}>
              <span className="tk"><Tk s={tokenText(p.id)} /></span>
              <span className="b">
                <i style={{ width: `${(p.p / pmax) * 100}%` }} />
              </span>
              <span className="pc">{p.p >= 0.001 ? `${(p.p * 100).toFixed(1)}%` : p.p > 0 ? "<0.1%" : "cut"}</span>
            </div>
          ))}
          <div className="lr-bar" style={{ opacity: 0.8 }}>
            <span className="tk nv-muted">{k > 0 ? `the other ${(50257 - Math.min(k, 12)).toLocaleString()}` : "50,000+ others"}</span>
            <span className="b">
              <i style={{ width: `${Math.min(100, ((1 - shown.reduce((a, b) => a + b.p, 0)) / pmax) * 100)}%`, background: pal.muted }} />
            </span>
            <span className="pc">{((1 - shown.reduce((a, b) => a + b.p, 0)) * 100).toFixed(1)}%</span>
          </div>
        </div>
        <div style={{ marginTop: "1rem" }}>
          <div className="lr-stats" style={{ marginTop: 0 }}>
            <Stat k="top choice" v={`${(d.probs[0].p * 100).toFixed(0)}%`} accent />
            <Stat k="effectively choosing among" v={d.eff < 10 ? d.eff.toFixed(1) : d.eff.toFixed(0)} unit="tokens" />
          </div>
          <div className="lr-controls" style={{ marginTop: "1rem" }}>
            <button className="lr-btn solid" onClick={() => draw(1)}>
              draw 1
            </button>
            <button className="lr-btn" onClick={() => draw(20)}>
              draw 20
            </button>
            {draws.length > 0 && (
              <button className="lr-btn" onClick={() => setDraws([])}>
                clear
              </button>
            )}
          </div>
          <div style={{ lineHeight: 2 }}>
            {draws.map((id, i) => (
              <Chip key={draws.length - i} tone={id === d.probs[0].id ? 0 : id === "tail" ? 5 : 1}>
                {id === "tail" ? "(rare)" : <Tk s={tokenText(id)} />}
              </Chip>
            ))}
            {!draws.length && <span className="lr-small">draws appear here</span>}
          </div>
        </div>
      </div>
    </Fig>
  );
}

export default function Sampling() {
  return (
    <>
      <p>
        The model's job ends with a list of probabilities. Choosing a token from that list is a separate, tiny step called{" "}
        <b>sampling</b>, and it controls how predictable the text is.
      </p>
      <p>
        The obvious choice is to always take the top token (called <b>greedy</b>). It sounds safe, but it writes dull,
        repetitive text and often gets stuck in loops. Rolling a weighted die reads more naturally, but now and then picks
        something absurd from the long tail. Two settings steer between the two. <b>Temperature</b> reshapes the
        probabilities: low makes the favourite win more often, high flattens them. <b>Top-k</b> keeps only the k most likely
        tokens and drops the rest.
      </p>
      <Sampler />
      <Idea>Sampling is a cheap step after the expensive one. Temperature reshapes the probabilities; top-k cuts off the long tail; then you roll a die.</Idea>
      <Deeper title="Why this is free, in engine terms">
        <p>
          Producing the scores took a full forward pass: about 250 million multiply-adds and a read of about 500 MB of
          weights. Sampling from them is a few thousand operations. So the lab can re-sample a token at a different
          temperature instantly. The expensive part is done, and the scores are kept.
        </p>
        <p>
          With a temperature of 0 the output is deterministic, which is how the engine is tested: the same prompt must give
          token-for-token the same text as the reference implementation.
        </p>
      </Deeper>
    </>
  );
}
