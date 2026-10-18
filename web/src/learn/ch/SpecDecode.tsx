import { useEffect, useMemo, useState } from "react";
import { Tip, useWidth } from "../../ui/Figure";
import { usePalette } from "../../ui/theme";
import { chartColors, Deeper, Fig, G, Idea, Seg, nf, useLearnData, useOnScreen, type SpecPair, type SpecPrompt } from "../kit";

// Speculative decoding on the M4 GPU (bench/specdec.json via public/learn/data.json):
// a GPT-2 small drafter (4-bit) guessing for GPT-2 medium or large (16-bit).

const KIND: Record<string, string> = {
  license: "licence text",
  list: "a list",
  qa: "questions and answers",
  story: "a story",
  prose: "an essay",
  wikitext: "a Wikipedia article",
  code: "Python code",
};
const SHORT: Record<string, string> = { qa: "Q&A", wikitext: "Wikipedia", license: "licence", prose: "essay", story: "story", list: "list", code: "code" };
const ORDER = ["license", "list", "qa", "story", "prose", "wikitext", "code"];
const MODEL: Record<string, string> = { "gpt2-large": "GPT-2 large", "gpt2-medium": "GPT-2 medium" };
const PARAMS: Record<string, string> = { "gpt2-large": "774M", "gpt2-medium": "355M" };

interface Round {
  kept: string[];
  own: string;
  wrong: number;
}

/** Playback rows are [token, 1 = accepted guess | 0 = the big model's own token]; a round
 *  ends at each 0. Guesses after the first wrong one are thrown away (k - kept of them). */
function rounds(p: SpecPrompt, k = 4): Round[] {
  const out: Round[] = [];
  let kept: string[] = [];
  for (const [t, a] of p.playback) {
    if (a) kept.push(t);
    else {
      out.push({ kept, own: t, wrong: k - kept.length });
      kept = [];
    }
  }
  return out;
}

const show = (t: string) => t.replace(/\n/g, "↵\n");

function Playback({ pairs }: { pairs: SpecPair[] }) {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [target, setTarget] = useState(pairs[0].target);
  const [kind, setKind] = useState("license");
  const pair = pairs.find((p) => p.target === target) ?? pairs[0];
  const pr = pair.prompts.find((p) => p.prompt === kind) ?? pair.prompts[0];
  const rs = useMemo(() => rounds(pr), [pr]);
  const reduced = typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const [n, setN] = useState(reduced ? rs.length : 0);
  const [playing, setPlaying] = useState(!reduced);
  useEffect(() => {
    setN(reduced ? rs.length : 0);
    setPlaying(!reduced);
  }, [rs, reduced]);
  const [onScreen, vis] = useOnScreen<HTMLElement>();
  useEffect(() => {
    if (!playing || !vis) return;
    const t = setInterval(() => setN((v) => (v >= rs.length ? (setPlaying(false), v) : v + 1)), 520);
    return () => clearInterval(t);
  }, [playing, vis, rs.length]);
  const done = rs.slice(0, n);
  const toks = done.reduce((a, r) => a + r.kept.length + 1, 0);
  const cur = n > 0 ? rs[n - 1] : null;
  const box = (bg: string, line: string) => ({
    background: bg,
    boxShadow: `inset 0 -2px 0 ${line}`,
    color: pal.fg,
    borderRadius: 3,
    padding: "0.08rem 0",
    whiteSpace: "pre-wrap" as const,
    lineHeight: 1.9,
    boxDecorationBreak: "clone" as const,
    WebkitBoxDecorationBreak: "clone" as const,
  });
  const keptStyle = box(`color-mix(in srgb, ${cc.c2} 16%, transparent)`, `color-mix(in srgb, ${cc.c2} 70%, transparent)`);
  const ownStyle = box(`color-mix(in srgb, ${cc.c1} 22%, transparent)`, cc.c1);
  return (
    <Fig
      figRef={onScreen}
      n="18.1"
      title="Guess, then check: a real run"
      prov="measured"
      caption={
        <>
          A recorded run on the M4's GPU. GPT-2 small (4-bit) guesses 4 tokens at a time for {MODEL[pair.target]} ({PARAMS[pair.target]} weights,
          16-bit), which checks all 4 in one pass. <span style={keptStyle}>Blue</span>: a guess the big model agreed with, kept
          for free. <span style={ownStyle}>Ember</span>: the big model's own token, which every check adds: either the fix for the
          first wrong guess, or a bonus when all 4 were right. <span style={{ color: pal.muted }}>✕</span> marks each guess thrown away
          (the recording keeps how many, not what they were). The text is exactly what {MODEL[pair.target]} writes on its own.
        </>
      }
      tryit={<>Play the licence text, then an essay. Watch how many tokens each check adds.</>}
    >
      <div className="lr-controls">
        <Seg options={ORDER.filter((o) => pair.prompts.some((p) => p.prompt === o)).map((v) => ({ v, label: KIND[v] }))} value={kind} onChange={setKind} label="text" />
      </div>
      <div className="lr-controls">
        <Seg options={pairs.map((p) => ({ v: p.target, label: `checker: ${MODEL[p.target]}` }))} value={target} onChange={setTarget} label="big model" />
        <button className="lr-btn" onClick={() => (n >= rs.length ? (setN(0), setPlaying(true)) : setPlaying((p) => !p))}>
          {playing ? "❚❚ pause" : n >= rs.length ? "⟲ replay" : "▶ play"}
        </button>
        <button className="lr-btn" onClick={() => (setPlaying(false), setN((v) => Math.min(rs.length, v + 1)))} disabled={n >= rs.length}>
          one check →
        </button>
      </div>
      <div style={{ fontSize: "0.86rem", minHeight: "9.5rem", fontFamily: "var(--nv-mono, inherit)" }} aria-live="off">
        <span style={{ color: pal.muted, whiteSpace: "pre-wrap" }}>…{pr.text.slice(-70)}</span>
        {done.map((r, i) => (
          <span key={i}>
            {r.kept.length > 0 && <span style={keptStyle}>{show(r.kept.join(""))}</span>}
            {r.wrong > 0 && (
              <span style={{ color: pal.muted, fontSize: "0.7rem", letterSpacing: "-0.05em", margin: "0 0.1rem" }} aria-label={`${r.wrong} guesses thrown away`}>
                {"✕".repeat(r.wrong)}
              </span>
            )}
            <span style={ownStyle}>{show(r.own)}</span>
          </span>
        ))}
      </div>
      <div className="lr-note">
        {cur ? (
          <>
            This check: {cur.wrong ? `${cur.kept.length} of 4 guesses right, ${cur.wrong} thrown away` : "all 4 guesses right"}, plus the big model's own token
            ={" "}
            <b>
              {cur.kept.length + 1} token{cur.kept.length ? "s" : ""} for one pass
            </b>
            . So far: <b>{toks} tokens</b> in <b>{n} passes</b> of the big model, where writing alone would take {toks} passes.
          </>
        ) : (
          <>Press play.</>
        )}{" "}
        Over the whole run: {pr.k4.perForward.toFixed(1)} tokens per pass, {pr.k4.speedup.toFixed(1)}× faster ({nf(pr.k4.tokS)} vs {nf(pr.target)} tok/s
        alone).
      </div>
    </Fig>
  );
}

function ByText({ pairs }: { pairs: SpecPair[] }) {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [target, setTarget] = useState(pairs[0].target);
  const [hover, setHover] = useState<number | null>(null);
  const [ref, w] = useWidth<HTMLDivElement>();
  const pair = pairs.find((p) => p.target === target) ?? pairs[0];
  const rows = [...pair.prompts].sort((a, b) => b.best - a.best);
  const narrow = w > 0 && w < 480;
  const W = Math.max(300, w), labelW = narrow ? 76 : Math.min(150, W * 0.36), row = 30, bar = 14, top = 20;
  const H = top + rows.length * row + 30;
  const xmax = Math.ceil(Math.max(...pairs.flatMap((p) => p.prompts.map((q) => q.best))) * 2) / 2;
  const right = narrow ? 44 : 132;
  const x = (v: number) => labelW + 8 + ((v - 1) / (xmax - 1)) * (W - labelW - 8 - right);
  const ticks = Array.from({ length: Math.floor(xmax) }, (_, i) => i + 1);
  return (
    <Fig
      n="18.2"
      title="How much it helps depends on the text"
      prov="measured"
      caption={
        <>
          One row per kind of text, 128 tokens each, on the M4's GPU. The bar is how much faster {MODEL[pair.target]} writes with a
          GPT-2 small guesser than alone (best number of guesses per check, shown as k), so a bar at 1× means no help at all. On
          the right: how often a guess was right, with 4 guesses per check.
        </>
      }
      tryit={<>Switch the checker to GPT-2 medium: the same texts, smaller gains, because the guesser is now a bigger share of the work.</>}
    >
      <div className="lr-controls">
        <Seg options={pairs.map((p) => ({ v: p.target, label: MODEL[p.target] }))} value={target} onChange={setTarget} label="big model" />
      </div>
      <div ref={ref} style={{ position: "relative" }} onMouseLeave={() => setHover(null)}>
        {w > 0 && (
          <svg width={W} height={H} role="img" aria-label={`speed-up from speculative decoding by kind of text, ${MODEL[pair.target]}`}>
            {ticks.map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={top} y2={H - 26} stroke={pal.rule} />
                <text x={x(t)} y={H - 12} textAnchor="middle" fontSize={10} fill={pal.muted}>
                  {t}×
                </text>
              </g>
            ))}
            <text x={W - 2} y={11} textAnchor="end" fontSize={10} fill={pal.muted}>
              {narrow ? "right" : "guesses right"}
            </text>
            <text x={x(1)} y={H - 1} fontSize={10} fill={pal.muted}>
              faster than the big model alone →
            </text>
            {rows.map((r, i) => {
              const y0 = top + i * row + (row - bar) / 2;
              const on = hover === i;
              const x1 = x(r.best);
              const x0 = x(1);
              return (
                <g key={r.prompt} onMouseEnter={() => setHover(i)} onClick={() => setHover(i)} opacity={hover != null && !on ? 0.6 : 1}>
                  <rect x={0} y={top + i * row} width={W} height={row} fill="transparent" />
                  <text x={labelW} y={y0 + bar / 2 + 4} textAnchor="end" fontSize={11.5} fill={on ? pal.fg : pal.sub}>
                    {(narrow ? SHORT : KIND)[r.prompt] ?? r.prompt}
                  </text>
                  <path d={`M${x0},${y0} H${x1 - 4} a4,4 0 0 1 4,4 V${y0 + bar - 4} a4,4 0 0 1 -4,4 H${x0} Z`} fill={cc.c1} />
                  <text x={x1 + 6} y={y0 + bar / 2 + 4} fontSize={11} fill={pal.fg} fontWeight={700}>
                    {r.best.toFixed(1)}×
                  </text>
                  <text x={W - 2} y={y0 + bar / 2 + 4} textAnchor="end" fontSize={11} fill={pal.sub}>
                    {Math.round(r.alpha * 100)}%
                  </text>
                </g>
              );
            })}
          </svg>
        )}
        {hover != null && rows[hover] && (
          <Tip x={Math.min(Math.max(x(rows[hover].best), 120), W - 120)} y={top + hover * row + 4}>
            <b>{KIND[rows[hover].prompt]}</b>
            <span className="nv-muted"> · {MODEL[pair.target]}</span>
            <br />
            {nf(rows[hover].target)} → <b>{nf(rows[hover].bestTokS)}</b> tok/s with k = {rows[hover].bestK} guesses
            <br />
            {Math.round(rows[hover].alpha * 100)}% of guesses right (k = 4)
          </Tip>
        )}
      </div>
    </Fig>
  );
}

export default function SpecDecode() {
  const d = useLearnData();
  const pairs = d ? ["gpt2-large", "gpt2-medium"].map((t) => d.spec[t]).filter(Boolean) : [];
  const large = pairs[0];
  const top = large ? large.prompts.reduce((a, b) => (b.best > a.best ? b : a)) : null;
  const med = pairs[1];
  const medAlone = med ? Math.max(...med.prompts.map((p) => p.target)) : 0;
  const draftShare = large ? large.prompts.reduce((a, p) => a + p.c, 0) / large.prompts.length : 0;
  const lo = large ? large.prompts.reduce((a, b) => (b.alpha < a.alpha ? b : a)) : null;
  const medCode = med?.prompts.find((p) => p.prompt === "code");
  const lgCode = large?.prompts.find((p) => p.prompt === "code");
  return (
    <>
      <h2 style={{ marginTop: "2.2rem" }}>The bottleneck</h2>
      <p>
        Batching helps a server with many users. It does nothing for <i>one</i> user waiting on a big model: each token
        needs the one before it, so the tokens of a single answer can't be put through the model together. Or can they?
      </p>
      <p>
        Chapter 17 showed the useful fact. While decode is waiting on memory, pushing 5 tokens through the model costs about
        the same as pushing 1: the weights are read once either way. The trouble is only that we don't know the next 5 tokens
        yet.
      </p>
      <h2>The idea: let a small model guess</h2>
      <p>
        Most of the time, the next few tokens are easy: the end of a word, a comma, “the”. A model ten times smaller gets
        those right too. So let the small model, the <b>drafter</b>, quickly guess the next 4 tokens. Then run the big model
        once over all 4 guesses. In that one pass it says what <i>it</i> would have written at every position. Keep the
        guesses up to the first one it disagrees with, and add the big model's own token at that point. This is{" "}
        <G k="speculative decoding" />.
      </p>
      <p>
        Every check writes at least one token (the big model's own), so it is never slower in tokens per pass, and the text
        comes out exactly as the big model alone would have written it, word for word. If 3 guesses are right, one pass
        writes 4 tokens.
      </p>
      {pairs.length ? <Playback pairs={pairs} /> : <div className="lr-empty">loading…</div>}
      <h2>The result</h2>
      {large && top && (
        <p>
          With GPT-2 large checking (774 million weights, 16-bit), and GPT-2 small guessing, the best case went from{" "}
          {nf(top.target)} to <span className="k">{nf(top.bestTokS)} tokens per second, {top.best.toFixed(2)}× faster</span>. That's
          faster than GPT-2 medium, a model half the size, manages on its own ({nf(medAlone)}). The drafter costs little: one of
          its steps takes about {Math.round(draftShare * 100)}% of the big model's.
        </p>
      )}
      <p>But that best case was a licence. How often the guesses are right depends on what is being written:</p>
      {pairs.length ? <ByText pairs={pairs} /> : null}
      {lo && (
        <p>
          Boilerplate and lists, where the next word is nearly forced, get almost every guess right. Stories and essays, with
          more real choices, get fewer. Code was the odd one: GPT-2 medium agreed with the small model's code{" "}
          {medCode ? `${Math.round(medCode.alpha * 100)}%` : ""} of the time, GPT-2 large only{" "}
          {lgCode ? `${Math.round(lgCode.alpha * 100)}%` : ""}. Large simply writes different code from small. How well the trick
          works is a property of the two models <i>and</i> the text, which is why it is measured, not assumed.
        </p>
      )}
      <Idea>A cheap guesser lets one pass of the big model write several tokens, without changing a word of its output. It pays off exactly as often as the small model can predict the big one.</Idea>
      <Deeper title="How many guesses per check?">
        <p>
          More guesses means more tokens per pass when they're right, but the drafter spends longer guessing and the check gets a
          little more expensive. If a fraction α of guesses are right, one pass writes on average (1 − α<sup>k+1</sup>) ÷ (1 − α)
          tokens with k guesses, and the measured tokens per pass match that closely.
        </p>
        {large && (
          <p>
            The best k was {Math.max(...large.prompts.map((p) => p.bestK))} where nearly every guess is right, and{" "}
            {Math.min(...large.prompts.map((p) => p.bestK))}–6 elsewhere. For GPT-2 medium, checking 8 tokens took only 27% longer than checking 1;
            the 9th makes the GPU switch to a bigger, slower kernel, which is where the gains stop.
          </p>
        )}
      </Deeper>
      <Deeper title="A caveat: GPT-2 repeats itself">
        <p>
          Asked for the most likely next token every time, GPT-2 falls into loops after a few dozen tokens: the story in fig 18.1
          keeps saying she was a very good cook. Repetition is easy to guess, so every acceptance rate here is higher than it would
          be for a modern chat model writing something new. The mechanism and the measurement are real; treat the bigger
          speed-ups as a best case.
        </p>
      </Deeper>
    </>
  );
}
