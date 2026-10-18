import { useMemo, useState } from "react";
import { SpaceKey, Tk } from "../../ui/Tok";
import { decodeTokens, engine, tokenText, useEngine } from "../../engine/client";
import { LiveButton } from "../../ui/Figure";
import { softmaxAt } from "../../lib/recorded";
import { usePalette } from "../../ui/theme";
import { log } from "../../lib/fmt";
import { Tip, useWidth } from "../../ui/Figure";
import { Deeper, Fig, G, Idea, Seg, liveForward, useRecorded } from "../kit";

type Dist = { id: number; p: number }[];

/** fig 1.1: the model's real guesses; click one to keep writing. Two sources, never mixed:
 *  the recorded example (step through a real run) or your own prompt (live, in this tab). */
function Guesser() {
  const rec = useRecorded();
  const status = useEngine((s) => s.status);
  const live = status === "ready";
  const [mode, setMode] = useState<"example" | "mine">("example");
  const [step, setStep] = useState(0);
  // your own prompt (live)
  const [prompt, setPrompt] = useState("The best way to learn something is to");
  const [ctx, setCtx] = useState<number[] | null>(null);
  const [nPrompt, setNPrompt] = useState(0);
  const [dist, setDist] = useState<Dist | null>(null);
  const [busy, setBusy] = useState(false);

  const recTokens = rec?.run.tokens ?? [];
  const firstGen = recTokens.findIndex((t) => !t.prompt);
  const genCount = firstGen < 0 ? 0 : recTokens.length - firstGen;

  const guess = async (ids: number[]) => {
    setBusy(true);
    try {
      const tr = await liveForward("learn-next", ids);
      if (tr) {
        setCtx(ids);
        setDist(tr.top);
      }
    } finally {
      setBusy(false);
    }
  };
  const start = async () => {
    if (!prompt.trim()) return;
    const ids = await engine.encode(prompt);
    setNPrompt(Math.min(ids.length, 200));
    await guess(ids.slice(-200));
  };
  const reset = () => {
    setMode("example");
    setCtx(null);
    setDist(null);
    setStep(0);
  };

  const mine = mode === "mine";
  const showingMine = mine && !!ctx && !!dist;
  let shownCtx: number[] = [], shownDist: Dist | null = null, chosen: number | null = null, nCtxPrompt = 0;
  if (showingMine) {
    shownCtx = ctx!;
    shownDist = dist;
    nCtxPrompt = nPrompt;
  } else if (!mine && rec && firstGen >= 0) {
    const i = firstGen + Math.min(step, genCount - 1);
    shownCtx = recTokens.slice(0, i).map((t) => t.id);
    shownDist = recTokens[i].top;
    chosen = recTokens[i].id;
    nCtxPrompt = firstGen;
  }
  const picked = shownCtx.length - nCtxPrompt;

  return (
    <Fig
      n="1.1"
      title="Guess the next word"
      prov={mine ? "live" : "recorded"}
      caption={
        mine ? (
          <>
            Your prompt (grey), any tokens you've picked (black), and below it GPT-2's ten most likely next tokens with their
            probabilities (out of all 50,257). Each guess is computed when you ask, in this tab.
          </>
        ) : (
          <>
            A real sentence GPT-2 wrote, replayed one token at a time. Grey is the prompt a person typed; black is what the model
            has added so far. The bars are the model's ten best guesses for the next token, and the highlighted bar is the one
            that was picked.
          </>
        )
      }
      tryit={
        mine ? (
          <>Click any bar to add that token and see the next guesses. Deliberately pick unlikely tokens and watch the story bend.</>
        ) : (
          <>Press <b>next word</b> a few times. Notice the model is often unsure (no bar near 100%), and the pick isn't always the top bar. Then switch to <b>your prompt</b>.</>
        )
      }
    >
      <div className="lr-guess-mode">
        <Seg
          label="what to continue"
          value={mode}
          onChange={(v) => setMode(v)}
          options={[
            { v: "example", label: "recorded example" },
            { v: "mine", label: "your prompt" },
          ]}
        />
        {mine && (ctx || picked > 0) && (
          <button className="lr-btn" onClick={reset} title="back to the recorded example">
            ⟲ reset
          </button>
        )}
      </div>
      {mine && !live ? (
        <div className="lr-empty lr-guess-load">
          <span>To continue your own text, GPT-2 has to run in this tab (it downloads once, then works offline).</span>
          <LiveButton label="load GPT-2" />
        </div>
      ) : mine ? (
        <form
          className="lr-controls lr-guess-input"
          onSubmit={(e) => {
            e.preventDefault();
            void start();
          }}
        >
          <input className="lr-input" value={prompt} onChange={(e) => setPrompt(e.target.value)} aria-label="your prompt" placeholder="Type the start of a sentence" spellCheck={false} />
          <button className="lr-btn solid" type="submit" disabled={busy || !prompt.trim()}>
            guess
          </button>
        </form>
      ) : null}
      {(!mine || showingMine) && (
        <div>
          <div className="lr-text lr-guess-text">
            <span className="ctx">{decodeTokens(shownCtx.slice(0, nCtxPrompt))}</span>
            <span>{decodeTokens(shownCtx.slice(nCtxPrompt))}</span>
            <span className="lr-cursor" />
          </div>
          <div style={{ marginTop: "1rem" }}>
            <div className="lr-small lr-guess-head">
              <span>{shownDist ? "top 10 guesses for the next token" : "…"}</span>
              <SpaceKey />
            </div>
            <div className="lr-bars">
              {shownDist?.map((d) => {
                const Tag = showingMine ? "button" : "div";
                return (
                  <Tag
                    key={d.id}
                    className={`lr-bar${d.id === chosen ? " hi" : ""}`}
                    onClick={showingMine && !busy ? () => guess([...shownCtx, d.id]) : undefined}
                    disabled={showingMine ? busy : undefined}
                    title={showingMine ? `add “${tokenText(d.id)}” and guess again` : undefined}
                  >
                    <span className="tk">
                      <Tk s={tokenText(d.id)} />
                    </span>
                    <span className="b">
                      <i style={{ width: `${Math.max(1, d.p * 100)}%` }} />
                    </span>
                    <span className="pc">{(d.p * 100).toFixed(1)}%</span>
                  </Tag>
                );
              })}
            </div>
          </div>
        </div>
      )}
      {mine && live && !showingMine && <div className="lr-empty">Type the start of a sentence and press guess: GPT-2 will score every possible next token.</div>}
      {!mine && (
        <div className="lr-controls" style={{ marginTop: "1rem", marginBottom: 0 }}>
          <button className="lr-btn" onClick={() => setStep(0)} disabled={step === 0}>
            ⟲ start
          </button>
          <button className="lr-btn solid" onClick={() => setStep((s) => Math.min(genCount - 1, s + 1))} disabled={step >= genCount - 1}>
            next word →
          </button>
          <span className="lr-small">
            word {step + 1} of {genCount || "…"}
          </span>
        </div>
      )}
      {showingMine && (
        <div className="lr-small" style={{ marginTop: "0.8rem" }}>
          {picked > 0 ? `${picked} token${picked === 1 ? "" : "s"} picked by you · ` : ""}live · computed in this tab
        </div>
      )}
    </Fig>
  );
}

/** fig 1.2: the full list is 50,257 long; most of it is nearly zero. */
function LongTail() {
  const rec = useRecorded();
  const pal = usePalette();
  const [ref, w] = useWidth<HTMLDivElement>();
  const probs = useMemo(() => {
    if (!rec) return null;
    const { probs, tailMass } = softmaxAt(rec.walk.logits, 1);
    return { probs, tailMass };
  }, [rec]);
  const [hover, setHover] = useState<number | null>(null);
  const H = 170, padL = 44, padB = 26;
  const W = Math.max(280, w);
  const x = log(1, 50257, padL, W - 8);
  const y = log(1e-6, 1, H - padB, 6);
  // the tail: spread the remaining mass over ranks 257..50257 using the histogram shape
  const tail = useMemo(() => {
    if (!rec || !probs) return [];
    const { counts, min, max } = rec.walk.logits.tail;
    const top0 = rec.walk.logits.top[0][1];
    const wbin = (max - min) / counts.length;
    // softmax normaliser from the top tokens + tail
    const z = probs.probs[0].p > 0 ? Math.exp(0) / probs.probs[0].p : 1;
    const pts: [number, number][] = [];
    let rank = probs.probs.length;
    for (let b = counts.length - 1; b >= 0; b--) {
      if (!counts[b]) continue;
      const p = Math.exp(min + (b + 0.5) * wbin - top0) / z;
      rank += counts[b];
      pts.push([rank, p]);
    }
    return pts;
  }, [rec, probs]);
  return (
    <Fig
      n="1.2"
      title="The whole list"
      prov="recorded"
      caption={
        <>
          All 50,257 probabilities from one real step, sorted from most to least likely. Across: the rank (1st, 10th,
          100th…). Up: the probability. Both axes are logarithmic, so each gridline is 100× the one below. The top 10 hold{" "}
          <b>{probs ? (probs.probs.slice(0, 10).reduce((a, b) => a + b.p, 0) * 100).toFixed(0) : "…"}%</b> of the total; the
          dotted line is the long tail of tokens that each get almost nothing.
        </>
      }
      tryit={<>Move along the dots to see which tokens they are.</>}
    >
      <div ref={ref} style={{ position: "relative" }}>
        {probs && w > 0 && (
          <svg
            width={W}
            height={H}
            onMouseLeave={() => setHover(null)}
            onPointerMove={(e) => {
              // nearest dot by rank: readers aim at a position, not at a 2px dot
              const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
              const rank = x.invert(e.clientX - r.left);
              const i = Math.round(rank) - 1;
              setHover(i >= 0 && i < probs.probs.length && rank < probs.probs.length * 1.15 ? i : null);
            }}
          >
            {[1e-6, 1e-4, 1e-2, 1].map((t) => (
              <g key={t}>
                <line x1={padL} x2={W - 8} y1={y(t)} y2={y(t)} stroke={pal.rule} />
                <text x={padL - 6} y={y(t) + 4} textAnchor="end" fontSize={10} fill={pal.muted}>
                  {t === 1 ? "100%" : t === 1e-2 ? "1%" : t === 1e-4 ? "0.01%" : "1e-6"}
                </text>
              </g>
            ))}
            {[1, 10, 100, 1000, 10000, 50257].map((r) => (
              <text key={r} x={x(r)} y={H - 8} textAnchor="middle" fontSize={10} fill={pal.muted}>
                {r === 50257 ? "50,257" : r.toLocaleString()}
              </text>
            ))}
            <polyline
              fill="none"
              stroke={pal.muted}
              strokeWidth={1.5}
              strokeDasharray="2 3"
              points={tail.map(([r, p]) => `${x(r)},${y(Math.max(1e-6, p))}`).join(" ")}
            />
            {probs.probs.map((d, i) => (
              <circle
                key={d.id}
                cx={x(i + 1)}
                cy={y(Math.max(1e-6, d.p))}
                r={hover === i ? 4 : i < 10 ? 2.6 : 1.4}
                fill={i < 10 ? pal.accent : pal.heat[3]}
              />
            ))}
            {hover != null && <line x1={x(hover + 1)} x2={x(hover + 1)} y1={6} y2={H - padB} stroke={pal.muted} strokeWidth={1} opacity={0.5} />}
            <text x={W - 8} y={14} textAnchor="end" fontSize={10} fill={pal.muted}>
              rank →
            </text>
          </svg>
        )}
        {probs && hover != null && (
          <Tip x={Math.min(Math.max(x(hover + 1), 90), W - 90)} y={y(Math.max(1e-6, probs.probs[hover].p)) - 6}>
            <b>{(probs.probs[hover].p * 100).toPrecision(2)}%</b> <Tk s={tokenText(probs.probs[hover].id)} />
            <span className="nv-muted"> · rank {hover + 1}</span>
          </Tip>
        )}
      </div>
    </Fig>
  );
}

export default function NextToken() {
  return (
    <>
      <p>
        Strip away the chat window and a <G k="language model" /> does exactly one thing. You hand it some text. It hands back a list
        of every piece of text that could come next, each with a <G k="probability" />.
      </p>
      <p>
        That's it. Answering questions, writing code, holding a conversation: all of it is that one trick, done over and
        over. The pieces of text are called <G k="token">tokens</G> (the next chapter is about them); for now, think of a
        token as a word.
      </p>
      <Guesser />
      <p>
        To write more than one word, you pick one of the guesses, stick it on the end, and ask again. The sentence above was
        written by doing that 32 times. Nobody planned the whole sentence. Each word was chosen knowing only the words
        before it.
      </p>
      <Idea>A language model is a function: text in, a probability for every possible next token out. Writing is just calling it in a loop.</Idea>
      <p>
        The list really is complete. GPT-2 knows <span className="k">50,257</span> tokens, and every single time it's asked,
        it scores all of them. Most get almost nothing, but the model still has to compute each one. Hold on to that: doing
        the same big computation for every single word is what part two of this course is about.
      </p>
      <Deeper title="See the whole list of 50,257">
        <LongTail />
      </Deeper>
      <Deeper title="Isn't this just autocomplete?">
        <p>
          Yes, in the same way a chess engine is "just" picking moves. The mechanism is simple; what's remarkable is how good
          the guesses get. To guess the next word of a physics paper well, it helps to know some physics. Training squeezes
          that kind of knowledge into the model's numbers, because it makes the guesses better.
        </p>
        <p>
          The model here is <b>GPT-2 small</b> (124 million numbers, OpenAI, 2019). It's tiny by today's standards, which is
          why it fits in a browser tab. The engine running it is nanoinfer, written from scratch in Rust. Part three of this
          course is the story of making that engine fast.
        </p>
      </Deeper>
    </>
  );
}
