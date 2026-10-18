import { useEffect, useMemo, useState } from "react";
import { SpaceKey, Tk } from "../../ui/Tok";
import { engine, useEngine } from "../../engine/client";
import { loadTokenizer, type Tokenizer } from "../bpe";
import { Chip, Deeper, Fig, G, Idea, Stat, useStepper, useOnScreen } from "../kit";

function useTokenizer(): [Tokenizer | null, string | null] {
  const [t, setT] = useState<Tokenizer | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    loadTokenizer().then(setT, (e) => setErr(String(e.message ?? e)));
  }, []);
  return [t, err];
}

const DEFAULT = "Tokenization isn't magic. Unbelievably, the dog ate 1234567 cookies 🙂";

/** fig 2.1: a live tokenizer */
function LiveTokenizer({ tok, onPickWord }: { tok: Tokenizer; onPickWord: (w: string) => void }) {
  const [text, setText] = useState(DEFAULT);
  const [hover, setHover] = useState<number | null>(null);
  const pieces = useMemo(() => tok.encode(text), [tok, text]);
  const ready = useEngine((s) => s.status === "ready");
  const [agree, setAgree] = useState<boolean | null>(null);
  // cross-check against the engine's own (Rust) tokenizer when it's loaded
  useEffect(() => {
    if (!ready) return;
    let live = true;
    const t = setTimeout(async () => {
      const ids = await engine.encode(text);
      if (live) setAgree(ids.length === pieces.length && ids.every((id, i) => id === pieces[i].id));
    }, 250);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [ready, text, pieces]);
  const words = useMemo(() => {
    const m = new Map<number, string>();
    for (const p of pieces) m.set(p.word, (m.get(p.word) ?? "") + p.text);
    return m;
  }, [pieces]);
  return (
    <Fig
      n="2.1"
      title="Type anything"
      prov="weights"
      caption={
        <>
          Your text, cut into GPT-2's tokens: each coloured chip is one token (colours just separate neighbours), with its ID
          number underneath. A small <Tk s=" " /> bracket means a space: most words carry the space in front of them. This is GPT-2's real tokenizer (its own vocabulary files), running in your
          browser.
          {agree != null && (
            <span className="nv-accent">
              {" "}
              {agree ? "✓ The engine's own Rust tokenizer agrees, ID for ID." : "The engine's tokenizer disagrees here (please report this)."}
            </span>
          )}
        </>
      }
      tryit={
        <>
          Type a rare word, a long number, or another language, and watch the token count jump. Click a chip to see how its
          word was assembled (in the fold below).
        </>
      }
    >
      <textarea className="lr-input" rows={2} value={text} onChange={(e) => setText(e.target.value)} aria-label="text to tokenize" spellCheck={false} />
      <div style={{ marginTop: "0.8rem", lineHeight: 2.3 }}>
        {pieces.map((p, i) => (
          <span key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} style={{ display: "inline-flex", flexDirection: "column", alignItems: "center", verticalAlign: "top" }}>
            <Chip tone={i} on={hover === i} onClick={() => onPickWord(words.get(p.word) ?? p.text)} title={`token ${p.id}`}>
              <Tk s={p.text} />
            </Chip>
            <span className="lr-small" style={{ fontSize: "0.62rem", lineHeight: 1 }}>
              {p.id}
            </span>
          </span>
        ))}
      </div>
      <div className="lr-keyrow">
        <SpaceKey />
      </div>
      <div className="lr-stats">
        <Stat k="characters" v={text.length} />
        <Stat k="tokens" v={pieces.length} accent />
        <Stat k="characters per token" v={(text.length / Math.max(1, pieces.length)).toFixed(1)} />
        <Stat k="vocabulary" v="50,257" />
      </div>
    </Fig>
  );
}

/** fig 2.2: BPE merges, step by step */
function Merges({ tok, word, onWord }: { tok: Tokenizer; word: string; onWord: (w: string) => void }) {
  const steps = useMemo(() => tok.steps(word), [tok, word]);
  const [playing, setPlaying] = useState(true);
  const [onScreen, vis] = useOnScreen<HTMLElement>();
  const [i, setI] = useStepper(steps.length + 2, 900, vis && playing);
  useEffect(() => setI(0), [word, setI]);
  const k = Math.min(i, steps.length - 1);
  const cur = steps[k];
  const prev = k > 0 ? steps[k - 1] : null;
  // which piece is new at this step
  let merged = -1;
  if (prev) {
    for (let j = 0; j < cur.length; j++) if (cur[j] !== prev[j]) {
      merged = j;
      break;
    }
  }
  const rank = prev && merged >= 0 ? tok.rank(prev[merged], prev[merged + 1]) : undefined;
  return (
    <Fig
      figRef={onScreen}
      n="2.2"
      title={`How “${word.trim()}” gets its tokens`}
      prov="weights"
      caption={
        <>
          One word being built. It starts as single bytes (one chip each). At every step the most common neighbouring pair,
          according to GPT-2's training text, is glued into one piece (highlighted). The merge number is that pair's rank in
          GPT-2's list of 50,000 merges: small numbers are very common pairs. When no merge applies, the chips left are the
          tokens.
        </>
      }
      tryit={<>Pick one of the example words, or click a chip in fig 2.1.</>}
    >
      <div style={{ minHeight: "5.4rem" }}>
        <div className="lr-small" style={{ marginBottom: "0.5rem" }}>
          step {k} of {steps.length - 1}
          {rank != null && (
            <>
              {" "}
              · merged “<Tk s={prev![merged]} />” + “<Tk s={prev![merged + 1]} />” <span className="nv-accent">(merge #{rank.toLocaleString()})</span>
            </>
          )}
          {k === 0 && " · one piece per byte"}
          {k === steps.length - 1 && k > 0 && " · done"}
        </div>
        <div style={{ lineHeight: 2.2 }}>
          {cur.map((p, j) => (
            <Chip key={`${k}-${j}`} tone={j === merged ? 0 : 5} on={j === merged}>
              <Tk s={p} />
            </Chip>
          ))}
        </div>
      </div>
      <div className="lr-controls" style={{ marginBottom: 0, marginTop: "0.6rem" }}>
        <button className="lr-btn" onClick={() => setPlaying((p) => !p)}>
          {playing ? "❚❚ pause" : "▶ play"}
        </button>
        <button className="lr-btn" onClick={() => (setPlaying(false), setI(Math.max(0, k - 1)))}>
          ←
        </button>
        <button className="lr-btn" onClick={() => (setPlaying(false), setI(Math.min(steps.length - 1, k + 1)))}>
          →
        </button>
        <span className="lr-small">try: </span>
        {[" unbelievably", " tokenization", " 2024", " antidisestablishment"].map((w) => (
          <button key={w} className="lr-btn" onClick={() => (setPlaying(true), onWord(w))} aria-pressed={w === word}>
            {w.trim()}
          </button>
        ))}
      </div>
    </Fig>
  );
}

export default function Tokens() {
  const [tok, err] = useTokenizer();
  const [word, setWord] = useState(" Unbelievably");
  return (
    <>
      <p>
        The model can't read letters. It reads numbers, so the text is chopped into pieces called <G k="token">tokens</G> first, and each
        token is replaced by its ID: a number between 0 and 50,256. A token can be a word, part of a word, or a single byte. A
        leading space usually travels with the word after it, and a rare emoji can take several single-byte tokens.
      </p>
      <p>Common words get their own token. Rare words get built from smaller pieces. Type something and see:</p>
      {tok ? (
        <LiveTokenizer
          tok={tok}
          onPickWord={(w) => {
            setWord(w);
            const d = document.getElementById("bpe");
            if (d && !d.hasAttribute("open")) d.setAttribute("open", "");
            d?.scrollIntoView({ behavior: "smooth", block: "start" });
          }}
        />
      ) : (
        <Fig n="2.1" title="Type anything" prov="weights">
          <div className="lr-empty">{err ? `Couldn't load the tokenizer files (${err}).` : "loading GPT-2's vocabulary (1 MB)…"}</div>
        </Fig>
      )}
      <p>
        Notice the leading spaces: <code><Tk s=" dog" /></code> (with its space) and <code>dog</code> are different tokens. Notice the
        numbers, which split in odd places. And notice that the count of tokens, not characters or words, is what matters
        from here on.
      </p>
      <Idea>Text becomes a list of integers. Every token costs one trip through the model, so the token count is the real unit of work.</Idea>
      <h2>Where the pieces come from</h2>
      <p>
        Nobody chose the vocabulary by hand. It was learned by a simple greedy rule called byte-pair encoding: start with
        the 256 possible bytes, find the pair that appears side by side most often in a mountain of text, make it a new
        token, and repeat 50,000 times.
      </p>
      <Deeper title="Watch a word get built, merge by merge" id="bpe">
        {tok && <Merges tok={tok} word={word} onWord={setWord} />}
      </Deeper>
      <Deeper title="Why not just use letters, or whole words?">
        <p>
          <b>Letters</b> would make every sentence four or five times longer, and the model does one full pass per token, so
          everything would be four or five times slower. <b>Whole words</b> would need millions of entries and still choke
          on typos, names and new words. Byte pieces are the compromise: any text at all can be written down (in the worst
          case, one byte at a time), and common text is short.
        </p>
        <p>
          50,257 = 256 bytes + 50,000 merges + one special "end of text" token. The vocabulary size matters for speed too: the
          model's last step scores every one of those tokens, which turns out to be about a third of GPT-2's work per token
          (chapter 10).
        </p>
      </Deeper>
    </>
  );
}
