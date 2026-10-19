import { useEffect, useMemo, useRef, useState } from "react";
import { SpaceKey, Tk } from "../ui/Tok";
import { decodes, engine, loadModel, median, tokenText, useEngine, useRunSource, weightLabel, type Tok } from "../engine/client";
import { generate } from "../engine/generate";
import { NeedsEngine, useFigureSource, useWidth } from "../ui/Figure";
import { css, heatAt, usePalette, useReducedMotion } from "../ui/theme";
import { fmtBytes, fmtMs, linear, sig } from "../lib/fmt";
import { stageLabel } from "../lib/stages";
import { StagePicker, noteSpeed, recommendedStage } from "./StagePicker";

const DEFAULT_PROMPT = "In a shocking finding, scientists discovered a herd of unicorns living in";

export { fallback } from "../lib/stages";
export { StagePicker } from "./StagePicker";

export default function RunIt() {
  const status = useEngine((s) => s.status);
  const run = useEngine((s) => s.run);
  const busy = useEngine((s) => s.busy);
  const stages = useEngine((s) => s.stages);
  const cfg = useEngine((s) => s.config);
  const pal = usePalette();
  const reduced = useReducedMotion();
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [stage, setStage] = useState("");
  const [n, setN] = useState(48);
  const [temp, setTemp] = useState(0.8);
  const [sel, setSel] = useState<number | null>(null);
  const abort = useRef({ aborted: false });
  const running = busy === "hero";
  const ready = status === "ready";
  const source = useRunSource();
  const recorded = source === "recorded";
  useFigureSource(source);

  // default: the fastest version that really runs in this browser, re-picked when the
  // live stage list arrives (with the GPU rungs), until the reader picks one themselves
  const picked = useRef(false);
  const stageRef = useRef(stage);
  stageRef.current = stage;
  useEffect(() => {
    if (!stages.length) return;
    if (!picked.current || !stages.some((s) => s.name === stageRef.current)) {
      const rec = recommendedStage(stages) ?? stages.at(-1)!.name;
      stageRef.current = rec;
      setStage(rec);
    }
  }, [stages]);
  const pick = (s: string) => {
    picked.current = true;
    stageRef.current = s;
    setStage(s);
  };

  const go = async (prefix?: Tok[], forced?: Parameters<typeof generate>[0]["forced"]) => {
    abort.current = { aborted: false };
    setSel(null);
    let pre = prefix;
    if (!pre) {
      const ids = await engine.encode(prompt);
      pre = ids.map((id) => ({ id, prompt: true, p: null, top: null, entropy: null }));
    }
    if (!pre.length && !forced) return;
    await generate({ stage: stageRef.current, prefix: pre, n: forced ? n - 1 : n, temperature: temp, topK: 40, forced, signal: abort.current });
  };

  // Auto-run once when the engine first becomes ready: the reader should see
  // GPT-2 talk before reading anything.
  const autoRan = useRef(false);
  useEffect(() => {
    if (ready && !autoRan.current && (!run || recorded)) {
      autoRan.current = true;
      void go();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  const branch = (i: number, alt: { id: number; p: number }) => {
    if (!run || running) return;
    const t = run.tokens[i];
    void go(run.tokens.slice(0, i), { id: alt.id, p: alt.p, top: t.top, entropy: t.entropy });
  };

  const dec = decodes(run);
  const recent = dec.slice(-8).map((f) => f.ms);
  const tokS = recent.length ? 1000 / median(recent) : NaN;
  const last = run?.forwards[run.forwards.length - 1];
  const nGen = run ? run.tokens.filter((t) => !t.prompt).length : 0;
  const nextDist = run && !run.done && last ? last.top : null;
  const shown = sel != null && run?.tokens[sel]?.top ? run.tokens[sel] : null;
  const dist = shown?.top ?? nextDist ?? run?.tokens[run.tokens.length - 1]?.top ?? null;
  const chosen = shown ? shown.id : null;
  const kvBytes = last && last.kvLen >= 0 && cfg ? 2 * cfg.nLayer * last.kvLen * cfg.nEmbd * 4 : null;

  // Remember how fast each engine version ran here, for the picker.
  useEffect(() => {
    if (run?.done && !recorded && dec.length >= 4) noteSpeed(run.stage, tokS);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run?.done, recorded]);

  // While showing the recording, put its prompt in the box.
  useEffect(() => {
    if (recorded && run) setPrompt(decodeRecordedPrompt(run.tokens));
  }, [recorded, run]);

  if (!ready && !run) return <NeedsEngine>Type a prompt and GPT-2 writes the rest, one token at a time, in this tab.</NeedsEngine>;

  return (
    <div>
      <input
        className="nv-input"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && !running && ready && go()}
        aria-label="prompt"
        spellCheck={false}
      />
      <div style={{ marginTop: "0.9rem" }}>
        <StagePicker value={stage} onChange={pick} disabled={running} />
      </div>
      <div className="nv-controls" style={{ marginTop: "0.9rem" }}>
        {running ? (
          <button className="nv-btn" onClick={() => (abort.current.aborted = true)}>
            ■ Stop
          </button>
        ) : !ready ? (
          <button className="nv-btn" onClick={() => loadModel()} disabled={status === "loading" || status === "boot"}
            title="downloads GPT-2 once (cached in your browser) and runs it on your CPU">
            {status === "loading" ? "loading GPT-2…" : <>▶ Run live <span style={{ opacity: 0.7, fontWeight: 400 }}>· {weightLabel()}</span></>}
          </button>
        ) : (
          <button className="nv-btn" onClick={() => go()} disabled={!ready || !!busy}>
            ▶ Generate
          </button>
        )}
        <label className="nv-field">
          tokens
          <input type="range" min={8} max={160} step={8} value={n} onChange={(e) => setN(+e.target.value)} disabled={running} />
          <span className="nv-num" style={{ width: "2rem" }}>{n}</span>
        </label>
        <label className="nv-field">
          temperature
          <input type="range" min={0} max={1.5} step={0.05} value={temp} onChange={(e) => setTemp(+e.target.value)} disabled={running} />
          <span className="nv-num" style={{ width: "2.4rem" }}>{temp === 0 ? "greedy" : temp.toFixed(2)}</span>
        </label>
      </div>

      <div className="nv-run" style={{ marginTop: "1.4rem" }}>
        <div>
          <div className="nv-stream" aria-live="polite">
            {run?.tokens.map((t, i) => {
              const heat = t.p == null ? undefined : css(heatAt(pal, 0.25 + 0.75 * t.p));
              const fresh = !reduced && !t.prompt && i >= run.tokens.length - 1 && !run.done;
              return t.prompt ? (
                <span key={i} className="nv-t prompt">
                  {tokenText(t.id)}
                </span>
              ) : (
                <button
                  key={i}
                  className={`nv-t${sel === i ? " sel" : ""}${fresh ? " fresh" : ""}`}
                  style={{ ["--nv-heat" as string]: heat }}
                  onClick={() => setSel(sel === i ? null : i)}
                  title={`p = ${sig(t.p ?? 0, 2)} · click to see the alternatives`}
                >
                  {tokenText(t.id)}
                </button>
              );
            })}
            {running && <span className="nv-caret" />}
          </div>
          <div className="nv-stats" style={{ marginTop: "1.2rem" }}>
            <Stat k={run && !recorded ? `speed · ${stageLabel(run.stage)}` : "decode speed"} v={isFinite(tokS) ? sig(tokS, 3) : "–"} u="tok/s" accent />
            <Stat k="last token" v={last ? fmtMs(last.ms) : "–"} u={last ? `${last.nComputed} ${last.nComputed === 1 ? "row" : "rows"} computed` : undefined} />
            <Stat k="kv cache" v={kvBytes != null ? fmtBytes(kvBytes) : "off"} u={run ? `${run.tokens.length} tokens · ${nGen} generated` : undefined} />
          </div>
          <MsSpark ms={dec.map((f) => f.ms)} />
        </div>
        <div>
          <div className="nv-muted nv-probs-head">
            <span>
            {shown
              ? recorded
                ? `what it could have said at position ${sel}`
                : `what it could have said at position ${sel} (click one to branch)`
              : running
                ? "next token, as it is being chosen"
                : "the last choice · click any token to inspect it"}
            </span>
            <SpaceKey />
          </div>
          <div className="nv-probs">
            {dist ? (
              dist.map((d) => (
                <button
                  key={d.id}
                  className={`nv-prob${chosen === d.id ? " chosen" : ""}`}
                  onClick={() => shown && sel != null && branch(sel, d)}
                  disabled={!shown || running || chosen === d.id || !ready}
                  title={shown && chosen !== d.id ? `branch: continue with "${tokenText(d.id)}"` : undefined}
                >
                  <span className="tk"><Tk id={d.id} /></span>
                  <span className="b">
                    <i style={{ width: `${Math.max(1, d.p * 100)}%` }} />
                  </span>
                  <span className="pc">{(d.p * 100).toFixed(1)}%</span>
                </button>
              ))
            ) : (
              <span className="nv-muted">…</span>
            )}
          </div>
          {shown?.entropy != null && (
            <div className="nv-muted" style={{ fontSize: "var(--t-xs)", marginTop: "0.6rem" }}>
              entropy {shown.entropy.toFixed(2)} nats · ≈ {Math.exp(shown.entropy).toFixed(1)} equally likely choices
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function decodeRecordedPrompt(toks: Tok[]): string {
  return toks.filter((t) => t.prompt).map((t) => tokenText(t.id)).join("");
}

export function Stat({ k, v, u, accent }: { k: string; v: string; u?: string; accent?: boolean }) {
  return (
    <div className="nv-stat">
      <span className="k">{k}</span>
      <span className={`v${accent ? " nv-accent" : ""}`}>
        {v}
        {u && <small>{u}</small>}
      </span>
    </div>
  );
}

/** ms per decoded token, one bar per token. Flat = KV cache; a ramp = recompute. */
function MsSpark({ ms }: { ms: number[] }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const pal = usePalette();
  const h = 36;
  const max = useMemo(() => Math.max(1, ...ms), [ms]);
  const N = Math.max(ms.length, 32);
  const y = linear(0, max, h, 2);
  const bw = w / N;
  return (
    <div ref={ref} style={{ marginTop: "1rem" }}>
      <div className="nv-muted" style={{ fontSize: "var(--t-xs)", display: "flex", justifyContent: "space-between" }}>
        <span>ms per token →</span>
        <span>max {fmtMs(max)}</span>
      </div>
      <svg className="nv-svg nv-spark" viewBox={`0 0 ${w || 1} ${h}`} height={h} preserveAspectRatio="none">
        <line x1={0} x2={w} y1={h - 0.5} y2={h - 0.5} stroke={pal.rule} />
        {ms.map((m, i) => (
          <rect key={i} x={i * bw + 0.5} width={Math.max(1, bw - 1.5)} y={y(m)} height={h - y(m)} fill={i === ms.length - 1 ? pal.accent : pal.muted} fillOpacity={i === ms.length - 1 ? 1 : 0.55} />
        ))}
      </svg>
    </div>
  );
}
