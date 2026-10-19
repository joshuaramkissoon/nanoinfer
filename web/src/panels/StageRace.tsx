import { useEffect, useRef, useState } from "react";
import { claim, decodeTokens, engine, getState, isGpu, release, useEngine } from "../engine/client";
import { LiveButton, NeedsEngine, useFigureSource, useInView } from "../ui/Figure";
import { usePalette } from "../ui/theme";
import { fmtMs, sig } from "../lib/fmt";
import { fallback } from "./RunIt";

const PROMPT = "The fastest way to read a book is";
const TARGET = 40;
const BUDGET_MS = 6000; // engine time per lane

interface Lane {
  stage: string;
  note?: string;
  tokens: number[];
  /** measured ms of each decode forward */
  ms: number[];
  /** cumulative engine time after each token */
  t: number[];
  done: boolean;
}

/**
 * Every stage writes the same greedy continuation of the same prompt. Each lane
 * shows its k-th token at the moment its own measured forward passes add up to,
 * so the lanes move at their stages' real speeds.
 *
 * Live, there's one engine thread, so the lanes are computed time-sliced (the
 * lane with the least engine time used goes next) and the clock can't run
 * ahead of the slowest unfinished lane: playback runs slower than real time
 * until the fast lanes finish. Recorded, it plays back at true speed.
 */
export default function StageRace() {
  const status = useEngine((s) => s.status);
  const stages = useEngine((s) => s.stages);
  const busy = useEngine((s) => s.busy);
  const rec = useEngine((s) => s.recorded);
  const pal = usePalette();
  const [viewRef, inView] = useInView<HTMLDivElement>();
  const [lanes, setLanes] = useState<Lane[] | null>(null);
  const [mode, setMode] = useState<"live" | "recorded" | null>(null);
  const [clock, setClock] = useState(0);
  const [running, setRunning] = useState(false);
  const [promptText, setPromptText] = useState(PROMPT);
  const lanesRef = useRef<Lane[] | null>(null);
  const stop = useRef(false);
  const t0 = useRef(0);
  const raf = useRef(0);
  const ready = status === "ready";
  useFigureSource(mode ?? (rec?.race ? "recorded" : "none"));

  // The display clock: real time since start, capped (live) by what's computed.
  const tick = () => {
    const ls = lanesRef.current;
    if (!ls) return;
    let t = performance.now() - t0.current;
    const open = ls.filter((l) => !l.done);
    if (open.length) t = Math.min(t, Math.min(...open.map((l) => l.t.at(-1) ?? 0)));
    const end = Math.max(...ls.map((l) => l.t.at(-1) ?? 0));
    setClock(Math.min(t, end));
    if (open.length || t < end) raf.current = requestAnimationFrame(tick);
  };

  const replay = () => {
    const r = rec?.race;
    if (!r) return;
    cancelAnimationFrame(raf.current);
    const ls = r.lanes.map((l) => {
      const t: number[] = [];
      l.ms.reduce((a, m) => (t.push(a + m), a + m), 0);
      return { stage: l.stage, note: rec?.meta.stages.find((s) => s.name === l.stage)?.wasmNote, tokens: l.tokens, ms: l.ms, t, done: true };
    });
    lanesRef.current = ls;
    setLanes(ls);
    setMode("recorded");
    setPromptText(r.text);
    t0.current = performance.now();
    raf.current = requestAnimationFrame(tick);
  };

  // Recorded: start the replay the first time the figure scrolls into view.
  const started = useRef(false);
  useEffect(() => {
    if (inView && !started.current && rec?.race && !ready) {
      started.current = true;
      replay();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inView, rec, ready]);
  useEffect(() => () => cancelAnimationFrame(raf.current), []);

  const race = async () => {
    if (!claim("stage-race")) return;
    setRunning(true);
    stop.current = false;
    cancelAnimationFrame(raf.current);
    const slots: number[] = [];
    try {
      await engine.capture(0);
      const prompt = await engine.encode(PROMPT);
      const ls: Lane[] = getState().stages.map((s) => ({ stage: s.name, note: s.wasmNote, tokens: [], ms: [], t: [], done: false }));
      const next: number[] = [];
      for (const l of ls) {
        const slot = await engine.open(l.stage, prompt.length + TARGET + 8);
        slots.push(slot);
        await engine.forward(slot, prompt); // prefill: same work for every lane, not raced
        next.push((await engine.sample(slot, 0, 0)).id);
      }
      lanesRef.current = ls;
      setLanes(ls.map((l) => ({ ...l })));
      setMode("live");
      setPromptText(PROMPT);
      t0.current = performance.now();
      raf.current = requestAnimationFrame(tick);
      while (!stop.current) {
        const open = ls.map((_, i) => i).filter((i) => !ls[i].done);
        if (!open.length) break;
        // fair: the lane that has used the least engine time runs next
        const i = open.reduce((p, c) => ((ls[c].t.at(-1) ?? 0) < (ls[p].t.at(-1) ?? 0) ? c : p));
        const l = ls[i];
        if (isGpu(l.stage)) {
          // WebGPU lanes decode on their fastest greedy path (GPU argmax,
          // several tokens per submit where the stage has it), a few tokens
          // per turn; each token is charged an equal share of the chunk.
          const k = Math.min(4, TARGET - l.tokens.length);
          const r = await engine.gpuGreedy(slots[i], next[i], k);
          const fed = [next[i], ...r.tokens.slice(0, k - 1)];
          for (const t of fed) {
            l.tokens.push(t);
            l.ms.push(r.ms / k);
            l.t.push((l.t.at(-1) ?? 0) + r.ms / k);
          }
          next[i] = r.tokens[k - 1];
          if (l.tokens.length >= TARGET || l.t.at(-1)! >= BUDGET_MS) l.done = true;
          setLanes(ls.map((x) => ({ ...x })));
          continue;
        }
        const tr = await engine.forward(slots[i], [next[i]]);
        l.tokens.push(next[i]);
        l.ms.push(tr.ms);
        l.t.push((l.t.at(-1) ?? 0) + tr.ms);
        if (l.tokens.length >= TARGET || l.t.at(-1)! >= BUDGET_MS) l.done = true;
        else next[i] = (await engine.sample(slots[i], 0, 0)).id;
        setLanes(ls.map((x) => ({ ...x })));
      }
      for (const l of ls) l.done = true;
      raf.current = requestAnimationFrame(tick);
    } finally {
      for (const s of slots) await engine.close(s);
      await engine.capture(10);
      setRunning(false);
      release("stage-race");
    }
  };

  if (!lanes && !ready) return <div ref={viewRef}><NeedsEngine>Every stage, racing to write the same sentence.</NeedsEngine></div>;

  const shown = lanes ?? stages.map((s) => ({ stage: s.name, note: s.wasmNote, tokens: [], ms: [], t: [], done: false }));
  const visible = (l: Lane) => l.t.filter((t) => t <= clock).length;
  const realElapsed = mode ? performance.now() - t0.current : 0;
  const speed = mode === "live" && running && realElapsed > 300 ? clock / realElapsed : 1;
  const fastest = Math.max(...shown.map((l) => (l.tokens.length ? l.tokens.length / (l.t.at(-1)! / 1000) : 0)));

  return (
    <div ref={viewRef}>
      <div className="nv-controls" style={{ justifyContent: "space-between" }}>
        <span style={{ display: "flex", gap: "0.8rem", alignItems: "center", flexWrap: "wrap" }}>
          {running ? (
            <button className="nv-btn" onClick={() => (stop.current = true)}>■ Stop</button>
          ) : ready ? (
            <button className="nv-btn" onClick={race} disabled={!!busy}>▶ Race every stage in this tab</button>
          ) : (
            <>
              <button className="nv-btn" onClick={replay} disabled={!rec?.race}>↻ Replay</button>
              <LiveButton label="race them in your tab" />
            </>
          )}
        </span>
        <span className="nv-race-clock">
          <b>{(clock / 1000).toFixed(2)} s</b>
          <span className="nv-muted">
            {" "}
            of engine time per lane
            {mode === "live" && running && speed < 0.95 ? ` · playing at ${sig(speed, 2)}× (${shown.length} lanes share one core)` : mode === "recorded" ? " · real speed" : ""}
          </span>
        </span>
      </div>
      <div className="nv-race">
        {shown.map((l) => {
          const n = visible(l);
          const tokS = n > 0 ? n / (l.t[n - 1] / 1000) : 0;
          const finished = l.done && n === l.tokens.length && n > 0;
          const isFb = fallback({ wasmNote: l.note, browserOnly: l.stage.startsWith("wasm-") });
          return (
            <div key={l.stage} className={`nv-race-lane${finished ? " fin" : ""}`}>
              <span className="nm" title={l.note || undefined}>
                {l.stage}
                {isFb && <span className="nv-muted">*</span>}
              </span>
              <span className="tx">
                <span className="bar" style={{ width: `${(100 * n) / TARGET}%`, background: l.stage.startsWith("wasm-") ? pal.accent : isGpu(l.stage) ? pal.cool : pal.heat[3] }} />
                <bdi>
                  <span className="nv-muted">{promptText.replace(/\n/g, "⏎")}</span>
                  {n ? decodeTokens(l.tokens.slice(0, n)).replace(/\n/g, "⏎") : ""}
                  {!finished && n < TARGET && mode && <span className="nv-caret" />}
                </bdi>
              </span>
              <span className="st nv-num">
                {n > 0 ? (
                  <>
                    <b style={{ color: tokS >= fastest * 0.98 ? pal.accent : undefined }}>{sig(tokS, 3)}</b> tok/s
                    {finished && <span className="nv-muted"> · {n} in {fmtMs(l.t[n - 1])}</span>}
                  </>
                ) : (
                  <span className="nv-muted">{mode ? "…" : ""}</span>
                )}
              </span>
            </div>
          );
        })}
      </div>
      <div className="nv-muted" style={{ fontSize: "var(--t-xs)", marginTop: "0.6rem" }}>
        greedy, so every f32 lane writes the same words (quantised lanes round the weights and can pick different ones) · {TARGET} tokens or {BUDGET_MS / 1000} s of engine time per lane · prompt prefill not raced ·
        lanes run one forward at a time on one engine thread, and each token appears when that lane's own measured forward times add
        up to it · * = this stage runs another stage's code in wasm
      </div>
    </div>
  );
}
