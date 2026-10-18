// The front door: one question, one demo, the map of the journey.

import { useEffect, useRef, useState } from "react";
import { boot, claim, decodeTokens, engine, loadModel, release, useEngine, weightLabel } from "../engine/client";
import { bestStage, useRecorded } from "../learn/kit";
import { CHAPTERS, chapterHref } from "../learn/chapters";
import { linkTo } from "../learn/router";
import { Journey } from "./Journey";
import { TopBar } from "./TopBar";
import "../learn/learn.css";

const MB_PER_TOKEN = 494.6;

function Demo() {
  const rec = useRecorded();
  const status = useEngine((s) => s.status);
  const progress = useEngine((s) => s.progress);
  const live = status === "ready";
  const [prompt, setPrompt] = useState("The best way to learn something new is");
  const [promptIds, setPromptIds] = useState<number[]>([]);
  const [gen, setGen] = useState<number[]>([]);
  const [speed, setSpeed] = useState<number | null>(null);
  const [running, setRunning] = useState(false);
  const [src, setSrc] = useState<"recorded" | "live">("recorded");
  const stop = useRef(false);

  // recorded: replay the recorded run at its measured pace
  const replay = async () => {
    if (!rec) return;
    stop.current = false;
    setRunning(true);
    setSrc("recorded");
    const toks = rec.run.tokens;
    const nP = toks.filter((t) => t.prompt).length;
    setPromptIds(toks.slice(0, nP).map((t) => t.id));
    setGen([]);
    setSpeed(null);
    const fw = rec.run.forwards;
    let ms = 0;
    for (let i = 0; i < toks.length - nP && !stop.current; i++) {
      const f = fw[i];
      await new Promise((r) => setTimeout(r, f ? f.ms : 10));
      if (i > 0) ms += f?.ms ?? 10;
      setGen(toks.slice(nP, nP + i + 1).map((t) => t.id));
      if (i > 0) setSpeed((i * 1000) / ms);
    }
    setRunning(false);
  };

  const runLive = async () => {
    if (!claim("landing")) return;
    stop.current = false;
    setRunning(true);
    setSrc("live");
    setGen([]);
    setSpeed(null);
    let slot = -1;
    try {
      const ids = (await engine.encode(prompt)).slice(-200);
      setPromptIds(ids);
      slot = await engine.open(bestStage(), ids.length + 64);
      await engine.capture(10);
      await engine.seed(1 + Math.floor(Math.random() * 2 ** 31));
      await engine.forward(slot, ids);
      const out: number[] = [];
      let t0 = 0;
      for (let i = 0; i < 48 && !stop.current; i++) {
        const { id } = await engine.sample(slot, 0.8, 40);
        out.push(id);
        setGen([...out]);
        if (i === 0) t0 = performance.now();
        else setSpeed((i * 1000) / (performance.now() - t0));
        if (i < 47) await engine.forward(slot, [id]);
      }
    } finally {
      if (slot >= 0) await engine.close(slot);
      release("landing");
      setRunning(false);
    }
  };

  useEffect(() => {
    if (rec && src === "recorded" && !gen.length && !running) void replay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rec]);
  useEffect(() => () => void (stop.current = true), []);

  const words = speed ? speed * 0.75 : null;
  return (
    <div className="ld-demo" data-ask-section>
      <div className="ld-demo-head">
        <span className={`lr-prov ${src === "live" ? "live" : "recorded"}`}>
          {src === "live" ? "live · GPT-2 running in this tab" : "a recorded run of GPT-2 in the engine"}
        </span>
      </div>
      {live ? (
        <div className="lr-controls" style={{ flexWrap: "nowrap", margin: "0 0 0.9rem" }}>
          <input className="lr-input" value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={(e) => e.key === "Enter" && !running && runLive()} aria-label="prompt" />
          <button className="lr-btn solid" onClick={() => (running ? (stop.current = true) : runLive())}>
            {running ? "stop" : "write"}
          </button>
        </div>
      ) : null}
      <div className="lr-text ld-out">
        <span className="ctx">{decodeTokens(promptIds)}</span>
        <span>{decodeTokens(gen)}</span>
        {running && <span className="lr-cursor" />}
      </div>
      <div className="ld-speed">
        {speed ? (
          <>
            <b>{speed.toFixed(0)} tokens a second</b>, about {words!.toFixed(0)} words. For every one of those tokens, the computer
            read all {MB_PER_TOKEN.toFixed(0)} MB of GPT-2's numbers from memory: about {((speed * MB_PER_TOKEN) / 1000).toFixed(0)} GB every
            second. That is why it isn't faster, and the course explains how to change it.
          </>
        ) : (
          <span className="nv-muted">{running ? "writing…" : " "}</span>
        )}
      </div>
      <div className="ld-actions">
        {!live && (
          <>
            <button className="lr-btn" onClick={() => !running && replay()} disabled={running || !rec}>
              ⟲ replay
            </button>
            <button className="lr-btn solid" onClick={() => loadModel()} disabled={status === "loading" || status === "boot"}>
              {status === "loading"
                ? `loading GPT-2${progress?.total ? ` · ${Math.round((100 * progress.loaded) / progress.total)}%` : "…"}`
                : `type your own · loads GPT-2 (${weightLabel()}) into this tab`}
            </button>
          </>
        )}
      </div>
    </div>
  );
}

export default function Landing() {
  useEffect(() => {
    document.title = "nanoinfer · how a language model runs, and how fast";
    void boot();
  }, []);
  const first = CHAPTERS[0];
  return (
    <main className="nv lr ld" data-askable data-ask-title="nanoinfer: home">
      <TopBar here="home" />
      <header className="nv-col ld-hero" data-ask-section>
        <div className="nv-kicker">an inference engine written from scratch · and a course in how it works</div>
        <h1 className="ld-h1">
          What happens when a model writes the next word, <em>and how fast can one computer do it?</em>
        </h1>
        <p className="nv-lead">
          Below, GPT-2 is writing, run by nanoinfer: a small engine written from scratch in Rust, compiled to run in your
          browser. The course starts from nothing and builds up to why this is slow, then follows the engine as it gets
          1,700× faster, one measured step at a time.
        </p>
      </header>
      <div className="nv-col">
        <Demo />
        <div className="lr-start" style={{ marginTop: "2.2rem" }}>
          <a className="nv-btn" href={chapterHref(first)} onClick={linkTo(chapterHref(first))}>
            Start the course · {first.title} →
          </a>
          <span className="nv-muted" style={{ fontSize: "0.78rem" }}>
            {CHAPTERS.length} short chapters · select any text to ask about it
          </span>
        </div>
      </div>
      <section className="nv-col ld-map">
        <h2 className="ld-h2">The journey</h2>
        <Journey />
      </section>
      <footer className="nv-col nv-footer">
        nanoinfer · GPT-2 124M weights by OpenAI (MIT) · engine, kernels and tokenizer written from scratch in Rust · every
        number on this site comes from the engine's own measurements on an Apple M4 Mac mini, or from your tab.
      </footer>
    </main>
  );
}
