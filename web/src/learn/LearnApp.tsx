// The Foundations track: /learn (contents) and /learn/<chapter>.

import { useEffect, useState } from "react";
import { boot } from "../engine/client";
import { CHAPTERS, PARTS, chapterHref, type Chapter } from "./chapters";
import { Journey } from "../site/Journey";
import { linkTo, navigate, usePath } from "./router";
import { TopBar } from "../site/TopBar";
import "./learn.css";

const VISITED_KEY = "nanoinfer-learn-visited";
function readVisited(): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem(VISITED_KEY) ?? "[]"));
  } catch {
    return new Set();
  }
}
function markVisited(slug: string) {
  const v = readVisited();
  if (v.has(slug)) return;
  v.add(slug);
  try {
    localStorage.setItem(VISITED_KEY, JSON.stringify([...v]));
  } catch {
    /* storage blocked */
  }
}

export default function LearnApp() {
  const path = usePath();
  useEffect(() => void boot(), []);
  const slug = path.replace(/^\/learn\/?/, "").replace(/\/$/, "");
  const idx = CHAPTERS.findIndex((c) => c.slug === slug);
  const ch = idx >= 0 ? CHAPTERS[idx] : null;

  useEffect(() => {
    document.title = ch ? `${ch.title} · nanoinfer foundations` : "Foundations · nanoinfer";
    if (ch) markVisited(ch.slug);
  }, [ch]);

  // ← / → between chapters
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (e.metaKey || e.ctrlKey || e.altKey || t.closest("input, textarea, select, [contenteditable]")) return;
      if (e.key === "ArrowRight" && idx < CHAPTERS.length - 1) navigate(chapterHref(CHAPTERS[idx + 1]));
      if (e.key === "ArrowLeft") navigate(idx > 0 ? chapterHref(CHAPTERS[idx - 1]) : "/learn");
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [idx]);

  return (
    <main className="nv lr">
      <div className="lr-progress" aria-hidden>
        <i style={{ width: ch ? `${((idx + 1) / CHAPTERS.length) * 100}%` : "0%" }} />
      </div>
      <TopBar
        here="learn"
        crumb={
          <>
            {" / "}
            <a href="/learn" onClick={linkTo("/learn")}>
              course
            </a>
            {ch && <span className="nv-muted"> / {String(idx + 1).padStart(2, "0")}</span>}
          </>
        }
      />
      {ch ? <ChapterPage ch={ch} idx={idx} key={ch.slug} /> : <Index />}
    </main>
  );
}

function Rail({ idx }: { idx: number }) {
  const visited = readVisited();
  return (
    <nav className="lr-rail" aria-label="chapters">
      {CHAPTERS.map((c, i) => {
        const first = i === 0 || CHAPTERS[i - 1].part !== c.part;
        return (
          <div key={c.slug} className={first ? "lr-rail-first" : undefined}>
            {first && (
              <span className="lr-rail-part">
                part {c.part + 1} · {PARTS[c.part].toLowerCase()}
              </span>
            )}
            <a
              href={chapterHref(c)}
              onClick={linkTo(chapterHref(c))}
              className={`${i === idx ? "on" : ""} ${visited.has(c.slug) ? "seen" : ""}`}
              aria-current={i === idx ? "page" : undefined}
              aria-label={`${String(i + 1).padStart(2, "0")} ${c.title}`}
            >
              <span className="d" />
              <span className="t">
                <span className="nv-muted">{String(i + 1).padStart(2, "0")}</span> {c.title}
              </span>
            </a>
          </div>
        );
      })}
    </nav>
  );
}

function ChapterPage({ ch, idx }: { ch: Chapter; idx: number }) {
  const prev = idx > 0 ? CHAPTERS[idx - 1] : null;
  const next = idx < CHAPTERS.length - 1 ? CHAPTERS[idx + 1] : null;
  const C = ch.Component;
  return (
    <>
      <Rail idx={idx} />
      <article className="lr-ch" data-askable data-ask-title={ch.title}>
        <header className="lr-head" data-ask-section>
          <div className="lr-kicker">
            <span className="nv-accent">{String(idx + 1).padStart(2, "0")}</span>
            <span className="nv-muted">
              {" · "}part {ch.part + 1} · {PARTS[ch.part].toLowerCase()}
            </span>
          </div>
          <h1 className="lr-h1">{ch.title}</h1>
          <p className="lr-q">{ch.q}</p>
        </header>
        <div className="lr-body">
          <C />
        </div>
        <nav className="lr-pn">
          {prev ? (
            <a href={chapterHref(prev)} onClick={linkTo(chapterHref(prev))} className="prev">
              <span className="k">← {String(idx).padStart(2, "0")}</span>
              <span className="t">{prev.title}</span>
            </a>
          ) : (
            <a href="/learn" onClick={linkTo("/learn")} className="prev">
              <span className="k">← contents</span>
              <span className="t">Foundations</span>
            </a>
          )}
          {next ? (
            <a href={chapterHref(next)} onClick={linkTo(chapterHref(next))} className="next">
              <span className="k">next · {String(idx + 2).padStart(2, "0")} →</span>
              <span className="t">{next.title}</span>
              <span className="q">{next.q}</span>
            </a>
          ) : (
            <a href="/" className="next">
              <span className="k">the end of the beginning →</span>
              <span className="t">Now watch it all run, live</span>
              <span className="q">The demo page: the same engine, every stage of the climb.</span>
            </a>
          )}
        </nav>
        <p className="lr-hint nv-muted">
          ← → to move between chapters · select any text to ask about it
        </p>
      </article>
    </>
  );
}

function Index() {
  const [visited] = useState(readVisited);
  const firstUnseen = CHAPTERS.find((c) => !visited.has(c.slug)) ?? CHAPTERS[0];
  return (
    <div className="lr-index" data-askable data-ask-title="The course: contents">
      <header className="nv-col lr-ihero" data-ask-section>
        <div className="nv-kicker">The course · {CHAPTERS.length} short chapters · about an hour</div>
        <h1 className="nv-h1">
          How a language model runs, <em>from the ground up</em>.
        </h1>
        <p className="nv-lead">
          One idea per page, each with a figure you can poke. Start with a machine that guesses the next word; end knowing
          why it's slow, what made our engine 1,700× faster, and how to work out the speed limit of any model on any computer.
          No maths beyond multiplying and adding.
        </p>
        <div className="lr-start">
          <a className="nv-btn" href={chapterHref(firstUnseen)} onClick={linkTo(chapterHref(firstUnseen))}>
            {visited.size ? "Continue" : "Start"} · {firstUnseen.title} →
          </a>
          {visited.size > 0 && (
            <span className="nv-muted" style={{ fontSize: "0.78rem" }}>
              {visited.size} of {CHAPTERS.length} read
            </span>
          )}
        </div>
      </header>
      <div className="nv-col lr-toc">
        <Journey showQuestions />
        <p className="nv-muted lr-foot">
          Numbers come from nanoinfer's own logs: GPT-2 124M on an Apple M4 Mac mini (CPU and GPU), and the same Rust compiled
          to WebAssembly in your browser. Toy figures are labelled as toys.
        </p>
      </div>
    </div>
  );
}
