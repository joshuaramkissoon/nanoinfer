// The course: /learn (contents) and /learn/<chapter>.

import { useEffect, useState } from "react";
import { boot } from "../engine/client";
import { CHAPTERS, GUIDE_HREF, PARTS, chapterHref, chapterNum, type Chapter } from "./chapters";
import { checksFor, CheckYourself } from "./Check";
import FieldGuide from "./FieldGuide";
import { climbPct } from "./Climb";
import { useLearnData } from "./kit";
import { Journey } from "../site/Journey";
import { linkTo, navigate, usePath } from "./router";
import { TopBar } from "../site/TopBar";
import { useAskAvailable } from "../ask/AskLayer";
import "./learn.css";
import "./p0.css";

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
  const guide = slug === "field-guide";

  useEffect(() => {
    document.title = ch ? `${ch.title} · The Climb` : guide ? "Field guide · The Climb" : "The course · The Climb";
    if (ch) markVisited(ch.slug);
  }, [ch, guide]);

  // ← / → between chapters
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (guide || e.metaKey || e.ctrlKey || e.altKey || t.closest("input, textarea, select, [contenteditable]")) return;
      if (e.key === "ArrowRight" && idx < CHAPTERS.length - 1) navigate(chapterHref(CHAPTERS[idx + 1]));
      if (e.key === "ArrowLeft") navigate(idx > 0 ? chapterHref(CHAPTERS[idx - 1]) : "/learn");
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [idx, guide]);

  return (
    <main className="nv lr">
      <div className="lr-progress" aria-hidden>
        <i style={{ width: ch ? `${((idx + 1) / CHAPTERS.length) * 100}%` : "0%" }} />
      </div>
      <TopBar
        here="learn"
        crumb={
          <>
            <span className={ch || guide ? "lr-hide-sm" : undefined}>
              {" / "}
              <a href="/learn" onClick={linkTo("/learn")}>
                course
              </a>
            </span>
            {ch && <span className="nv-muted"> / {chapterNum(ch)}</span>}
            {guide && <span className="nv-muted"> / field guide</span>}
          </>
        }
      />
      {ch ? (
        <ChapterPage ch={ch} idx={idx} key={ch.slug} />
      ) : guide ? (
        <div id="content" tabIndex={-1}>
          <FieldGuide />
        </div>
      ) : (
        <Index />
      )}
    </main>
  );
}

function Rail({ idx }: { idx: number }) {
  const visited = readVisited();
  // step aside when the prev/next cards at the end of the chapter scroll into view
  const [away, setAway] = useState(false);
  useEffect(() => {
    const pn = document.querySelector(".lr-pn");
    if (!pn) return;
    const io = new IntersectionObserver(([e]) => setAway(e.isIntersecting), { rootMargin: "0px 0px -35% 0px" });
    io.observe(pn);
    return () => io.disconnect();
  }, [idx]);
  return (
    <nav className={`lr-rail${away ? " away" : ""}`} aria-label="chapters">
      {CHAPTERS.map((c, i) => {
        const first = i === 0 || CHAPTERS[i - 1].part !== c.part;
        return (
          <div key={c.slug} className={first ? "lr-rail-first" : undefined}>
            {first && (
              <span className="lr-rail-part">
                part {c.part} · {PARTS[c.part].toLowerCase()}
              </span>
            )}
            <a
              href={chapterHref(c)}
              onClick={linkTo(chapterHref(c))}
              className={`${i === idx ? "on" : ""} ${visited.has(c.slug) ? "seen" : ""}`}
              aria-current={i === idx ? "page" : undefined}
              aria-label={`${chapterNum(c)} ${c.title}`}
            >
              <span className="d">{chapterNum(c)}</span>
              <span className="t">{c.title}</span>
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
  const ask = useAskAvailable();
  return (
    <>
      <Rail idx={idx} />
      <article className="lr-ch" id="content" tabIndex={-1} data-askable data-ask-title={ch.title}>
        <header className="lr-head" data-ask-section>
          <div className="lr-kicker">
            <span className="nv-accent">{chapterNum(ch)}</span>
            <span className="nv-muted">
              {" · "}part {ch.part} · {PARTS[ch.part].toLowerCase()}
            </span>
          </div>
          <h1 className="lr-h1">{ch.title}</h1>
          <p className="lr-q">{ch.q}</p>
        </header>
        <div className="lr-body">
          <C />
          <CheckYourself qs={checksFor(ch.slug)} />
        </div>
        <nav className="lr-pn" aria-label="previous and next chapter">
          {prev ? (
            <a href={chapterHref(prev)} onClick={linkTo(chapterHref(prev))} className="prev">
              <span className="k">← previous · {chapterNum(prev)}</span>
              <span className="t">{prev.title}</span>
            </a>
          ) : (
            <a href="/learn" onClick={linkTo("/learn")} className="prev">
              <span className="k">← contents</span>
              <span className="t">The course</span>
            </a>
          )}
          {next ? (
            <a href={chapterHref(next)} onClick={linkTo(chapterHref(next))} className="next">
              <span className="k">next · {chapterNum(next)} →</span>
              <span className="t">{next.title}</span>
              <span className="q">{next.q}</span>
            </a>
          ) : (
            <a href="/lab" className="next">
              <span className="k">next · the lab →</span>
              <span className="t">The lab: every stage, running in your tab</span>
              <span className="q">The same engine with nothing simplified away.</span>
            </a>
          )}
        </nav>
        <p className="lr-hint nv-muted">
          ← → to move between chapters ·{ask ? " select any text to ask about it ·" : ""}{" "}
          <a href={GUIDE_HREF} onClick={linkTo(GUIDE_HREF)}>
            field guide
          </a>
        </p>
      </article>
    </>
  );
}

function Index() {
  const [visited] = useState(readVisited);
  const firstUnseen = CHAPTERS.find((c) => !visited.has(c.slug)) ?? CHAPTERS[0];
  const pct = climbPct(useLearnData());
  return (
    <div className="lr-index" id="content" tabIndex={-1} data-askable data-ask-title="The course: contents">
      <header className="nv-col lr-ihero" data-ask-section>
        <div className="nv-kicker">The course · {CHAPTERS.length} short chapters in 4 parts · about 90 minutes</div>
        <h1 className="nv-h1">How a language model runs, from the ground up.</h1>
        <p className="nv-lead">
          One idea per page, each with a figure to explore. Part 0 frames the field for people who know systems but not ML:
          what a model is, why serving one is expensive, the hardware and the jargon. Part 1 builds a machine that guesses the
          next token. Part 2 shows why it is slow and how to work out the speed limit of any model on any computer. Part 3
          follows our engine {pct ? `from ${pct.first}% to ${pct.best}% of that limit` : "towards that limit"}. No maths beyond
          multiplying and adding.
        </p>
        <div className="lr-start">
          <a className="nv-btn" href={chapterHref(firstUnseen)} onClick={linkTo(chapterHref(firstUnseen))}>
            {visited.size ? "Continue" : "Start"} · {firstUnseen.title} →
          </a>
          {visited.size > 0 && (
            <span className="nv-muted" style={{ fontSize: "var(--t-sm)" }}>
              {visited.size} of {CHAPTERS.length} read
            </span>
          )}
        </div>
      </header>
      <div className="nv-col lr-toc">
        <Journey showQuestions />
        <a className="lr-guide-card" href={GUIDE_HREF} onClick={linkTo(GUIDE_HREF)} data-ask-section>
          <span className="k">reference</span>
          <span className="t">The field guide →</span>
          <span className="q">
            MoE, GQA, bf16, AWQ, FlashAttention, TTFT, goodput, NVLink and the rest, each in one plain sentence, with the
            limit it works around.
          </span>
        </a>
        <p className="nv-muted lr-foot">
          Numbers come from nanoinfer's own logs: GPT-2 small on an Apple M4 Mac mini (CPU and GPU), and the same Rust compiled
          to WebAssembly in your browser. Toy figures are labelled as toys.
        </p>
      </div>
    </div>
  );
}
