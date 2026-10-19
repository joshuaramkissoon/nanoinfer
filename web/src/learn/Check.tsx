// "Check yourself": 2–3 intuition questions at the end of every chapter. Pick an answer and the reasoning appears;
// nothing is scored or stored. The questions live in checks.ts, keyed by chapter slug.

import { useState, type ReactNode } from "react";
import { CHECKS, type Q } from "./checks";

export const checksFor = (slug: string): Q[] => CHECKS[slug] ?? [];

/** The answers are written right-answer-first; show them in a stable, question-specific order instead. */
function order(q: Q): number[] {
  let h = 0;
  for (const ch of q.q) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  const n = q.options.length;
  const r = h % n;
  return Array.from({ length: n }, (_, i) => (i + r) % n);
}

function Question({ q, i }: { q: Q; i: number }) {
  const [pick, setPick] = useState<number | null>(null);
  const done = pick != null;
  const ord = order(q);
  return (
    <li className={`lr-q1${done ? " done" : ""}`}>
      <p className="lr-q1-q">
        <span className="n">{i + 1}</span>
        {q.q}
      </p>
      <div className="lr-q1-opts" role="group" aria-label={`question ${i + 1}`}>
        {ord.map((j, pos) => {
          const o = q.options[j];
          const state = !done ? "" : j === q.answer ? " right" : j === pick ? " wrong" : " dim";
          return (
            <button key={j} className={`lr-q1-opt${state}`} onClick={() => setPick(j)} aria-pressed={pick === j}>
              <span className="m" aria-hidden>
                {done && j === q.answer ? "✓" : done && j === pick ? "×" : ""}
              </span>
              <span className="nv-sr">{String.fromCharCode(97 + pos)}. </span>
              <span>{o}</span>
            </button>
          );
        })}
      </div>
      {done && (
        <p className="lr-q1-why" aria-live="polite">
          <span className="k">{pick === q.answer ? "Yes." : "Not quite."}</span>
          {q.why}
        </p>
      )}
    </li>
  );
}

export function CheckYourself({ qs, title = "Check yourself" }: { qs: Q[]; title?: ReactNode }) {
  if (!qs.length) return null;
  return (
    <section className="lr-check" data-ask-section>
      <div className="lr-check-head">
        <h2>{title}</h2>
        <span className="lr-small">pick an answer to see the reasoning</span>
      </div>
      <ol>
        {qs.map((q, i) => (
          <Question key={i} q={q} i={i} />
        ))}
      </ol>
    </section>
  );
}
