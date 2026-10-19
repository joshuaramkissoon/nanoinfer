import { CHAPTERS, PARTS, PART_BLURB, chapterHref, chapterNum } from "../learn/chapters";
import { linkTo } from "../learn/router";
import { climbPct } from "../learn/Climb";
import { useLearnData } from "../learn/kit";

const VISITED_KEY = "nanoinfer-learn-visited";
export function readVisited(): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem(VISITED_KEY) ?? "[]"));
  } catch {
    return new Set();
  }
}

/** The map of the course: parts 0–3 as stops on a route, then the lab. Part 0 is flagged as the recommended start.
 *  `compact` (the landing) shows each part with its chapter count and leaves the chapter rows to /learn. */
export function Journey({ showQuestions = false, compact = false }: { showQuestions?: boolean; compact?: boolean }) {
  const visited = readVisited();
  // the headline metric: % of the speed limit, first stage → best single-user stage
  const pct = climbPct(useLearnData());
  const climb = pct ? `${pct.first}% → ${pct.best}% of the speed limit` : null;
  return (
    <ol className="jr">
      {PARTS.map((p, pi) => (
        <li key={p} className={`jr-part${pi === 0 ? " p0" : ""}`} data-ask-section>
          <span className="jr-node" aria-hidden>
            {pi}
          </span>
          <div className="jr-head">
            <span className="jr-k">
              part {pi}
              {pi === 0 && <span className="jr-rec"> · start here if you know systems but not ML</span>}
            </span>
            <h3>
              {compact ? (
                <a href={chapterHref(CHAPTERS.find((c) => c.part === pi)!)} onClick={linkTo(chapterHref(CHAPTERS.find((c) => c.part === pi)!))}>
                  {p}
                </a>
              ) : (
                p
              )}
              {compact && (
                <span className="jr-count">
                  {CHAPTERS.filter((c) => c.part === pi).length} chapters{pi === 3 && climb ? ` · ${climb}` : ""}
                </span>
              )}
            </h3>
            <p>{PART_BLURB[pi]}</p>
          </div>
          {!compact && (
          <ol className="jr-ch">
            {CHAPTERS.map((c) =>
              c.part !== pi ? null : (
                <li key={c.slug}>
                  <a href={chapterHref(c)} onClick={linkTo(chapterHref(c))} className={visited.has(c.slug) ? "seen" : ""}>
                    <span className="n">{chapterNum(c)}</span>
                    <span className="t">{c.title}</span>
                    {showQuestions && <span className="q">{c.q}</span>}
                  </a>
                </li>
              ),
            )}
          </ol>
          )}
        </li>
      ))}
      <li className="jr-part lab" data-ask-section>
        <span className="jr-node" aria-hidden>
          →
        </span>
        <div className="jr-head">
          <span className="jr-k">then</span>
          <h3>
            <a href="/lab" onClick={linkTo("/lab")}>
              The lab →
            </a>
          </h3>
          <p>
            For after the course. Every stage of the engine runs in your tab, with real per-operation timings, all 144
            attention heads, the roofline and the full ladder of results.
          </p>
        </div>
      </li>
    </ol>
  );
}
