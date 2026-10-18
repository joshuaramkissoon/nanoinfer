import { CHAPTERS, PARTS, PART_BLURB, chapterHref } from "../learn/chapters";
import { linkTo } from "../learn/router";

const VISITED_KEY = "nanoinfer-learn-visited";
export function readVisited(): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem(VISITED_KEY) ?? "[]"));
  } catch {
    return new Set();
  }
}

/** The map of the course: the three parts as stops on a route, then the lab. */
export function Journey({ showQuestions = false }: { showQuestions?: boolean }) {
  const visited = readVisited();
  return (
    <ol className="jr">
      {PARTS.map((p, pi) => (
        <li key={p} className="jr-part" data-ask-section>
          <span className="jr-node" aria-hidden>
            {pi + 1}
          </span>
          <div className="jr-head">
            <span className="jr-k">part {pi + 1}</span>
            <h3>{p}</h3>
            <p>{PART_BLURB[pi]}</p>
          </div>
          <ol className="jr-ch">
            {CHAPTERS.map((c, i) =>
              c.part !== pi ? null : (
                <li key={c.slug}>
                  <a href={chapterHref(c)} onClick={linkTo(chapterHref(c))} className={visited.has(c.slug) ? "seen" : ""}>
                    <span className="n">{String(i + 1).padStart(2, "0")}</span>
                    <span className="t">{c.title}</span>
                    {showQuestions && <span className="q">{c.q}</span>}
                  </a>
                </li>
              ),
            )}
          </ol>
        </li>
      ))}
      <li className="jr-part lab" data-ask-section>
        <span className="jr-node" aria-hidden>
          4
        </span>
        <div className="jr-head">
          <span className="jr-k">part 4</span>
          <h3>
            <a href="/lab" onClick={linkTo("/lab")}>
              The lab →
            </a>
          </h3>
          <p>
            The engine room, for after the course: every stage of the engine runnable in your tab, the real per-operation
            timings, all 144 attention heads, the roofline, and the full ladder of results.
          </p>
        </div>
      </li>
    </ol>
  );
}
