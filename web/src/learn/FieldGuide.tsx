// /learn/field-guide: the reference page. Entry point is the constraint map (what limits inference → the techniques
// that work around it); below it, the terms grouped as a dictionary: term on the left, one plain sentence beside it,
// the chapter on the right. An entry opens in place into why it exists and where you'll meet it.

import { useEffect, useMemo, useRef, useState } from "react";
import { useAskAvailable } from "../ask/AskLayer";
import { ConstraintMap } from "./ConstraintMap";
import { chapterBySlug, chapterHref, chapterNum } from "./chapters";
import { chartColors } from "./kit";
import { GROUPS, type Entry } from "./guide/entries";
import { linkTo } from "./router";
import { usePalette } from "../ui/theme";

function Term({ e, open, onToggle }: { e: Entry; open: boolean; onToggle: () => void }) {
  const ch = e.ch ? chapterBySlug(e.ch) : undefined;
  return (
    <div id={e.id} className={`fg-entry${open ? " open" : ""}${open && e.big ? " big" : ""}`} data-ask-section>
      <button className="fg-entry-h" onClick={onToggle} aria-expanded={open}>
        <span className="fg-term">
          {e.term}
          {e.aka && <span className="fg-aka">{e.aka}</span>}
        </span>
        <span className="fg-plain">{e.plain}</span>
        <span className="fg-ch">{ch ? `ch ${chapterNum(ch)}` : ""}</span>
      </button>
      {open && (
        <div className="fg-entry-b">
          <p>
            <span className="lr-cap-k">Why it exists</span>
            {e.why}
          </p>
          {e.visual && <div className="fg-visual">{e.visual()}</div>}
          <p>
            <span className="lr-cap-k">Where you'll see it</span>
            {e.where}
          </p>
          {ch && (
            <p>
              <a className="p0-tchip" href={chapterHref(ch)} onClick={linkTo(chapterHref(ch))}>
                chapter {chapterNum(ch)} · {ch.title} →
              </a>
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default function FieldGuide() {
  const [open, setOpen] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const scrollTo = useRef<string | null>(null);
  const smooth = useRef(true);

  const openId = (id: string, scroll = true) => {
    setOpen(id);
    setQ("");
    if (scroll) scrollTo.current = id;
    try {
      history.replaceState(null, "", `/learn/field-guide#${id}`);
    } catch {
      /* ignore */
    }
  };

  // deep links: /learn/field-guide#moe (on load, and when a tooltip link changes the hash)
  useEffect(() => {
    const fromHash = () => {
      const id = decodeURIComponent(location.hash.slice(1));
      if (id && document.getElementById(id)) {
        smooth.current = false;
        openId(id);
      }
    };
    fromHash();
    window.addEventListener("popstate", fromHash);
    window.addEventListener("hashchange", fromHash);
    return () => {
      window.removeEventListener("popstate", fromHash);
      window.removeEventListener("hashchange", fromHash);
    };
  }, []);

  useEffect(() => {
    const id = scrollTo.current;
    if (!id) return;
    scrollTo.current = null;
    const behavior = smooth.current ? "smooth" : "auto";
    smooth.current = true;
    requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView({ block: "start", behavior }));
  }, [open]);

  const needle = q.trim().toLowerCase();
  const groups = useMemo(
    () =>
      GROUPS.map((g) => ({
        ...g,
        entries: needle ? g.entries.filter((e) => `${e.term} ${e.aka ?? ""} ${e.plain}`.toLowerCase().includes(needle)) : g.entries,
      })).filter((g) => g.entries.length),
    [needle],
  );
  const count = GROUPS.reduce((a, g) => a + g.entries.length, 0);
  const ask = useAskAvailable();
  const cc = chartColors(usePalette().dark);

  return (
    <div className="fg" data-askable data-ask-title="Field guide">
      <header className="fg-col fg-hero" data-ask-section>
        <div className="nv-kicker">reference · {count} terms</div>
        <h1 className="nv-h1">The field guide</h1>
        <p className="nv-lead">
          Each term in one plain sentence, the constraint it exists to work around, and where you'll meet it. Start from the
          map: almost every term is a workaround for one of six limits.
        </p>
      </header>

      <section className="fg-col fg-map" data-ask-section>
        <h2 className="fg-h2">The map: six limits and their workarounds</h2>
        <ConstraintMap onPick={(id) => openId(id)} />
        <p className="lr-small fg-legend">
          <span className="sw" style={{ background: cc.c2 }} /> memory <span className="sw" style={{ background: cc.c1 }} /> compute{" "}
          <span className="sw" style={{ background: cc.c3 }} /> communication
        </p>
      </section>

      <div className="fg-col fg-search">
        <input className="lr-input" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter: try “cache”, “bits” or “GPU”" aria-label="filter terms" spellCheck={false} />
        <nav className="fg-jump lr-small" aria-label="groups">
          {GROUPS.map((g) => (
            <a key={g.key} href={`#g-${g.key}`} onClick={(e) => (e.preventDefault(), document.getElementById(`g-${g.key}`)?.scrollIntoView({ behavior: "smooth" }))}>
              {g.name}
            </a>
          ))}
        </nav>
      </div>

      {groups.map((g) => (
        <section key={g.key} id={`g-${g.key}`} className="fg-col fg-group">
          <h2 className="fg-h2">{g.name}</h2>
          <p className="fg-blurb">{g.blurb}</p>
          <div className="fg-list">
            {g.entries.map((e) => (
              <Term key={e.id} e={e} open={open === e.id} onToggle={() => (open === e.id ? setOpen(null) : openId(e.id, false))} />
            ))}
          </div>
        </section>
      ))}
      {!groups.length && <p className="fg-col nv-muted">No term matches “{q}”.</p>}

      <p className="fg-col lr-hint nv-muted">
        If you are new, the course starts with{" "}
        <a href="/learn/what-is-inference" onClick={linkTo("/learn/what-is-inference")}>
          part 0: inference for systems engineers
        </a>
        .{ask && " Select any text to ask about it."}
      </p>
    </div>
  );
}
