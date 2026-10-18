// /learn/field-guide: the reference page. Entry point is the constraint map (what limits inference → the techniques
// that work around it); below it, the terms grouped as calm cards that open into a compact explainer.

import { useEffect, useMemo, useRef, useState } from "react";
import { ConstraintMap, useRoleColor } from "./ConstraintMap";
import { chapterBySlug, chapterHref, chapterNum } from "./chapters";
import { GROUPS, type Entry } from "./guide/entries";
import { linkTo } from "./router";

function Card({ e, open, onToggle }: { e: Entry; open: boolean; onToggle: () => void }) {
  const color = useRoleColor();
  const ch = e.ch ? chapterBySlug(e.ch) : undefined;
  return (
    <div id={e.id} className={`fg-card${open ? " open" : ""}${open && e.big ? " big" : ""}`} style={{ ["--rc" as string]: color(e.role) }} data-ask-section>
      <button className="fg-card-h" onClick={onToggle} aria-expanded={open}>
        <span className="fg-term">
          <i className="dot" />
          {e.term}
        </span>
        {e.aka && <span className="fg-aka">{e.aka}</span>}
        <span className="fg-plain">{e.plain}</span>
      </button>
      {open && (
        <div className="fg-card-b">
          <p>
            <span className="lr-cap-k">why it exists</span>
            {e.why}
          </p>
          {e.visual && <div className="fg-visual">{e.visual()}</div>}
          <p>
            <span className="lr-cap-k">where you'll see it</span>
            {e.where}
          </p>
          {ch && (
            <a className="p0-tchip" href={chapterHref(ch)} onClick={linkTo(chapterHref(ch))}>
              chapter {chapterNum(ch)} · {ch.title} →
            </a>
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

  return (
    <div className="fg" data-askable data-ask-title="Field guide">
      <header className="fg-col fg-hero" data-ask-section>
        <div className="nv-kicker">reference · {count} terms</div>
        <h1 className="nv-h1">
          The field guide: <em>what people mean</em>.
        </h1>
        <p className="nv-lead">
          Each term in one plain sentence, the constraint it exists to work around, and where you'll meet it. Start from the
          map: almost every term is a workaround for one of six limits.
        </p>
      </header>

      <section className="fg-col fg-map" data-ask-section>
        <h2 className="fg-h2">The map: limits → workarounds</h2>
        <ConstraintMap onPick={(id) => openId(id)} />
        <p className="lr-small fg-legend">
          <span className="sw" style={{ background: "var(--fg-mem)" }} /> memory <span className="sw" style={{ background: "var(--fg-cmp)" }} /> compute{" "}
          <span className="sw" style={{ background: "var(--fg-comm)" }} /> communication <span className="sw" style={{ background: "var(--m-muted)" }} /> time · the
          same colours mark every card below
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
          <h2 className="fg-h2">
            {g.name} <span className="nv-muted">· {g.blurb}</span>
          </h2>
          <div className="fg-grid">
            {g.entries.map((e) => (
              <Card key={e.id} e={e} open={open === e.id} onToggle={() => (open === e.id ? setOpen(null) : openId(e.id, false))} />
            ))}
          </div>
        </section>
      ))}
      {!groups.length && <p className="fg-col nv-muted">No term matches “{q}”.</p>}

      <p className="fg-col lr-hint nv-muted">
        New here? The course starts with{" "}
        <a href="/learn/what-is-inference" onClick={linkTo("/learn/what-is-inference")}>
          part 0: inference for systems engineers
        </a>
        . Select any text to ask about it.
      </p>
    </div>
  );
}
