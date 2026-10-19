// The constraint → technique map, drawn as a stack of hairline rows: the constraint on the left with a small swatch for
// its role (DESIGN.md: memory blue, compute ember, communication olive; anything else gets none), the techniques that
// work around it as plain text links on the right. Used in chapter 0.2 and the field guide.

import { useState } from "react";
import { usePalette } from "../ui/theme";
import { CONSTRAINTS, techHref, type Role } from "./concepts";
import { chartColors } from "./kit";
import { linkTo } from "./router";

export function useRoleColor() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  return (r: Role) => (r === "memory" ? cc.c2 : r === "compute" ? cc.c1 : r === "comm" ? cc.c3 : "transparent");
}

/** `collapsible`: one constraint open at a time (chapter 0.2, where most technique names are still new); the field
 *  guide shows the whole map at once. */
export function ConstraintMap({ onPick, collapsible }: { onPick?: (id: string) => void; collapsible?: boolean }) {
  const color = useRoleColor();
  const [open, setOpen] = useState(CONSTRAINTS[0].key);
  return (
    <div className={`p0-cmap${collapsible ? " fold" : ""}`}>
      {CONSTRAINTS.map((c) => {
        const shut = collapsible && open !== c.key;
        return (
        <div key={c.key} className={`p0-cmap-row${shut ? " shut" : ""}`} style={{ ["--rc" as string]: color(c.role) }}>
          {collapsible ? (
            <button className="p0-cmap-c" aria-expanded={!shut} onClick={() => setOpen(c.key)}>
              <span className={`dot${c.role === "memory" || c.role === "compute" || c.role === "comm" ? "" : " none"}`} aria-hidden />
              <b>{c.name}</b>
              <span className="n">{shut ? `${c.techniques.length} techniques` : ""}</span>
              {!shut && <span className="why">{c.why}</span>}
            </button>
          ) : (
            <div className="p0-cmap-c">
              <span className={`dot${c.role === "memory" || c.role === "compute" || c.role === "comm" ? "" : " none"}`} aria-hidden />
              <b>{c.name}</b>
              <span className="why">{c.why}</span>
            </div>
          )}
          {!shut && (
          <div className="p0-cmap-t">
            {c.techniques.map((t, ti) => {
              const href = techHref(t);
              const guide = t.to.startsWith("#");
              return (
                <span key={t.label}>
                {ti > 0 && <span className="sep" aria-hidden>·</span>}
                <a
                  href={href}
                  title={guide ? "in the field guide" : "the chapter that explains it"}
                  className={`p0-tchip${guide ? " guide" : ""}`}
                  onClick={(e) => {
                    if (guide && onPick && !(e.metaKey || e.ctrlKey)) {
                      e.preventDefault();
                      onPick(t.to.slice(1));
                      return;
                    }
                    linkTo(href)(e);
                  }}
                >
                  {t.label}
                </a>
                </span>
              );
            })}
          </div>
          )}
        </div>
        );
      })}
    </div>
  );
}
