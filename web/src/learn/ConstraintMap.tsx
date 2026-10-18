// The constraint → technique map, drawn as a stack of rows: a coloured constraint on the left (colour = its role in
// DESIGN.md), the techniques that work around it as chips on the right. Used in chapter 0.2 and the field guide.

import { useState } from "react";
import { usePalette } from "../ui/theme";
import { CONSTRAINTS, techHref, type Role } from "./concepts";
import { chartColors } from "./kit";
import { linkTo } from "./router";

export function useRoleColor() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  return (r: Role) => (r === "memory" ? cc.c2 : r === "compute" ? cc.c1 : r === "comm" ? cc.c3 : pal.muted);
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
              <span className="dot" />
              <b>{c.name}</b>
              <span className="n">{shut ? `${c.techniques.length} techniques` : ""}</span>
              {!shut && <span className="why">{c.why}</span>}
            </button>
          ) : (
            <div className="p0-cmap-c">
              <span className="dot" />
              <b>{c.name}</b>
              <span className="why">{c.why}</span>
            </div>
          )}
          {!shut && (
          <div className="p0-cmap-t">
            {c.techniques.map((t) => {
              const href = techHref(t);
              const guide = t.to.startsWith("#");
              return (
                <a
                  key={t.label}
                  href={href}
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
                  <span className="to">{guide ? "guide" : "ch"}</span>
                </a>
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
