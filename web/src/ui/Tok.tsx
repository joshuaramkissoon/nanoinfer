// Token text with its whitespace made visible. GPT-2 tokens usually carry their
// leading space (" the"), and that space matters, so it's drawn as a small muted
// ␣ bracket rather than hidden or shown as an ambiguous dot. Newlines and tabs get
// ⏎ and ⇥. For SVG/canvas text, use the string forms (showToken / visible), which
// use the same ␣ character.

import { Fragment } from "react";
import { tokenText } from "../engine/client";

const MARK: Record<string, string> = { "\n": "⏎", "\t": "⇥" };

/** Render token text with visible whitespace. Pass the text (`s`) or a token id. */
export function Tk({ s, id }: { s?: string; id?: number }) {
  const text = s ?? (id != null ? tokenText(id) : "");
  const parts = text.split(/([ \n\t])/);
  return (
    <span className="tk-t">
      {parts.map((p, i) =>
        p === " " ? (
          <span key={i} className="tk-sp" role="img" aria-label="space" />
        ) : MARK[p] ? (
          <span key={i} className="tk-ws" aria-label={p === "\n" ? "newline" : "tab"}>
            {MARK[p]}
          </span>
        ) : (
          <Fragment key={i}>{p}</Fragment>
        ),
      )}
    </span>
  );
}

/** The tiny legend that goes with token figures: "␣ = space". */
export function SpaceKey({ className }: { className?: string }) {
  return (
    <span className={`tk-key${className ? ` ${className}` : ""}`}>
      <span className="tk-sp" aria-hidden /> = a space
    </span>
  );
}
