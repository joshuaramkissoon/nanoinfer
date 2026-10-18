// A tiny, safe markdown renderer for streamed answers: paragraphs, lists, code
// fences, `code`, **bold**, *italic*, ==highlight==. Builds React nodes (no HTML strings).

import type { ReactNode } from "react";

function inline(s: string, key = 0): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(`[^`]+`|\*\*[^*]+\*\*|==[^=]+==|\*[^*\s][^*]*\*)/g;
  let last = 0, m: RegExpExecArray | null, k = key;
  while ((m = re.exec(s))) {
    if (m.index > last) out.push(s.slice(last, m.index));
    const t = m[0];
    if (t.startsWith("`")) out.push(<code key={k++}>{t.slice(1, -1)}</code>);
    else if (t.startsWith("**")) out.push(<b key={k++}>{t.slice(2, -2)}</b>);
    else if (t.startsWith("==")) out.push(<mark key={k++}>{t.slice(2, -2)}</mark>);
    else out.push(<i key={k++}>{t.slice(1, -1)}</i>);
    last = m.index + t.length;
  }
  if (last < s.length) out.push(s.slice(last));
  return out;
}

export function Markdown({ src }: { src: string }) {
  const blocks: ReactNode[] = [];
  const parts = src.split(/(```[\s\S]*?(?:```|$))/);
  let k = 0;
  for (const part of parts) {
    if (part.startsWith("```")) {
      const body = part.replace(/^```[^\n]*\n?/, "").replace(/```$/, "");
      blocks.push(
        <pre key={k++}>
          <code>{body}</code>
        </pre>,
      );
      continue;
    }
    for (const para of part.split(/\n{2,}/)) {
      const lines = para.split("\n").filter((l) => l.trim());
      if (!lines.length) continue;
      if (lines.every((l) => /^\s*([-*]|\d+\.)\s/.test(l))) {
        blocks.push(
          <ul key={k++}>
            {lines.map((l, i) => (
              <li key={i}>{inline(l.replace(/^\s*([-*]|\d+\.)\s/, ""))}</li>
            ))}
          </ul>,
        );
      } else blocks.push(<p key={k++}>{inline(lines.join(" "))}</p>);
    }
  }
  return <>{blocks}</>;
}
