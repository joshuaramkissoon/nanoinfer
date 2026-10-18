// Select any text inside [data-askable] → a small toolbar → ask about it, or one
// click for "simpler" / "why?" / "example". Answers stream into a side panel, with
// the surrounding section ([data-ask-section]) sent as context. The model call goes
// through /api/ask (server/ask.ts), so no key ever reaches the browser.

import { useCallback, useEffect, useRef, useState } from "react";
import { Markdown } from "./md";
import "./ask.css";

interface Pending {
  text: string;
  rect: DOMRect;
  context: string;
  title: string;
}
interface Turn {
  id: number;
  kind: string;
  sel: string;
  q: string;
  context: string;
  title: string;
  a: string;
  done: boolean;
  error: string | null;
}
type Status = { available: boolean; keyEnv: string; model: string } | null;

const LABEL: Record<string, string> = { simpler: "Explain it more simply", why: "Why?", example: "Show me an example" };

function contextOf(node: Node | null): { context: string; title: string } {
  const el = ((node?.nodeType === 3 ? node.parentElement : node) as HTMLElement | null) ?? null;
  const sec = el?.closest("[data-ask-section]") as HTMLElement | null;
  const root = el?.closest("[data-askable]") as HTMLElement | null;
  const title = root?.dataset.askTitle ?? document.title;
  // the section itself, plus the paragraphs right around it, for a little more context
  let context = (sec?.innerText ?? el?.innerText ?? "").trim();
  if (sec && context.length < 1500) {
    const prev = sec.previousElementSibling as HTMLElement | null;
    const next = sec.nextElementSibling as HTMLElement | null;
    context = [prev?.innerText ?? "", context, next?.innerText ?? ""].filter(Boolean).join("\n\n");
  }
  return { context: context.slice(0, 6000), title };
}

let seq = 0;

export function AskLayer() {
  const [pending, setPending] = useState<Pending | null>(null);
  const [asking, setAsking] = useState(false);
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [status, setStatus] = useState<Status>(null);
  const [mode, setMode] = useState<"quick" | "think">("quick");
  const [follow, setFollow] = useState("");
  const [model, setModel] = useState("");
  const bar = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const turnsRef = useRef(turns);
  turnsRef.current = turns;

  useEffect(() => {
    fetch("/api/ask")
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => setStatus(j ? { available: !!j.available, keyEnv: j.keyEnv ?? "TOGETHER_API_KEY", model: j.model ?? "" } : { available: false, keyEnv: "TOGETHER_API_KEY", model: "" }))
      .catch(() => setStatus({ available: false, keyEnv: "TOGETHER_API_KEY", model: "" }));
  }, []);

  // selection → toolbar
  useEffect(() => {
    let t: ReturnType<typeof setTimeout>;
    const on = () => {
      clearTimeout(t);
      t = setTimeout(() => {
        if (document.activeElement === input.current) return;
        const sel = getSelection();
        if (!sel || !sel.rangeCount || sel.isCollapsed) {
          setPending(null);
          setAsking(false);
          return;
        }
        const range = sel.getRangeAt(0);
        const start = range.startContainer.nodeType === 3 ? range.startContainer.parentElement : (range.startContainer as Element);
        if (!start?.closest("[data-askable]") || start.closest(".ask-panel, input, textarea, button, .ask-bar")) return;
        const text = sel.toString().replace(/\s+/g, " ").trim();
        if (text.length < 2 || text.length > 1500) return;
        setPending({ text, rect: range.getBoundingClientRect(), ...contextOf(range.commonAncestorContainer) });
        setAsking(false);
        setQ("");
      }, 140);
    };
    document.addEventListener("selectionchange", on);
    const hide = () => document.activeElement !== input.current && setPending(null);
    window.addEventListener("scroll", hide, { passive: true });
    return () => {
      document.removeEventListener("selectionchange", on);
      window.removeEventListener("scroll", hide);
    };
  }, []);

  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (pending) {
        setPending(null);
        getSelection()?.removeAllRanges();
      } else if (open) setOpen(false);
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [pending, open]);

  const run = useCallback(
    async (turn: Turn, history: { role: string; content: string }[]) => {
      const patch = (p: Partial<Turn>) => setTurns((ts) => ts.map((x) => (x.id === turn.id ? { ...x, ...p } : x)));
      const ctrl = new AbortController();
      let stall = setTimeout(() => ctrl.abort(), 30_000);
      try {
        const r = await fetch("/api/ask", {
          method: "POST",
          signal: ctrl.signal,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ kind: turn.kind, selection: turn.sel, question: turn.q, title: turn.title, context: turn.context, mode, history }),
        });
        if (!r.ok) {
          let msg = r.statusText;
          try {
            msg = (await r.json()).error || msg;
          } catch {
            /* keep statusText */
          }
          throw new Error(msg);
        }
        setModel(r.headers.get("X-Model") ?? "");
        const reader = r.body!.getReader();
        const dec = new TextDecoder();
        let a = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          clearTimeout(stall);
          stall = setTimeout(() => ctrl.abort(), 30_000);
          a += dec.decode(value, { stream: true });
          const z = a.indexOf("\u0000");
          if (z >= 0) {
            console.warn("ask:", a.slice(z + 1));
            a = a.slice(0, z);
            patch({ a });
            throw new Error("The model stopped partway. Try again.");
          }
          patch({ a });
          const sc = scroller.current;
          if (sc && sc.scrollHeight - sc.scrollTop - sc.clientHeight < 80) sc.scrollTop = sc.scrollHeight;
        }
        patch({ done: true });
      } catch (e) {
        const err = e as Error;
        patch({ done: true, error: err.name === "AbortError" ? "This took too long; the connection may have dropped." : err instanceof TypeError ? "Couldn't reach the server." : err.message });
      } finally {
        clearTimeout(stall);
      }
    },
    [mode],
  );

  const ask = (kind: string, question = "") => {
    if (!pending) return;
    const turn: Turn = { id: ++seq, kind, sel: pending.text, q: question, context: pending.context, title: pending.title, a: "", done: false, error: null };
    setTurns((ts) => [...ts, turn]);
    setOpen(true);
    setPending(null);
    setAsking(false);
    getSelection()?.removeAllRanges();
    if (status?.available) void run(turn, []);
    setTimeout(() => scroller.current && (scroller.current.scrollTop = scroller.current.scrollHeight), 30);
  };

  const followUp = () => {
    const last = turnsRef.current[turnsRef.current.length - 1];
    const text = follow.trim();
    if (!last || !text) return;
    const history: { role: string; content: string }[] = [];
    for (const t of turnsRef.current) {
      if (t.sel !== last.sel || t.error) continue;
      history.push({ role: "user", content: t.kind === "followup" ? t.q : `${LABEL[t.kind] ?? t.q} (“${t.sel}”)` }, { role: "assistant", content: t.a });
    }
    const turn: Turn = { id: ++seq, kind: "followup", sel: last.sel, q: text, context: last.context, title: last.title, a: "", done: false, error: null };
    setTurns((ts) => [...ts, turn]);
    setFollow("");
    if (status?.available) void run(turn, history);
  };

  // toolbar position
  let style: React.CSSProperties | undefined;
  if (pending) {
    const w = asking ? 300 : 330, coarse = matchMedia("(pointer: coarse)").matches;
    let top = coarse ? pending.rect.bottom + 10 : pending.rect.top - 48;
    if (top < 10) top = pending.rect.bottom + 10;
    style = { top, left: Math.max(8, Math.min(innerWidth - w - 8, pending.rect.left + pending.rect.width / 2 - w / 2)) };
  }

  return (
    <>
      {pending && (
        <div className="ask-bar" ref={bar} style={style} role="toolbar" aria-label="Ask about the selection" onMouseDown={(e) => !(e.target as HTMLElement).closest("input") && e.preventDefault()}>
          {asking ? (
            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (q.trim()) ask("ask", q.trim());
              }}
            >
              <input ref={input} autoFocus value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about this…" aria-label="your question" />
              <button type="submit" disabled={!q.trim()}>
                ↵
              </button>
            </form>
          ) : (
            <>
              <button className="primary" onClick={() => setAsking(true)}>
                <span className="spark">✦</span> Ask
              </button>
              <button onClick={() => ask("simpler")}>Simpler</button>
              <button onClick={() => ask("why")}>Why?</button>
              <button onClick={() => ask("example")}>Example</button>
            </>
          )}
        </div>
      )}
      {!open && turns.length > 0 && (
        <button className="ask-reopen" onClick={() => setOpen(true)}>
          <span className="spark">✦</span> {turns.length} {turns.length === 1 ? "answer" : "answers"}
        </button>
      )}
      <aside className={`ask-panel${open ? " open" : ""}`} aria-hidden={!open} aria-label="Answers">
        <header>
          <span>
            <span className="spark">✦</span> Ask
            {model && <span className="ask-model"> · {model.split("/").pop()}</span>}
          </span>
          <span style={{ display: "flex", gap: "0.4rem", alignItems: "center" }}>
            <span className="ask-modes" role="group" aria-label="answer mode">
              <button aria-pressed={mode === "quick"} onClick={() => setMode("quick")} title="fast answer">
                quick
              </button>
              <button aria-pressed={mode === "think"} onClick={() => setMode("think")} title="the model thinks first: slower, more careful">
                think
              </button>
            </span>
            <button className="ask-x" onClick={() => setOpen(false)} aria-label="close">
              ×
            </button>
          </span>
        </header>
        <div className="ask-scroll" ref={scroller}>
          {status && !status.available && (
            <div className="ask-off">
              <b>Ask isn't switched on here.</b> It needs a model API key on the server (never in the browser). For local dev, put{" "}
              <code>{status.keyEnv}=…</code> in <code>web/.env.local</code> and restart <code>npm run dev</code>; on Vercel, add it
              to the project's environment variables.
            </div>
          )}
          {!turns.length && status?.available && <div className="ask-empty">Select any text on the page, then ask about it.</div>}
          {turns.map((t, i) => {
            const newThread = i === 0 || turns[i - 1].sel !== t.sel;
            return (
              <section key={t.id} className="ask-turn">
                {newThread && <blockquote>{t.sel.length > 220 ? t.sel.slice(0, 220) + "…" : t.sel}</blockquote>}
                <div className="ask-q">{t.kind === "ask" || t.kind === "followup" ? t.q : LABEL[t.kind]}</div>
                <div className="ask-a">
                  {t.a ? <Markdown src={t.a} /> : !t.done && status?.available ? <span className="ask-dots">thinking</span> : null}
                  {t.error && (
                    <div className="ask-err">
                      {t.error}{" "}
                      <button
                        onClick={() => {
                          setTurns((ts) => ts.map((x) => (x.id === t.id ? { ...x, a: "", done: false, error: null } : x)));
                          void run({ ...t, a: "", done: false, error: null }, []);
                        }}
                      >
                        retry
                      </button>
                    </div>
                  )}
                </div>
              </section>
            );
          })}
        </div>
        {turns.length > 0 && status?.available && (
          <form
            className="ask-follow"
            onSubmit={(e) => {
              e.preventDefault();
              followUp();
            }}
          >
            <input value={follow} onChange={(e) => setFollow(e.target.value)} placeholder="Ask a follow-up…" aria-label="follow-up question" />
          </form>
        )}
      </aside>
    </>
  );
}
