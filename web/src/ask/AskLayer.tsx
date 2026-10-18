// Select any text inside [data-askable] → a small toolbar with the same actions as
// Tangent: Ask (type a question), Explain, Define, Simplify ¶ (the whole paragraph in
// plain words), ✦ Deeper (a short reading on it) and a hold-to-talk mic. Answers stream
// into a side panel, with the surrounding section ([data-ask-section]) sent as context.
// The model call goes through /api/ask (server/ask.ts), so no key ever reaches the browser.

import { useCallback, useEffect, useRef, useState } from "react";
import { Markdown } from "./md";
import "./ask.css";

interface Pending {
  text: string;
  /** the paragraph the selection sits in (what "Simplify ¶" rewrites) */
  para: string;
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
type Status = { available: boolean; keyEnv: string; model: string; provider: string } | null;

/** In production the tutor only exists when the server has a key; setup hints are for local dev. */
const DEV = import.meta.env.DEV;

let statusP: Promise<Status> | null = null;
/** GET /api/ask once per page; shared by the layer and useAskAvailable. */
function fetchStatus(): Promise<Status> {
  statusP ??= fetch("/api/ask")
    .then((r) => (r.ok ? r.json() : null))
    .then((j) =>
      j
        ? { available: !!j.available, keyEnv: j.keyEnv ?? "TOGETHER_API_KEY", model: j.model ?? "", provider: j.provider ?? "" }
        : { available: false, keyEnv: "TOGETHER_API_KEY", model: "", provider: "" },
    )
    .catch(() => ({ available: false, keyEnv: "TOGETHER_API_KEY", model: "", provider: "" }));
  return statusP;
}

/** Is select-to-ask usable here? For hints like "select any text to ask about it":
 *  false until known, and false in production without a server key. */
export function useAskAvailable(): boolean {
  const [ok, setOk] = useState(false);
  useEffect(() => {
    let live = true;
    void fetchStatus().then((st) => live && setOk(!!st?.available));
    return () => {
      live = false;
    };
  }, []);
  return ok;
}

/** How each one-click action is labelled in the answers panel (Tangent's labels). */
const LABEL: Record<string, string> = { explain: "Explain", define: "Define", simplify: "Plain words", deeper: "Deeper" };

const PARA = "p, li, blockquote, figcaption, td, h1, h2, h3, .lr-idea, .lr-q, .nv-lead";

function contextOf(node: Node | null): { context: string; title: string; para: string } {
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
  const p = (el?.closest(PARA) as HTMLElement | null) ?? sec ?? el;
  return { context: context.slice(0, 6000), title, para: (p?.innerText ?? "").replace(/\s+/g, " ").trim().slice(0, 1500) };
}

// ---- voice: the browser's own speech recognition (Chrome, Edge, Safari) ----------------

interface Recognition {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onresult: ((e: { resultIndex: number; results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }> }) => void) | null;
  onerror: ((e: { error: string }) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}
const SpeechRec = (window as unknown as { SpeechRecognition?: new () => Recognition; webkitSpeechRecognition?: new () => Recognition }).SpeechRecognition ??
  (window as unknown as { webkitSpeechRecognition?: new () => Recognition }).webkitSpeechRecognition;

function MicIcon() {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden>
      <path
        fill="currentColor"
        d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.9V21h2v-2.1a7 7 0 0 0 6-6.9h-2Z"
      />
    </svg>
  );
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
  /** voice: null, or what's been heard so far */
  const [heard, setHeard] = useState<string | null>(null);
  const [voiceMsg, setVoiceMsg] = useState("");
  const bar = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const turnsRef = useRef(turns);
  turnsRef.current = turns;
  const pendingRef = useRef(pending);
  pendingRef.current = pending;
  const voice = useRef<{ rec: Recognition | null; text: string; startedAt: number; toggled: boolean; send: boolean }>({ rec: null, text: "", startedAt: 0, toggled: false, send: false });

  useEffect(() => {
    void fetchStatus().then(setStatus);
  }, []);
  // The selection toolbar appears only where the tutor works (always in dev, to show the setup hint).
  const enabledRef = useRef(false);
  enabledRef.current = DEV || !!status?.available;

  // selection → toolbar
  useEffect(() => {
    let t: ReturnType<typeof setTimeout>;
    const on = () => {
      clearTimeout(t);
      t = setTimeout(() => {
        if (!enabledRef.current || document.activeElement === input.current || voice.current.rec) return;
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
        setPending({ text, rect: range.getBoundingClientRect(), ...contextOf(range.startContainer) });
        setAsking(false);
        setQ("");
      }, 140);
    };
    document.addEventListener("selectionchange", on);
    const hide = () => document.activeElement !== input.current && !voice.current.rec && setPending(null);
    window.addEventListener("scroll", hide, { passive: true });
    return () => {
      document.removeEventListener("selectionchange", on);
      window.removeEventListener("scroll", hide);
    };
  }, []);

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
    const pend = pendingRef.current;
    if (!pend) return;
    // "Simplify ¶" works on the whole paragraph, not just the selected words
    const sel = kind === "simplify" ? pend.para || pend.text : pend.text;
    const turn: Turn = { id: ++seq, kind, sel, q: question, context: pend.context, title: pend.title, a: "", done: false, error: null };
    setTurns((ts) => [...ts, turn]);
    setOpen(true);
    setPending(null);
    setAsking(false);
    getSelection()?.removeAllRanges();
    if (status?.available) void run(turn, []);
    setTimeout(() => scroller.current && (scroller.current.scrollTop = scroller.current.scrollHeight), 30);
  };
  const askRef = useRef(ask);
  askRef.current = ask;

  const openAsk = (initial: string) => {
    setQ(initial);
    setAsking(true);
  };

  // ---- voice: hold the mic (or V) and talk; a quick tap keeps listening until the next tap
  const stopVoice = useCallback((send: boolean) => {
    const v = voice.current;
    if (!v.rec) return;
    v.send = send;
    v.rec.stop();
  }, []);
  const startVoice = useCallback(() => {
    const v = voice.current;
    if (v.rec || !pendingRef.current) return;
    if (!SpeechRec) {
      setVoiceMsg("Voice isn't supported in this browser");
      setTimeout(() => setVoiceMsg(""), 2500);
      return;
    }
    const rec = new SpeechRec();
    rec.lang = navigator.language || "en-GB";
    rec.interimResults = true;
    rec.continuous = true;
    Object.assign(v, { rec, text: "", startedAt: Date.now(), toggled: false, send: false });
    setHeard("");
    rec.onresult = (e) => {
      let text = "";
      for (let i = 0; i < e.results.length; i++) text += e.results[i][0].transcript;
      v.text = text.trim();
      setHeard(v.text);
    };
    rec.onerror = (e) => {
      if (e.error === "not-allowed" || e.error === "service-not-allowed") setVoiceMsg("Microphone access was blocked");
      else if (e.error !== "aborted" && e.error !== "no-speech") setVoiceMsg("Couldn't hear that. Try again");
      v.send = false;
      setTimeout(() => setVoiceMsg(""), 2500);
    };
    rec.onend = () => {
      const text = v.text;
      const send = v.send && !!text;
      v.rec = null;
      setHeard(null);
      if (send) askRef.current("ask", text);
    };
    try {
      rec.start();
    } catch {
      v.rec = null;
      setHeard(null);
    }
  }, []);
  const micDown = (e: React.PointerEvent) => {
    e.preventDefault();
    if (voice.current.rec && voice.current.toggled) return stopVoice(true);
    startVoice();
  };
  const micUp = () => {
    const v = voice.current;
    if (!v.rec) return;
    if (Date.now() - v.startedAt < 350) v.toggled = true;
    else stopVoice(true);
  };

  // keys: Esc closes; typing while the toolbar is up starts a question; hold V to talk
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (voice.current.rec) stopVoice(false);
        if (pendingRef.current) {
          setPending(null);
          getSelection()?.removeAllRanges();
        } else if (open) setOpen(false);
        return;
      }
      if (!pendingRef.current || asking || e.metaKey || e.ctrlKey || e.altKey) return;
      if ((e.target as HTMLElement).closest?.("input, textarea, select, [contenteditable]")) return;
      if (e.code === "KeyV") {
        e.preventDefault();
        if (!e.repeat) startVoice();
        return;
      }
      if (e.key.length === 1 && !voice.current.rec) {
        e.preventDefault();
        openAsk(e.key);
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "KeyV" && voice.current.rec && !voice.current.toggled) stopVoice(true);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [open, asking, startVoice, stopVoice]);

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
    const w = asking || heard != null ? 320 : 430, coarse = matchMedia("(pointer: coarse)").matches;
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
              <input
                ref={input}
                autoFocus
                value={q}
                onChange={(e) => setQ(e.target.value)}
                onBlur={() => setTimeout(() => document.activeElement !== input.current && !input.current?.value && setPending(null), 150)}
                placeholder="Ask about this…"
                aria-label="your question"
              />
              <button type="submit" className="ask-send" disabled={!q.trim()} aria-label="ask">
                ↵
              </button>
            </form>
          ) : heard != null ? (
            <div className="ask-listening" aria-live="polite">
              <span className="ask-rec" aria-hidden />
              <span className={heard ? "" : "ask-hint"}>{heard || (voice.current.toggled ? "Listening. Tap the mic again to send" : "Listening…")}</span>
              <button className="mic held" onPointerDown={micDown} onPointerUp={micUp} aria-label="stop and send">
                <MicIcon />
              </button>
            </div>
          ) : voiceMsg ? (
            <div className="ask-listening">
              <span className="ask-hint">{voiceMsg}</span>
            </div>
          ) : (
            <>
              <span className="ask-ai" title="Answers come from an AI model and can be wrong">AI</span>
              <button className="ask-primary" onClick={() => openAsk("")}>
                Ask
              </button>
              <button onClick={() => ask("explain")}>Explain</button>
              <button onClick={() => ask("define")}>Define</button>
              <button onClick={() => ask("simplify")} title="Plain-words version of this paragraph">
                Simplify ¶
              </button>
              <button className="ask-deeper" onClick={() => ask("deeper")} title="Go down the rabbit hole: a short reading on this">
                <span className="spark">✦</span> Deeper
              </button>
              <button className="mic" onPointerDown={micDown} onPointerUp={micUp} onContextMenu={(e) => e.preventDefault()} aria-label="Hold to ask out loud" title="Hold to talk (or hold V)">
                <MicIcon />
              </button>
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
          {status && !status.available && DEV && (
            <div className="ask-off">
              <b>Ask isn't switched on here.</b> It needs a model API key on the server (never in the browser). For local dev, put{" "}
              <code>{status.keyEnv}=…</code> in <code>web/.env.local</code> and restart <code>npm run dev</code>; on Vercel, add it
              to the project's environment variables. (Only shown in dev: in production the tutor is hidden without a key.)
            </div>
          )}
          {status && !status.available && !DEV && <div className="ask-off">The AI tutor is unavailable right now.</div>}
          {status?.available && (
            <div className="ask-note">
              Answers are written by an AI model ({(model || status.model).split("/").pop()}
              {status.provider ? ` via ${status.provider}` : ""}), paid for by this site. They can be wrong: check them against the page.
            </div>
          )}
          {!turns.length && status?.available && <div className="ask-empty">Select any text on the page, then ask about it.</div>}
          {turns.map((t, i) => {
            const newThread = i === 0 || turns[i - 1].sel !== t.sel;
            return (
              <section key={t.id} className="ask-turn">
                {newThread && <blockquote>{t.sel.length > 220 ? t.sel.slice(0, 220) + "…" : t.sel}</blockquote>}
                <div className="ask-q">{t.kind === "ask" || t.kind === "followup" ? t.q : (LABEL[t.kind] ?? t.kind)}</div>
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
