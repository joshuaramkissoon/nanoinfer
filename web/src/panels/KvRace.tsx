import { useEffect, useMemo, useRef, useState } from "react";
import { claim, decodeTokens, engine, release, useEngine } from "../engine/client";
import { LiveButton, NeedsEngine, fitCanvas, useFigureSource, useWidth } from "../ui/Figure";
import { usePalette } from "../ui/theme";
import { fmtBytes, fmtMs, linear, sig } from "../lib/fmt";

interface Lane {
  stage: string;
  slot: number;
  tokens: number[];
  ms: number[];
  pos: number[];
  rows: number;
  used: number;
  kvLen: number;
}

const PROMPT = "The KV cache is";
const BUDGET_MS = 30000;

/**
 * Two stages race on the same prompt under a fair scheduler: whichever lane has
 * used less engine time goes next. Same words come out of both (greedy), so the
 * only difference is cost.
 */
export default function KvRace() {
  const status = useEngine((s) => s.status);
  const stages = useEngine((s) => s.stages);
  const busy = useEngine((s) => s.busy);
  const cfg = useEngine((s) => s.config);
  const pal = usePalette();
  const [a, setA] = useState("naive");
  const [b, setB] = useState("kv-cache");
  const [target, setTarget] = useState(96);
  const [lanes, setLanes] = useState<Lane[] | null>(null);
  const [running, setRunning] = useState(false);
  const stop = useRef(false);
  const [ref, w] = useWidth<HTMLDivElement>();
  const grid = useRef<HTMLCanvasElement>(null);
  const rec = useEngine((s) => s.recorded);
  const ready = status === "ready";
  // Before anything live: the recorded race (naive vs kv-cache, same prompt, greedy).
  const recLanes = useMemo<Lane[] | null>(() => {
    const r = rec?.race;
    if (!r) return null;
    const pick = (st: string) => r.lanes.find((l) => l.stage === st);
    const a0 = pick("naive"), b0 = pick("kv-cache");
    if (!a0 || !b0) return null;
    return [a0, b0].map((l, i) => ({
      stage: l.stage,
      slot: -1,
      tokens: l.tokens,
      ms: l.ms,
      pos: l.tokens.map((_, j) => r.prompt.length + j + 1),
      rows: i === 0 ? l.tokens.reduce((acc, _, j) => acc + r.prompt.length + j + 1, 0) : l.tokens.length,
      used: l.ms.reduce((x, y) => x + y, 0),
      kvLen: i === 0 ? -1 : r.prompt.length + l.tokens.length,
    }));
  }, [rec]);
  const shownLanes = lanes ?? (ready ? null : recLanes);
  const promptText = lanes || !rec?.race ? PROMPT : rec.race.text;
  useFigureSource(lanes ? "live" : shownLanes ? "recorded" : "none");

  useEffect(() => {
    if (!stages.length) return;
    const names = stages.map((s) => s.name);
    if (!names.includes(a)) setA(names[0]);
    if (!names.includes(b)) setB(names[names.length - 1]);
  }, [stages, a, b]);

  const race = async () => {
    if (!claim("race")) return;
    setRunning(true);
    stop.current = false;
    const opened: number[] = [];
    try {
      await engine.capture(0);
      const prompt = await engine.encode(PROMPT);
      const ls: Lane[] = [];
      for (const st of [a, b]) {
        const slot = await engine.open(st, prompt.length + target + 1);
        opened.push(slot);
        ls.push({ stage: st, slot, tokens: [], ms: [], pos: [], rows: 0, used: 0, kvLen: -1 });
      }
      // prefill both (not timed in the race: identical prompt work)
      const next: number[] = [];
      for (const l of ls) {
        await engine.forward(l.slot, prompt);
        next.push((await engine.sample(l.slot, 0, 0)).id);
      }
      setLanes(ls.map((l) => ({ ...l })));
      const t0 = performance.now();
      while (!stop.current && performance.now() - t0 < BUDGET_MS) {
        // fair scheduling: lane with less engine time runs
        const live = ls.map((_, i) => i).filter((i) => ls[i].tokens.length < target);
        if (!live.length) break;
        const i = live.reduce((p, c) => (ls[c].used < ls[p].used ? c : p));
        const l = ls[i];
        const tok = next[i];
        const tr = await engine.forward(l.slot, [tok]);
        l.tokens.push(tok);
        l.ms.push(tr.ms);
        l.pos.push(tr.contextLen);
        l.rows += tr.nComputed;
        l.used += tr.ms;
        l.kvLen = tr.kvLen;
        next[i] = (await engine.sample(l.slot, 0, 0)).id;
        setLanes(ls.map((x) => ({ ...x, tokens: [...x.tokens], ms: [...x.ms], pos: [...x.pos] })));
      }
    } finally {
      for (const s of opened) await engine.close(s);
      await engine.capture(10);
      setRunning(false);
      release("race");
    }
  };

  // KV grid: layers x positions, filling as the cached lane writes
  const kvLane = shownLanes?.find((l) => l.kvLen >= 0);
  useEffect(() => {
    const c = grid.current;
    if (!c || !w || !cfg) return;
    const L = cfg.nLayer;
    const cols = Math.max(target + PROMPT.split(" ").length + 2, 32);
    const cw = Math.max(2, Math.floor((w - 40) / cols));
    const ch = 7;
    const ctx = fitCanvas(c, w, L * (ch + 2) + 4);
    ctx.clearRect(0, 0, w, c.height);
    const filled = kvLane?.kvLen ?? 0;
    ctx.font = "9px JetBrains Mono, monospace";
    ctx.textBaseline = "middle";
    for (let l = 0; l < L; l++) {
      ctx.fillStyle = pal.muted;
      ctx.fillText(`L${l}`, 0, l * (ch + 2) + ch / 2 + 1);
      for (let p = 0; p < cols; p++) {
        ctx.fillStyle = p < filled ? (p === filled - 1 ? pal.accent : pal.cool) : pal.code;
        ctx.globalAlpha = p < filled ? 0.55 + 0.45 * (l % 2) : 1;
        ctx.fillRect(40 + p * cw, l * (ch + 2) + 1, cw - 1, ch);
      }
    }
    ctx.globalAlpha = 1;
  }, [kvLane?.kvLen, w, cfg, pal, target]);

  if (!ready && !shownLanes) return <NeedsEngine>Race two stages on the same prompt and watch one of them drown in recomputation.</NeedsEngine>;

  const H = 220;
  const m = { l: 44, r: 12, t: 10, b: 28 };
  const maxPos = Math.max(lanes ? target + 6 : 0, ...(shownLanes?.flatMap((l) => l.pos) ?? [0]));
  const maxMs = Math.max(10, ...(shownLanes?.flatMap((l) => l.ms) ?? [0]));
  const x = linear(0, maxPos, m.l, (w || 1) - m.r);
  const y = linear(0, maxMs * 1.08, H - m.b, m.t);
  const colors = [pal.up, pal.cool];
  const same =
    shownLanes && shownLanes.length === 2
      ? shownLanes[0].tokens.every((t, i) => i >= shownLanes[1].tokens.length || shownLanes[1].tokens[i] === t)
      : true;

  return (
    <div>
      <div className="nv-controls">
        {running ? (
          <button className="nv-btn" onClick={() => (stop.current = true)}>■ Stop</button>
        ) : ready ? (
          <button className="nv-btn" onClick={race} disabled={!!busy}>▶ Race</button>
        ) : (
          <LiveButton label="race them in your tab" />
        )}
        {[
          [a, setA],
          [b, setB],
        ].map(([v, set], i) => (
          <label className="nv-field" key={i}>
            <i style={{ width: 10, height: 10, background: colors[i], borderRadius: 2, display: "inline-block" }} />
            <select value={v as string} disabled={running} onChange={(e) => (set as (s: string) => void)(e.target.value)}
              style={{ background: "transparent", border: "1px solid var(--m-rule)", borderRadius: 4, padding: "0.3rem" }}>
              {stages.map((s) => <option key={s.name}>{s.name}</option>)}
            </select>
          </label>
        ))}
        <label className="nv-field">
          to
          <input type="range" min={32} max={256} step={16} value={target} disabled={running} onChange={(e) => setTarget(+e.target.value)} />
          <span className="nv-num">{target} tokens</span>
        </label>
      </div>

      <div style={{ display: "grid", gap: "0.6rem", marginTop: "1.2rem" }}>
        {(shownLanes ?? [a, b].map((s) => ({ stage: s, tokens: [], used: 0, rows: 0 }) as unknown as Lane)).map((l, i) => (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "6.5rem 1fr 9rem", gap: "0.8rem", alignItems: "center", fontSize: "0.78rem" }}>
            <span style={{ color: colors[i], fontWeight: 700 }}>{l.stage}</span>
            <div style={{ position: "relative", height: "1.6rem", background: pal.code, borderRadius: 4, overflow: "hidden" }}>
              <div style={{ position: "absolute", inset: 0, width: `${(100 * l.tokens.length) / (lanes ? target : 40)}%`, background: colors[i], opacity: 0.22, transition: "width 0.2s" }} />
              <div style={{ position: "absolute", inset: "0 0.5rem", display: "flex", alignItems: "center", whiteSpace: "pre", overflow: "hidden", direction: "rtl", color: pal.fg }}>
                <bdi>{promptText}{l.tokens.length ? decodeTokens(l.tokens) : ""}</bdi>
              </div>
            </div>
            <span className="nv-num nv-sub" style={{ textAlign: "right" }}>
              {l.tokens.length} tok · {fmtMs(l.used)}
            </span>
          </div>
        ))}
      </div>

      <div ref={ref} style={{ marginTop: "1.2rem" }}>
        <svg className="nv-svg" viewBox={`0 0 ${w || 1} ${H}`} height={H} role="img" aria-label="ms per token vs context length">
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <g key={f}>
              <line x1={m.l} x2={w - m.r} y1={y(maxMs * f)} y2={y(maxMs * f)} stroke={pal.rule} strokeDasharray="2 3" />
              <text x={m.l - 6} y={y(maxMs * f)} textAnchor="end" dominantBaseline="middle">{fmtMs(maxMs * f)}</text>
            </g>
          ))}
          <text x={(w + m.l) / 2} y={H - 4} textAnchor="middle">context length (tokens) →</text>
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <text key={f} x={x(maxPos * f)} y={H - m.b + 13} textAnchor="middle">{Math.round(maxPos * f)}</text>
          ))}
          {shownLanes?.map((l, i) => (
            <g key={i}>
              <polyline fill="none" stroke={colors[i]} strokeWidth={2} points={l.pos.map((p, j) => `${x(p)},${y(l.ms[j])}`).join(" ")} />
              {l.pos.length > 0 && <circle cx={x(l.pos.at(-1)!)} cy={y(l.ms.at(-1)!)} r={3.5} fill={colors[i]} />}
            </g>
          ))}
        </svg>
      </div>

      <div style={{ marginTop: "1rem" }}>
        <div className="nv-muted" style={{ fontSize: "0.72rem", marginBottom: "0.4rem", display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "0.5rem" }}>
          <span>KV cache: one K and one V row per layer per token{kvLane ? ` (${kvLane.stage})` : ""}</span>
          <span>
            {kvLane && cfg ? `${kvLane.kvLen} positions · ${fmtBytes(2 * cfg.nLayer * kvLane.kvLen * cfg.nEmbd * 4)} · full 1024 would be ${fmtBytes(2 * cfg.nLayer * 1024 * cfg.nEmbd * 4)}` : "no cached lane yet"}
          </span>
        </div>
        <canvas ref={grid} className="nv-canvas" />
      </div>

      {shownLanes && shownLanes.every((l) => l.tokens.length > 0) && (
        <div className="nv-stats" style={{ marginTop: "1.2rem" }}>
          {shownLanes.map((l, i) => (
            <div className="nv-stat" key={i}>
              <span className="k" style={{ color: colors[i] }}>{l.stage}: rows pushed through the network</span>
              <span className="v">{l.rows.toLocaleString()}<small>for {l.tokens.length} tokens</small></span>
            </div>
          ))}
          <div className="nv-stat">
            <span className="k">same tokens out?</span>
            <span className="v" style={{ color: same ? pal.down : pal.up }}>{same ? "identical" : "diverged"}</span>
          </div>
          {shownLanes[0].tokens.length > 2 && shownLanes[1].tokens.length > 2 && (
            <div className="nv-stat">
              <span className="k">speed-up so far</span>
              <span className="v nv-accent">
                {sig(Math.max(shownLanes[0].tokens.length / shownLanes[0].used, shownLanes[1].tokens.length / shownLanes[1].used) /
                  Math.min(shownLanes[0].tokens.length / shownLanes[0].used, shownLanes[1].tokens.length / shownLanes[1].used), 3)}×
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
