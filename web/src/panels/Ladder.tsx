import { useState } from "react";
import { getState, useEngine } from "../engine/client";
import { benchStage } from "../engine/generate";
import { Tip, useWidth } from "../ui/Figure";
import { usePalette } from "../ui/theme";
import { log, logTicks, pct, sig } from "../lib/fmt";
import { M4_GBS, latestByStage, useNative } from "../lib/native";
import { STAGE_NOTES } from "../lib/stages";

/**
 * The hillclimb as one chart: one rung per stage, Josh's native M4 number next to
 * the number your browser just measured, under both speed-of-light lines.
 */
export default function Ladder() {
  const pal = usePalette();
  const stages = useEngine((s) => s.stages);
  const status = useEngine((s) => s.status);
  const busy = useEngine((s) => s.busy);
  const bench = useEngine((s) => s.bench);
  const cfg = useEngine((s) => s.config);
  const bw = useEngine((s) => s.bandwidthGBs);
  const native = useNative();
  const [ref, w] = useWidth<HTMLDivElement>();
  const [running, setRunning] = useState<string | null>(null);
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null);

  const nat = native ? latestByStage(native) : new Map();
  // rung order: the engine's stage list, then any native-only stages (e.g. ones
  // that only exist on a branch, or structural steps that are native only)
  const names = [...stages.map((s) => s.name), ...[...nat.keys()].filter((k) => !stages.some((s) => s.name === k))];
  const weightBytes = cfg?.decodeWeightBytes ?? 497_759_232;
  const nativeCeil = [...nat.values()].find((r) => r.decode_ceiling_tok_s)?.decode_ceiling_tok_s ?? (M4_GBS * 1e9) / weightBytes;
  const browserCeil = bw ? (bw * 1e9) / weightBytes : null;

  const values = [
    ...[...nat.values()].map((r) => r.decode_tok_s),
    ...Object.values(bench).map((b) => b.tokS),
    nativeCeil,
    browserCeil ?? 1,
  ].filter((v) => v > 0);
  const Y0 = Math.max(0.05, Math.min(...values, 1) / 2);
  const Y1 = Math.max(...values) * 2.2;
  const H = 320;
  const m = { l: 46, r: 16, t: 16, b: 54 };
  const y = log(Y0, Y1, H - m.b, m.t);
  const step = names.length && w > 100 ? (w - m.l - m.r) / names.length : 0;
  const cx = (i: number) => m.l + step * (i + 0.5);

  const runAll = async () => {
    for (const s of getState().stages) {
      setRunning(s.name);
      await benchStage(s.name);
    }
    setRunning(null);
  };

  return (
    <div>
      <div className="nv-controls" style={{ justifyContent: "space-between" }}>
        <button className="nv-btn" disabled={status !== "ready" || !!busy} onClick={runAll}>
          {running ? `measuring ${running}…` : "▶ Benchmark every stage in this browser"}
        </button>
        <span className="nv-muted" style={{ fontSize: "0.75rem" }}>
          {native == null ? "loading native results…" : native.length ? `${native.length} native runs in bench/results.jsonl` : "no native runs recorded yet (bench/results.jsonl)"}
        </span>
      </div>
      <div ref={ref} style={{ position: "relative", marginTop: "0.8rem" }} onPointerLeave={() => setHover(null)}>
        <svg className="nv-svg" viewBox={`0 0 ${w || 1} ${H}`} height={H} role="img" aria-label="tok/s per stage, native and browser">
          {logTicks(Y0, Y1).map((t) => (
            <g key={t}>
              <line x1={m.l} x2={w - m.r} y1={y(t)} y2={y(t)} stroke={pal.rule} strokeDasharray="2 3" />
              <text x={m.l - 6} y={y(t)} textAnchor="end" dominantBaseline="middle">{sig(t, 1)}</text>
            </g>
          ))}
          <text transform={`translate(11 ${(H - m.b) / 2}) rotate(-90)`} textAnchor="middle">decode tok/s (log) →</text>
          {[
            { v: nativeCeil, l: `M4 speed of light · f32 · ${sig(nativeCeil, 3)} tok/s`, c: pal.fg },
            browserCeil && { v: browserCeil, l: `this tab's speed of light · ${sig(browserCeil, 3)} tok/s`, c: pal.accent },
          ]
            .filter(Boolean)
            .map((c) => {
              const cc = c as { v: number; l: string; c: string };
              return (
                <g key={cc.l}>
                  <line x1={m.l} x2={w - m.r} y1={y(cc.v)} y2={y(cc.v)} stroke={cc.c} strokeWidth={1.4} strokeDasharray="6 4" />
                  <text x={m.l + 6} y={y(cc.v) - 5} style={{ fill: cc.c }}>{cc.l}</text>
                </g>
              );
            })}
          {names.map((n, i) => {
            const nr = nat.get(n);
            const br = bench[n];
            const x = cx(i);
            const bar = Math.min(26, step * 0.22);
            return (
              <g key={n}>
                <text x={x} y={H - m.b + 16} textAnchor="middle" style={{ fill: pal.fg, fontSize: step < 90 ? 9 : 11 }}>{n}</text>
                <text x={x} y={H - m.b + 30} textAnchor="middle" style={{ fontSize: 9 }}>
                  {stages.some((s) => s.name === n) ? `stage ${i}` : "native only"}
                </text>
                <text x={x} y={H - m.b + 42} textAnchor="middle" style={{ fontSize: 9 }}>
                  {STAGE_NOTES[n]?.wasm ? "wasm fallback" : ""}
                </text>
                {nr && (
                  <g onPointerEnter={() => setHover({ x: x - bar * 0.7, y: y(nr.decode_tok_s), text: `native ${n}: ${sig(nr.decode_tok_s, 3)} tok/s · ${pct(nr.decode_tok_s / nativeCeil, 1)} of ceiling${nr.machine ? ` · ${nr.machine}` : ""}${nr.git_sha ? ` · ${String(nr.git_sha).slice(0, 7)}` : ""}` })}>
                    <rect x={x - bar * 1.2} width={bar} y={y(nr.decode_tok_s)} height={H - m.b - y(nr.decode_tok_s)} fill={pal.fg} opacity={0.85} />
                    <text x={x - bar * 0.7} y={y(nr.decode_tok_s) - 5} textAnchor="middle" style={{ fill: pal.fg }}>{sig(nr.decode_tok_s, 3)}</text>
                  </g>
                )}
                {br && (
                  <g onPointerEnter={() => setHover({ x: x + bar * 0.7, y: y(br.tokS), text: `your browser ${n}: ${sig(br.tokS, 3)} tok/s (${br.promptLen}-token prompt, ${br.genTokens} decoded)${browserCeil ? ` · ${pct(br.tokS / browserCeil, 1)} of this tab's ceiling` : ""}` })}>
                    <rect x={x + bar * 0.2} width={bar} y={y(br.tokS)} height={H - m.b - y(br.tokS)} fill={pal.accent} />
                    <text x={x + bar * 0.7} y={y(br.tokS) - 5} textAnchor="middle" style={{ fill: pal.accent }}>{sig(br.tokS, 3)}</text>
                  </g>
                )}
                {(() => {
                  // per-rung ceiling when a stage reads fewer bytes per weight
                  const st = stages.find((s) => s.name === n);
                  if (!st || !st.bitsPerWeight || st.bitsPerWeight >= 32) return null;
                  const c = (nativeCeil * 32) / st.bitsPerWeight;
                  return (
                    <g>
                      <line x1={x - step * 0.4} x2={x + step * 0.4} y1={y(c)} y2={y(c)} stroke={pal.fg} strokeDasharray="3 3" />
                      <text x={x} y={y(c) - 4} textAnchor="middle" style={{ fontSize: 9 }}>{st.weightFormat} ceiling {sig(c, 3)}</text>
                    </g>
                  );
                })()}
                {!nr && !br && (
                  <text x={x} y={H - m.b - 8} textAnchor="middle" style={{ fontSize: 9 }}>not measured</text>
                )}
              </g>
            );
          })}
        </svg>
        {hover && <Tip x={Math.min(Math.max(hover.x, 140), w - 140)} y={hover.y}>{hover.text}</Tip>}
      </div>
      <div className="nv-legend">
        <span><i style={{ background: pal.fg }} />native · Mac mini M4 · from bench/results.jsonl</span>
        <span><i style={{ background: pal.accent }} />your browser · measured just now · wasm simd128 · 1 thread</span>
      </div>
    </div>
  );
}
