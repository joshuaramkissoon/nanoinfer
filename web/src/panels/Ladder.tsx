import { T } from "../learn/kit";
import { useState } from "react";
import { benchKey, getState, useEngine } from "../engine/client";
import { benchStage } from "../engine/generate";
import type { BuildName } from "../engine/types";
import { LiveButton, Tip, useFigureSource, useWidth } from "../ui/Figure";
import { usePalette } from "../ui/theme";
import { log, logTicks, pct, sig } from "../lib/fmt";
import { M4_GBS, latestByStage, useNative } from "../lib/native";
import { decodeBytes, gpuBandwidthOf } from "../lib/roofs";
import { fallback } from "./RunIt";

const BUILD_LABEL: Record<string, string> = { simd128: "simd128", relaxed: "relaxed-SIMD", threads: "threads" };

/**
 * The hillclimb as one chart: one rung per stage, the native M4 number next to
 * the number a browser measured (yours, live, or the recording), under the
 * speed-of-light lines.
 */
export default function Ladder() {
  const pal = usePalette();
  const liveStages = useEngine((s) => s.stages);
  const status = useEngine((s) => s.status);
  const busy = useEngine((s) => s.busy);
  const bench = useEngine((s) => s.bench);
  const cfg = useEngine((s) => s.config);
  const bw = useEngine((s) => s.bandwidthGBs);
  const gpuBwLive = useEngine((s) => s.gpuBandwidthGBs);
  const build = useEngine((s) => s.build);
  const rec = useEngine((s) => s.recorded);
  const live = status === "ready";
  const native = useNative();
  const [ref, w] = useWidth<HTMLDivElement>();
  const [running, setRunning] = useState<string | null>(null);
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null);
  const [metric, setMetric] = useState<"decode" | "prefill">("decode");
  const [showAll, setShowAll] = useState(false);
  // which of your saved builds to show (default: the one running now)
  const savedBuilds = [...new Set(Object.values(bench).map((b) => b.build).filter(Boolean))] as BuildName[];
  const [viewBuild, setViewBuild] = useState<BuildName | null>(null);
  const shownBuild = viewBuild ?? build ?? savedBuilds[0] ?? null;
  const mine = (n: string) => bench[benchKey(shownBuild, n)];
  const anyMine = Object.keys(bench).some((k) => k.startsWith(`${shownBuild}/`));
  useFigureSource(anyMine ? "live" : rec ? "recorded" : "none");

  const stages = liveStages.length ? liveStages : (rec?.meta.stages ?? []);
  const nat = native ? latestByStage(native, metric) : new Map();
  const nativeOnly = [...nat.keys()].filter((k) => !stages.some((s) => s.name === k));
  // rung order: the engine's stage list (browser rungs last), then native-only
  // stages (Metal GPU, Accelerate, ...) behind a toggle
  const names = [...stages.map((s) => s.name), ...(showAll ? nativeOnly : [])];
  const weightBytes = cfg?.decodeWeightBytes ?? 497_759_232;
  const nativeCeil = [...nat.values()].find((r) => r.decode_ceiling_tok_s)?.decode_ceiling_tok_s ?? (M4_GBS * 1e9) / weightBytes;
  const browserCeil = bw ? (bw * 1e9) / weightBytes : null;

  const nv = (n: string) => {
    const r = nat.get(n);
    return r ? (metric === "decode" ? r.decode_tok_s : (r.prefill_tok_s ?? 0)) : 0;
  };
  const mv = (n: string) => {
    const b = mine(n);
    return b ? (metric === "decode" ? b.tokS : (b.prefillTokS ?? 0)) : 0;
  };
  const rv = (n: string) => {
    const b = rec?.ladder[n];
    return b ? (metric === "decode" ? b.tokS : b.prefillTokS) : 0;
  };

  // M4 speed limit for a stage in its own weight and LM-head formats
  const fmtCeil = (st: (typeof stages)[number]) =>
    cfg ? (M4_GBS * 1e9) / decodeBytes(cfg, st, 0).total : (nativeCeil * 32) / st.bitsPerWeight;
  const rungCeil = (n: string) => {
    const st = stages.find((s) => s.name === n);
    return metric === "decode" && st && st.bitsPerWeight && st.bitsPerWeight < 32 ? fmtCeil(st) : 0;
  };
  // this tab's limit for one browser bar: its backend's bandwidth over its own bytes per token
  const gpuBw = gpuBwLive ?? gpuBandwidthOf(rec?.webgpu?.probe);
  const tabCeil = (st: (typeof stages)[number] | undefined, promptLen: number) => {
    const b = st?.backend === "webgpu" ? gpuBw : bw;
    if (!b || !cfg || !st) return null;
    return (b * 1e9) / decodeBytes(cfg, st, promptLen).total;
  };
  const values = [...names.flatMap((n) => [nv(n), mv(n), rv(n), rungCeil(n)]), ...(metric === "decode" ? [nativeCeil, browserCeil ?? 1] : [])].filter((v) => v > 0);
  const Y0 = Math.max(0.05, Math.min(...values, 1) / 2);
  const Y1 = Math.max(...values, 10) * 1.6;
  const H = 340;
  const m = { l: 46, r: 16, t: 16, b: 58 };
  const y = log(Y0, Y1, H - m.b, m.t);
  const step = names.length && w > 100 ? (w - m.l - m.r) / names.length : 0;
  const cx = (i: number) => m.l + step * (i + 0.5);
  const small = step < 70;
  const tiny = step < 48;

  // label each per-format ceiling once (again only after a long gap), so the
  // alternating q8/q4 rungs don't stack a dozen identical labels
  const ceilLabelled = new Map<string, number>();

  const runAll = async () => {
    setViewBuild(null);
    for (const s of getState().stages) {
      setRunning(s.name);
      await benchStage(s.name);
    }
    setRunning(null);
  };

  return (
    <div>
      <div className="nv-controls" style={{ justifyContent: "space-between" }}>
        {live ? (
          <button className="nv-btn" disabled={!!busy} onClick={runAll}>
            {running ? `measuring ${running}…` : `▶ Benchmark every stage in this browser (${BUILD_LABEL[build ?? "simd128"]})`}
          </button>
        ) : (
          <LiveButton label="run live to benchmark your browser" />
        )}
        <div className="nv-seg" role="group" aria-label="metric">
          <button aria-pressed={metric === "decode"} onClick={() => setMetric("decode")}>decode</button>
          <button aria-pressed={metric === "prefill"} onClick={() => setMetric("prefill")}>prefill (64)</button>
        </div>
      </div>
      <div className="nv-controls" style={{ justifyContent: "space-between", marginTop: "0.6rem", fontSize: "var(--t-sm)" }}>
        <span className="nv-muted">
          {native == null ? "loading native results…" : native.length ? `${native.length} native runs · best untraced run per stage at a 64-token prompt` : "no native runs recorded yet"}
        </span>
        <span style={{ display: "flex", gap: "0.8rem", alignItems: "center", flexWrap: "wrap" }}>
          {savedBuilds.length > 1 && (
            <span className="nv-seg" role="group" aria-label="browser build">
              {savedBuilds.map((b) => (
                <button key={b} aria-pressed={b === shownBuild} onClick={() => setViewBuild(b)}>{BUILD_LABEL[b]}</button>
              ))}
            </span>
          )}
          {nativeOnly.length > 0 && (
            <label className="nv-field">
              <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} /> + {nativeOnly.length} native-only rungs
            </label>
          )}
        </span>
      </div>
      <div ref={ref} style={{ position: "relative", marginTop: "0.8rem" }} onPointerLeave={() => setHover(null)}>
        <svg className="nv-svg" viewBox={`0 0 ${w || 1} ${H}`} height={H} role="img" aria-label={`${metric} tok/s per stage, native and browser`}>
          {logTicks(Y0, Y1).map((t) => (
            <g key={t}>
              <line x1={m.l} x2={w - m.r} y1={y(t)} y2={y(t)} stroke={pal.rule} strokeDasharray="2 3" />
              <text x={m.l - 6} y={y(t)} textAnchor="end" dominantBaseline="middle">{sig(t, 1)}</text>
            </g>
          ))}
          <text transform={`translate(11 ${(H - m.b) / 2}) rotate(-90)`} textAnchor="middle">{metric} tok/s (log) →</text>
          {metric === "decode" &&
            [
              { v: nativeCeil, l: `M4 speed limit (f32) ${sig(nativeCeil, 3)}`, c: pal.fg },
              browserCeil && { v: browserCeil, l: `${live ? "this" : "recorded"} tab's ${sig(browserCeil, 3)}`, c: pal.accent },
            ]
              .filter(Boolean)
              .map((c) => {
                const cc = c as { v: number; l: string; c: string };
                return (
                  <g key={cc.l}>
                    <line x1={m.l} x2={w - m.r} y1={y(cc.v)} y2={y(cc.v)} stroke={cc.c} strokeWidth={1.4} strokeDasharray="6 4" />
                    <text x={m.l + 6} y={y(cc.v) - 5} textAnchor="start" style={{ fill: cc.c, fontSize: T.xs }}>{cc.l}</text>
                  </g>
                );
              })}
          {names.map((n, i) => {
            const st = stages.find((s) => s.name === n);
            const nr = nat.get(n);
            const x = cx(i);
            const bar = Math.max(4, Math.min(22, step * 0.22));
            const nVal = nv(n), mVal = mv(n), rVal = rv(n);
            const bVal = mVal || rVal;
            const recBar = !mVal && rVal > 0;
            const tag = !st
              ? nr?.backend === "metal" ? "GPU · native" : nr?.backend === "wgpu-metal" ? "WebGPU · native" : "native only"
              : st.backend === "webgpu" ? "WebGPU"
              : st.browserOnly ? "browser only"
              : st.wasmNote?.startsWith("no OS threads") ? "wasm: 1 thread"
              : fallback(st) ? "wasm: fallback" : `stage ${i}`;
            return (
              <g key={n}>
                <text x={x} y={H - m.b + 16} textAnchor="middle" style={{ fill: pal.fg, fontSize: T.xs }}
                  transform={small ? `rotate(-28 ${x} ${H - m.b + 16})` : undefined}>{n}</text>
                {!small && <text x={x} y={H - m.b + 30} textAnchor="middle" style={{ fontSize: T.xs, fill: st?.browserOnly ? pal.accent : undefined }}>{tag}</text>}
                {nVal > 0 && (
                  <g onPointerEnter={() => setHover({ x: x - bar * 0.7, y: y(nVal), text: `native ${nr?.backend === "wgpu-metal" ? "(wgpu on Metal) " : ""}${n}: ${sig(nVal, 3)} tok/s ${metric}${metric === "decode" ? ` · ${pct(nVal / nativeCeil, 1)} of the speed limit` : ""}${nr?.threads && nr.threads !== 1 ? ` · ${nr.threads} threads` : ""}${nr?.note ? ` · ${nr.note}` : ""}` })}>
                    <rect x={x - bar * 1.2} width={bar} y={y(nVal)} height={H - m.b - y(nVal)} fill={pal.fg} opacity={0.85} />
                    {!tiny && <text x={x - bar * 0.7} y={y(nVal) - 5} textAnchor="middle" style={{ fill: pal.fg, fontSize: T.xs }}>{sig(nVal, small ? 2 : 3)}</text>}
                  </g>
                )}
                {bVal > 0 && (
                  <g onPointerEnter={() => setHover({ x: x + bar * 0.7, y: y(bVal), text: recBar ? `recorded browser ${n}: ${sig(bVal, 3)} tok/s ${metric} (${st?.backend === "webgpu" ? "WebGPU in Chrome" : `${rec?.meta.build} build, V8`} on the M4)` : `your browser ${n}: ${sig(bVal, 3)} tok/s ${metric} (${st?.backend === "webgpu" ? `WebGPU${mine(n)?.dispatches ? `, ${sig(mine(n)!.dispatches!, 3)} dispatches and ${sig(mine(n)!.submits!, 2)} submits per token` : ""}` : `${BUILD_LABEL[shownBuild ?? "simd128"]} build`})${(() => {
                    const c = metric === "decode" ? tabCeil(st, mine(n)?.promptLen ?? 64) : null;
                    const lim = `this tab's ${st?.backend === "webgpu" ? "GPU" : "CPU"} limit`;
                    return !c ? "" : bVal <= c ? ` · ${pct(bVal / c, 1)} of ${lim}` : ` · at ${lim} (the bandwidth probe read low)`;
                  })()}` })}>
                    <rect x={x + bar * 0.2} width={bar} y={y(bVal)} height={H - m.b - y(bVal)}
                      fill={recBar ? "none" : pal.accent} stroke={pal.accent} strokeWidth={recBar ? 1.3 : 0} strokeDasharray={recBar ? "3 2" : undefined} />
                    {!tiny && <text x={x + bar * 0.7} y={y(bVal) - (small ? 14 : 5)} textAnchor="middle" style={{ fill: pal.fg, fontSize: T.xs }}>{sig(bVal, small ? 2 : 3)}</text>}
                  </g>
                )}
                {(() => {
                  // per-rung ceiling when a stage reads fewer bytes per weight
                  if (metric !== "decode" || !st || !st.bitsPerWeight || st.bitsPerWeight >= 32) return null;
                  const c = fmtCeil(st);
                  const key = sig(c, 3);
                  const last = ceilLabelled.get(key);
                  const label = last == null || i - last > 10;
                  if (label) ceilLabelled.set(key, i);
                  return (
                    <g>
                      <line x1={x - step * 0.4} x2={x + step * 0.4} y1={y(c)} y2={y(c)} stroke={pal.fg} strokeDasharray="3 3" opacity={0.6} />
                      {label && (
                        <text x={x - step * 0.4} y={y(c) - 4} textAnchor="start" style={{ fontSize: T.xs }}>
                          {st.weightFormat} limit {key}
                        </text>
                      )}
                    </g>
                  );
                })()}
                {!nVal && !bVal && <text x={x} y={H - m.b - 8} textAnchor="middle" style={{ fontSize: T.xs }}>–</text>}
              </g>
            );
          })}
        </svg>
        {hover && <Tip x={Math.min(Math.max(hover.x, 140), w - 140)} y={hover.y}>{hover.text}</Tip>}
      </div>
      <div className="nv-legend">
        <span><i style={{ background: pal.fg }} />native · Mac mini M4 · bench/results.jsonl</span>
        {anyMine && <span><i style={{ background: pal.accent }} />your browser · {BUILD_LABEL[shownBuild ?? "simd128"]} build · 1 thread unless the stage says</span>}
        {rec && Object.keys(rec.ladder).length > 0 && (
          <span><i style={{ background: "none", outline: `1.3px dashed ${pal.accent}`, outlineOffset: -1.3 }} />browser, recorded · {rec.meta.build} build in V8 on the M4</span>
        )}
      </div>
    </div>
  );
}
