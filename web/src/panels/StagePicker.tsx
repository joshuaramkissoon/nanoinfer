// The engine-version picker for "Run it": every stage of the climb, grouped into
// families, each with a plain name, one line on what changed, and its measured speed
// (natively on the M4, and in this browser when we have a number). Picking one changes
// which engine writes the text. Stages written for the Mac's CPU that a browser can't
// really run are marked ≈ and say what they fall back to.

import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { benchKey, useEngine } from "../engine/client";
import type { StageInfo } from "../engine/types";
import { FAMILIES, fallback, familyOf, stageLabel, stageWhat, type Family } from "../lib/stages";
import { latestByStage, useNative } from "../lib/native";
import { sig } from "../lib/fmt";

// ---- decode speeds measured by generating text in this tab (this session) ----------
const here = new Map<string, number>();
const subs = new Set<() => void>();
let version = 0;
export function noteSpeed(stage: string, tokS: number) {
  if (!isFinite(tokS) || tokS <= 0) return;
  here.set(stage, tokS);
  version++;
  subs.forEach((f) => f());
}
function useHereVersion() {
  return useSyncExternalStore(
    (f) => (subs.add(f), () => void subs.delete(f)),
    () => version,
  );
}

/** Speed in this browser: generated just now, else your Lab benchmark, else the recording. */
function useBrowserSpeeds(): (name: string) => { v: number; how: "now" | "bench" | "recorded" } | null {
  useHereVersion();
  const bench = useEngine((s) => s.bench);
  const build = useEngine((s) => s.build);
  const rec = useEngine((s) => s.recorded);
  return (name) => {
    const h = here.get(name);
    if (h) return { v: h, how: "now" };
    const b = bench[benchKey(build, name)];
    if (b?.tokS) return { v: b.tokS, how: "bench" };
    const r = rec?.ladder[name];
    if (r?.tokS) return { v: r.tokS, how: "recorded" };
    return null;
  };
}

/** The fastest stage that really runs in this browser: by measurement where we have
 *  one, else the newest GPU rung, else the best CPU rung. */
export function recommendedStage(stages: StageInfo[], speed?: (n: string) => number | undefined): string | null {
  const real = stages.filter((s) => !fallback(s));
  if (!real.length) return stages.at(-1)?.name ?? null;
  const measured = speed ? real.map((s) => [s.name, speed(s.name) ?? 0] as const).filter(([, v]) => v > 0) : [];
  const gpu = real.filter((s) => familyOf(s) === "gpu");
  // a GPU we haven't measured yet beats any CPU number we have
  const gpuMeasured = measured.some(([n]) => gpu.some((g) => g.name === n));
  if (measured.length && (gpuMeasured || !gpu.length)) return measured.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
  if (gpu.length) return gpu.at(-1)!.name;
  for (const n of ["wasm-threads", "wasm-fma", "wasm-simd", "unrolled", "kv-cache"]) if (real.some((s) => s.name === n)) return n;
  return real.at(-1)!.name;
}

/** What the ≈ stage actually runs in a browser, in a few words. */
function fallbackNote(s: StageInfo): string {
  const n = s.wasmNote ?? "";
  if (/NEON/.test(n) && /unrolled/.test(n)) return "no NEON in a browser: runs the 16-totals code";
  if (/scalar code/.test(n) && /tile/.test(n)) return "no NEON in a browser: the tile runs as plain code";
  if (/single-threaded/.test(n) && /weights/.test(n)) return "one thread here, and the small weights are unpacked slowly";
  if (/single-threaded/.test(n)) return "no OS threads in a browser: runs on one core";
  if (/scalar/.test(n)) return "the small weights are unpacked with plain code here";
  return n;
}

const fmt = (v: number) => (v >= 100 ? Math.round(v).toLocaleString("en-GB") : sig(v, 2));

export function StagePicker({ value, onChange, disabled }: { value: string; onChange: (s: string) => void; disabled?: boolean }) {
  const liveStages = useEngine((s) => s.stages);
  const rec = useEngine((s) => s.recorded);
  const stages = liveStages.length ? liveStages : (rec?.meta.stages ?? []);
  const native = useNative();
  const nat = useMemo(() => (native ? latestByStage(native, "decode") : new Map()), [native]);
  const browser = useBrowserSpeeds();
  const [open, setOpen] = useState(false);
  const [showNative, setShowNative] = useState(false);
  const rec1 = recommendedStage(stages, (n) => {
    const b = browser(n);
    return b && b.how !== "recorded" ? b.v : undefined;
  });
  const cur = stages.find((s) => s.name === value);

  useEffect(() => {
    if (!open) return;
    const on = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [open]);

  const groups = FAMILIES.map((f) => ({ ...f, items: stages.filter((s) => familyOf(s) === f.key) })).filter((g) => g.items.length);
  const famTitle = (k: Family) => FAMILIES.find((f) => f.key === k)!.title;
  const curSpeed = cur ? browser(cur.name) : null;

  return (
    <div className={`nv-sp${open ? " open" : ""}`}>
      <button
        className="nv-sp-current"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-controls="nv-sp-list"
        disabled={disabled || !stages.length}
      >
        <span className="nv-sp-k">engine version</span>
        <span className="nv-sp-v">
          {cur ? (
            <>
              <b>{stageLabel(cur.name)}</b>
              <span className="nv-muted"> · {famTitle(familyOf(cur))}</span>
              {fallback(cur) && <span className="nv-sp-fb" title={fallbackNote(cur)}> ≈</span>}
            </>
          ) : (
            "…"
          )}
        </span>
        <span className="nv-sp-speed">
          {curSpeed ? (
            <>
              {fmt(curSpeed.v)} <small>tok/s {curSpeed.how === "now" ? "here" : curSpeed.how === "bench" ? "here (benchmark)" : "in a browser (recorded)"}</small>
            </>
          ) : nat.get(value) ? (
            <>
              {fmt(nat.get(value)!.decode_tok_s)} <small>tok/s on the Mac</small>
            </>
          ) : null}
        </span>
        <span className="nv-sp-caret" aria-hidden>
          <svg viewBox="0 0 12 12" width="10" height="10">
            <path d="M2.5 4.5 6 8l3.5-3.5" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
      </button>
      {open && (
        <div className="nv-sp-list" id="nv-sp-list" role="radiogroup" aria-label="engine version">
          <div className="nv-sp-cols" aria-hidden>
            <span>Each version is a real step of the climb: pick one and press Generate, and that engine writes the text.</span>
            <span className="nv-sp-num">Mac</span>
            <span className="nv-sp-num">browser</span>
          </div>
          {groups.map((g) => {
            const folded = g.key === "native" && !showNative && !g.items.some((s) => s.name === value);
            return (
            <section key={g.key} className="nv-sp-group">
              <h4>
                {g.title}
                <span>{g.blurb}</span>
              </h4>
              {folded && (
                <button className="nv-sp-more" onClick={() => setShowNative(true)}>
                  show all {g.items.length}
                </button>
              )}
              {!folded && g.items.map((s) => {
                const fb = fallback(s);
                const b = browser(s.name);
                const n = nat.get(s.name)?.decode_tok_s;
                const on = s.name === value;
                return (
                  <button
                    key={s.name}
                    role="radio"
                    aria-checked={on}
                    className={`nv-sp-opt${on ? " on" : ""}${fb ? " fb" : ""}`}
                    onClick={() => {
                      onChange(s.name);
                      setOpen(false);
                    }}
                    title={`${s.name} · kernels: ${s.kernels}${s.wasmNote ? ` · in wasm: ${s.wasmNote}` : ""}`}
                  >
                    <span className="nv-sp-dot" aria-hidden />
                    <span className="nv-sp-name">
                      <b>{stageLabel(s.name)}</b>
                      {s.name === rec1 && <span className="nv-sp-rec">recommended</span>}
                      <code>{s.name}</code>
                      <span className="nv-sp-what">
                        {fb ? (
                          <>
                            <span className="nv-sp-fb">≈</span> {fallbackNote(s)}
                          </>
                        ) : (
                          stageWhat(s)
                        )}
                      </span>
                    </span>
                    <span className="nv-sp-num">{n ? fmt(n) : "–"}</span>
                    <span className={`nv-sp-num${b?.how === "recorded" ? " rec" : ""}`} title={b ? (b.how === "now" ? "measured just now, in this tab" : b.how === "bench" ? "your Lab benchmark, this browser" : "recorded on the M4 in a browser engine") : undefined}>
                      {b ? fmt(b.v) : "–"}
                    </span>
                  </button>
                );
              })}
            </section>
            );
          })}
          <p className="nv-sp-foot">
            Decode speed, tokens per second. <b>Mac</b>: native code on the M4 Mac mini (bench/results.jsonl). <b>browser</b>: measured when
            you generate here, or from your benchmark in the ladder below; grey numbers are a recording. <span className="nv-sp-fb">≈</span> marks
            versions written for the Mac's CPU: a browser can't run their key trick, so here they run simpler code and their speed says
            little about the idea.
          </p>
        </div>
      )}
    </div>
  );
}
