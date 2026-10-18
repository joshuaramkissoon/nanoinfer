import { useEffect } from "react";
import { PlusIcon } from "./learn/kit";
import { boot, loadModel, useEngine, engine, setState, WEIGHTS_BASE, weightLabel } from "./engine/client";
import { Figure } from "./ui/Figure";
import { TopBar } from "./site/TopBar";
import { PANELS } from "./panels/registry";
import { fmtBytes, fmtMs, sig } from "./lib/fmt";

export default function App() {
  useEffect(() => {
    document.title = "The lab · nanoinfer";
    void boot();
  }, []);
  return (
    <main className="nv lr" data-askable data-ask-title="The lab">
      <TopBar here="lab" crumb={<span> / lab</span>} />

      <header className="nv-col nv-hero" data-ask-section>
        <div className="nv-kicker">the lab</div>
        <h1 className="nv-h1">
          The lab: every stage of the engine, running in this tab.
        </h1>
        <p className="nv-lead">
          This is for after the <a href="/learn">course</a>. Every figure here is drawn from the engine as it runs, with
          nothing simplified away, and each one links back to the chapter that explains its idea. Load GPT-2 below (once,
          cached afterwards) to drive them all live; until then they show a recorded run.
        </p>
        <EngineCard />
      </header>

      {PANELS.map((p, i) => (
        <section key={p.id} data-ask-section>
          {p.intro && <div className="nv-col nv-prose">{p.intro}</div>}
          <Figure id={p.id} n={i + 1} title={p.title} caption={p.caption} narrow={p.narrow}>
            <p.Component />
          </Figure>
        </section>
      ))}

      <footer className="nv-col nv-footer">
        nanoinfer · GPT-2 124M weights by OpenAI (MIT) · engine, kernels and tokenizer written from scratch · three wasm builds
        (simd128, relaxed-SIMD, threads), picked by what your browser supports · weights downloaded as f16 and widened to f32
        on load · WebGPU stages run where the browser supports them.
      </footer>
    </main>
  );
}

function EngineCard() {
  const s = useEngine((s) => s);
  const pr = s.progress;
  const frac = pr ? pr.loaded / Math.max(1, pr.total) : 0;
  const dot = s.status === "ready" ? "live" : s.status === "loading" ? "busy" : s.status === "error" ? "err" : "";
  return (
    <div className="nv-engine" id="engine">
      <div className="nv-engine-row">
        <span>
          <span className={`nv-dot ${dot}`} />
          {s.status === "boot" && "starting the engine worker…"}
          {s.status === "idle" && "engine compiled · weights not loaded"}
          {s.status === "loading" &&
            (pr?.phase === "parse"
              ? "parsing safetensors and transposing into kernel layout…"
              : pr
                ? `${pr.cached ? "reading from cache" : "downloading"} model.safetensors · ${fmtBytes(pr.loaded)} / ${fmtBytes(pr.total)}`
                : "fetching tokenizer…")}
          {s.status === "ready" && s.config && (
            <>
              GPT-2 loaded · {(s.config.nParams / 1e6).toFixed(1)}M params · {s.config.nLayer} layers × {s.config.nHead} heads ·{" "}
              {fmtBytes(s.config.decodeWeightBytes)} read per token
            </>
          )}
          {s.status === "error" && <span style={{ color: "var(--m-up)" }}>{s.error}</span>}
        </span>
        {(s.status === "idle" || s.status === "error") && (
          <button className="nv-btn" onClick={() => loadModel()}>
            Load GPT-2 · {weightLabel()} · runs locally
          </button>
        )}
      </div>
      {s.status === "loading" && (
        <div className="nv-bar">
          <i style={{ width: `${(frac * 100).toFixed(1)}%` }} />
        </div>
      )}
      <details className="nv-engine-more">
        <summary className="nv-muted">
          <PlusIcon />
          engine details
        </summary>
      <div className="nv-engine-row nv-muted" style={{ fontSize: "0.72rem" }}>
        <span>stages: {s.stages.map((x) => x.name).join(" → ") || "…"}</span>
        <span>
          wasm build: {s.build ?? "…"}
          {s.threads > 1 ? ` · ${s.threads} workers` : ""}
          {s.builds.length > 1 ? ` (this browser can run: ${s.builds.join(", ")})` : ""}
        </span>
        <span>{s.isolated ? "cross-origin isolated" : "not isolated"} · timer {s.timerResUs ? sig(s.timerResUs, 2) : "?"} µs</span>
        {s.bandwidthGBs != null && <span>stream read {sig(s.bandwidthGBs, 3)} GB/s{s.source === "recorded" ? " (recorded)" : ""}</span>}
        {s.peakGflops != null && <span>simd peak {sig(s.peakGflops, 3)} GFLOP/s{s.source === "recorded" ? " (recorded)" : ""}</span>}
        {s.load && (
          <span>
            loaded in {fmtMs(s.load.ms)} {s.load.fromCache ? "(from OPFS cache)" : ""}{" "}
            <button
              className="nv-iconbtn"
              style={{ fontSize: "0.72rem", textDecoration: "underline", padding: 0 }}
              onClick={async () => {
                await engine.clearCache();
                setState({ load: { ...s.load!, fromCache: false } });
              }}
            >
              clear cache
            </button>
          </span>
        )}
        <span>
          weights from {WEIGHTS_BASE} as {s.dtype}
          {s.dtype === "f16" ? " (rounded to f16 for the download, widened to f32 on load)" : ""}
        </span>
      </div>
      </details>
    </div>
  );
}
