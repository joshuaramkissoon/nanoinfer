import { useEffect, useState } from "react";
import { boot, loadModel, useEngine, engine, setState, WEIGHTS_BASE } from "./engine/client";
import { Figure } from "./ui/Figure";
import { PANELS } from "./panels/registry";
import { fmtBytes, fmtMs, sig } from "./lib/fmt";

type Theme = "light" | "dark";
const THEME_KEY = "minimal-theme"; // same key as jkmr, so the choice carries over

function useTheme(): [Theme, () => void] {
  const [t, setT] = useState<Theme>(() => {
    try {
      const s = localStorage.getItem(THEME_KEY);
      if (s === "light" || s === "dark") return s;
    } catch {
      /* ignore */
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  useEffect(() => {
    document.body.dataset.theme = t;
  }, [t]);
  return [
    t,
    () =>
      setT((x) => {
        const n = x === "dark" ? "light" : "dark";
        try {
          localStorage.setItem(THEME_KEY, n);
        } catch {
          /* ignore */
        }
        return n;
      }),
  ];
}

export default function App() {
  const [theme, toggle] = useTheme();
  useEffect(() => void boot(), []);
  return (
    <main className="nv">
      <div className="nv-wide nv-top">
        <span className="nv-brand">
          nanoinfer<span>.</span>live
        </span>
        <span style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
          <a className="nv-muted" href="https://github.com/joshuaramkissoon/nanoinfer" style={{ textDecoration: "none" }}>
            source
          </a>
          <button className="nv-iconbtn" onClick={toggle} aria-label="toggle theme">
            {theme === "dark" ? "☀" : "☾"}
          </button>
        </span>
      </div>

      <header className="nv-col nv-hero">
        <div className="nv-kicker">GPT-2 124M · from scratch in Rust · compiled to WebAssembly</div>
        <h1 className="nv-h1">
          A language model, running <em>in this tab</em>, with the lid off.
        </h1>
        <p className="nv-lead">
          Every token re-reads the whole book of weights to say one word. Below, the same engine that's being hillclimbed
          natively on a Mac mini runs on your CPU. Nothing is a mock-up or a screenshot: each figure is drawn from the timings
          the engine reports as it runs.
        </p>
        <EngineCard />
      </header>

      {PANELS.map((p, i) => (
        <section key={p.id}>
          {p.intro && <div className="nv-col nv-prose">{p.intro}</div>}
          <Figure id={p.id} n={i + 1} title={p.title} caption={p.caption} narrow={p.narrow}>
            <p.Component />
          </Figure>
        </section>
      ))}

      <footer className="nv-col nv-footer">
        nanoinfer · GPT-2 124M weights by OpenAI (MIT) · engine, kernels and tokenizer written from scratch · single-threaded
        wasm with SIMD128 for now; threads, f16/int8 and WebGPU are later rungs of the ladder.
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
            Load GPT-2 · f32 · 548 MB · runs locally
          </button>
        )}
      </div>
      {s.status === "loading" && (
        <div className="nv-bar">
          <i style={{ width: `${(frac * 100).toFixed(1)}%` }} />
        </div>
      )}
      <div className="nv-engine-row nv-muted" style={{ fontSize: "0.72rem" }}>
        <span>stages: {s.stages.map((x) => x.name).join(" → ") || "…"}</span>
        <span>{s.isolated ? "cross-origin isolated" : "not isolated"} · timer {s.timerResUs ? sig(s.timerResUs, 2) : "?"} µs</span>
        {s.bandwidthGBs != null && <span>stream read {sig(s.bandwidthGBs, 3)} GB/s</span>}
        {s.peakGflops != null && <span>simd peak {sig(s.peakGflops, 3)} GFLOP/s</span>}
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
        <span>weights from {WEIGHTS_BASE}</span>
      </div>
    </div>
  );
}
