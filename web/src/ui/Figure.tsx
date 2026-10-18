import { createContext, useCallback, useContext, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { loadModel, useEngine, weightLabel } from "../engine/client";
import { fmtBytes } from "../lib/fmt";

export type FigSource = "live" | "recorded" | "none";
const FigCtx = createContext<(s: FigSource) => void>(() => {});

/** A panel tells its frame where the data it's drawing came from. Defaults to
 *  the shared run's source. */
export function useFigureSource(src: FigSource) {
  const set = useContext(FigCtx);
  useLayoutEffect(() => set(src), [src, set]);
}

/** Frame shared by every panel: number, title, provenance, body, caption. */
export function Figure({
  id,
  n,
  title,
  caption,
  children,
  narrow,
  prov,
}: {
  id: string;
  n: number;
  title: string;
  caption: ReactNode;
  children: ReactNode;
  narrow?: boolean;
  /** override the provenance line */
  prov?: ReactNode;
}) {
  const global = useEngine((s) => s.source);
  const [src, setSrc] = useState<FigSource | null>(null);
  return (
    <FigCtx.Provider value={setSrc}>
      <figure className={`nv-fig${narrow ? " narrow" : ""}`} id={id} style={{ margin: undefined }} data-source={src ?? global}>
        <div className="nv-fig-head">
          <div className="nv-fig-title">
            <span className="n">fig {n}</span>
            {title}
          </div>
          <div className="nv-prov">{prov ?? <Provenance source={src ?? global} />}</div>
        </div>
        <div className="nv-fig-body">{children}</div>
        <figcaption className="nv-cap">{caption}</figcaption>
      </figure>
    </FigCtx.Provider>
  );
}

const BUILD_LABEL: Record<string, string> = { simd128: "wasm simd128", relaxed: "wasm relaxed-simd", threads: "wasm threads" };

export function Provenance({ source }: { source: FigSource }) {
  const s = useEngine((s) => s.status);
  const iso = useEngine((s) => s.isolated);
  const res = useEngine((s) => s.timerResUs);
  const build = useEngine((s) => s.build);
  const threads = useEngine((s) => s.threads);
  const rec = useEngine((s) => s.recorded);
  const pr = useEngine((s) => s.progress);
  if (source === "live" && s === "ready")
    return (
      <span>
        <span className="nv-dot live" />
        live · this tab · {BUILD_LABEL[build ?? "simd128"]} · {threads > 1 ? `${threads} threads` : "1 thread"} · timer{" "}
        {res < 1 ? res.toFixed(2) : res.toFixed(0)} µs
        {iso ? "" : " (not isolated)"}
      </span>
    );
  if (source === "recorded" && rec)
    return (
      <span className="nv-prov-rec">
        <span className="nv-dot rec" />
        <span title={`${rec.meta.runtime} · ${rec.meta.build} build · ${rec.meta.recordedAt}`}>
          recorded on an {rec.meta.machine.replace(/^Apple /, "")} · same wasm, in V8
        </span>
        <LiveButton />
      </span>
    );
  if (s === "loading")
    return (
      <span>
        <span className="nv-dot busy" />
        loading GPT-2{pr && pr.phase === "download" ? ` · ${fmtBytes(pr.loaded)} / ${fmtBytes(pr.total)}` : pr ? " · parsing" : "…"}
      </span>
    );
  return <span>waiting for the engine</span>;
}

/** "Run it live" affordance, shown wherever a figure is drawing recorded data. */
export function LiveButton({ label }: { label?: string }) {
  const s = useEngine((s) => s.status);
  const pr = useEngine((s) => s.progress);
  useEngine((s) => s.dtype);
  if (s === "ready") return null;
  if (s === "loading")
    return (
      <span className="nv-live-btn busy">
        loading{pr && pr.total ? ` ${Math.round((100 * pr.loaded) / pr.total)}%` : "…"}
      </span>
    );
  return (
    <button className="nv-live-btn" onClick={() => loadModel()} disabled={s === "boot"} title="downloads GPT-2 once (cached in your browser) and runs it on your CPU">
      ▶ {label ?? "run live in your tab"} <span className="nv-muted">· {weightLabel()}</span>
    </button>
  );
}

/** Placeholder shown before the engine is loaded. */
export function NeedsEngine({ children }: { children?: ReactNode }) {
  return (
    <div className="nv-empty">
      <div>
        {children ?? "Live figure."} <br />
        <a href="#engine">Load the engine</a> to drive it with the real thing.
      </div>
    </div>
  );
}

/** Element width, tracked with ResizeObserver. Uses a callback ref so it works
 *  for elements that mount later (panels render a placeholder first). */
export function useWidth<T extends HTMLElement>(): [(el: T | null) => void, number] {
  const [w, setW] = useState(0);
  const ro = useRef<ResizeObserver | null>(null);
  const ref = useCallback((el: T | null) => {
    ro.current?.disconnect();
    ro.current = null;
    if (el) {
      ro.current = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width)));
      ro.current.observe(el);
    }
  }, []);
  return [ref, w];
}

/** True once the element has been on screen (lazy-init figures). */
export function useInView<T extends HTMLElement>(): [(el: T | null) => void, boolean] {
  const [v, setV] = useState(false);
  const io = useRef<IntersectionObserver | null>(null);
  const ref = useCallback((el: T | null) => {
    io.current?.disconnect();
    if (el) {
      io.current = new IntersectionObserver(([e]) => e.isIntersecting && setV(true), { rootMargin: "200px" });
      io.current.observe(el);
    }
  }, []);
  return [ref, v];
}

/** Size a canvas to its CSS box at device pixel ratio; returns a 2D context. */
export function fitCanvas(c: HTMLCanvasElement, w: number, h: number): CanvasRenderingContext2D {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
  }
  c.style.height = `${h}px`;
  const ctx = c.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

export function Tip({ x, y, children }: { x: number; y: number; children: ReactNode }) {
  return (
    <div className="nv-tip" style={{ left: x, top: y }}>
      {children}
    </div>
  );
}
