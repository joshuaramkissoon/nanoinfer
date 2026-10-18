import { useCallback, useRef, useState, type ReactNode } from "react";
import { useEngine } from "../engine/client";

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
  return (
    <figure className={`nv-fig${narrow ? " narrow" : ""}`} id={id} style={{ margin: undefined }}>
      <div className="nv-fig-head">
        <div className="nv-fig-title">
          <span className="n">fig. {String(n).padStart(2, "0")}</span>
          {title}
        </div>
        <div className="nv-prov">{prov ?? <Provenance />}</div>
      </div>
      <div className="nv-fig-body">{children}</div>
      <figcaption className="nv-cap">{caption}</figcaption>
    </figure>
  );
}

export function Provenance() {
  const s = useEngine((s) => s.status);
  const iso = useEngine((s) => s.isolated);
  const res = useEngine((s) => s.timerResUs);
  if (s !== "ready") return <span>waiting for the engine</span>;
  return (
    <span>
      <span className="nv-dot live" />
      live · this tab · wasm simd128 · 1 thread · f32 · timer {res < 1 ? res.toFixed(2) : res.toFixed(0)} µs
      {iso ? "" : " (not isolated)"}
    </span>
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
