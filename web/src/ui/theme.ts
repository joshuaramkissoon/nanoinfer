// Theme: read the --m-* CSS variables so canvas/SVG figures match whatever theme
// the host page (this app, or a jkmr post) is in.

import { useEffect, useState } from "react";

export interface Palette {
  bg: string;
  fg: string;
  sub: string;
  muted: string;
  rule: string;
  accent: string;
  code: string;
  heat: [string, string, string, string, string];
  up: string;
  down: string;
  cool: string;
  dark: boolean;
}

function host(): HTMLElement {
  return (document.querySelector(".minimal") as HTMLElement) ?? document.body;
}

export function readPalette(): Palette {
  const cs = getComputedStyle(host());
  const v = (n: string, d: string) => cs.getPropertyValue(n).trim() || d;
  return {
    bg: v("--m-bg", "#f6f4ef"),
    fg: v("--m-fg", "#1f1d1a"),
    sub: v("--m-sub", "#57534c"),
    muted: v("--m-muted", "#8b867d"),
    rule: v("--m-rule", "#e2ded5"),
    accent: v("--m-accent", "#b8541f"),
    code: v("--m-code-bg", "#ece8df"),
    heat: [
      v("--m-heat-0", "#e9e5dc"),
      v("--m-heat-1", "#f1cfae"),
      v("--m-heat-2", "#e6a26c"),
      v("--m-heat-3", "#cf6d33"),
      v("--m-heat-4", "#96431a"),
    ],
    up: v("--m-up", "#b0566a"),
    down: v("--m-down", "#3f8a4f"),
    cool: v("--nv-cool", "#3c6e8f"),
    dark: host().dataset.theme === "dark",
  };
}

/** Re-renders on data-theme changes of the .minimal host. */
export function usePalette(): Palette {
  const [p, setP] = useState(readPalette);
  useEffect(() => {
    const mo = new MutationObserver(() => setP(readPalette()));
    mo.observe(host(), { attributes: true, attributeFilter: ["data-theme", "class"] });
    // the theme may have been applied between this component's first render and now
    // (a sibling's effect sets data-theme), which the observer can't see: re-read once
    setP((old) => {
      const now = readPalette();
      return now.bg === old.bg && now.fg === old.fg ? old : now;
    });
    return () => mo.disconnect();
  }, []);
  return p;
}

export function useReducedMotion(): boolean {
  const q = "(prefers-reduced-motion: reduce)";
  const [r, setR] = useState(() => window.matchMedia(q).matches);
  useEffect(() => {
    const m = window.matchMedia(q);
    const f = () => setR(m.matches);
    m.addEventListener("change", f);
    return () => m.removeEventListener("change", f);
  }, []);
  return r;
}

/** Parse #rgb/#rrggbb to [r,g,b]. */
export function rgb(hex: string): [number, number, number] {
  let h = hex.replace("#", "");
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Sample the 5-stop heat ramp at t in [0,1]. */
export function heatAt(p: Palette, t: number): [number, number, number] {
  const stops = p.heat.map(rgb);
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
export const css = (c: [number, number, number]) => `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;

/** Op kind -> colour. Weight-reading ops are warm (they are the bytes); the
 *  rest are muted, attention/KV get the one cool hue. Unknown kinds (new ones
 *  the hillclimb adds) still get a stable colour. */
export function kindColor(p: Palette, kind: string): string {
  switch (kind) {
    case "lm_head":
      return p.accent;
    case "matmul":
      return p.heat[3];
    case "attn":
      return p.cool;
    case "kv_write":
      return p.down;
    case "gelu":
      return p.heat[2];
    case "ln":
      return p.sub;
    case "residual":
      return p.muted;
    case "embed":
      return p.heat[1];
    default: {
      let h = 0;
      for (const ch of kind) h = (h * 31 + ch.charCodeAt(0)) | 0;
      return [p.up, p.heat[4], p.heat[2], p.cool][Math.abs(h) % 4];
    }
  }
}

export const KIND_LABEL: Record<string, string> = {
  lm_head: "lm head",
  matmul: "matmul",
  attn: "attention",
  kv_write: "kv write",
  gelu: "gelu",
  ln: "layernorm",
  residual: "residual",
  embed: "embed",
};

const lutCache = new WeakMap<Palette, Uint8ClampedArray>();
/** 256-step RGB lookup table of the heat ramp (fast per-pixel painting). */
export function heatLut(p: Palette): Uint8ClampedArray {
  let l = lutCache.get(p);
  if (!l) {
    l = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const c = heatAt(p, i / 255);
      l[i * 3] = c[0];
      l[i * 3 + 1] = c[1];
      l[i * 3 + 2] = c[2];
    }
    lutCache.set(p, l);
  }
  return l;
}
