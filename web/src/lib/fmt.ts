// Number formatting and tiny scales (no d3: a few lines do).

export const fmtBytes = (b: number) =>
  b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(b >= 1e8 ? 0 : 1)} MB` : b >= 1e3 ? `${(b / 1e3).toFixed(1)} KB` : `${b | 0} B`;

export const fmtMs = (ms: number) =>
  ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : ms >= 10 ? `${ms.toFixed(0)} ms` : ms >= 1 ? `${ms.toFixed(1)} ms` : `${(ms * 1000).toFixed(0)} µs`;

export function sig(x: number, n = 3): string {
  if (!isFinite(x)) return "–";
  if (x === 0) return "0";
  const d = Math.max(0, n - 1 - Math.floor(Math.log10(Math.abs(x))));
  return x.toFixed(Math.min(d, 6));
}

export const pct = (x: number, d = 0) => `${(x * 100).toFixed(d)}%`;

export function linear(d0: number, d1: number, r0: number, r1: number) {
  const k = (r1 - r0) / (d1 - d0 || 1);
  const f = (x: number) => r0 + (x - d0) * k;
  f.invert = (y: number) => d0 + (y - r0) / k;
  return f;
}

export function log(d0: number, d1: number, r0: number, r1: number) {
  const l0 = Math.log10(d0), l1 = Math.log10(d1);
  const k = (r1 - r0) / (l1 - l0 || 1);
  const f = (x: number) => r0 + (Math.log10(Math.max(x, 1e-12)) - l0) * k;
  f.invert = (y: number) => 10 ** (l0 + (y - r0) / k);
  return f;
}

/** Decade ticks (1, 10, 100, ...) plus 2 and 5 subdivisions when the range is short. */
export function logTicks(d0: number, d1: number): number[] {
  const out: number[] = [];
  const decades = Math.log10(d1 / d0);
  for (let e = Math.floor(Math.log10(d0)); e <= Math.ceil(Math.log10(d1)); e++) {
    for (const m of decades < 2.5 ? [1, 2, 5] : [1]) {
      const v = m * 10 ** e;
      if (v >= d0 * 0.999 && v <= d1 * 1.001) out.push(v);
    }
  }
  return out;
}

export const short = (x: number) =>
  x >= 1e9 ? `${sig(x / 1e9, 2)}G` : x >= 1e6 ? `${sig(x / 1e6, 2)}M` : x >= 1e3 ? `${sig(x / 1e3, 2)}k` : sig(x, 2);
