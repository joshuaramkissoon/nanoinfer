import type { ForwardTrace } from "../engine/types";

export interface OpAgg {
  name: string;
  kind: string;
  calls: number;
  ms: number;
  flops: number;
  bytes: number;
}

/** Sum ops by name across forwards (all layers together). */
export function aggregateByName(fwds: ForwardTrace[]): { rows: OpAgg[]; totalMs: number } {
  const m = new Map<string, OpAgg>();
  let totalMs = 0;
  for (const f of fwds) {
    totalMs += f.ms;
    for (const o of f.ops) {
      let a = m.get(o.name);
      if (!a) m.set(o.name, (a = { name: o.name, kind: o.kind, calls: 0, ms: 0, flops: 0, bytes: 0 }));
      a.calls++;
      a.ms += o.dur;
      a.flops += o.flops;
      a.bytes += o.bytes;
    }
  }
  const rows = [...m.values()].sort((a, b) => b.ms - a.ms);
  return { rows, totalMs };
}
