// Tiny explainer visuals for field-guide entries. Same visual language as the course (web/DESIGN.md):
// blue = memory / stored bytes, ember = compute, olive = communication, grey = everything else.

import { useState } from "react";
import { usePalette } from "../../ui/theme";
import { chartColors, Seg } from "../kit";

// ---- number formats ---------------------------------------------------------------------------------

const FMT = [
  { name: "fp32", s: 1, e: 8, m: 23, note: "training master copy; this site's original weights" },
  { name: "bf16", s: 1, e: 8, m: 7, note: "fp32's range, less precision: today's default for weights" },
  { name: "fp16", s: 1, e: 5, m: 10, note: "more precision, less range: overflows more easily" },
  { name: "fp8", s: 1, e: 4, m: 3, note: "E4M3: H100-era tensor cores run it natively" },
  { name: "int8", s: 0, e: 0, m: 8, int: true, note: "integers × a shared scale per block" },
  { name: "int4", s: 0, e: 0, m: 4, int: true, note: "16 levels × a scale per block of 32–128" },
];

export function FormatBits() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const cell = 15;
  return (
    <div className="p0-bits">
      {FMT.map((f) => {
        const bits = f.s + f.e + f.m;
        return (
          <div key={f.name} className="p0-bits-row">
            <span className="nm">{f.name}</span>
            <svg width={32 * cell * 0.62 + 4} height={cell} viewBox={`0 0 ${32 * cell + 8} ${cell + 4}`} preserveAspectRatio="xMinYMid meet" style={{ maxWidth: "100%" }}>
              {Array.from({ length: bits }, (_, i) => {
                const kind = f.int ? "m" : i < f.s ? "s" : i < f.s + f.e ? "e" : "m";
                const fill = kind === "s" ? pal.muted : kind === "e" ? cc.c2 : cc.c2;
                return <rect key={i} x={i * cell} y={2} width={cell - 2} height={cell} rx={2} fill={fill} fillOpacity={kind === "e" ? 0.95 : kind === "m" ? 0.4 : 0.8} />;
              })}
            </svg>
            <span className="note lr-small">
              {bits} bits · 7B model = {Math.round((7 * bits) / 8)} GB · {f.note}
            </span>
          </div>
        );
      })}
      <div className="lr-small" style={{ marginTop: "0.3rem" }}>
        grey: sign · solid blue: exponent (range) · faded blue: mantissa or integer (precision)
      </div>
    </div>
  );
}

// ---- MHA / GQA / MQA --------------------------------------------------------------------------------

export function HeadSharing() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [kv, setKv] = useState(2);
  const Q = 8;
  const W = 420;
  const qx = (i: number) => 30 + i * ((W - 60) / (Q - 1));
  const kx = (j: number) => (kv === 1 ? W / 2 : 30 + j * ((W - 60) / Math.max(1, kv - 1)));
  return (
    <div>
      <div className="lr-controls">
        <Seg
          options={[
            { v: 8, label: "MHA: 8 KV heads" },
            { v: 2, label: "GQA: 2" },
            { v: 1, label: "MQA: 1" },
          ]}
          value={kv}
          onChange={setKv}
          label="KV heads"
        />
      </div>
      <svg viewBox={`0 0 ${W} 120`} style={{ width: "100%", maxWidth: 460 }} role="img" aria-label={`${Q} query heads sharing ${kv} key-value heads`}>
        {Array.from({ length: Q }, (_, i) => {
          const j = Math.floor(i / (Q / kv));
          return <line key={i} x1={qx(i)} y1={36} x2={kx(j)} y2={86} stroke={pal.muted} />;
        })}
        {Array.from({ length: Q }, (_, i) => (
          <rect key={i} x={qx(i) - 11} y={18} width={22} height={18} rx={4} fill={cc.c1} fillOpacity={0.8} />
        ))}
        {Array.from({ length: kv }, (_, j) => (
          <rect key={j} x={kx(j) - 13} y={86} width={26} height={18} rx={3} fill={cc.c2} />
        ))}
        <text x={W - 4} y={10} textAnchor="end" fontSize={10} fill={pal.sub}>
          query heads (compute)
        </text>
        <text x={W - 4} y={118} textAnchor="end" fontSize={10} fill={pal.sub}>
          KV heads (cached) · KV cache = {kv}/8 of MHA
        </text>
      </svg>
    </div>
  );
}

// ---- parallelism patterns ---------------------------------------------------------------------------

export type ParMode = "data" | "tensor" | "pipeline" | "expert" | "sequence";

export function ParallelMini({ mode }: { mode: ParMode }) {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const N = 4, W = 440, gw = 88, gh = 76, gy = 26;
  const gx = (i: number) => 14 + i * ((W - 28 - gw) / (N - 1));
  const panel = pal.dark ? "#262421" : "#ffffff";
  const layers = 4;
  const cellsFor = (g: number) => {
    const out: Cell[] = [];
    for (let l = 0; l < layers; l++) {
      for (let c = 0; c < 4; c++) {
        let held = true;
        if (mode === "tensor") held = c === g;
        if (mode === "pipeline") held = l === g;
        if (mode === "expert") held = c === g || l === 0;
        out.push({ l, c, held });
      }
    }
    return out;
  };
  const label: Record<ParMode, string> = {
    data: "every GPU holds the whole model; each serves different requests",
    tensor: "every GPU holds a slice of every layer; all-reduce after each",
    pipeline: "every GPU holds some layers; activations hop GPU to GPU",
    expert: "shared layer everywhere; each GPU holds different experts; tokens shuffle all-to-all",
    sequence: "every GPU holds a chunk of one long sequence's tokens; attention exchanges keys/values",
  };
  return (
    <svg viewBox={`0 0 ${W} 140`} style={{ width: "100%", maxWidth: 480 }} role="img" aria-label={label[mode]}>
      {/* links */}
      {mode === "tensor" && <path d={`M${gx(0) + gw / 2},${gy + gh + 8} H${gx(N - 1) + gw / 2}`} stroke={cc.c3} strokeWidth={2} />}
      {mode === "tensor" && Array.from({ length: N }, (_, i) => <line key={i} x1={gx(i) + gw / 2} x2={gx(i) + gw / 2} y1={gy + gh} y2={gy + gh + 8} stroke={cc.c3} strokeWidth={2} />)}
      {mode === "pipeline" && Array.from({ length: N - 1 }, (_, i) => <path key={i} d={`M${gx(i) + gw + 2},${gy + gh / 2} H${gx(i + 1) - 2}`} stroke={cc.c3} strokeWidth={2} />)}
      {mode === "expert" &&
        Array.from({ length: N }, (_, i) =>
          Array.from({ length: N }, (_, j) => (j > i ? <path key={`${i}${j}`} d={`M${gx(i) + gw / 2},${gy + gh} Q${(gx(i) + gx(j)) / 2 + gw / 2},${gy + gh + 30 + (j - i) * 6} ${gx(j) + gw / 2},${gy + gh}`} fill="none" stroke={cc.c3} strokeWidth={1.2} /> : null)),
        )}
      {mode === "sequence" && Array.from({ length: N - 1 }, (_, i) => <path key={i} d={`M${gx(i) + gw + 2},${gy + gh / 2} H${gx(i + 1) - 2}`} stroke={cc.c3} strokeWidth={2} />)}
      {Array.from({ length: N }, (_, g) => (
        <g key={g}>
          <rect x={gx(g)} y={gy} width={gw} height={gh} rx={8} fill={panel} stroke={pal.sub} />
          <text x={gx(g) + 6} y={gy - 6} fontSize={9.5} fill={pal.muted}>
            GPU {g + 1}
          </text>
          {mode === "sequence"
            ? Array.from({ length: 8 }, (_, t) => <rect key={t} x={gx(g) + 8 + (t % 4) * 19} y={gy + 14 + Math.floor(t / 4) * 26} width={15} height={18} rx={4} fill={pal.code} stroke={pal.muted} />)
            : cellsFor(g).map(({ l, c, held }) => (
                <rect key={`${l}-${c}`} x={gx(g) + 8 + c * 19} y={gy + 8 + l * 16} width={16} height={12} rx={2} fill={held ? (mode === "expert" && l > 0 ? cc.c1 : cc.c2) : pal.rule} fillOpacity={held ? 0.75 : 0.35} />
              ))}
          {mode === "data" && (
            <text x={gx(g) + gw / 2} y={gy + gh + 16} textAnchor="middle" fontSize={9.5} fill={pal.sub}>
              users {g * 2 + 1}–{g * 2 + 2}
            </text>
          )}
        </g>
      ))}
      <text x={W / 2} y={136} textAnchor="middle" fontSize={9.5} fill={pal.sub}>
        {label[mode]}
      </text>
    </svg>
  );
}

interface Cell {
  l: number;
  c: number;
  held: boolean;
}
