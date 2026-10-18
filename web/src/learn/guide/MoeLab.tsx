// Mixture of experts, hands-on: a router sends each token to its top-k of 8 experts. Shows why compute per token is
// small while memory (and, in a batch, bytes read) is not, and what a hot expert does to an expert-parallel deployment.

import { useMemo, useState } from "react";
import { usePalette } from "../../ui/theme";
import { chartColors, Seg, Stat, Swatch, nf } from "../kit";

const E = 8;
// Mixtral-8x7B-shaped numbers (Mistral AI, 2023): 46.7B total, 12.9B active with top-2.
const PER_EXPERT = (46.7 - 12.9) / 6; // ≈ 5.63B params in each expert's feed-forward weights (all layers)
const SHARED = 12.9 - 2 * PER_EXPERT; // ≈ 1.63B: attention, embeddings, norms, router
const TOTAL = SHARED + E * PER_EXPERT;

const WORDS = "The cat sat on the warm mat while rain fell on the old tin roof and the kettle began to sing softly in the kitchen".split(" ");

/** small deterministic hash → [0, 1) */
function h(a: number, b: number) {
  let x = (a * 374761393 + b * 668265263) ^ 0x5bf03635;
  x = Math.imul(x ^ (x >>> 13), 1274126177);
  return ((x ^ (x >>> 16)) >>> 0) / 4294967296;
}

function route(n: number, k: number, skew: boolean) {
  const out: number[][] = [];
  for (let t = 0; t < n; t++) {
    const s = Array.from({ length: E }, (_, e) => h(t + 1, e + 1) + (skew && e === 2 ? 0.55 : 0) + (skew && e === 5 ? 0.2 : 0));
    out.push(
      s
        .map((v, e) => [v, e] as const)
        .sort((a, b) => b[0] - a[0])
        .slice(0, k)
        .map((x) => x[1]),
    );
  }
  return out;
}

export function MoeLab() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [k, setK] = useState(2);
  const [n, setN] = useState(8);
  const [skew, setSkew] = useState(false);
  const [hov, setHov] = useState<number | null>(null);
  const routes = useMemo(() => route(n, k, skew), [n, k, skew]);
  const load = Array.from({ length: E }, (_, e) => routes.filter((r) => r.includes(e)).length);
  const touched = load.filter((l) => l > 0).length;
  const fair = (n * k) / E;
  const hot = Math.max(...load);
  const shownTok = Math.min(n, 8);
  const active = SHARED + k * PER_EXPERT;
  const readGB = (SHARED + touched * PER_EXPERT) * 2; // bf16
  const W = 560, tokY = 34, expY = 168;
  const tx = (i: number) => 40 + i * ((W - 80) / Math.max(1, shownTok - 1 || 1));
  const ex = (e: number) => 44 + e * ((W - 88) / (E - 1));
  const sel = hov != null && hov < shownTok ? hov : null;
  const panel = pal.dark ? "#262421" : "#ffffff";
  return (
    <div className="p0-moe">
      <div className="lr-controls">
        <Seg options={[1, 2, 4].map((v) => ({ v, label: `top-${v}` }))} value={k} onChange={setK} label="experts per token" />
        <Seg options={[1, 8, 64].map((v) => ({ v, label: `${v} token${v > 1 ? "s" : ""} / step` }))} value={n} onChange={(v) => (setN(v), setHov(null))} label="tokens per step" />
        <Seg
          options={[
            { v: 0, label: "balanced router" },
            { v: 1, label: "skewed router" },
          ]}
          value={skew ? 1 : 0}
          onChange={(v) => setSkew(!!v)}
          label="router"
        />
      </div>
      <svg viewBox={`0 0 ${W} 250`} style={{ width: "100%" }} onPointerLeave={() => setHov(null)} role="img" aria-label={`each token routed to ${k} of 8 experts`}>
        <text x={8} y={14} fontSize={10} fill={pal.muted}>
          tokens in this step{n > 8 ? ` (first 8 of ${n})` : ""}
        </text>
        {routes.slice(0, shownTok).map((r, i) =>
          r.map((e) => (
            <path
              key={`${i}-${e}`}
              d={`M${tx(i)},${tokY + 12} C${tx(i)},${(tokY + expY) / 2} ${ex(e)},${(tokY + expY) / 2} ${ex(e)},${expY - 18}`}
              fill="none"
              stroke={cc.c3}
              strokeWidth={sel === i ? 2.2 : 1}
              opacity={sel == null ? 0.55 : sel === i ? 1 : 0.12}
            />
          )),
        )}
        {Array.from({ length: shownTok }, (_, i) => (
          <g key={i} onPointerEnter={() => setHov(i)} style={{ cursor: "pointer" }}>
            <rect x={tx(i) - 26} y={tokY - 10} width={52} height={22} rx={5} fill={panel} stroke={sel === i ? pal.accent : pal.rule} strokeWidth={sel === i ? 1.6 : 1} />
            <text x={tx(i)} y={tokY + 5} textAnchor="middle" fontSize={10} fill={pal.fg}>
              {WORDS[i % WORDS.length]}
            </text>
          </g>
        ))}
        <rect x={W / 2 - 78} y={(tokY + expY) / 2 - 9} width={156} height={16} rx={8} fill={pal.dark ? "#211f1d" : "#fbfaf7"} stroke={pal.rule} />
        <text x={W / 2} y={(tokY + expY) / 2 + 3} textAnchor="middle" fontSize={9.5} fill={pal.sub}>
          router: top-{k} of 8 per token
        </text>
        {Array.from({ length: E }, (_, e) => {
          const l = load[e];
          const on = sel != null ? routes[sel].includes(e) : l > 0;
          const barH = (l / Math.max(hot, 1)) * 44;
          return (
            <g key={e}>
              <rect x={ex(e) - 24} y={expY - 18} width={48} height={36} rx={6} fill={on ? cc.c1 : pal.rule} fillOpacity={on ? 0.85 : 0.6} />
              <text x={ex(e)} y={expY + 4} textAnchor="middle" fontSize={10} fill={on ? "#fff" : pal.sub} fontWeight={on ? 700 : 400}>
                E{e + 1}
              </text>
              <rect x={ex(e) - 10} y={expY + 26 + (44 - barH)} width={20} height={Math.max(barH, l ? 2 : 0)} rx={2} fill={cc.c1} fillOpacity={0.45} />
              <text x={ex(e)} y={expY + 82} textAnchor="middle" fontSize={9.5} fill={pal.sub}>
                {l}
              </text>
            </g>
          );
        })}
        {n > 1 && (
          <line x1={20} x2={W - 20} y1={expY + 26 + 44 - (fair / Math.max(hot, 1)) * 44} y2={expY + 26 + 44 - (fair / Math.max(hot, 1)) * 44} stroke={pal.muted} strokeDasharray="3 3" />
        )}
      </svg>
      <div className="lr-stats">
        <Stat k="compute per token" v={`${nf(active, 1)}B`} unit={`of ${nf(TOTAL, 1)}B params`} accent />
        <Stat k="must sit in memory" v={`${nf(TOTAL * 2)} GB`} unit="bf16, all experts" />
        <Stat k="read this step" v={`${nf(readGB)} GB`} unit={`${touched} of 8 experts touched`} />
        {n > 1 && <Stat k="busiest expert" v={`${(hot / fair).toFixed(1)}×`} unit="its fair share" />}
      </div>
      <p className="lr-note">
        Lines (<Swatch c={cc.c3}>olive</Swatch>) are the router's dispatch; <Swatch c={cc.c1}>ember</Swatch> experts are doing
        work, bars under them count tokens received (dashed line: a perfectly even share). Hover a token to see its route.
        With one token per step, only {k} experts' weights are read: it decodes like a {nf(active, 0)}B model. At 64 tokens
        nearly every expert is touched, so each step reads almost all {nf(TOTAL * 2)} GB, shared across the batch.
      </p>
    </div>
  );
}
