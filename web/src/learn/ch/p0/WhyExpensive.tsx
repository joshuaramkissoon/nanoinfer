import { useMemo, useState } from "react";
import { Tip, useWidth } from "../../../ui/Figure";
import { usePalette } from "../../../ui/theme";
import { log } from "../../../lib/fmt";
import { ConstraintMap } from "../../ConstraintMap";
import { chartColors, Deeper, Fig, G, Idea, Seg, Slider, Stat, Swatch, nf, useLearnData, T } from "../../kit";
import { FORMATS, MACHINES, MODELS, SOURCES, fmtBytes, fmtRate } from "./specs";

// ---- fig 0.2.1: bytes per token → the speed limit on each machine ---------------------------------------

/** our engine's measured GPT-2 decode, by format (docs/LOG.md; GPU int4 from data.json) */
const MEASURED: Record<string, Record<string, number>> = {
  m4cpu: { fp32: 146, bf16: 249, int8: 425, int4: 497 },
  m4gpu: { fp32: 133, bf16: 245, int8: 365, int4: 817 },
};

function BytesPerToken() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const d = useLearnData();
  const [mk, setMk] = useState("7b");
  const [fk, setFk] = useState("bf16");
  const m = MODELS.find((x) => x.key === mk)!;
  const f = FORMATS.find((x) => x.key === fk)!;
  const bytes = m.params * f.bytes;
  const x = log(0.1, 1e5, 0, 100);
  const measured = (mach: string) => {
    if (mk !== "gpt2") return null;
    if (mach === "m4gpu" && fk === "int4" && d) return d.headline.gpu.decode;
    return MEASURED[mach]?.[fk] ?? null;
  };
  return (
    <Fig
      n="0.2.1"
      title="Every token reads the whole model, so bandwidth sets a speed limit"
      prov="estimate"
      caption={
        <>
          Top: bytes read and arithmetic done for <b>one token, one user</b>. Below, each machine's ceiling: its memory
          bandwidth divided by those bytes (<Swatch c={cc.c2}>blue</Swatch>, log scale: each gridline is 10×). Rows marked{" "}
          <i>doesn't fit</i> can't hold the weights at all. For GPT-2, <Swatch c={cc.c1}>ember</Swatch> rings are what our
          engine actually measured.
        </>
      }
      tryit={
        <>
          A 7B model in bf16 is 14 GB per token: even an H100 tops out near 240 tokens a second. Now pick 70B: one H100 can't
          hold it at all. Switch to int4 and it fits, at 4× the speed limit. Then pick GPT-2 and compare our measured rings.
        </>
      }
      source={SOURCES + " Ceilings ignore the KV cache and quantisation scales (≈ +6–12% bytes)."}
    >
      <div className="lr-controls">
        <Seg options={MODELS.map((x) => ({ v: x.key, label: x.key === "gpt2" ? "GPT-2 124M" : x.key.toUpperCase() }))} value={mk} onChange={setMk} label="model" />
        <Seg options={FORMATS.map((x) => ({ v: x.key, label: x.name }))} value={fk} onChange={setFk} label="number format" />
      </div>
      <div className="p0-hero">
        <div>
          <div className="p0-hero-v">
            {fmtBytes(bytes)}
          </div>
          <div className="lr-small">read from memory per token</div>
        </div>
        <div>
          <div className="p0-hero-v">
            {fmtRate((2 * m.params) / 1e9)} GFLOP
          </div>
          <div className="lr-small">of arithmetic per token (2 per weight)</div>
        </div>
        <div className="p0-hero-note lr-small">
          {m.name}: {m.blurb}. {nf(m.params / 1e9, m.params < 1e9 ? 3 : 0)} billion weights × {f.bytes} byte{f.bytes === 1 ? "" : "s"}.
        </div>
      </div>
      <div className="p0-ceil">
        {MACHINES.map((mach) => {
          const fits = bytes <= mach.cap * 1e9 * 0.95;
          const ceil = (mach.bw * 1e9) / bytes;
          const meas = measured(mach.key);
          return (
            <div key={mach.key} className={`p0-ceil-row${fits ? "" : " nofit"}`} title={mach.note}>
              <span className="nm">
                {mach.name}
                <span className="lr-small">
                  {" "}
                  {mach.bw >= 1000 ? `${nf(mach.bw / 1000, 1)} TB/s` : `${mach.bw} GB/s`} · {mach.cap} GB
                </span>
              </span>
              <span className="track">
                {fits && [1, 10, 100, 1000, 10000].map((g) => <i key={g} className="grid" style={{ left: `${x(g)}%` }} />)}
                {fits ? (
                  <>
                    <i className="guide" />
                    <i className="dot" style={{ left: `${x(ceil)}%`, background: cc.c2 }} />
                    {meas != null && <i className="meas" style={{ left: `${x(meas)}%`, borderColor: cc.c1 }} title={`our engine: ${meas.toFixed(0)} tok/s`} />}
                  </>
                ) : (
                  <span className="nofit-k">
                    <span>doesn't fit: needs {fmtBytes(bytes)}, has {mach.cap} GB</span>
                  </span>
                )}
              </span>
              <span className="v">{fits ? `${fmtRate(ceil)} tok/s` : "–"}</span>
              {fits && meas != null && (
                <span className="measv lr-small" style={{ color: pal.sub }}>
                  measured {meas.toFixed(0)} tok/s ({Math.round((meas / ceil) * 100)}% of ceiling)
                </span>
              )}
            </div>
          );
        })}
        <div className="p0-ceil-axis lr-small">
          {[1, 10, 100, 1000, 10000].map((g) => (
            <span key={g} style={{ left: `${x(g)}%` }}>
              {g >= 1000 ? `${g / 1000}k` : g}
            </span>
          ))}
          <span className="u">tokens per second, one user →</span>
        </div>
      </div>
    </Fig>
  );
}

// ---- fig 0.2.2: how much maths each byte could carry -----------------------------------------------

function IdleMaths() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [mk, setMk] = useState<"m4gpu" | "h100">("h100");
  const [e, setE] = useState(0);
  const B = 2 ** e;
  const mach = MACHINES.find((x) => x.key === mk)!;
  // FLOPs the chip can do per byte read, vs FLOPs decode does per byte (bf16: 2 FLOPs per 2-byte weight, per user)
  const ratio = Math.round((mach.tflops * 1e12) / (mach.bw * 1e9));
  const used = Math.min(B, ratio);
  const busy = used / ratio;
  const cols = mk === "h100" ? 25 : 9;
  return (
    <Fig
      n="0.2.2"
      title="While one byte arrives, the chip could do hundreds of multiply-adds"
      prov="estimate"
      caption={
        <>
          Each square is one FLOP the chip could do in the time it takes to read one byte of weights:{" "}
          {mk === "h100" ? "~989 TFLOP/s ÷ 3.35 TB/s ≈ 295" : "~4.3 TFLOP/s ÷ 105 GB/s ≈ 41"}. Decode in bf16 does one FLOP
          per byte per user (<Swatch c={cc.c1}>ember</Swatch>); the rest of the chip's arithmetic (grey) waits for memory. This
          counts weight bytes only. Each user's KV cache isn't shared, so real speeds do fall as users join: fig 0.2.3 adds it.
        </>
      }
      tryit={
        <>
          Drag the users slider. Each extra user's token reuses the same bytes, so the ember fills in for free until the
          chip runs out of maths. That fill point is the ridge from chapter 09, and the whole case for batching.
        </>
      }
      source="H100 SXM: 989 TFLOP/s dense bf16 and 3.35 TB/s HBM3 (NVIDIA datasheet). M4 GPU: ~4.3 TFLOP/s f32 and 105 GB/s measured (docs/LOG.md)."
    >
      <div className="lr-controls">
        <Seg
          options={[
            { v: "h100", label: "H100" },
            { v: "m4gpu", label: "M4 GPU" },
          ]}
          value={mk}
          onChange={(v) => setMk(v as "h100" | "m4gpu")}
          label="chip"
        />
        <Slider label="users per step" value={e} min={0} max={9} onChange={setE} fmt={() => nf(B)} />
      </div>
      <div className="p0-lanes" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)`, maxWidth: mk === "h100" ? 520 : 240 }} aria-label={`${used} of ${ratio} lanes busy`}>
        {Array.from({ length: ratio }, (_, i) => (
          <i key={i} style={{ background: i < used ? cc.c1 : pal.rule }} />
        ))}
      </div>
      <div className="lr-stats">
        <Stat k="arithmetic busy" v={`${(busy * 100).toFixed(busy < 0.1 ? 1 : 0)}%`} accent />
        <Stat k="FLOPs per byte, decode" v={nf(B)} unit={`of ${ratio} possible`} />
        <Stat k="each user's speed, weights only" v={B <= ratio ? "unchanged" : `÷ ${(B / ratio).toFixed(1)}`} unit={B <= ratio ? "memory-bound" : "compute-bound"} />
      </div>
    </Fig>
  );
}

// ---- fig 0.2.3: cost per token vs batch size ------------------------------------------------------

const PRICE = 2.5; // $ per H100-hour, a typical on-demand rental (estimate)
const W7 = 14e9; // 7B in bf16
const KV_PER_TOK = 2 * 32 * 4096 * 2; // Llama-2-7B: K and V, 32 layers, 4096 wide, bf16 = 0.5 MB / token
const H100 = MACHINES.find((x) => x.key === "h100")!;

function costAt(B: number, ctx: number, gqa: boolean) {
  const kv = (KV_PER_TOK / (gqa ? 4 : 1)) * ctx;
  const memS = (W7 + B * kv) / (H100.bw * 1e9);
  const cmpS = (B * 2 * 7e9) / (H100.tflops * 1e12);
  const step = Math.max(memS, cmpS);
  const per = 1 / step;
  const total = B * per;
  return { step, per, total, usd: (PRICE / 3600 / total) * 1e6, kvGB: (B * kv) / 1e9, memBound: memS >= cmpS };
}

function CostCurve() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [ref, w] = useWidth<HTMLDivElement>();
  const [ctx, setCtx] = useState(1000);
  const [gqa, setGqa] = useState(false);
  const [e, setE] = useState(4);
  const [hover, setHover] = useState<number | null>(null);
  const kvPer = (KV_PER_TOK / (gqa ? 4 : 1)) * ctx;
  const maxB = Math.max(1, Math.floor((H100.cap * 1e9 * 0.95 - W7) / kvPer));
  const Bs = useMemo(() => {
    const out: number[] = [];
    for (let b = 1; b <= Math.min(512, maxB); b = Math.max(b + 1, Math.round(b * 1.15))) out.push(b);
    if (out[out.length - 1] !== Math.min(512, maxB)) out.push(Math.min(512, maxB));
    return out;
  }, [maxB]);
  const B = Math.min(2 ** e, maxB);
  const shown = hover ?? B;
  const cur = costAt(shown, ctx, gqa);
  const W = Math.max(280, w), H = 240, pl = 46, pr = 16, top = 26, pb = 34;
  const x = log(1, 512, pl, W - pr);
  const y = log(0.01, 10, H - pb, top);
  const pts = Bs.map((b) => [x(b), y(costAt(b, ctx, gqa).usd)] as const);
  const path = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
  const ring = pal.dark ? "#211f1d" : "#fbfaf7";
  const pick = (px: number) => {
    const b = Math.round(x.invert(px));
    setHover(Math.max(1, Math.min(maxB, 512, b)));
  };
  return (
    <Fig
      n="0.2.3"
      title="Batching makes tokens ~20× cheaper, until the KV cache fills memory"
      prov="estimate"
      caption={
        <>
          One H100 serving a 7B model in bf16, at ${PRICE.toFixed(2)} an hour. Across: users per step (log). Up: dollars per
          million tokens (log). Each step reads the weights once (shared) plus every user's KV cache (not shared). The line
          stops where weights + KV caches fill the 80 GB.
        </>
      }
      tryit={
        <>
          Raise the context to 4k tokens: the line gets shorter, because each user's cache is 2 GB. Then turn on GQA, which
          shrinks the cache 4× (the field guide explains how). Longer line, cheaper tokens.
        </>
      }
      source={`H100 spec numbers as in fig 0.2.1; ${PRICE.toFixed(2)} $/GPU-hour is a typical 2025–26 on-demand rental (estimate). KV cache: Llama-2-7B shape (32 layers × 4096 × K and V × 2 bytes = 0.5 MB per token). Ideal kernels; real servers reach maybe half of this.`}
    >
      <div className="lr-controls">
        <span className="p0-ctl">
          <span className="lr-small">context, tokens</span>
          <Seg
            options={[256, 1000, 4000].map((v) => ({ v, label: `${v >= 1000 ? `${v / 1000}k` : v}` }))}
            value={ctx}
            onChange={setCtx}
            label="context, tokens"
          />
        </span>
        <span className="p0-ctl">
        <span className="lr-small">KV cache</span>
        <Seg
          options={[
            { v: 0, label: "full" },
            { v: 1, label: "GQA (KV ÷ 4)" },
          ]}
          value={gqa ? 1 : 0}
          onChange={(v) => setGqa(!!v)}
          label="KV cache size"
        />
        </span>
      </div>
      <div className="lr-controls">
        <Slider label="users per step" value={e} min={0} max={9} onChange={(v) => (setE(v), setHover(null))} fmt={() => nf(B)} />
        <span className="lr-small">memory fits at most {nf(maxB)} users</span>
      </div>
      <div ref={ref} style={{ position: "relative" }}>
        {w > 0 && (
          <svg
            width={W}
            height={H}
            role="img"
            aria-label="dollars per million tokens falling as users per step grow"
            onPointerMove={(ev) => pick(ev.clientX - ev.currentTarget.getBoundingClientRect().left)}
            onPointerLeave={() => setHover(null)}
          >
            {[0.01, 0.1, 1, 10].map((t) => (
              <g key={t}>
                <line x1={pl} x2={W - pr} y1={y(t)} y2={y(t)} stroke={pal.rule} />
                <text x={pl - 6} y={y(t) + 4} textAnchor="end" fontSize={T.xs} fill={pal.muted}>
                  ${t}
                </text>
              </g>
            ))}
            {[1, 4, 16, 64, 256].map((t) => (
              <text key={t} x={x(t)} y={H - 16} textAnchor="middle" fontSize={T.xs} fill={pal.muted}>
                {t}
              </text>
            ))}
            <text x={W - pr} y={H - 2} textAnchor="end" fontSize={T.xs} fill={pal.muted}>
              users per step →
            </text>
            <text x={pl - 40} y={10} fontSize={T.xs} fill={pal.muted}>
              $ per million tokens
            </text>
            <path d={path} fill="none" stroke={cc.c1} strokeWidth={2} strokeLinejoin="round" />
            {maxB < 512 && (
              <g>
                <line x1={pts[pts.length - 1][0]} x2={pts[pts.length - 1][0]} y1={top} y2={H - pb} stroke={cc.c2} strokeDasharray="3 3" />
                <text x={pts[pts.length - 1][0] + (pts[pts.length - 1][0] > W - 130 ? -6 : 6)} y={top + 24} fontSize={T.xs} fill={pal.sub} textAnchor={pts[pts.length - 1][0] > W - 130 ? "end" : "start"}>
                  memory full
                </text>
              </g>
            )}
            <line x1={x(shown)} x2={x(shown)} y1={top} y2={H - pb} stroke={pal.muted} opacity={0.5} />
            <circle cx={x(shown)} cy={y(cur.usd)} r={5} fill={cc.c1} stroke={ring} strokeWidth={2} />
          </svg>
        )}
        {w > 0 && hover != null && (
          <Tip x={Math.min(Math.max(x(shown), 90), W - 90)} y={y(cur.usd) - 6}>
            <b>
              {shown} {shown === 1 ? "user" : "users"}
            </b>
            <br />${cur.usd.toFixed(cur.usd < 0.1 ? 3 : 2)} per M tokens · each {cur.per.toFixed(0)} tok/s
          </Tip>
        )}
      </div>
      <div className="lr-stats">
        <Stat k="cost" v={`$${cur.usd.toFixed(cur.usd < 0.1 ? 3 : 2)}`} unit="per million tokens" accent />
        <Stat k="each user" v={cur.per.toFixed(0)} unit="tok/s" />
        <Stat k="all users" v={nf(cur.total)} unit="tok/s" />
      </div>
      <p className="lr-note">
        Each step reads 14 GB of weights once plus {cur.kvGB.toFixed(1)} GB of KV caches, so it's{" "}
        {cur.memBound ? "still memory-bound" : "compute-bound"}: {(cur.step * 1000).toFixed(1)} ms per step.
      </p>
    </Fig>
  );
}

export default function WhyExpensive() {
  return (
    <>
      <p>
        Here is the fact the whole field is organised around. To write one token, the model runs its program once, and that
        program touches <b>every weight</b>. Each weight is read from memory and used for one multiply and one add. Then the
        next token reads them all again.
      </p>
      <p>
        So the cost of a token is roughly: <b>bytes = weights × bytes per weight</b>, and{" "}
        <b>arithmetic = 2 × weights</b>. Neither depends on how clever the question is.
      </p>
      <p>
        Two words you'll see everywhere: <G k="prefill" /> is the first pass, which reads your whole prompt at once;{" "}
        <G k="decode" /> is every pass after it, each writing exactly one new token. Decode is what you watch streaming in,
        and it's where this chapter's cost bites (chapter 08 replays both on a real run).
      </p>
      <BytesPerToken />
      <p>
        That ceiling is a back-of-envelope SLO you can compute before writing a line of code (chapter 10 does it properly).
        Real engines reach 50–80% of it. Notice two different walls: <b>bandwidth</b> (how fast) and <b>capacity</b>{" "}
        (whether it runs at all).
      </p>

      <h2>Memory, not maths</h2>
      <p>
        If you've profiled a service that's I/O-bound while the CPUs sit idle, you know this shape. Modern accelerators can
        do far more arithmetic per second than their memory can feed them. Decode gives each byte almost nothing to do.
      </p>
      <IdleMaths />
      <p>
        One user leaves more than 99% of an H100's arithmetic idle. That's why serving is a game of finding more useful work
        for each byte: more users per step (<G k="batch">batching</G>), more tokens per user per step (speculative
        decoding), or fewer bytes per weight (<G k="quantization">quantisation</G>).
      </p>

      <h2>The one piece of state: the KV cache</h2>
      <p>
        To write the next token, the model has to look back at every token before it (that's <G k="attention" />, chapter
        05). From each earlier token it needs two small vectors per layer, a <b>key</b> and a <b>value</b>, and those never
        change once computed. So every engine memoises them in an append-only, per-request log: the <G k="KV cache" />.
      </p>
      <p>
        You know this shape: it's per-session state that grows. One entry per token, per layer. For a 7B model (32 layers,
        4,096 numbers each for K and V, 2 bytes) that's <b>0.5 MB per token</b>, so a 4,000-token conversation holds 2 GB, and
        unlike the weights it can't be shared between users. It's read in full on every step, so it costs bandwidth as well
        as space. Chapter 12 shows it being built.
      </p>

      <h2>Latency, throughput, and the bill</h2>
      <p>
        Batching trades a little latency per user for a lot of <G k="throughput" />. The operator pays per GPU-hour, so cost
        per token is GPU price ÷ total tokens per second. The catch is that every user brings their own KV cache, which
        takes memory that doesn't shrink with batching.
      </p>
      <CostCurve />
      <p>
        That's the economics of every LLM API in one chart: cheap tokens come from big batches, big batches need memory, and
        long contexts eat it. Users meanwhile care about <G k="latency" />: <G k="TTFT">time to first token</G> and{" "}
        <G k="TPOT">time per output token</G>. The engine's job is to sit as far down this curve as the latency targets
        allow.
      </p>

      <h2>The constraints, and what works around them</h2>
      <p>Every technique you'll hear about exists to work around one of a handful of limits:</p>
      <Fig
        n="0.2.4"
        title="Six constraints, and the techniques that work around each"
        prov="diagram"
        caption={
          <>
            Colour is the kind of limit: blue for memory, ember for compute, olive for talking between chips, grey for time.
            Open one limit at a time; each chip opens the chapter (<b>ch</b>) or field-guide entry (<b>guide</b>) that
            explains it. Don't try to learn the names yet: the rest of the course meets them one by one.
          </>
        }
      >
        <ConstraintMap collapsible />
      </Fig>
      <Idea>
        Each token reads every weight once and does almost no arithmetic with it, so decode is limited by memory bandwidth,
        and what fits is limited by memory capacity. Nearly every inference trick either reads fewer bytes, shares each read
        across more work, or fits more into memory.
      </Idea>
      <Deeper title="Why not just keep the weights in cache?">
        <p>
          Same instinct as caching a hot dataset, wrong scale. An H100 has ~50 MB of L2 cache and ~30 MB of on-chip SRAM
          spread across its cores; a 7B model is 14 GB. Every token streams the weights through the chip once; there's no
          reuse within a token to exploit, only across tokens of a batch, which is exactly what batching does. (Cerebras and
          Groq build chips with hundreds of MB of SRAM and shard a model across many of them to get around this.)
        </p>
      </Deeper>
    </>
  );
}
