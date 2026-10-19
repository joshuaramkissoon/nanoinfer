import { useState } from "react";
import { useEngine } from "../../engine/client";
import { usePalette } from "../../ui/theme";
import { chartColors, Deeper, Fig, G, Idea, Seg, useLearnData, type LearnData } from "../kit";

type Fmt = 32 | 16 | 8 | 4;
const BYTES_PER_WEIGHT: Record<Fmt, number> = { 32: 4, 16: 2, 8: 1.0625, 4: 0.5625 }; // 8/4-bit: blocks of 32 + a 16-bit scale

const MODELS = [
  { v: "gpt2", label: "GPT-2 small", params: 123.65e6 }, // weights read per token (excludes the position table)
  { v: "8b", label: "an 8B model", params: 8e9 },
  { v: "70b", label: "a 70B model", params: 70e9 },
] as const;

const MACHINES = [
  { v: "m4", label: "M4 Mac mini (spec)", bw: 120, mem: 24e9, note: "Apple's spec figure for the base M4" },
  { v: "core", label: "one M4 core", bw: 57, mem: 24e9, note: "measured: what one CPU core can stream" },
  { v: "gpu", label: "M4 GPU", bw: 105, mem: 24e9, note: "measured: the GPU's practical read speed, on a quiet machine" },
  { v: "h100", label: "datacentre GPU", bw: 3350, mem: 80e9, note: "NVIDIA H100's spec (HBM3), 80 GB" },
] as const;

// Our best measured decode for GPT-2 on the M4 (docs/LOG.md), by machine and format.
// The GPU row and its read speed come from the bench files (public/learn/data.json):
// one user through the batched path (bench/batching.json, B = 1) and the single-stream
// best (bench/results.jsonl), both on a quiet machine.
const MEASURED_CPU: Record<string, Partial<Record<Fmt, number>>> = {
  m4: { 32: 146, 16: 249, 8: 425, 4: 497 },
  core: { 32: 100.5 },
};
const DT_BITS: Record<string, Fmt> = { f32: 32, f16: 16, q8_0: 8, q4_0: 4 };
function measuredGpu(d: LearnData | null): Partial<Record<Fmt, number>> {
  if (!d) return {};
  const out: Partial<Record<Fmt, number>> = {};
  for (const s of d.batching.series) if (DT_BITS[s.label]) out[DT_BITS[s.label]] = s.points[0][1];
  out[4] = Math.max(out[4] ?? 0, d.headline.gpu.decode);
  return out;
}

function Calculator() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [model, setModel] = useState<(typeof MODELS)[number]["v"]>("gpt2");
  const [fmt, setFmt] = useState<Fmt>(32);
  const [mach, setMach] = useState<(typeof MACHINES)[number]["v"]>("m4");
  const tabBw = useEngine((s) => (s.source === "live" ? s.bandwidthGBs : null));
  const d = useLearnData();
  const M = MODELS.find((m) => m.v === model)!;
  const C0 = MACHINES.find((m) => m.v === mach)!;
  const C = C0.v === "gpu" && d ? { ...C0, bw: d.headline.gpuReadGBs } : C0;
  const bytes = M.params * BYTES_PER_WEIGHT[fmt];
  const limit = (C.bw * 1e9) / bytes;
  const fits = bytes <= C.mem;
  const meas = model === "gpt2" ? (mach === "gpu" ? measuredGpu(d)[fmt] : MEASURED_CPU[mach]?.[fmt]) : undefined;
  const fmtBytes = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${(b / 1e6).toFixed(0)} MB`);
  const fmtTok = (x: number) => (x >= 100 ? Math.round(x).toLocaleString() : x >= 10 ? x.toFixed(0) : x.toFixed(1));
  return (
    <Fig
      n="10.1"
      title="Work out the speed limit"
      prov="measured"
      caption={
        <>
          Pick a model, a number format and a machine. The big number is the fastest that machine could possibly write
          tokens for one user: memory speed divided by the bytes of weights read per token. The bar below compares it with
          the best our engine actually measured, where we have a measurement (GPT-2 on the M4).
        </>
      }
      tryit={
        <>
          Start at GPT-2, 32-bit, M4: 243. Switch to 4-bit: the limit goes up about 7× (chapter 15 is about this). Then pick a 70B
          model on the Mac: it doesn't even fit.
          {tabBw ? ` (Your tab measured ${tabBw.toFixed(0)} GB/s.)` : ""}
        </>
      }
    >
      <div style={{ display: "grid", gap: "0.6rem", marginBottom: "1.1rem" }}>
        <Row label="model">
          <Seg options={MODELS.map((m) => ({ v: m.v, label: m.label }))} value={model} onChange={setModel} label="model" />
        </Row>
        <Row label="number format">
          <Seg
            options={([32, 16, 8, 4] as Fmt[]).map((f) => ({ v: f, label: `${f}-bit` }))}
            value={fmt}
            onChange={setFmt}
            label="number format"
          />
        </Row>
        <Row label="machine">
          <Seg options={MACHINES.map((m) => ({ v: m.v, label: m.label }))} value={mach} onChange={setMach} label="machine" />
        </Row>
      </div>
      <div style={{ fontSize: "var(--t-sm)", color: pal.sub }}>
        {C.bw.toLocaleString()} GB/s ÷ {fmtBytes(bytes)} per token =
      </div>
      <div style={{ fontSize: "clamp(var(--t-xl), 7vw, var(--t-2xl))", fontWeight: 700, letterSpacing: "-0.04em", lineHeight: 1.1, fontVariantNumeric: "proportional-nums" }}>
        {fits ? (
          <>
            {fmtTok(limit)} <span style={{ fontSize: "var(--t-md)", fontWeight: 400, color: pal.sub, letterSpacing: 0 }}>tokens per second, at most</span>
          </>
        ) : (
          <span style={{ fontSize: "var(--t-lg)" }}>doesn't fit: {fmtBytes(bytes)} of weights, {fmtBytes(C.mem)} of memory</span>
        )}
      </div>
      <div className="lr-small" style={{ marginTop: "0.2rem" }}>
        {C.note}
      </div>
      {fits && meas != null && (
        <div style={{ marginTop: "1.1rem" }}>
          <div className="lr-bar" style={{ gridTemplateColumns: "7.5rem 1fr 6rem" }}>
            <span className="tk">speed limit</span>
            <span className="b">
              <i style={{ width: "100%", background: pal.muted }} />
            </span>
            <span className="pc">{fmtTok(limit)} tok/s</span>
          </div>
          <div className="lr-bar" style={{ gridTemplateColumns: "7.5rem 1fr 6rem" }}>
            <span className="tk">we measured</span>
            <span className="b">
              <i style={{ width: `${Math.min(100, (meas / limit) * 100)}%`, background: cc.c1 }} />
            </span>
            <span className="pc">
              {fmtTok(meas)} · {((meas / limit) * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      )}
    </Fig>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "0.3rem 0.8rem" }}>
      <span className="lr-small" style={{ width: "6.5rem" }}>
        {label}
      </span>
      {children}
    </div>
  );
}

export default function SpeedOfLight() {
  return (
    <>
      <p>
        If decode is waiting on memory, we can work out how fast it could ever go, before writing a line of code. To write
        one token, the computer has to read every weight once. (If you read chapter 0.2, you've seen this division; here it
        becomes the yardstick for every step of the engine.) So:
      </p>
      <code className="lr-math">{"fastest possible tokens per second  =  memory speed  ÷  bytes of weights"}</code>
      <p>
        For GPT-2 small: 124 million weights at 4 bytes each is 495 MB per token. Our Mac mini's memory is rated at 120 GB/s
        (<G k="GB" />
        /s). 120 GB/s ÷ 495 MB = <span className="k">243 tokens per second</span>. No engine, however clever, can write
        faster than that on this machine, for one user, with 32-bit weights.
      </p>
      <Calculator />
      <p>
        That one division explains a lot about the AI industry. It's why people pay for chips with faster memory, why models
        are shrunk to fewer bits, and why a 70-billion-weight model writes slower than a small one on the same machine: it's
        just more bytes to read per token.
      </p>
      <Idea>Speed limit = memory bandwidth ÷ bytes per token. It's a ceiling you can compute on the back of an envelope, and every optimisation in part three is measured against it.</Idea>
      <Deeper title="Why can't we ever quite reach it?">
        <p>
          The spec figure is a best case; real programs get 70–90% of it. Our CPU cores together can stream about 98 GB/s
          (82% of spec), the GPU about 105 (88%) when nothing else is running, and much less when something is. On top of that, each token has fixed costs that don't shrink with the bytes:
          starting threads, waiting for the GPU, the small operations between the big ones. The faster the model gets, the
          bigger those fixed costs loom.
        </p>
        <p>
          The limit also moves. Store each weight in 4 bits instead of 32, and there are about 7× fewer bytes to read (8×, less the scale each block of 32 carries). Serve 64
          users at once and each weight read does 64 users' work. Part three does both.
        </p>
      </Deeper>
    </>
  );
}
