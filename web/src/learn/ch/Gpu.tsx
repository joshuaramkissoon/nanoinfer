import { useState } from "react";
import { useWidth } from "../../ui/Figure";
import { usePalette } from "../../ui/theme";
import { Climb } from "../Climb";
import { chartColors, Deeper, Fig, G, Idea, useLearnData, type LearnData } from "../kit";

// One decode token on the M4 GPU, four ways: quiet-machine runs from bench/webgpu.json
// (the paired Metal rungs) and bench/results.jsonl, via public/learn/data.json.
type Step = LearnData["headline"]["gpuSteps"][number];

function FourWays({ STEPS }: { STEPS: Step[] }) {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [sel, setSel] = useState(0);
  const [ref, w] = useWidth<HTMLDivElement>();
  const W = Math.max(300, w), labelW = Math.min(160, W * 0.36), row = 40, bar = 16, top = 6;
  const H = top + STEPS.length * row + 30;
  const xmax = Math.ceil((STEPS[0].work + STEPS[0].idle) / 10) * 10;
  const x = (ms: number) => labelW + 8 + (ms / xmax) * (W - labelW - 100);
  return (
    <Fig
      n="16.1"
      title="One token on the GPU, four ways"
      prov="measured"
      caption={
        <>
          The time to write one token on the M4's GPU, in milliseconds (shorter is better). Ember: the GPU doing work. Grey:
          the GPU sitting idle, waiting for the CPU (estimated: the number of waits per token × the 130 µs one wait was
          measured to take). Each row is a later version of the engine's GPU code; the number at the end is its speed in
          tokens per second. Measured on a quiet machine, prompt 64 then 64 tokens.
        </>
      }
      tryit={<>Hover each row for what changed. Notice the second row: identical GPU code, just less waiting, {(STEPS[1].tokS / STEPS[0].tokS).toFixed(1)}× faster.</>}
    >
      <div ref={ref}>
        {w > 0 && (
          <svg width={W} height={H}>
            {Array.from({ length: xmax / 10 + 1 }, (_, i) => i * 10).map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={top} y2={H - 26} stroke={pal.rule} />
                <text x={x(t)} y={H - 12} textAnchor="middle" fontSize={10} fill={pal.muted}>
                  {t}
                </text>
              </g>
            ))}
            <text x={W - 4} y={H - 1} textAnchor="end" fontSize={10} fill={pal.muted}>
              ms per token →
            </text>
            {STEPS.map((s, i) => {
              const y0 = top + i * row + (row - bar) / 2;
              const on = i === sel;
              const xw = x(s.work), xi = x(s.work + s.idle);
              return (
                <g key={s.label} onMouseEnter={() => setSel(i)} onClick={() => setSel(i)} style={{ cursor: "pointer" }} opacity={on ? 1 : 0.6}>
                  <rect x={0} y={top + i * row} width={W} height={row} fill="transparent" />
                  <text x={labelW} y={y0 + bar / 2 + 4} textAnchor="end" fontSize={11.5} fill={on ? pal.fg : pal.sub} fontWeight={on ? 700 : 400}>
                    {s.label}
                  </text>
                  <rect x={x(0)} y={y0} width={Math.max(2, xw - x(0))} height={bar} rx={2} fill={cc.c1} />
                  {s.idle > 0.5 && <rect x={xw + 2} y={y0} width={Math.max(2, xi - xw - 2)} height={bar} rx={2} fill={pal.muted} opacity={0.6} />}
                  <text x={xi + 8} y={y0 + bar / 2 + 4} fontSize={11} fill={pal.fg} fontWeight={on ? 700 : 400}>
                    {s.tokS < 100 ? s.tokS.toFixed(1) : Math.round(s.tokS)} tok/s
                  </text>
                </g>
              );
            })}
          </svg>
        )}
      </div>
      <div className="lr-note">
        <b>{STEPS[sel].label}</b>: {STEPS[sel].note}.
      </div>
    </Fig>
  );
}

export default function Gpu() {
  const d = useLearnData();
  const h = d?.headline;
  const steps = h?.gpuSteps;
  const first = steps?.[0].tokS;
  const best = h?.gpu;
  return (
    <>
      <h2 style={{ marginTop: "2.2rem" }}>The bottleneck</h2>
      <p>
        A CPU has a handful of strong cores. A <G k="GPU" /> has thousands of small, simple ones, built to do the same
        arithmetic on lots of numbers at once. That's exactly what prefill needs. On our Mac it reads the same shared memory
        as the CPU (about {h ? h.gpuReadGBs : "…"} GB/s in practice), so for decode it should at least keep up. It should win easily.
      </p>
      <p>
        The first GPU version wrote <span className="k">{first ? first.toFixed(0) : "…"} tokens per second</span>: no faster
        than a single CPU core (19).
      </p>
      <h2>The idea: stop waiting, and think like a GPU</h2>
      <p>
        Two things were wrong. First, the CPU sent the GPU one operation at a time and waited to hear it was done before
        sending the next: 135 operations per token, each wait about 130 microseconds. Asking the GPU to do one small job
        costs less than a microsecond; asking whether it's finished costs 130. Second, the code was shaped for a CPU: some
        jobs ran on one GPU thread out of thousands, like asking one worker to carry a whole delivery while the rest watch.
      </p>
      {steps ? <FourWays STEPS={steps} /> : <div className="lr-empty">loading…</div>}
      <h2>The result</h2>
      <p>
        Decode <span className="k">{best ? Math.round(best.decode).toLocaleString("en-GB") : "…"} tokens per second</span> (4-bit
        weights), {best && first ? Math.round(best.decode / first) : "…"}× the first GPU version and the fastest single-user
        number in the project. That's each token in {best ? (1000 / best.decode).toFixed(2) : "…"} ms, reading 89 MB of
        weights at close to the {h ? h.gpuReadGBs : "…"} GB/s the GPU can read at all. Prefill, which can keep thousands of GPU
        threads busy, reads a 64-token prompt at <b>{best ? best.prefill.toLocaleString("en-GB") : "…"} tokens per second</b> using the
        GPU's built-in matrix-multiply hardware.
      </p>
      <Climb upto={5} n="16.2" />
      <Idea>A GPU wins only once the code is shaped for it, and only if the CPU stops making it wait. At this model size the last big gains came from removing waiting, not from faster maths.</Idea>
      <Deeper title="Fewer kernels didn't help; not waiting did">
        <p>
          A popular trick is fusing small GPU operations together so there are fewer of them. Halving the count (137 → 64)
          changed speed by about 5%, inside the noise. Letting the GPU read each next token from its own memory, so the CPU
          never has to wait between tokens, gave +30%. The expensive thing was never the number of kernels; it was the CPU and
          GPU taking turns.
        </p>
        <p>
          The last step was a better-shaped multiply: 8 GPU threads per row of 4-bit weights instead of 32, so each thread
          has several loads in flight at once. That took the narrow matrices from 38 to 58 GB/s and decode up 5–8%.
        </p>
      </Deeper>
      <Deeper title="Measuring on a busy machine">
        <p>
          The Mac's memory is shared by CPU and GPU, so anything else running eats into the GPU's budget. Our first runs of
          this chapter's best version, with other benchmarks and a search indexer competing for memory, gave 592 tokens per
          second and a GPU read speed of about 80 GB/s. Re-measured with nothing else running: {best ? Math.round(best.decode) : "…"} tokens per
          second and {h ? h.gpuReadGBs : "…"} GB/s. The numbers on this site are the quiet ones; the log keeps both.
        </p>
      </Deeper>
    </>
  );
}
