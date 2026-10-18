import { useState } from "react";
import { useWidth } from "../../ui/Figure";
import { usePalette } from "../../ui/theme";
import { Climb } from "../Climb";
import { chartColors, Deeper, Fig, G, Idea } from "../kit";

// One decode token on the M4 GPU (docs/LOG.md, GPU stages). ms of GPU work vs ms the GPU sat idle.
const STEPS = [
  { label: "ask after every step", tokS: 14.6, work: 41, idle: 27.5, note: "135 operations, and after each one the CPU waits ~150 µs to hear it's done" },
  { label: "ask once per token", tokS: 41.3, work: 23.5, idle: 0.7, note: "same kernels, all 135 sent in one batch, one wait" },
  { label: "GPU-shaped kernels", tokS: 137, work: 7.0, idle: 0.3, note: "256 threads per job instead of 1; attention 100× faster" },
  { label: "4-bit, never wait", tokS: 641, work: 1.56, idle: 0, note: "fewer bytes, and the next token is queued before this one finishes" },
];

function FourWays() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [sel, setSel] = useState(0);
  const [ref, w] = useWidth<HTMLDivElement>();
  const W = Math.max(300, w), labelW = Math.min(160, W * 0.36), row = 40, bar = 16, top = 6;
  const H = top + STEPS.length * row + 30;
  const xmax = 70;
  const x = (ms: number) => labelW + 8 + (ms / xmax) * (W - labelW - 100);
  return (
    <Fig
      n="16.1"
      title="One token on the GPU, four ways"
      prov="measured"
      caption={
        <>
          The time to write one token on the M4's GPU, in milliseconds (shorter is better). Ember: the GPU doing work. Grey:
          the GPU sitting idle, waiting for the CPU. Each row is a later version of the engine's GPU code; the number at the end
          is its speed in tokens per second.
        </>
      }
      tryit={<>Hover each row for what changed. Notice the second row: identical GPU code, just less waiting, 2.8× faster.</>}
    >
      <div ref={ref}>
        {w > 0 && (
          <svg width={W} height={H}>
            {[0, 10, 20, 30, 40, 50, 60, 70].map((t) => (
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
                    {s.tokS} tok/s
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
  return (
    <>
      <h2 style={{ marginTop: "2.2rem" }}>The bottleneck</h2>
      <p>
        A CPU has a handful of strong cores. A <G k="GPU" /> has thousands of small, simple ones, built to do the same
        arithmetic on lots of numbers at once. That's exactly what prefill needs. On our Mac it reads the same shared memory
        as the CPU (about 80 GB/s in practice), so for decode it should at least keep up. It should win easily.
      </p>
      <p>
        The first GPU version wrote <span className="k">14.6 tokens per second</span>: slower than a single CPU core (19).
      </p>
      <h2>The idea: stop waiting, and think like a GPU</h2>
      <p>
        Two things were wrong. First, the CPU sent the GPU one operation at a time and waited to hear it was done before
        sending the next: 135 operations per token, each wait about 150 microseconds. Asking the GPU to do nothing costs less
        than a microsecond; asking whether it's finished costs 150. Second, the code was shaped for a CPU: some jobs ran on
        one GPU thread out of thousands, like asking one worker to carry a whole delivery while the rest watch.
      </p>
      <FourWays />
      <h2>The result</h2>
      <p>
        Decode <span className="k">641 tokens per second</span> (4-bit weights), 44× the first GPU version and the fastest
        single-user number in the project. Prefill, which can keep thousands of GPU threads busy, reached <b>8,900 tokens per
        second</b> using the GPU's built-in matrix-multiply hardware: 3.7× the best CPU result.
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
          The Mac's memory is shared by CPU and GPU. Running a CPU benchmark at the same time cut GPU decode from 140 to ~100
          tokens per second: the CPU's reads came straight out of the GPU's budget.
        </p>
      </Deeper>
    </>
  );
}
