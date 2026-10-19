import { useState } from "react";
import { useWidth } from "../../ui/Figure";
import { usePalette } from "../../ui/theme";
import { Climb } from "../Climb";
import { chartColors, Deeper, Fig, G, Idea, Swatch, useLearnData, type LearnData, T } from "../kit";

// One decode token on the M4 GPU, four ways: quiet-machine runs from bench/webgpu.json
// (the paired Metal rungs) and bench/results.jsonl, via public/learn/data.json.
type Step = LearnData["headline"]["gpuSteps"][number];

function FourWays({ STEPS }: { STEPS: Step[] }) {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [sel, setSel] = useState(0);
  const [ref, w] = useWidth<HTMLDivElement>();
  const W = Math.max(300, w), labelW = Math.min(160, W * 0.36), row = 40, bar = 16, top = 6;
  const base = top + STEPS.length * row;
  const H = base + 36;
  const xmax = Math.ceil((STEPS[0].work + STEPS[0].idle) / 10) * 10;
  const x = (ms: number) => labelW + 8 + (ms / xmax) * (W - labelW - 100);
  return (
    <Fig
      n="16.1"
      title="One token on the GPU, four ways"
      prov="measured"
      caption={
        <>
          The time to write one token on the M4's GPU, in milliseconds (shorter is better). Each row is a later GPU stage, with
          its tokens per second at the end. <Swatch c={cc.c1}>Ember</Swatch> bars: the GPU working.{" "}
          <Swatch c={pal.muted}>Grey</Swatch> bars: the GPU idle, waiting for the CPU (an estimate: waits per token × 130 µs).
        </>
      }
      tryit={<>Select each row to see what changed. Notice which colour shrinks between the first two rows, and which between the last two.</>}
    >
      <div ref={ref}>
        {w > 0 && (
          <svg width={W} height={H}>
            {Array.from({ length: xmax / 10 + 1 }, (_, i) => i * 10).map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={top} y2={base} stroke={pal.rule} />
                <text x={x(t)} y={base + 14} textAnchor="middle" fontSize={T.xs} fill={pal.muted}>
                  {t}
                </text>
              </g>
            ))}
            <text x={x(xmax)} y={base + 32} textAnchor="end" fontSize={T.xs} fill={pal.muted}>
              ms per token →
            </text>
            {STEPS.map((s, i) => {
              const y0 = top + i * row + (row - bar) / 2;
              const on = i === sel;
              const xw = x(s.work), xi = x(s.work + s.idle);
              return (
                <g key={s.label} onMouseEnter={() => setSel(i)} onClick={() => setSel(i)} style={{ cursor: "pointer" }}>
                  <rect x={0} y={top + i * row} width={W} height={row} fill="transparent" />
                  <text x={labelW} y={y0 + bar / 2 + 4} textAnchor="end" fontSize={T.xs} fill={on ? pal.fg : pal.sub} fontWeight={on ? 700 : 400}>
                    {s.label}
                  </text>
                  <rect x={x(0)} y={y0} width={Math.max(2, xw - x(0))} height={bar} rx={2} fill={cc.c1} opacity={on ? 1 : 0.45} />
                  {s.idle > 0.5 && <rect x={xw + 2} y={y0} width={Math.max(2, xi - xw - 2)} height={bar} rx={2} fill={pal.muted} opacity={0.6} />}
                  <text x={xi + 8} y={y0 + bar / 2 + 4} fontSize={T.xs} fill={on ? pal.fg : pal.sub} fontWeight={on ? 700 : 400}>
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

// MB read per decoded token by metal-lanes: Q4_0 blocks + Q8_0 scoring matrix (docs/LOG.md, Metal Q4 table)
const Q4_MB = 89.3;
// one-user speed limit: 120 GB/s spec bandwidth ÷ 89.3 MB per token
const LIMIT = (120 * 1000) / Q4_MB;

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
        A CPU has a handful of strong cores. A <G k="GPU" /> has thousands of simple arithmetic lanes that do the same
        arithmetic on many numbers at once, which is what prefill needs. On the M4 Mac mini the GPU reads the same shared
        memory as the CPU, about {h ? h.gpuReadGBs : "…"} GB/s in practice. So for decode it should at least match the CPU.
      </p>
      <p>
        The first GPU stage wrote <span className="k">{first ? first.toFixed(0) : "…"} tokens per second</span>, no faster
        than one CPU core with a KV cache (19).
      </p>
      <h2>The idea: stop waiting, and think like a GPU</h2>
      <p>
        Two things were wrong. First, the CPU sent the GPU one operation at a time and waited for each to finish before
        sending the next. That is 135 operations per token, and each wait took about 130 µs. Sending the GPU a small job
        costs less than a microsecond; hearing that it has finished costs 130. Second, the code was shaped for a CPU. Some
        jobs ran on one GPU thread while thousands sat idle.
      </p>
      {steps ? <FourWays STEPS={steps} /> : <div className="lr-empty">loading…</div>}
      <h2>The result: {best ? Math.round((100 * best.decode) / LIMIT) : "…"}% of the speed limit</h2>
      <p>
        Decode reached <span className="k">{best ? Math.round(best.decode).toLocaleString("en-GB") : "…"} tokens per second</span>{" "}
        for one user with 4-bit weights. That is {best && first ? Math.round(best.decode / first) : "…"}× the first GPU stage,
        and {best ? Math.round((100 * best.decode) / LIMIT) : "…"}% of the {Math.round(LIMIT).toLocaleString("en-GB")} tok/s
        speed limit for 89 MB per token. Each token takes {best ? (1000 / best.decode).toFixed(2) : "…"} ms, an average of{" "}
        {best ? Math.round((Q4_MB / 1000) * best.decode) : "…"} GB/s, or{" "}
        {best && h ? Math.round(((Q4_MB / 1000) * best.decode * 100) / h.gpuReadGBs) : "…"}% of the {h ? h.gpuReadGBs : "…"} GB/s
        the GPU can read at all.
      </p>
      <p>
        The biggest matrix, the LM head, reads at the full 106 GB/s. The others manage 66 to 86 GB/s, and attention adds
        time without adding many bytes. This number is pipelined: the CPU queues each token's work before the last one
        finishes, so it never waits (the second fold below shows what streaming to a user costs). Prefill keeps thousands of
        GPU threads busy, and reads a 64-token prompt at <b>{best ? best.prefill.toLocaleString("en-GB") : "…"} tokens per
        second</b> on the GPU's matrix-multiply hardware.
      </p>
      <Climb upto={5} n="16.2" />
      <Idea>A GPU wins only once the code is shaped for it and the CPU stops making it wait. At this model size, the last big gains came from removing waits.</Idea>
      <Deeper title="Fewer kernels didn't help; not waiting did">
        <p>
          A popular trick is fusing small GPU operations so there are fewer of them. Halving the count (137 → 64) changed
          speed by about 5%, inside the noise. Letting the GPU read each next token from its own memory, so the CPU never
          waits between tokens, gave +30%. The cost was the CPU and GPU taking turns, so removing the waits mattered more
          than cutting the kernel count.
        </p>
        <p>
          The last step was a better-shaped multiply: 8 GPU threads per row of 4-bit weights instead of 32, so each thread
          has several memory reads in flight. That took the narrow matrices from 38 to 58 GB/s and decode up 5 to 8%.
        </p>
      </Deeper>
      <Deeper title="Pipelined decode isn't what a chat server sees">
        <p>
          A chat server has to read each token back to the CPU before it encodes the next pass. It needs the token to stream
          it, detokenise it and check for a stop. With that per-token synchronisation, which is also how llama.cpp's
          benchmark runs, each token costs about 0.37 ms more. Keeping one token in flight (sending pass t+1 before reading
          token t back) hides that cost. Decode tok/s for one user on the M4's GPU:
        </p>
        <ul>
          <li>4-bit: 760 pipelined, 593 with per-token sync, 760 with one token in flight. MLX 694, llama.cpp 622.</li>
          <li>8-bit: 589 pipelined, 486 with per-token sync, 577 with one token in flight. MLX 534, llama.cpp 502.</li>
          <li>16-bit: 347 pipelined, 305 with per-token sync, 339 with one token in flight. MLX 342, llama.cpp 333.</li>
          <li>32-bit: 186 pipelined, 174 with per-token sync, 185 with one token in flight. MLX 178, llama.cpp 192.</li>
        </ul>
        <p>
          So with per-token sync, nanoinfer is slightly behind llama.cpp and MLX at every format. With one token in flight it
          is ahead of both at 4-bit and 8-bit, level at 16-bit, and a little behind llama.cpp at 32-bit. A server can run that loop:
          it notices a stop one token late and discards the extra pass. Prefill is a clear loss. llama.cpp reads a prompt
          about 1.6 to 1.9× faster on this GPU, because it stages its matrix tiles in 16-bit and ours are 32-bit.
        </p>
        <p>
          Our rows come from one session on 19 Oct that ran about 3% slower than the 18 Oct head-to-head, where MLX and
          llama.cpp were measured (nanoinfer's pipelined 4-bit made 787 there). Every engine read the same bytes per token
          and had its quality checked against its own 32-bit model.
        </p>
      </Deeper>
      <Deeper title="Measuring on a busy machine">
        <p>
          The M4's memory is shared by the CPU and GPU, so anything else running takes part of the GPU's bandwidth. The first
          runs of this chapter's best stage competed with other benchmarks and a search indexer. They gave 592 tokens per
          second and about 80 GB/s. With nothing else running, the same stage made {best ? Math.round(best.decode) : "…"} tokens
          per second and {h ? h.gpuReadGBs : "…"} GB/s. The numbers on this site are the quiet ones, and the log keeps both.
        </p>
      </Deeper>
    </>
  );
}
