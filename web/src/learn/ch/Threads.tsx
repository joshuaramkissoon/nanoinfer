import { useState } from "react";
import { useWidth } from "../../ui/Figure";
import { usePalette } from "../../ui/theme";
import { Climb } from "../Climb";
import { chartColors, Deeper, Fig, G, Idea, T } from "../kit";

// `nanoinfer membw` on the M4 (docs/LOG.md, "The ceiling"): read GB/s by thread count.
const MEMBW: [number, number][] = [
  [1, 58],
  [2, 70],
  [4, 92],
  [6, 96],
  [8, 98],
  [10, 98],
];
const WEIGHTS_MB = 494.6;

function Pantry() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [sel, setSel] = useState(3);
  const [ref, w] = useWidth<HTMLDivElement>();
  const W = Math.max(280, w), H = 210, pl = 34, pb = 30, top = 22;
  const bw = (W - pl - 10) / MEMBW.length;
  const barW = Math.min(24, bw * 0.5);
  const y = (v: number) => H - pb - (v / 120) * (H - pb - top);
  const [t, gb] = MEMBW[sel];
  return (
    <Fig
      n="14.1"
      title="More cores, same memory"
      prov="measured"
      caption={
        <>
          The M4 Mac mini's CPU memory bandwidth with 1 to 10 cores reading at once, measured by the engine. Across: cores
          working. Up: GB/s. The dashed line is Apple's 120 GB/s spec. The last 4 cores are the M4's slower efficiency
          cores.
        </>
      }
      tryit={<>Select each bar. Read the speed limit each core count allows, and notice where it stops growing.</>}
    >
      <div ref={ref}>
        {w > 0 && (
          <svg width={W} height={H}>
            {[0, 40, 80, 120].map((v) => (
              <g key={v}>
                <line x1={pl} x2={W - 10} y1={y(v)} y2={y(v)} stroke={pal.rule} />
                <text x={pl - 6} y={y(v) + 4} textAnchor="end" fontSize={T.xs} fill={pal.muted}>
                  {v}
                </text>
              </g>
            ))}
            <line x1={pl} x2={W - 10} y1={y(120)} y2={y(120)} stroke={pal.fg} strokeDasharray="4 3" opacity={0.6} />
            <text x={W - 10} y={y(120) - 6} textAnchor="end" fontSize={T.xs} fill={pal.sub}>
              spec 120 GB/s
            </text>
            <text x={pl} y={10} fontSize={T.xs} fill={pal.sub}>
              GB/s read
            </text>
            {MEMBW.map(([n, v], i) => {
              const cx = pl + bw * i + bw / 2;
              const on = i === sel;
              return (
                <g key={n} onClick={() => setSel(i)} onMouseEnter={() => setSel(i)} style={{ cursor: "pointer" }}>
                  <rect x={cx - bw / 2} y={top} width={bw} height={H - pb - top} fill="transparent" />
                  <path d={`M${cx - barW / 2},${y(0)} V${y(v) + 4} a4,4 0 0 1 4,-4 H${cx + barW / 2 - 4} a4,4 0 0 1 4,4 V${y(0)} Z`} fill={on ? cc.c2 : pal.muted} opacity={on ? 1 : 0.5} />
                  <text x={cx} y={y(v) - 6} textAnchor="middle" fontSize={T.xs} fill={on ? pal.fg : pal.muted} fontWeight={on ? 700 : 400}>
                    {v}
                  </text>
                  <text x={cx} y={H - pb + 14} textAnchor="middle" fontSize={T.xs} fill={on ? pal.fg : pal.muted}>
                    {n}
                  </text>
                </g>
              );
            })}
            <text x={W - 10} y={H + 3} textAnchor="end" fontSize={T.xs} fill={pal.muted}>
              cores working →
            </text>
          </svg>
        )}
      </div>
      <div className="lr-note">
        {t} {t === 1 ? "core" : "cores"}: {gb} GB/s ÷ {WEIGHTS_MB} MB per token = a speed limit of <b>{Math.round((gb * 1000) / WEIGHTS_MB)} tokens per second</b>
        {sel > 0 && <> ({((gb / MEMBW[0][1]) * 1).toFixed(1)}× one core, for {t}× the cores)</>}
      </div>
    </Fig>
  );
}

export default function Threads() {
  return (
    <>
      <h2 style={{ marginTop: "2.2rem" }}>The bottleneck: one core's bandwidth</h2>
      <p>
        At 100 tokens per second, one core reads memory as fast as one core can, about 57 GB/s. The M4 has 10 cores, and
        the memory is rated at 120 GB/s. The obvious move is to split every matrix into slices and give each core a slice,
        each running its own <G k="thread" />.
      </p>
      <h2>The idea, and the catch: one memory bus</h2>
      <p>
        Ten cores can do 10 times the arithmetic, but they share one memory bus. In chapter 9's terms, that is one truck and
        10 factories:
      </p>
      <Pantry />
      <p>
        So threads can at best take decode from one core's 57 GB/s to the whole chip's 98 GB/s: <b>1.7×</b>, not 10×. There
        was a second catch. A token has 49 matrix multiplies, each only about a fifth of a millisecond. Waking a sleeping
        thread to help costs tens of microseconds, so 10 threads given a tiny job spend much of it getting started. The fix
        was a thread pool whose workers never fully sleep, so they see new work in about a tenth of a microsecond. The
        small operations between the matrices were threaded too.
      </p>
      <h2>The result: 146 tokens per second</h2>
      <p>
        Decode went from <span className="k">100 to 146 tokens per second</span> (6 threads), 60% of the speed limit. The big
        win was prefill, which is compute-bound, so more cores give more speed. Prefill went from <b>185 to 1,456 tokens per
        second</b>, and later to 2,415 with 8-bit integer arithmetic on all 10 cores.
      </p>
      <Climb upto={3} n="14.2" />
      <Idea>Cores share one memory system. More threads help a memory-bound job only until memory is saturated (here 1.7×), and only if handing out the work costs less than doing it.</Idea>
      <Deeper title="The thread pool that made things slower">
        <p>
          The first never-sleeping pool spun its threads at full speed. On a busy laptop, decode fell from 93 to <b>36</b>{" "}
          tokens per second with 10 threads. The spinning threads took CPU time from the one thread that was still finishing,
          and every thread waited for it. A pool that spins, then yields, then sleeps fixed it.
        </p>
        <p>
          The best thread count for decode turned out to be 4 to 6. Every step waits for the slowest thread, and the
          efficiency cores add stragglers without adding bandwidth.
        </p>
      </Deeper>
    </>
  );
}
