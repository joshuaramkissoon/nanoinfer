import { useState } from "react";
import { useWidth } from "../../ui/Figure";
import { usePalette } from "../../ui/theme";
import { Climb } from "../Climb";
import { chartColors, Deeper, Fig, G, Idea } from "../kit";

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
          How fast the M4's CPU can read memory with 1 to 10 cores working at once, measured by the engine. The dashed line is
          Apple's 120 GB/s spec. Each extra core adds less: by 6 cores the memory system is saturated at ~98 GB/s, and the last
          4 cores (the M4's slower “efficiency” cores) add nothing.
        </>
      }
      tryit={<>Click the bars. Read the speed limit each thread count allows, and notice where it stops growing.</>}
    >
      <div ref={ref}>
        {w > 0 && (
          <svg width={W} height={H}>
            {[0, 40, 80, 120].map((v) => (
              <g key={v}>
                <line x1={pl} x2={W - 10} y1={y(v)} y2={y(v)} stroke={pal.rule} />
                <text x={pl - 6} y={y(v) + 4} textAnchor="end" fontSize={10} fill={pal.muted}>
                  {v}
                </text>
              </g>
            ))}
            <line x1={pl} x2={W - 10} y1={y(120)} y2={y(120)} stroke={pal.fg} strokeDasharray="4 3" opacity={0.6} />
            <text x={W - 10} y={y(120) - 6} textAnchor="end" fontSize={10} fill={pal.sub}>
              spec 120 GB/s
            </text>
            <text x={pl} y={10} fontSize={10} fill={pal.sub}>
              GB/s read
            </text>
            {MEMBW.map(([n, v], i) => {
              const cx = pl + bw * i + bw / 2;
              const on = i === sel;
              return (
                <g key={n} onClick={() => setSel(i)} onMouseEnter={() => setSel(i)} style={{ cursor: "pointer" }}>
                  <rect x={cx - bw / 2} y={top} width={bw} height={H - pb - top} fill="transparent" />
                  <path d={`M${cx - barW / 2},${y(0)} V${y(v) + 4} a4,4 0 0 1 4,-4 H${cx + barW / 2 - 4} a4,4 0 0 1 4,4 V${y(0)} Z`} fill={on ? cc.c2 : pal.muted} opacity={on ? 1 : 0.5} />
                  <text x={cx} y={y(v) - 6} textAnchor="middle" fontSize={10.5} fill={on ? pal.fg : pal.muted} fontWeight={on ? 700 : 400}>
                    {v}
                  </text>
                  <text x={cx} y={H - pb + 14} textAnchor="middle" fontSize={10.5} fill={on ? pal.fg : pal.muted}>
                    {n}
                  </text>
                </g>
              );
            })}
            <text x={W - 10} y={H - 2} textAnchor="end" fontSize={10} fill={pal.muted}>
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
      <h2 style={{ marginTop: "2.2rem" }}>The bottleneck</h2>
      <p>
        At 100 tokens per second, one core reads memory as fast as one core can: about 57 GB/s. The M4 has ten cores, and the
        memory is rated at 120 GB/s. The obvious move: split every matrix into slices and give each core a slice, each running
        its own <G k="thread" />.
      </p>
      <h2>The idea, and the catch</h2>
      <p>
        Ten cooks in a kitchen can chop ten times as fast. But if they all fetch ingredients through one narrow pantry door,
        the door sets the pace. Memory is that door:
      </p>
      <Pantry />
      <p>
        So threads can at best take decode from one core's 57 GB/s to the whole chip's 98: <b>1.7×</b>, not 10×. And there was a
        second catch. A token has 49 matrix multiplies, each only about a fifth of a millisecond, and waking a sleeping thread
        to help costs tens of microseconds. Hand ten threads a tiny job and they spend much of it getting started. The fix was
        a thread pool whose workers never fully go to sleep, so they see new work in about a tenth of a microsecond, plus
        threading the small operations in between the matrices too.
      </p>
      <h2>The result</h2>
      <p>
        Decode <span className="k">100 → 146 tokens per second</span> (6 threads), 60% of the speed limit. The big win was
        prefill, which is compute-bound, so more cores really is more speed: <b>185 → 1,456 tokens per second</b>, and later
        2,415 with 8-bit integer arithmetic on all ten cores.
      </p>
      <Climb upto={3} n="14.2" />
      <Idea>Cores share one memory system. More threads help a memory-bound job only until memory is saturated (here 1.7×), and only if handing out the work costs less than doing it.</Idea>
      <Deeper title="The thread pool that made things slower">
        <p>
          The first never-sleeping pool spun its threads flat out, and on a busy laptop went from 93 to <b>36</b> tokens per
          second with ten threads: spinning threads stole the CPU from the one thread still finishing the last slice, and
          everyone waited for it. Spin, then yield, then sleep fixed it. A spin pool assumes you own the machine, and a laptop
          never does.
        </p>
        <p>
          The best thread count for decode turned out to be 4 to 6, not 10: every step waits for the slowest thread, and the
          efficiency cores add stragglers without adding bandwidth.
        </p>
      </Deeper>
    </>
  );
}
