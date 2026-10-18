import { useRef, useState } from "react";
import { useWidth } from "../../ui/Figure";
import { usePalette } from "../../ui/theme";
import { Climb } from "../Climb";
import { chartColors, Deeper, Fig, G, Idea, Seg, useFrame, useOnScreen } from "../kit";

const LAT = 4; // ticks an add takes before its result can be used
const MEM = 3; // numbers memory delivers per tick (so 16 lanes become memory-bound)

function Lanes() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [k, setK] = useState(1);
  const sim = useRef({ lanes: [] as number[], buf: 0, done: 0, ticks: 0, acc: 0, hist: [] as number[] });
  const [, setN] = useState(0);
  const reset = (kk: number) => {
    sim.current = { lanes: new Array(kk).fill(0), buf: 0, done: 0, ticks: 0, acc: 0, hist: [] };
  };
  if (sim.current.lanes.length !== k) reset(k);
  const [onScreen, vis] = useOnScreen<HTMLElement>();
  useFrame(vis, (dt) => {
    const s = sim.current;
    s.acc += dt * 5; // ticks per second
    while (s.acc >= 1) {
      s.acc -= 1;
      s.ticks++;
      s.buf = Math.min(24, s.buf + MEM);
      let fin = 0;
      for (let i = 0; i < s.lanes.length; i++) {
        if (s.lanes[i] > 0) {
          s.lanes[i]--;
          if (s.lanes[i] === 0) fin++;
        }
        if (s.lanes[i] === 0 && s.buf >= 1) {
          s.buf--;
          s.lanes[i] = LAT;
        }
      }
      s.done += fin;
      s.hist.push(fin);
      if (s.hist.length > 20) s.hist.shift();
    }
    setN((x) => x + 1);
  });
  const s = sim.current;
  const rate = s.hist.length ? s.hist.reduce((a, b) => a + b, 0) / s.hist.length : 0;
  const ideal = Math.min(k / LAT, MEM);
  const memBound = k / LAT >= MEM;
  const [ref, w] = useWidth<HTMLDivElement>();
  const W = Math.max(280, w), rowH = k > 4 ? 9 : 18, H = k * (rowH + 3) + 4;
  return (
    <Fig
      figRef={onScreen}
      n="13.1"
      title="One running total, or sixteen"
      prov="toy"
      caption={
        <>
          Each row is a running total (an “accumulator”). Adding a number to a total takes {LAT} ticks before the total can
          take the next one, so one total can only finish an add every {LAT} ticks, however fast the numbers arrive. With
          several totals, several adds are in flight at once (each bar is an add in progress). Memory here delivers {MEM}{" "}
          numbers per tick. The meter compares adds finished per tick with what memory delivers.
        </>
      }
      tryit={<>Go from 1 to 4 to 16 running totals. At 16 the totals are no longer the problem: memory is.</>}
    >
      <div className="lr-controls">
        <Seg
          options={[1, 4, 16].map((v) => ({ v, label: `${v} running total${v > 1 ? "s" : ""}` }))}
          value={k}
          onChange={(v) => {
            reset(v);
            setK(v);
          }}
          label="accumulators"
        />
      </div>
      <div ref={ref}>
        {w > 0 && (
          <svg width={W} height={H}>
            {s.lanes.map((p, i) => (
              <g key={i}>
                <rect x={0} y={2 + i * (rowH + 3)} width={W} height={rowH} rx={3} fill={pal.rule} opacity={0.6} />
                {p > 0 && <rect x={0} y={2 + i * (rowH + 3)} width={Math.min(1, (LAT - p + s.acc) / LAT) * W} height={rowH} rx={3} fill={cc.c1} />}
              </g>
            ))}
          </svg>
        )}
      </div>
      <div className="lr-bars" style={{ marginTop: "0.8rem" }}>
        <div className="lr-bar" style={{ gridTemplateColumns: "13.5rem 1fr 3rem" }}>
          <span className="tk">adds finished per tick</span>
          <span className="b">
            <i style={{ width: `${(Math.max(rate, 0) / MEM) * 100}%`, background: cc.c1 }} />
          </span>
          <span className="pc">{rate.toFixed(2)}</span>
        </div>
        <div className="lr-bar" style={{ gridTemplateColumns: "13.5rem 1fr 3rem" }}>
          <span className="tk">memory delivers per tick</span>
          <span className="b">
            <i style={{ width: "100%", background: cc.c2 }} />
          </span>
          <span className="pc">{MEM}</span>
        </div>
      </div>
      <div className="lr-note">
        → {memBound ? (
          <>
            <b>waiting on memory</b>: the totals keep up with everything memory delivers.
          </>
        ) : (
          <>
            <b>waiting on the adds</b>: memory could deliver {MEM} per tick, the totals only take {ideal.toFixed(2)}.
          </>
        )}
      </div>
    </Fig>
  );
}

export default function Arithmetic() {
  return (
    <>
      <h2 style={{ marginTop: "2.2rem" }}>The bottleneck</h2>
      <p>
        With the cache, decode should be waiting on memory. It wasn't: it read 10 GB/s, a twelfth of what the memory can do.
        The engine was waiting on its own arithmetic, and the culprit was the humblest line in the program, the dot product's
        running total:
      </p>
      <code className="lr-math">{"total = total + weight[i] × x[i]     // 768 times, one after another"}</code>
      <p>
        Every add needs the result of the previous add. A processor can start a new add every cycle, but each one takes a
        few cycles to finish, so with one running total it spends most of its time waiting on itself.
      </p>
      <h2>The idea</h2>
      <p>
        Keep sixteen running totals instead of one, each summing every sixteenth number, and add them together at the end.
        Now sixteen adds can be in flight at once. It's like opening more checkout lanes.
      </p>
      <Lanes />
      <p>
        There was a bonus. Once the program spelled out sixteen independent totals, the compiler grouped them four at a time
        into <G k="SIMD" /> instructions, which do the same operation on four numbers at once: a paint roller instead of a
        brush. The compiler hadn't done that before, because adding decimals in a different order can change the last digit
        of the answer, so it isn't allowed to reorder them on its own. Writing sixteen totals gave it permission.
      </p>
      <h2>The result</h2>
      <p>
        Sixteen variables instead of one: decode <span className="k">19 → 100 tokens per second</span>, 5.8× faster. Every
        matrix now read about 55 GB/s, essentially all that one core can pull from memory (57). Decode had become truly
        memory-bound.
      </p>
      <Climb upto={2} n="13.2" />
      <Idea>A single running total makes every add wait for the last one. Many independent totals (and SIMD) keep the arithmetic busy until memory, not maths, is the limit.</Idea>
      <Deeper title="The hand-written version that wasn't faster">
        <p>
          The next stage rewrote the loop by hand with the chip's own SIMD instructions (fused multiply-add, NEON). It was{" "}
          <b>exactly</b> as fast: 100.3 vs 100.5 tokens per second. Decode was waiting on memory, so fewer instructions bought
          nothing. A version that read four weight rows at once to save loads was even 25% <i>slower</i>: memory likes one
          long, straight read.
        </p>
        <p>
          The same trick did pay off for prefill, which is compute-bound: computing a 4 × 4 block of outputs at once, reusing
          each weight from a register four times, made prefill 2.5× faster (185 → ~460 tokens per second).
        </p>
      </Deeper>
    </>
  );
}
