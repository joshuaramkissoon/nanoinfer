import { useRef, useState } from "react";
import { useWidth } from "../../ui/Figure";
import { usePalette } from "../../ui/theme";
import { Climb } from "../Climb";
import { chartColors, Deeper, Fig, G, Idea, Seg, Swatch, useFrame, useOnScreen, PlayPause, usePlaying } from "../kit";

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
  const [playing, setPlaying] = usePlaying();
  useFrame(vis && playing, (dt) => {
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
          Each row is a running total (an “accumulator”), and each <Swatch c={cc.c1}>ember bar</Swatch> is an add in
          progress. An add takes {LAT} ticks before its total can take the next number. Memory delivers {MEM} numbers per
          tick. The meters compare adds finished per tick with what memory delivers.
        </>
      }
      tryit={<>Switch from 1 to 4 to 16 running totals. Notice which meter is full at each setting.</>}
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
        <PlayPause playing={playing} onToggle={() => setPlaying((p) => !p)} />
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
      <h2 style={{ marginTop: "2.2rem" }}>The bottleneck: one running total</h2>
      <p>
        With the KV cache, decode should be waiting on memory. It wasn't. It read 10 GB/s, a twelfth of what the memory can
        do. The engine was waiting on its own arithmetic, and the cause was the simplest line in the program, the dot
        product's running total:
      </p>
      <code className="lr-math">{"total = total + weight[i] × x[i]     // 768 times, one after another"}</code>
      <p>
        Every add needs the result of the previous add. A processor can start a new add every cycle, but each one takes a
        few cycles to finish. So with one running total, it spends most of its time waiting on itself.
      </p>
      <h2>The idea: sixteen totals</h2>
      <p>
        Keep 16 running totals instead of one, each summing every 16th number, and add them together at the end. Now 16
        adds can be in flight at once.
      </p>
      <Lanes />
      <p>
        There was a second gain. With 16 independent totals, the compiler grouped them into <G k="SIMD" /> instructions,
        which add 4 numbers at once. It couldn't do this before. Floating-point addition gives slightly different results in
        a different order, so a compiler may not reorder the adds by itself. Writing out 16 totals told it the order was
        allowed.
      </p>
      <h2>The result: 5.3× faster</h2>
      <p>
        With 16 variables instead of one, decode went from <span className="k">19 to 100 tokens per second</span>, 5.3× the
        KV-cache stage (fig 13.2). Run back to back in one session, the two stages measured 17.3 and 101.6, or 5.9×. Every
        matrix now read about 55 GB/s, close to the 57 GB/s one core can read. Decode was now memory-bound.
      </p>
      <Climb upto={2} n="13.2" />
      <Idea>A single running total makes every add wait for the last one. Many independent totals (and SIMD) keep the arithmetic busy until memory becomes the limit.</Idea>
      <Deeper title="The hand-written version that wasn't faster">
        <p>
          The next stage rewrote the loop by hand with the chip's own SIMD instructions (fused multiply-add, NEON). It was
          as fast to within noise: 100.3 vs 100.5 tokens per second. Decode was waiting on memory, so fewer instructions
          bought nothing. A version that read 4 weight rows at once to save loads was 25% <i>slower</i>, because memory
          streams one long, straight read best.
        </p>
        <p>
          The same trick did pay off for prefill, which is compute-bound. Computing a 4 × 4 tile of outputs at once reuses
          each weight from a register 4 times. That made prefill 2.5× faster, from 185 to about 460 tokens per second.
        </p>
      </Deeper>
    </>
  );
}
