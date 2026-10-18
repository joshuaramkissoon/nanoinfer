import { useRef, useState } from "react";
import { useEngine } from "../../engine/client";
import { useWidth } from "../../ui/Figure";
import { usePalette } from "../../ui/theme";
import { log, logTicks, short } from "../../lib/fmt";
import { Deeper, Fig, G, Idea, Seg, Slider, useFrame } from "../kit";

/** fig 9.1: a truck feeding a factory */
function Factory() {
  const pal = usePalette();
  const [I, setI] = useState(0.5); // work per box (FLOP per byte)
  const R = 1.6; // the tab's ridge: 97 GFLOP/s / 59 GB/s
  const [ref, w] = useWidth<HTMLDivElement>();
  const sim = useRef({ boxes: [] as number[], queue: 0, proc: 0, spawn: 0, busy: 0, idle: 0, out: 0 });
  const [, tick] = useState(0);
  const truckRate = 2.2; // boxes per second, visual
  useFrame(true, (dt) => {
    const s = sim.current;
    s.spawn += dt * truckRate;
    while (s.spawn >= 1 && s.queue < 9) {
      s.spawn -= 1;
      s.boxes.push(0);
    }
    if (s.queue >= 9) s.spawn = Math.min(s.spawn, 1);
    s.boxes = s.boxes.map((x) => x + dt * 0.55).filter((x) => {
      if (x >= 1) {
        s.queue++;
        return false;
      }
      return true;
    });
    const per = I / R / truckRate; // seconds of factory time per box
    if (s.queue > 0 || s.proc > 0) {
      if (s.proc <= 0 && s.queue > 0) {
        s.queue--;
        s.proc = per;
      }
      s.proc -= dt;
      s.busy += dt;
      if (s.proc <= 0) s.out++;
    } else s.idle += dt;
    s.busy *= 0.995;
    s.idle *= 0.995;
    tick((n) => n + 1);
  });
  const s = sim.current;
  const factoryBusy = Math.min(1, I / R), truckBusy = Math.min(1, R / I);
  const W = Math.max(300, w), H = 120;
  const road0 = 70, road1 = W - 120;
  const bound = I < R ? "memory" : "compute";
  return (
    <Fig
      n="9.1"
      title="The truck and the factory"
      prov="toy"
      caption={
        <>
          Left, the warehouse: memory, where the model's weights live. The road: memory <G k="bandwidth" />, delivering boxes
          of bytes at a steady rate. Right, the factory: the processor, doing arithmetic on what arrives. Each box needs some
          amount of work. The two bars say how busy each side is. The crossover is set to this browser tab's measured
          numbers: ~97 billion operations per second against ~59 GB/s, so <b>1.6 operations per byte</b>.
        </>
      }
      tryit={
        <>
          Press <b>decode (0.5)</b>: the factory mostly waits. Press <b>prefill</b>: now boxes pile up at the factory door. Slide
          slowly from one to the other and find the crossover.
        </>
      }
    >
      <div className="lr-controls">
        <Slider
          label="work per byte"
          value={Math.log2(I)}
          min={-3}
          max={6}
          step={0.05}
          onChange={(v) => {
            setI(2 ** v);
            sim.current.queue = Math.min(sim.current.queue, 3);
          }}
          fmt={() => `${I < 1 ? I.toFixed(2) : I.toFixed(1)} FLOP/B`}
          width="11rem"
        />
        <button className="lr-btn" onClick={() => setI(0.5)}>
          decode (0.5)
        </button>
        <button className="lr-btn" onClick={() => setI(32)}>
          prefill, 64 tokens (32)
        </button>
      </div>
      <div ref={ref}>
        {w > 0 && (
          <svg width={W} height={H}>
            {/* warehouse */}
            <rect x={4} y={30} width={56} height={56} rx={6} fill="none" stroke={pal.cool} />
            <text x={32} y={62} textAnchor="middle" fontSize={10} fill={pal.cool}>
              memory
            </text>
            <text x={32} y={102} textAnchor="middle" fontSize={9} fill={pal.muted}>
              weights
            </text>
            {/* road */}
            <line x1={road0} x2={road1} y1={58} y2={58} stroke={pal.rule} strokeWidth={14} strokeLinecap="round" />
            <line x1={road0} x2={road1} y1={58} y2={58} stroke={pal.muted} strokeDasharray="6 8" opacity={0.5} />
            {s.boxes.map((x, i) => (
              <rect key={i} x={road0 + x * (road1 - road0 - 12)} y={52} width={12} height={12} rx={2} fill={pal.heat[3]} />
            ))}
            {/* queue */}
            {Array.from({ length: Math.min(9, s.queue) }, (_, i) => (
              <rect key={i} x={road1 - 4 - (i % 3) * 14} y={78 + Math.floor(i / 3) * -14 + 14} width={12} height={12} rx={2} fill={pal.heat[3]} opacity={0.9} />
            ))}
            {/* factory */}
            <g transform={`translate(${W - 104},20)`}>
              <rect x={0} y={10} width={96} height={66} rx={6} fill={pal.accent} fillOpacity={s.proc > 0 ? 0.18 : 0.04} stroke={pal.accent} />
              <path d="M10,10 v-12 h14 v12 M34,10 v-8 h14 v8" fill="none" stroke={pal.accent} />
              <text x={48} y={40} textAnchor="middle" fontSize={10} fill={pal.accent}>
                processor
              </text>
              <text x={48} y={56} textAnchor="middle" fontSize={9.5} fill={pal.fg} fontWeight={700}>
                {s.proc > 0 ? "working" : "waiting…"}
              </text>
            </g>
          </svg>
        )}
      </div>
      <div className="lr-bars" style={{ marginTop: "0.4rem" }}>
        <div className="lr-bar" style={{ gridTemplateColumns: "11.5rem 1fr 3.4rem" }}>
          <span className="tk">truck (memory) busy</span>
          <span className="b">
            <i style={{ width: `${truckBusy * 100}%`, background: pal.cool }} />
          </span>
          <span className="pc">{(truckBusy * 100).toFixed(0)}%</span>
        </div>
        <div className="lr-bar" style={{ gridTemplateColumns: "11.5rem 1fr 3.4rem" }}>
          <span className="tk">factory (maths) busy</span>
          <span className="b">
            <i style={{ width: `${factoryBusy * 100}%`, background: pal.accent }} />
          </span>
          <span className="pc">{(factoryBusy * 100).toFixed(0)}%</span>
        </div>
      </div>
      <div className="lr-note">
        → <b>{bound === "memory" ? "memory-bound" : "compute-bound"}</b>:{" "}
        {bound === "memory" ? "a faster processor would change nothing; only a faster truck (or fewer bytes) helps." : "the truck is fast enough; only faster arithmetic helps."}
      </div>
    </Fig>
  );
}

/** fig 9.2: the roofline, built up */
function Roofline() {
  const pal = usePalette();
  const liveBw = useEngine((s) => (s.source === "live" ? s.bandwidthGBs : null));
  const livePk = useEngine((s) => (s.source === "live" ? s.peakGflops : null));
  const [m, setM] = useState<"tab" | "gpu">("tab");
  const [stage, setStage] = useState(4);
  const [P, setP] = useState(64);
  const [ref, w] = useWidth<HTMLDivElement>();
  const bw = m === "tab" ? liveBw ?? 59 : 80;
  const peak = m === "tab" ? livePk ?? 97 : 4300;
  const W = Math.max(300, w), H = 280, pl = 52, pb = 34;
  const x = log(0.1, 1000, pl, W - 10);
  const y = log(1, 10000, H - pb, 10);
  const ridge = peak / bw;
  const att = (I: number) => Math.min(peak, bw * I);
  const dec = 0.5, pre = P / 2;
  return (
    <Fig
      n="9.2"
      title="The roofline"
      prov={m === "tab" ? (liveBw ? "live" : "recorded") : "measured"}
      caption={
        <>
          The truck-and-factory story as engineers draw it (a “roofline”). Across: work per byte. Up: arithmetic per second
          you can actually get. Both axes are logarithmic. You're capped by the blue slanted roof (memory) or the ember flat
          one (compute), whichever is lower. Decode sits at 0.5 operations per byte; prefill moves right as the prompt grows.
          {m === "gpu" && " GPU numbers: ~80 GB/s measured on our M4; ~4.3 trillion operations per second is Apple's spec figure."}
        </>
      }
      tryit={<>Step through 1 → 4. Then drag the prompt length and find where prefill hits the flat roof.</>}
    >
      <div className="lr-controls">
        <Seg
          options={[
            { v: 1, label: "1 · memory roof" },
            { v: 2, label: "2 · compute roof" },
            { v: 3, label: "3 · decode" },
            { v: 4, label: "4 · prefill" },
          ]}
          value={stage}
          onChange={setStage}
          label="build up"
        />
        <Seg
          options={[
            { v: "tab", label: liveBw ? "this tab" : "browser tab (M4)" },
            { v: "gpu", label: "M4 GPU" },
          ]}
          value={m}
          onChange={setM}
          label="machine"
        />
        {stage >= 4 && <Slider label="prompt tokens" value={P} min={1} max={512} onChange={setP} width="7rem" />}
      </div>
      <div ref={ref}>
        {w > 0 && (
          <svg width={W} height={H}>
            {logTicks(0.1, 1000).map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={10} y2={H - pb} stroke={pal.rule} />
                <text x={x(t)} y={H - pb + 14} textAnchor="middle" fontSize={10} fill={pal.muted}>
                  {t}
                </text>
              </g>
            ))}
            {logTicks(1, 10000).map((t) => (
              <g key={t}>
                <line x1={pl} x2={W - 10} y1={y(t)} y2={y(t)} stroke={pal.rule} />
                <text x={pl - 6} y={y(t) + 4} textAnchor="end" fontSize={10} fill={pal.muted}>
                  {short(t)}
                </text>
              </g>
            ))}
            <text x={W - 10} y={H - 4} textAnchor="end" fontSize={10} fill={pal.sub}>
              work per byte (FLOP/B) →
            </text>
            <text x={pl} y={8} fontSize={10} fill={pal.sub}>
              GFLOP/s ↑
            </text>
            {/* memory roof */}
            <line x1={x(0.1)} y1={y(Math.max(1, bw * 0.1))} x2={x(stage >= 2 ? ridge : 1000)} y2={y(stage >= 2 ? peak : bw * 1000)} stroke={pal.cool} strokeWidth={2.5} opacity={stage >= 2 ? 1 : 0.9} />
            {stage === 1 && (
              <text x={x(2)} y={y(bw * 2) - 10} fontSize={11} fill={pal.cool}>
                memory: {bw.toFixed(0)} GB/s × work per byte
              </text>
            )}
            {stage >= 2 && (
              <>
                <line x1={x(ridge)} x2={x(1000)} y1={y(peak)} y2={y(peak)} stroke={pal.accent} strokeWidth={2.5} />
                <line x1={x(ridge)} x2={x(ridge)} y1={y(peak)} y2={H - pb} stroke={pal.muted} strokeDasharray="3 4" />
                <text x={x(ridge) + 6} y={H - pb - 6} fontSize={10} fill={pal.muted}>
                  ridge {ridge.toFixed(1)}
                </text>
                <text x={W - 14} y={y(peak) - 8} textAnchor="end" fontSize={11} fill={pal.accent}>
                  compute: {short(peak * 1e9)}FLOP/s
                </text>
              </>
            )}
            {stage >= 3 && (
              <g>
                <circle cx={x(dec)} cy={y(att(dec))} r={6} fill={pal.cool} stroke={pal.bg} strokeWidth={2} />
                <text x={x(dec) + 10} y={y(att(dec)) + 16} fontSize={11} fill={pal.fg} fontWeight={700}>
                  decode
                </text>
                <text x={x(dec) + 10} y={y(att(dec)) + 29} fontSize={10} fill={pal.muted}>
                  {att(dec).toFixed(0)} GFLOP/s · {((att(dec) / peak) * 100).toFixed(0)}% of the factory
                </text>
              </g>
            )}
            {stage >= 4 && (
              <g>
                <circle cx={x(pre)} cy={y(att(pre))} r={6} fill={pal.accent} stroke={pal.bg} strokeWidth={2} />
                <text x={Math.min(W - 160, x(pre) + 10)} y={y(att(pre)) + (pre > ridge ? 18 : -10)} fontSize={11} fill={pal.fg} fontWeight={700}>
                  prefill, {P} tokens
                </text>
                <text x={Math.min(W - 160, x(pre) + 10)} y={y(att(pre)) + (pre > ridge ? 31 : 3)} fontSize={10} fill={pal.muted}>
                  {pre} FLOP/B · {pre > ridge ? "compute-bound" : "memory-bound"}
                </text>
              </g>
            )}
          </svg>
        )}
      </div>
    </Fig>
  );
}

export default function BytesFlops() {
  return (
    <>
      <p>
        Part one ended with a fact: every decode step uses every weight in the model. Here's why that matters. To do a
        multiply-add, a processor needs two things: the numbers, and the time to do the arithmetic. The numbers live in
        memory and have to be carried over. So every computation is waiting on one of two things: the <b>truck</b> (memory
        bandwidth, bytes per second) or the <b>factory</b> (arithmetic, operations per second).
      </p>
      <Factory />
      <p>
        Now apply it to decode. Each weight is 4 bytes. It's fetched, used for exactly one multiply and one add, and not
        needed again until the next token. That's <span className="k">0.5 operations per byte</span>, and the crossover in a
        browser tab is 1.6. Decode is <G k="memory-bound" />: the factory spends most of its time waiting for the truck.
      </p>
      <p>
        Prefill is different. With a 64-token prompt, each weight is fetched once and used 64 times, once per token. Same
        bytes, 64 times the work: 32 operations per byte, far past the crossover. Prefill is <G k="compute-bound" />.
      </p>
      <Idea>Every program waits on memory or on maths. Decode does so little work per byte that it waits on memory: a faster processor won't help it, fewer bytes will.</Idea>
      <p>
        We saw exactly this in the engine. A browser build with faster multiply-add instructions (a faster factory) made
        prefill <b>40% faster</b>, 255 → 370 tokens per second, and decode <b>not at all</b>: it stayed at 98–101. Decode was
        never waiting on the arithmetic.
      </p>
      <Deeper title="The chart engineers draw for this: the roofline">
        <Roofline />
      </Deeper>
      <Deeper title="Same idea, other names">
        <p>
          “Work per byte” is called <b>arithmetic intensity</b>, and the crossover is the <b>ridge point</b>. On real hardware
          the ridge is far to the right: our M4's GPU is near 50 operations per byte, a datacenter GPU near 300. So for one
          user at a time, decode is memory-bound on essentially every computer that exists.
        </p>
        <p>
          The two numbers for this tab are measured by the engine itself, compiled to WebAssembly: a big streaming read
          (bandwidth) and a tight multiply-add loop (compute).
        </p>
      </Deeper>
    </>
  );
}
