import { useRef, useState } from "react";
import { useEngine } from "../../engine/client";
import { useWidth } from "../../ui/Figure";
import { usePalette } from "../../ui/theme";
import { log, logTicks } from "../../lib/fmt";
import { chartColors, Deeper, Fig, G, Idea, PlayPause, Seg, Slider, Swatch, useFrame, useOnScreen, usePlaying, T } from "../kit";
import { MACHINES } from "./p0/specs";

/** fig 9.1: a truck feeding a factory */
function Factory() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [I, setI] = useState(0.5); // work per box (FLOP per byte)
  const R = 1.6; // the tab's ridge: 97 GFLOP/s / 59 GB/s
  const [ref, w] = useWidth<HTMLDivElement>();
  const sim = useRef({ boxes: [] as number[], queue: 0, proc: 0, spawn: 0, busy: 0, idle: 0, out: 0 });
  const [, tick] = useState(0);
  const truckRate = 2.2; // boxes per second, visual
  const [onScreen, vis] = useOnScreen<HTMLElement>();
  const [playing, setPlaying] = usePlaying();
  useFrame(vis && playing, (dt) => {
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
  const W = Math.max(300, w), H = 132;
  const narrow = W < 460;
  const houseW = narrow ? 72 : 92, factW = narrow ? 96 : 120;
  const road0 = houseW + 14, road1 = W - factW - 18;
  const bound = I < R ? "memory" : "compute";
  return (
    <Fig
      figRef={onScreen}
      n="9.1"
      title="The truck and the factory"
      prov="toy"
      caption={
        <>
          Left, <Swatch c={cc.c2}>memory</Swatch>, where the weights live. The truck delivers boxes of bytes at a steady
          rate, its <G k="bandwidth" />. Right, the <Swatch c={cc.c1}>processor</Swatch> (the factory) does the arithmetic each
          box needs. The two bars show how busy each side is. The crossover uses this tab's measured numbers: about 97
          billion operations per second ÷ 59 GB/s, or <b>1.6 operations per byte</b>.
        </>
      }
      tryit={
        <>
          Press <b>decode</b>, then <b>prefill</b>. Then drag slowly from one to the other and find the crossover. Notice
          which side waits on each side of it.
        </>
      }
    >
      <div className="lr-controls lr-truck-controls">
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
          width="10rem"
        />
        <Seg
          label="presets"
          value={I === 0.5 ? "decode" : I === 32 ? "prefill" : ""}
          onChange={(v) => setI(v === "decode" ? 0.5 : 32)}
          options={[
            { v: "decode", label: "decode · 0.5" },
            { v: "prefill", label: "prefill · 32" },
          ]}
        />
        <PlayPause playing={playing} onToggle={() => setPlaying((p) => !p)} />
      </div>
      <div ref={ref}>
        {w > 0 && (
          <svg width={W} height={H}>
            {/* warehouse: memory (blue) */}
            <rect x={2} y={26} width={houseW} height={64} rx={8} fill={cc.c2} fillOpacity={0.1} stroke={cc.c2} />
            <text x={2 + houseW / 2} y={55} textAnchor="middle" fontSize={T.sm} fontWeight={700} fill={pal.fg}>
              memory
            </text>
            <text x={2 + houseW / 2} y={72} textAnchor="middle" fontSize={T.xs} fill={pal.sub}>
              the weights
            </text>
            {/* road: bandwidth, carrying boxes of bytes (blue) */}
            <line x1={road0} x2={road1} y1={58} y2={58} stroke={pal.rule} strokeWidth={16} strokeLinecap="round" />
            <text x={(road0 + road1) / 2} y={36} textAnchor="middle" fontSize={T.xs} fill={pal.sub}>
              bytes arriving at a fixed rate →
            </text>
            {s.boxes.map((x, i) => (
              <rect key={i} x={road0 + x * (road1 - road0 - 12)} y={52} width={12} height={12} rx={2} fill={cc.c2} />
            ))}
            {/* queue at the factory door */}
            {Array.from({ length: Math.min(9, s.queue) }, (_, i) => (
              <rect key={i} x={road1 - 4 - (i % 3) * 14} y={78 + Math.floor(i / 3) * -14 + 14} width={12} height={12} rx={2} fill={cc.c2} opacity={0.75} />
            ))}
            {/* factory: the processor (ember when working) */}
            <g transform={`translate(${W - factW - 4},18)`}>
              <rect x={0} y={10} width={factW} height={70} rx={8} fill={cc.c1} fillOpacity={s.proc > 0 ? 0.22 : 0.05} stroke={cc.c1} />
              <text x={factW / 2} y={40} textAnchor="middle" fontSize={T.sm} fontWeight={700} fill={pal.fg}>
                processor
              </text>
              <text x={factW / 2} y={58} textAnchor="middle" fontSize={T.xs} fill={s.proc > 0 ? pal.fg : pal.muted}>
                {s.proc > 0 ? "working" : "waiting…"}
              </text>
            </g>
            <text x={2} y={H - 4} fontSize={T.xs} fill={pal.muted}>
              {bound === "memory" ? "the processor is idle between boxes" : "boxes pile up at the processor's door"}
            </text>
          </svg>
        )}
      </div>
      <div className="lr-bars" style={{ marginTop: "0.4rem" }}>
        <div className="lr-bar lr-busy">
          <span className="tk">memory busy</span>
          <span className="b">
            <i style={{ width: `${truckBusy * 100}%`, background: cc.c2 }} />
          </span>
          <span className="pc">{(truckBusy * 100).toFixed(0)}%</span>
        </div>
        <div className="lr-bar lr-busy">
          <span className="tk">maths busy</span>
          <span className="b">
            <i style={{ width: `${factoryBusy * 100}%`, background: cc.c1 }} />
          </span>
          <span className="pc">{(factoryBusy * 100).toFixed(0)}%</span>
        </div>
      </div>
      <div className="lr-note lr-verdict">
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
  const bw = m === "tab" ? liveBw ?? 59 : 105;
  const peak = m === "tab" ? livePk ?? 97 : 4300;
  const W = Math.max(300, w), H = 300, pl = 44, pb = 44, pt = 26;
  const x = log(0.1, 1000, pl, W - 10);
  const y = log(1, 10000, H - pb, pt);
  const tk = (t: number) => (t >= 1000 ? `${t / 1000}k` : String(t));
  const flops = (g: number) => (g >= 1000 ? `${(g / 1000).toFixed(1)} TFLOP/s` : `${g.toFixed(0)} GFLOP/s`);
  /** labels get a surface-coloured halo so gridlines and roofs never run through them */
  const halo = { stroke: pal.bg, strokeWidth: 4, paintOrder: "stroke" as const, strokeLinejoin: "round" as const };
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
          The truck and the factory as engineers draw them, a “roofline”. Across: work per byte. Up: arithmetic per second
          you can get. Both axes are log scale. The lower of two roofs caps you: the <Swatch c={pal.cool}>blue</Swatch> slanted
          line (memory) or the <Swatch c={pal.accent}>ember</Swatch> flat line (compute).
          {m === "gpu" && " GPU numbers: about 105 GB/s measured on the M4 (quiet machine) and Apple's spec of about 4.3 trillion operations per second."}
        </>
      }
      tryit={<>Step through 1 to 4. Then drag the prompt length and notice where the prefill dot meets the flat roof.</>}
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
                <line x1={x(t)} x2={x(t)} y1={pt} y2={H - pb} stroke={pal.rule} />
                <text x={x(t)} y={H - pb + 16} textAnchor="middle" fontSize={T.xs} fill={pal.muted}>
                  {tk(t)}
                </text>
              </g>
            ))}
            {logTicks(1, 10000).map((t) => (
              <g key={t}>
                <line x1={pl} x2={W - 10} y1={y(t)} y2={y(t)} stroke={pal.rule} />
                <text x={pl - 6} y={y(t) + 4} textAnchor="end" fontSize={T.xs} fill={pal.muted}>
                  {tk(t)}
                </text>
              </g>
            ))}
            <text x={W - 10} y={H - 4} textAnchor="end" fontSize={T.xs} fill={pal.sub}>
              work per byte (FLOP/B, log scale) →
            </text>
            <text x={pl - 6} y={11} fontSize={T.xs} fill={pal.sub}>
              ↑ GFLOP/s achieved (log scale)
            </text>
            {/* memory roof */}
            <line x1={x(0.1)} y1={y(Math.max(1, bw * 0.1))} x2={x(stage >= 2 ? ridge : 1000)} y2={y(stage >= 2 ? peak : bw * 1000)} stroke={pal.cool} strokeWidth={2.5} opacity={stage >= 2 ? 1 : 0.9} />
            {stage === 1 && (
              <text x={x(2)} y={y(bw * 2) - 10} fontSize={T.xs} fill={pal.sub} {...halo}>
                memory roof: {bw.toFixed(0)} GB/s × work per byte
              </text>
            )}
            {stage >= 2 && (
              <>
                <line x1={x(ridge)} x2={x(1000)} y1={y(peak)} y2={y(peak)} stroke={pal.accent} strokeWidth={2.5} />
                <line x1={x(ridge)} x2={x(ridge)} y1={y(peak)} y2={H - pb} stroke={pal.muted} strokeDasharray="3 4" />
                <text x={x(ridge) + 6} y={H - pb - 6} fontSize={T.xs} fill={pal.muted} {...halo}>
                  ridge {ridge.toFixed(1)}
                </text>
                <text x={W - 14} y={y(peak) - 8} textAnchor="end" fontSize={T.xs} fill={pal.sub} {...halo}>
                  compute roof: {flops(peak)}
                </text>
              </>
            )}
            {stage >= 3 && (
              <g>
                <circle cx={x(dec)} cy={y(att(dec))} r={6} fill={pal.cool} stroke={pal.bg} strokeWidth={2} />
                <text x={x(dec) + 10} y={y(att(dec)) + 18} fontSize={T.xs} fill={pal.fg} fontWeight={700} {...halo}>
                  decode
                </text>
                <text x={x(dec) + 10} y={y(att(dec)) + 31} fontSize={T.xs} fill={pal.muted} {...halo}>
                  {att(dec).toFixed(0)} GFLOP/s · {((att(dec) / peak) * 100).toFixed(0)}% of the factory
                </text>
              </g>
            )}
            {stage >= 4 && (
              <g>
                <circle cx={x(pre)} cy={y(att(pre))} r={6} fill={pal.accent} stroke={pal.bg} strokeWidth={2} />
                <text x={Math.min(W - 160, x(pre) + 10)} y={y(att(pre)) + (pre > ridge ? 18 : -10)} fontSize={T.xs} fill={pal.fg} fontWeight={700} {...halo}>
                  prefill, {P} tokens
                </text>
                <text x={Math.min(W - 160, x(pre) + 10)} y={y(att(pre)) + (pre > ridge ? 31 : 3)} fontSize={T.xs} fill={pal.muted} {...halo}>
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

/** fig 9.3: the ridge on each machine the course uses, side by side */
function Ridges() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const liveBw = useEngine((s) => (s.source === "live" ? s.bandwidthGBs : null));
  const livePk = useEngine((s) => (s.source === "live" ? s.peakGflops : null));
  const [ref, w] = useWidth<HTMLDivElement>();
  const m = (k: string) => MACHINES.find((x) => x.key === k)!;
  const tabBw = liveBw ?? 59, tabPk = livePk ?? 97;
  const rows: { name: string; how: string; r: number; calc: string; fig: string }[] = [
    { name: "browser tab", how: "1 M4 core, WebAssembly, f32", r: tabPk / tabBw, calc: `${tabPk.toFixed(0)} GFLOP/s ÷ ${tabBw.toFixed(0)} GB/s`, fig: "fig 9.1" },
    { name: "M4 CPU", how: "10 cores, f32", r: (m("m4cpu").tflops * 1000) / m("m4cpu").bw, calc: `~${m("m4cpu").tflops * 1000} GFLOP/s ÷ ${m("m4cpu").bw} GB/s`, fig: "fig 0.3.1" },
    { name: "M4 GPU", how: "1,280 lanes, f32", r: (m("m4gpu").tflops * 1000) / m("m4gpu").bw, calc: `~${m("m4gpu").tflops} TFLOP/s ÷ ${m("m4gpu").bw} GB/s`, fig: "figs 0.2.2, 0.3.1" },
    { name: "H100", how: "tensor cores, bf16", r: (m("h100").tflops * 1000) / m("h100").bw, calc: `${m("h100").tflops} TFLOP/s ÷ ${(m("h100").bw / 1000).toFixed(2)} TB/s`, fig: "fig 0.2.2" },
  ];
  const W = Math.max(300, w);
  const narrow = W < 520;
  const labelW = narrow ? 92 : 170;
  const row = narrow ? 40 : 30, bar = 14, top = 22;
  const H = top + rows.length * row + 26;
  const x = log(0.1, 1000, labelW + 6, W - 56);
  return (
    <Fig
      n="9.3"
      title="The ridge on each machine in this course"
      prov="estimate"
      caption={
        <>
          Each grey dot is one machine's ridge: peak arithmetic ÷ memory bandwidth, in operations per byte. Across: log
          scale. The <Swatch c={cc.c2}>blue band</Swatch> is decode for one user: 0.5 operations per byte with 32-bit weights, 1
          with 16-bit.
        </>
      }
      source="Browser tab: measured by the engine in this page (recorded M4 values when not running). M4 CPU: 98 GB/s measured, ~0.6 TFLOP/s (estimate: 4 performance cores × 4 FMA pipes × 4 lanes). M4 GPU: 105 GB/s measured, ~4.3 TFLOP/s f32 (Apple). H100 SXM: NVIDIA datasheet."
    >
      <div ref={ref}>
        {w > 0 && (
          <svg width={W} height={H} role="img" aria-label="ridge points: browser tab 1.6, M4 CPU 6, M4 GPU 41, H100 295 operations per byte">
            {[0.1, 1, 10, 100, 1000].map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={top - 6} y2={H - 26} stroke={pal.rule} />
                <text x={x(t)} y={H - 12} textAnchor="middle" fontSize={T.xs} fill={pal.muted}>
                  {t}
                </text>
              </g>
            ))}
            <rect x={x(0.5)} y={top - 6} width={x(1) - x(0.5)} height={H - 26 - (top - 6)} fill={cc.c2} fillOpacity={0.22} />
            <text x={x(0.7)} y={top - 9} textAnchor="middle" fontSize={T.xs} fill={pal.sub}>
              decode
            </text>
            <text x={W - 4} y={H + 3} textAnchor="end" fontSize={T.xs} fill={pal.muted}>
              operations per byte (log scale) →
            </text>
            {rows.map((r, i) => {
              const y0 = top + i * row + (row - bar) / 2;
              return (
                <g key={r.name}>
                  <title>{`${r.name} (${r.how}): ${r.calc} ≈ ${r.r < 10 ? r.r.toFixed(1) : Math.round(r.r)} · used in ${r.fig}`}</title>
                  <text x={labelW} y={narrow ? y0 + 4 : y0 + bar / 2 + 4} textAnchor="end" fontSize={T.xs} fill={pal.fg}>
                    {r.name}
                  </text>
                  {narrow && (
                    <text x={labelW} y={y0 + 17} textAnchor="end" fontSize={T.xs} fill={pal.muted}>
                      {r.how}
                    </text>
                  )}
                  <line x1={x(0.1)} x2={x(r.r)} y1={y0 + bar / 2} y2={y0 + bar / 2} stroke={pal.muted} strokeWidth={1.5} />
                  <circle cx={x(r.r)} cy={y0 + bar / 2} r={5} fill={pal.sub} stroke={pal.bg} strokeWidth={2} />
                  <text x={x(r.r) + 10} y={y0 + bar / 2 + 4} fontSize={T.xs} fontWeight={700} fill={pal.fg}>
                    {r.r < 10 ? r.r.toFixed(1) : Math.round(r.r)}
                  </text>
                </g>
              );
            })}
          </svg>
        )}
      </div>
      <div className="lr-note">
        {rows.map((r, i) => (
          <span key={r.name}>
            {i > 0 && " · "}
            <b>{r.name}</b> ({r.how}): {r.calc}
          </span>
        ))}
      </div>
    </Fig>
  );
}

export default function BytesFlops() {
  return (
    <>
      <p>
        Every decode step reads every weight in the model. A multiply-add needs two things: the numbers, and time to do
        the arithmetic. The numbers live in memory and have to be carried to the processor. So every computation waits on
        one of two things. One is the <b>truck</b>, memory bandwidth in bytes per second. The other is the <b>factory</b>,
        arithmetic in operations per second.
      </p>
      <Factory />
      <p>
        Now apply it to decode. Each weight is 4 bytes. Decode reads it, uses it for one multiply and one add, and doesn't
        need it again until the next token. That's <span className="k">0.5 operations per byte</span>, and the crossover in a
        browser tab is 1.6. Decode is <G k="memory-bound" />, so the factory spends most of its time waiting for the truck.
        Chapter 0.2 said about 1 per byte, because it used 2-byte weights. Halving the bytes doubles the work per byte, and
        that is still far below any chip's crossover.
      </p>
      <p>
        Prefill is different. With a 64-token prompt, prefill reads each weight once and uses it 64 times, once per token.
        The same bytes do 64 times the work, 32 operations per byte, far past the crossover. Prefill is <G k="compute-bound" />.
      </p>
      <Idea>Every program waits on memory or on arithmetic. Decode does so little work per byte that it waits on memory, so fewer bytes help it and a faster processor doesn't.</Idea>
      <p>
        The engine shows this. A browser build with faster multiply-add instructions (a faster factory) made prefill{" "}
        <b>40% faster</b>, from 255 to 370 tokens per second. Decode didn't move: it stayed at 98 to 101 tokens per second,
        because it was never waiting on the arithmetic.
      </p>
      <Deeper title="The chart engineers draw for this: the roofline">
        <Roofline />
      </Deeper>
      <Deeper title="Same idea, other names">
        <p>
          “Work per byte” is called <b>arithmetic intensity</b>, and the crossover is the <b>ridge point</b>. On fast chips
          the ridge is far to the right: near 41 operations per byte for the M4's GPU and 295 for an H100 (fig 9.3). So for
          one user, decode is memory-bound on every common CPU and GPU.
        </p>
        <p>
          The two numbers for this tab are measured by the engine itself, compiled to WebAssembly: a big streaming read
          (bandwidth) and a tight multiply-add loop (compute).
        </p>
      </Deeper>
      <p>
        The course quotes three crossovers: about 1.6 for a browser tab on the M4, 41 for the M4's GPU and 295 for an H100.
        Each is right for its machine. A ridge divides two separate design choices: the chip's arithmetic (at its number
        format) and its memory bandwidth. An H100's tensor cores do 16-bit matrix arithmetic about 230× faster than the M4's
        GPU does 32-bit. Its memory is only about 32× faster.
      </p>
      <Ridges />
    </>
  );
}
