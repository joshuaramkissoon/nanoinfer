import { useState } from "react";
import { useWidth } from "../../ui/Figure";
import { usePalette } from "../../ui/theme";
import { log } from "../../lib/fmt";
import { Climb } from "../Climb";
import { chartColors, Deeper, Fig, G, Idea, Seg, useLearnData, type LearnData } from "../kit";

const DT_LABEL: Record<string, string> = { q4_0: "4-bit", q8_0: "8-bit", f16: "16-bit", f32: "32-bit" };

function BatchCurve({ d }: { d: LearnData }) {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [dt, setDt] = useState("f32");
  const [i, setI] = useState(3);
  const [ref, w] = useWidth<HTMLDivElement>();
  const series = d.batching.series.find((s) => s.label === dt) ?? d.batching.series[0];
  const pts = series.points;
  const k = Math.min(i, pts.length - 1);
  const [B, total, per, ms] = pts[k];
  const W = Math.max(300, w), H = 250, pl = 46, pb = 30, top = 12, pr = 12;
  const x = log(1, 256, pl, W - pr);
  const y = log(10, 10000, H - pb, top);
  const line = (j: 1 | 2) => pts.map((p, n) => `${n ? "L" : "M"}${x(p[0])},${y(p[j])}`).join(" ");
  return (
    <Fig
      n="17.1"
      title="One GPU, many users"
      prov="measured"
      caption={
        <>
          Measured on the M4's GPU. Across: how many users are served together in each step (the <G k="batch" /> size). Up:
          tokens per second, on a log scale. Ember: tokens per second the GPU produces <b>in total</b>. Blue: tokens per second
          each <b>individual user</b> gets.
        </>
      }
      tryit={
        <>
          With 32-bit weights, drag from 1 to 8 users: the total grows 7× while each user slows by only 13%, because the
          weights were being read anyway. Keep going and each user's speed falls away. Then switch to 4-bit: there's less
          waiting on memory to soak up, so the free ride ends sooner (around 4 users).
        </>
      }
    >
      <div className="lr-controls">
        <Seg options={["q4_0", "q8_0", "f16", "f32"].map((v) => ({ v, label: DT_LABEL[v] }))} value={dt} onChange={setDt} label="weights" />
        <label className="lr-slider">
          <span className="k">users at once</span>
          <input type="range" min={0} max={pts.length - 1} value={k} onChange={(e) => setI(+e.target.value)} />
          <span className="v">{B}</span>
        </label>
      </div>
      <div style={{ display: "flex", gap: "1.2rem", fontSize: "0.74rem", color: pal.sub, marginBottom: "0.3rem" }}>
        <span>
          <i style={{ display: "inline-block", width: 14, height: 2, background: cc.c1, verticalAlign: "middle", marginRight: 6 }} />
          total, all users
        </span>
        <span>
          <i style={{ display: "inline-block", width: 14, height: 2, background: cc.c2, verticalAlign: "middle", marginRight: 6 }} />
          each user
        </span>
      </div>
      <div ref={ref}>
        {w > 0 && (
          <svg width={W} height={H}>
            {[10, 100, 1000, 10000].map((t) => (
              <g key={t}>
                <line x1={pl} x2={W - pr} y1={y(t)} y2={y(t)} stroke={pal.rule} />
                <text x={pl - 6} y={y(t) + 4} textAnchor="end" fontSize={10} fill={pal.muted}>
                  {t >= 1000 ? `${t / 1000}k` : t}
                </text>
              </g>
            ))}
            {[1, 4, 16, 64, 256].map((t) => (
              <text key={t} x={x(t)} y={H - 12} textAnchor="middle" fontSize={10} fill={pal.muted}>
                {t}
              </text>
            ))}
            <text x={W - pr} y={H - 1} textAnchor="end" fontSize={10} fill={pal.muted}>
              users per step →
            </text>
            <line x1={x(B)} x2={x(B)} y1={top} y2={H - pb} stroke={pal.muted} />
            <path d={line(1)} fill="none" stroke={cc.c1} strokeWidth={2} strokeLinejoin="round" />
            <path d={line(2)} fill="none" stroke={cc.c2} strokeWidth={2} strokeLinejoin="round" />
            <circle cx={x(B)} cy={y(total)} r={5} fill={cc.c1} stroke={pal.dark ? "#211f1d" : "#fbfaf7"} strokeWidth={2} />
            <circle cx={x(B)} cy={y(per)} r={5} fill={cc.c2} stroke={pal.dark ? "#211f1d" : "#fbfaf7"} strokeWidth={2} />
            <text x={Math.min(W - 70, x(B) + 8)} y={y(total) - 8} fontSize={11} fill={pal.fg} fontWeight={700}>
              {total.toLocaleString()}
            </text>
            <text x={Math.min(W - 70, x(B) + 8)} y={y(per) + 16} fontSize={11} fill={pal.fg}>
              {per.toFixed(0)}
            </text>
          </svg>
        )}
      </div>
      <div className="lr-note">
        {B} {B === 1 ? "user" : "users"} at once: each gets <b>{per.toFixed(0)} tok/s</b> (a token every {ms.toFixed(1)} ms); the GPU makes{" "}
        <b>{total.toLocaleString()} tok/s</b> in total, {(total / pts[0][1]).toFixed(1)}× serving one user at a time.
      </div>
    </Fig>
  );
}

function Gantt({ d }: { d: LearnData }) {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [pol, setPol] = useState<"static" | "continuous">("static");
  const [ref, w] = useWidth<HTMLDivElement>();
  const run = d.serving["rate=16"]?.[pol];
  if (!run) return null;
  const tmax = Math.max(...Object.values(d.serving["rate=16"]).flatMap((r) => r.gantt.map((g) => g[3])));
  const W = Math.max(300, w), row = 7, H = run.gantt.length * (row + 2) + 24;
  const x = (t: number) => 4 + (t / tmax) * (W - 8);
  return (
    <Fig
      n="17.2"
      title="Static vs continuous batching"
      prov="measured"
      caption={
        <>
          The first 24 of 128 requests arriving at random (16 a second), served on the M4's GPU. Each row is one request, left
          to right in time. Grey: waiting in the queue. Ember: being served, from first token to last.
        </>
      }
      tryit={<>Switch between the two policies and compare how long the grey waits are.</>}
    >
      <div className="lr-controls">
        <Seg
          options={[
            { v: "static", label: "static batches" },
            { v: "continuous", label: "continuous" },
          ]}
          value={pol}
          onChange={setPol}
          label="policy"
        />
        <span className="lr-small">
          median wait for the first word: <b style={{ color: pal.fg }}>{run.ttft_p50_ms >= 1000 ? `${(run.ttft_p50_ms / 1000).toFixed(1)} s` : `${run.ttft_p50_ms.toFixed(0)} ms`}</b>
        </span>
      </div>
      <div ref={ref}>
        {w > 0 && (
          <svg width={W} height={H}>
            {run.gantt.map(([arr, , first, fin], i) => (
              <g key={i}>
                <rect x={x(arr)} y={i * (row + 2)} width={Math.max(1, x(first) - x(arr))} height={row} rx={2} fill={pal.muted} opacity={0.45} />
                <rect x={x(first) + 1} y={i * (row + 2)} width={Math.max(2, x(fin) - x(first) - 1)} height={row} rx={2} fill={cc.c1} />
              </g>
            ))}
            <text x={W - 4} y={H - 4} textAnchor="end" fontSize={10} fill={pal.muted}>
              time → ({tmax.toFixed(1)} s shown)
            </text>
          </svg>
        )}
      </div>
    </Fig>
  );
}

export default function Batching() {
  const d = useLearnData();
  return (
    <>
      <h2 style={{ marginTop: "2.2rem" }}>The bottleneck</h2>
      <p>
        Serving one user, the GPU reads every weight for every token and does one multiply-add with it. In chapter 9's terms,
        the factory is almost always idle, waiting for the truck. A server has the same problem, multiplied: many users, and
        a chip that's mostly waiting.
      </p>
      <h2>The idea</h2>
      <p>
        Read each weight once and use it for many users. Put the next token of 64 different conversations through the model
        together, as a <G k="batch" />, and each weight fetched from memory now does 64 multiply-adds instead of one. The
        truck delivers the same, the factory finally has work.
      </p>
      {d ? <BatchCurve d={d} /> : <div className="lr-empty">loading…</div>}
      <p>
        With 32-bit weights, up to about 8 users is nearly free: the GPU was waiting on memory anyway. Past that, the arithmetic catches up with
        the bytes (the crossover from chapter 9), and from there more users share a fixed amount of maths, so each one slows
        down.
      </p>
      <h2>The result</h2>
      <p>
        With 4-bit weights, one GPU went from 788 tokens per second for one user to <span className="k">6,453 tokens per
        second</span> in total for 256, about 25 each. And at large batches every number format lands between 5,600 and
        6,500: once the job is compute-bound, smaller numbers stop helping.
      </p>
      <Climb upto={6} n="17.3" />
      <Idea>Batching turns a memory-bound job into a compute-bound one: each weight read serves many users. It's why serving many people costs far less per person than serving one.</Idea>
      <Deeper title="Continuous batching: a bus that never waits at the terminal">
        <p>
          A simple server collects a batch, runs it until the <i>longest</i> answer is done, then takes the next batch.
          Short answers sit in their seats doing nothing, and new arrivals wait for the whole batch. A continuous server
          re-forms the batch every single step: finished requests leave, waiting ones join.
        </p>
        {d && <Gantt d={d} />}
        <p>
          Same GPU, same model: the median wait for the first word fell from <b>4.4 seconds to 62 ms</b>, and total throughput
          rose 1.8×. The static server was busy 100% of the time and still made people wait four seconds.
        </p>
      </Deeper>
    </>
  );
}
