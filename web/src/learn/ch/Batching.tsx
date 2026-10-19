import { useState } from "react";
import { Tip, useWidth } from "../../ui/Figure";
import { usePalette } from "../../ui/theme";
import { log } from "../../lib/fmt";
import { Climb, bestBatch } from "../Climb";
import { chartColors, Deeper, Fig, G, Idea, Seg, Swatch, nf, useLearnData, type LearnData, type ServingRate, T } from "../kit";

const DT_LABEL: Record<string, string> = { q4_0: "4-bit", q8_0: "8-bit", f16: "16-bit", f32: "32-bit" };
const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);

/** fig 17.1: the trade-off. Two lines in the same unit on one log axis: the GPU's total
 *  output, and what each user gets, as more users share each step. */
function BatchCurve({ d }: { d: LearnData }) {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [dt, setDt] = useState("q4_0");
  const [i, setI] = useState(4);
  const [over, setOver] = useState(false);
  const [ref, w] = useWidth<HTMLDivElement>();
  const series = d.batching.series.find((s) => s.label === dt) ?? d.batching.series[0];
  const pts = series.points;
  const k = Math.min(i, pts.length - 1);
  const [B, total, per, ms] = pts[k];
  const W = Math.max(300, w), H = 260, pl = 44, pb = 30, top = 14, pr = W < 480 ? 64 : 84;
  const x = log(1, 256, pl, W - pr);
  const y = log(10, 10000, H - pb, top);
  const line = (j: 1 | 2) => pts.map((p, n) => `${n ? "L" : "M"}${x(p[0])},${y(p[j])}`).join(" ");
  const ring = pal.dark ? "#211f1d" : "#fbfaf7";
  const last = pts[pts.length - 1];
  // nearest batch size to the pointer, so the whole plot is the hit target
  const pick = (px: number) => {
    let best = 0;
    for (let n = 1; n < pts.length; n++) if (Math.abs(x(pts[n][0]) - px) < Math.abs(x(pts[best][0]) - px)) best = n;
    setI(best);
  };
  return (
    <Fig
      n="17.1"
      title="One GPU, many users: the trade-off"
      prov="measured"
      caption={
        <>
          GPT-2 small on the M4's GPU, quiet machine, 64-token prompts then 64 tokens each. Across: users per step (the{" "}
          <G k="batch" /> size), doubling. Up: tok/s, log scale. <Swatch c={cc.c1}>Ember line</Swatch>: total for all users.{" "}
          <Swatch c={pal.sub}>Grey line</Swatch>: what one user gets.
        </>
      }
      tryit={
        <>
          With 4-bit weights, drag from 1 user to the most. Notice where the grey line starts to fall. Then switch to 32-bit
          and compare where that happens.
        </>
      }
    >
      <div className="lr-controls">
        <label className="lr-slider">
          <span className="k">users at once</span>
          <input type="range" min={0} max={pts.length - 1} value={k} onChange={(e) => setI(+e.target.value)} />
          <span className="v">{B}</span>
        </label>
        <Seg options={["q4_0", "q8_0", "f16", "f32"].map((v) => ({ v, label: DT_LABEL[v] }))} value={dt} onChange={setDt} label="number format" />
      </div>
      <div className="nv-legend" style={{ marginBottom: "0.3rem" }}>
        <span>
          <i style={{ display: "inline-block", width: 14, height: 2, background: cc.c1, verticalAlign: "middle", marginRight: 6 }} />
          total, all users
        </span>
        <span>
          <i style={{ display: "inline-block", width: 14, height: 2, background: pal.sub, verticalAlign: "middle", marginRight: 6 }} />
          each user
        </span>
      </div>
      <div ref={ref} style={{ position: "relative", touchAction: "pan-y" }} onPointerEnter={() => setOver(true)} onPointerLeave={() => setOver(false)}>
        {w > 0 && (
          <svg
            width={W}
            height={H}
            role="img"
            aria-label={`total and per-user tokens per second for 1 to 256 users, ${DT_LABEL[dt]} weights`}
            onPointerMove={(e) => pick(e.clientX - e.currentTarget.getBoundingClientRect().left)}
            onPointerDown={(e) => pick(e.clientX - e.currentTarget.getBoundingClientRect().left)}
          >
            {[10, 100, 1000, 10000].map((t) => (
              <g key={t}>
                <line x1={pl} x2={W - pr} y1={y(t)} y2={y(t)} stroke={pal.rule} />
                <text x={pl - 6} y={y(t) + 4} textAnchor="end" fontSize={T.xs} fill={pal.muted}>
                  {t >= 1000 ? `${t / 1000}k` : t}
                </text>
              </g>
            ))}
            {pts.map((p) => (
              <text key={p[0]} x={x(p[0])} y={H - 12} textAnchor="middle" fontSize={T.xs} fill={p[0] === B ? pal.fg : pal.muted}>
                {W < 420 && ![1, 4, 16, 64, 256].includes(p[0]) ? "" : p[0]}
              </text>
            ))}
            <text x={W - pr} y={H + 3} textAnchor="end" fontSize={T.xs} fill={pal.muted}>
              users per step →
            </text>
            <line x1={x(B)} x2={x(B)} y1={top} y2={H - pb} stroke={pal.muted} opacity={0.6} />
            <path d={line(1)} fill="none" stroke={cc.c1} strokeWidth={2} strokeLinejoin="round" />
            <path d={line(2)} fill="none" stroke={pal.sub} strokeWidth={2} strokeLinejoin="round" />
            {/* direct labels at the line ends */}
            <text x={x(last[0]) + 8} y={y(last[1]) + 4} fontSize={T.xs} fill={pal.sub}>
              total
            </text>
            <text x={x(last[0]) + 8} y={y(last[2]) + 4} fontSize={T.xs} fill={pal.sub}>
              each user
            </text>
            <circle cx={x(B)} cy={y(total)} r={5} fill={cc.c1} stroke={ring} strokeWidth={2} />
            <circle cx={x(B)} cy={y(per)} r={5} fill={pal.sub} stroke={ring} strokeWidth={2} />
          </svg>
        )}
        {w > 0 && over && (
          <Tip x={Math.min(Math.max(x(B), 100), W - 100)} y={y(total) - 4}>
            <b>{B} {B === 1 ? "user" : "users"}</b>
            <span className="nv-muted"> · {DT_LABEL[dt]}</span>
            <br />
            total <b>{nf(total)}</b> tok/s · each <b>{per.toFixed(0)}</b> tok/s
          </Tip>
        )}
      </div>
      <div className="lr-note">
        {B} {B === 1 ? "user" : "users"} at once: each gets <b>{per.toFixed(0)} tok/s</b> (a token every {ms.toFixed(1)} ms); the GPU
        writes <b>{nf(total)} tok/s</b> in total, {(total / pts[0][1]).toFixed(1)}× serving one user at a time.
      </div>
    </Fig>
  );
}

/** fig 17.3: the same requests, served two ways, on one time axis. */
function Timeline({ srv }: { srv: ServingRate }) {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<{ pol: "static" | "continuous"; i: number } | null>(null);
  const runs = { static: srv.static, continuous: srv.continuous };
  const tmax = Math.max(...Object.values(runs).flatMap((r) => r.gantt.map((g) => g[3])));
  const W = Math.max(300, w), pl = 6, pr = 6;
  const n = srv.static.gantt.length;
  const narrow = W < 520;
  const row = narrow ? 5 : 6, gap = 2, head = narrow ? 34 : 22, panel = head + n * (row + gap) + 10;
  const H = panel * 2 + 26;
  const x = (t: number) => pl + (t / tmax) * (W - pl - pr);
  const ticks = Array.from({ length: Math.floor(tmax / 2) + 1 }, (_, i) => i * 2);
  const tip = hover ? runs[hover.pol].gantt[hover.i] : null;
  return (
    <Fig
      n="17.3"
      title="The same requests, served two ways"
      prov="measured"
      caption={
        <>
          The first {n} of {srv.requests} requests, arriving at random (about 16 a second) on the M4's GPU, up to{" "}
          {srv.maxBatch} at once. Each row is one request over time. <Swatch c={pal.sub}>Grey bar</Swatch>: waiting for its
          first token. <Swatch c={cc.c1}>Ember bar</Swatch>: first token to last. Top: static batches. Bottom: continuous
          batching.
        </>
      }
      tryit={<>Compare the grey bars in the two panels. Then select a row to see one request's timings.</>}
    >
      <div ref={ref} style={{ position: "relative" }} onMouseLeave={() => setHover(null)}>
        {w > 0 && (
          <svg width={W} height={H} role="img" aria-label="request timelines under static and continuous batching">
            {ticks.map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={0} y2={H - 22} stroke={pal.rule} />
                <text x={x(t)} y={H - 8} textAnchor="middle" fontSize={T.xs} fill={pal.muted}>
                  {t} s
                </text>
              </g>
            ))}
            {(["static", "continuous"] as const).map((pol, pi) => {
              const r = runs[pol];
              const y0 = pi * panel;
              return (
                <g key={pol}>
                  <text x={pl} y={y0 + 14} fontSize={T.xs} fill={pal.fg} fontWeight={700}>
                    {pol === "static" ? "static batches" : "continuous batching"}
                    {!narrow && (
                      <tspan fontWeight={400} fill={pal.sub}>
                        {"  "}median wait for the first token: {fmtMs(r.ttft_p50_ms)}
                      </tspan>
                    )}
                  </text>
                  {narrow && (
                    <text x={pl} y={y0 + 27} fontSize={T.xs} fill={pal.sub}>
                      median wait for the first token: {fmtMs(r.ttft_p50_ms)}
                    </text>
                  )}
                  {r.gantt.map(([arr, , first, fin], i) => {
                    const yy = y0 + head + i * (row + gap);
                    const on = hover?.pol === pol && hover.i === i;
                    return (
                      <g key={i} onMouseEnter={() => setHover({ pol, i })} onClick={() => setHover({ pol, i })} opacity={hover && !on ? 0.55 : 1}>
                        <rect x={0} y={yy - 1} width={W} height={row + gap} fill="transparent" />
                        {/* the message is the waiting, so the grey carries the weight here and the ember steps back */}
                        <rect x={x(arr)} y={yy} width={Math.max(1.5, x(first) - x(arr))} height={row} rx={2} fill={pal.sub} opacity={0.85} />
                        <rect x={x(first) + 1} y={yy} width={Math.max(2, x(fin) - x(first) - 1)} height={row} rx={2} fill={cc.c1} opacity={on ? 1 : 0.5} />
                      </g>
                    );
                  })}
                </g>
              );
            })}
          </svg>
        )}
        {tip && hover && (
          <Tip x={Math.min(Math.max(x((tip[0] + tip[3]) / 2), 110), W - 110)} y={hover.pol === "static" ? head + hover.i * (row + gap) : panel + head + hover.i * (row + gap)}>
            <b>request {hover.i + 1}</b>
            <span className="nv-muted"> · {hover.pol}</span>
            <br />
            arrived at {tip[0].toFixed(2)} s · waited <b>{fmtMs((tip[2] - tip[0]) * 1000)}</b>
            <br />
            then {tip[4]} tokens in {(tip[3] - tip[2]).toFixed(2)} s
          </Tip>
        )}
      </div>
      <div className="lr-note">
        Over all {srv.requests} requests: median wait <b>{fmtMs(srv.static.ttft_p50_ms)}</b> with static batches,{" "}
        <b>{fmtMs(srv.continuous.ttft_p50_ms)}</b> with continuous batching; total output {nf(srv.static.throughput)} →{" "}
        {nf(srv.continuous.throughput)} tok/s.
      </div>
    </Fig>
  );
}

/** fig 17.4 (folded): where each request's KV cache actually lives. */
function PagedKv({ srv }: { srv: ServingRate }) {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const kv = srv.kv;
  const [si, setSi] = useState(Math.min(5, kv.snapshots.length - 1));
  const snap = kv.snapshots[si];
  const [sel, setSel] = useState<number>(kv.snapshots[Math.min(5, kv.snapshots.length - 1)].seqs[0]?.[0] ?? 0);
  const [ref, w] = useWidth<HTMLDivElement>();
  const owner = new Map<number, number>();
  for (const [id, , pages] of snap.seqs) for (const p of pages) owner.set(p, id);
  const shown = Math.max(...kv.snapshots.flatMap((s) => s.seqs.flatMap((q) => q[2]))) + 1;
  const seq = snap.seqs.find((q) => q[0] === sel) ?? snap.seqs[0];
  const W = Math.max(300, w);
  const cols = W < 480 ? 32 : 48;
  const cs = Math.floor(W / cols) - 2;
  const rows = Math.ceil(shown / cols);
  const H = rows * (cs + 2);
  const used = owner.size;
  return (
    <Fig
      n="17.4"
      title="Paged KV cache: a request's memory, scattered"
      prov="measured"
      caption={
        <>
          Snapshots of the continuous server's KV cache as it ran. Each square is one <G k="page" /> of 16 tokens' keys and
          values (the first {shown} of {nf(kv.pages)} pages). <Swatch c={cc.c2}>Blue squares</Swatch>: in use. Outlined: free.{" "}
          <Swatch c={cc.c1}>Ember squares</Swatch>: the chosen request's pages. The boxes above are its <b>block table</b>, 16
          tokens per box.
        </>
      }
      tryit={<>Drag through time and watch the pool fill and empty. Then select a request that started late, and notice where its pages are.</>}
    >
      <div className="lr-controls">
        <label className="lr-slider">
          <span className="k">step</span>
          <input
            type="range"
            min={0}
            max={kv.snapshots.length - 1}
            value={si}
            onChange={(e) => {
              const s = kv.snapshots[+e.target.value];
              setSi(+e.target.value);
              if (!s.seqs.some((q) => q[0] === sel) && s.seqs.length) setSel(s.seqs[0][0]);
            }}
          />
          <span className="v">{snap.iter}</span>
        </label>
        <span className="lr-small">
          {snap.seqs.length} requests live · {used} pages in use
        </span>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.15rem", marginBottom: "0.6rem" }} role="radiogroup" aria-label="request">
        {snap.seqs.map(([id, len]) => (
          <button key={id} role="radio" aria-checked={id === sel} className={`lr-chip btn${id === sel ? " on" : ""}`} onClick={() => setSel(id)} title={`${len} tokens`}>
            #{id}
          </button>
        ))}
      </div>
      {seq && (
        <div style={{ marginBottom: "0.7rem" }}>
          <div className="lr-small" style={{ marginBottom: "0.25rem" }}>
            request #{seq[0]}: {seq[1]} tokens in {seq[2].length} pages
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
            {seq[2].map((p, j) => (
              <span
                key={j}
                title={`tokens ${j * 16}–${Math.min(seq[1], (j + 1) * 16) - 1} → page ${p}`}
                style={{
                  minWidth: "2.1rem",
                  textAlign: "center",
                  fontSize: "var(--t-xs)",
                  padding: "0.12rem 0.2rem",
                  borderRadius: 4,
                  border: `1px solid ${cc.c1}`,
                  color: pal.fg,
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {p}
              </span>
            ))}
          </div>
        </div>
      )}
      <div ref={ref}>
        {w > 0 && (
          <svg width={W} height={H} role="img" aria-label="KV cache pages in the pool">
            {Array.from({ length: shown }, (_, p) => {
              const o = owner.get(p);
              const mine = o === sel;
              return (
                <rect
                  key={p}
                  x={(p % cols) * (cs + 2) + 0.5}
                  y={Math.floor(p / cols) * (cs + 2) + 0.5}
                  width={cs - 1}
                  height={cs - 1}
                  rx={2}
                  fill={mine ? cc.c1 : o != null ? cc.c2 : "transparent"}
                  opacity={mine ? 1 : o != null ? 0.42 : 1}
                  stroke={o == null ? pal.rule : "none"}
                  style={{ cursor: o != null ? "pointer" : undefined }}
                  onClick={() => o != null && setSel(o)}
                >
                  <title>{o != null ? `page ${p}: request #${o}` : `page ${p}: free`}</title>
                </rect>
              );
            })}
          </svg>
        )}
      </div>
      <div className="lr-note">
        On average {kv.stats.avg_live_seqs.toFixed(0)} requests were live, each holding about {kv.stats.avg_pages_per_seq.toFixed(0)} pages;
        only {(kv.stats.internal_fragmentation * 100).toFixed(1)}% of the memory handed out sat unused (the empty end of each
        request's last page). Static batching, which reserves room for the longest possible answer, peaked at {kv.staticPeak} pages; this
        peaked at {kv.contPeak}.
      </div>
    </Fig>
  );
}

export default function Batching() {
  const d = useLearnData();
  const srv = d?.serving["rate=16"];
  const q4 = d?.batching.series.find((s) => s.label === "q4_0");
  const [B, total, per] = d ? bestBatch(d) : [0, 0, 0];
  const sizes = d ? d.batching.series.filter((s) => s.label !== "q4_0").map((s) => s.points[s.points.length - 1][1]) : [];
  return (
    <>
      <h2 style={{ marginTop: "2.2rem" }}>The bottleneck: an idle factory</h2>
      <p>
        Serving one user, the GPU reads every weight for every token and does one multiply-add with it. In chapter 9's terms,
        the factory is almost always idle, waiting for the truck. A server has the same problem with many users.
      </p>
      <h2>The idea: one weight read, many users</h2>
      <p>
        Read each weight once and use it for many users. Put the next token of 64 different conversations through the model
        together, as a <G k="batch" />. Each weight read from memory now does 64 multiply-adds instead of 1. In chapter 9's
        terms, the same truckload now keeps the factory busy.
      </p>
      <p>This has a cost. More users per step means more arithmetic per step, so each step takes longer, and every user waits for every step.</p>
      {d ? <BatchCurve d={d} /> : <div className="lr-empty">loading…</div>}
      <p>
        Every server picks a point on this curve. While the GPU is still waiting on memory, extra users cost almost nothing.
        Past the crossover from chapter 9, the step is compute-bound. Total throughput then rises slowly and each user's speed
        falls.
      </p>
      <h2>The result: thousands of tokens per second in total</h2>
      <p>
        With 4-bit weights, one GPU went from {q4 ? nf(q4.points[0][1]) : "…"} tokens per second for one user to{" "}
        <span className="k">{nf(total)} tokens per second</span> in total for {B} users, about {Math.round(per)} each. That is
        still faster than anyone reads. At large batches every number format ends up close together
        {sizes.length ? ` (${nf(Math.min(...sizes, total))} to ${nf(Math.max(...sizes, total))} tok/s total)` : ""}. Once the job
        is compute-bound, smaller numbers stop helping.
      </p>
      <Climb upto={6} n="17.2" />
      <h2>Users don't arrive in neat batches</h2>
      <p>
        A simple server collects a batch, runs it until the <i>longest</i> answer is done, then takes the next batch.
        Finished requests hold their slots until the longest request ends, and new arrivals wait for the whole batch. A
        continuous server re-forms the batch at every step: finished requests leave, and waiting ones join. Here are the same
        GPU, model and requests, served both ways:
      </p>
      {srv ? <Timeline srv={srv} /> : <div className="lr-empty">loading…</div>}
      {srv && (
        <p>
          The median wait for the first token fell from <b>{fmtMs(srv.static.ttft_p50_ms)} to {fmtMs(srv.continuous.ttft_p50_ms)}</b>,
          and the total output rose {(srv.continuous.throughput / srv.static.throughput).toFixed(1)}×. The static server was busy
          the whole time and still made people wait seconds, because finished requests held many of its slots.
        </p>
      )}
      <Idea>Batching turns a memory-bound job into a compute-bound one, because each weight read serves many users. Continuous batching keeps the batch full without making anyone wait for a slot.</Idea>
      <p>
        Continuous batching needs one more thing: memory for KV caches that grow and finish at unpredictable times. Servers
        manage it the way an operating system does, with pages (the fold below).
      </p>
      <Deeper title="Paged attention: virtual memory for the KV cache">
        <p>
          Every live request has a <G k="KV cache" /> that grows by one token per step. If the server reserves room for the
          longest possible answer, most of that room is never used. So the server manages the KV cache like an operating
          system manages memory. It splits memory into fixed-size pages of 16 tokens. It gives a request a new page when the
          last one is full. A per-request <b>block table</b> maps each token position to its page. Attention reads through
          that table, so a request's pages can be anywhere in memory. Waste is at most the unfilled end of each request's last
          page.
        </p>
        {srv && <PagedKv srv={srv} />}
      </Deeper>
    </>
  );
}
