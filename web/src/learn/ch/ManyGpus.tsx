import { useState } from "react";
import { usePalette } from "../../ui/theme";
import { chartColors, Deeper, Fig, Idea, Seg, useFrame } from "../kit";

/** Ring all-reduce: N GPUs each own a vector split into N chunks; after 2(N−1) steps everyone has the sum. */
function Ring() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [n, setN] = useState(4);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const steps = 2 * (n - 1);
  useFrame(playing, (dt) => setT((v) => (v + dt * 0.9) % (steps + 1.5)));
  const step = Math.min(steps, Math.floor(t));
  const frac = t - Math.floor(t);
  const R = 92, cx = 150, cy = 120;
  const pos = (i: number) => [cx + R * Math.sin((2 * Math.PI * i) / n), cy - R * Math.cos((2 * Math.PI * i) / n)];
  // fraction of each GPU's chunks that are complete sums
  const reducePhase = step < n - 1;
  const doneChunks = step <= n - 1 ? (step === n - 1 ? 1 : 0) : 1 + (step - (n - 1));
  return (
    <Fig
      n="19.1"
      title="Passing dishes round a table"
      prov="toy"
      caption={
        <>
          GPUs in a ring, each holding its own partial result split into {n} chunks (the segments). At every step each GPU
          passes one chunk to its neighbour and adds in the one it receives. After {n - 1} steps every chunk has a complete sum
          somewhere; after {steps} steps everyone has every complete chunk. No link is ever idle, so the time hardly depends on
          how many GPUs sit at the table.
        </>
      }
      tryit={<>Change the number of GPUs: more steps, but each step sends a smaller chunk.</>}
    >
      <div className="lr-controls">
        <Seg options={[2, 4, 8].map((v) => ({ v, label: `${v} GPUs` }))} value={n} onChange={(v) => (setN(v), setT(0))} label="GPUs" />
        <button className="lr-btn" onClick={() => setPlaying((p) => !p)}>
          {playing ? "❚❚ pause" : "▶ play"}
        </button>
        <span className="lr-small">
          step {step} / {steps} · {reducePhase ? "adding up" : step < steps ? "sharing the sums" : "done: everyone has the total"}
        </span>
      </div>
      <svg viewBox="0 0 300 240" style={{ width: "100%", maxWidth: 340, display: "block", margin: "0 auto" }}>
        <circle cx={cx} cy={cy} r={R} fill="none" stroke={pal.rule} />
        {Array.from({ length: n }, (_, i) => {
          const [x, y] = pos(i);
          const [x2, y2] = pos((i + 1) % n);
          const moving = step < steps && playing;
          return (
            <g key={i}>
              {moving && <circle cx={x + (x2 - x) * frac} cy={y + (y2 - y) * frac} r={4} fill={cc.c3} />}
              <rect x={x - 22} y={y - 14} width={44} height={28} rx={6} fill={pal.dark ? "#262421" : "#fff"} stroke={pal.sub} />
              {Array.from({ length: n }, (_, c) => (
                <rect
                  key={c}
                  x={x - 18 + (36 / n) * c + 0.5}
                  y={y + 2}
                  width={36 / n - 1}
                  height={7}
                  rx={1}
                  fill={(c + i) % n < doneChunks ? cc.c1 : pal.rule}
                />
              ))}
              <text x={x} y={y - 3} textAnchor="middle" fontSize={9} fill={pal.fg}>
                GPU {i + 1}
              </text>
            </g>
          );
        })}
      </svg>
    </Fig>
  );
}

export default function ManyGpus() {
  return (
    <>
      <p>
        Everything so far fits on one chip. Big models don't. A 70-billion-weight model at 16 bits is 140 GB of weights, and
        one datacentre GPU holds 80. So the model is split across several GPUs, and now there's a new cost: they have to talk.
      </p>
      <p>Two ways to split it:</p>
      <ul className="lr-list">
        <li>
          <b>By layer</b> (pipeline): GPU 1 runs layers 1–40, GPU 2 runs 41–80. An assembly line: simple, but one token still
          has to go through every station in turn.
        </li>
        <li>
          <b>Within each layer</b> (tensor parallel): every GPU holds a slice of every matrix and does a slice of every
          multiply. Chefs working on one dish at once: fast, but after each layer they must combine their partial results.
        </li>
      </ul>
      <p>Combining partial results so every GPU ends up with the total is called an <b>all-reduce</b>, and the classic way to do it is a ring:</p>
      <Ring />
      <p>
        The same arithmetic from part two decides whether this is worth it. Inside a GPU, memory moves ~3,350 GB/s. Between
        GPUs in one server, the fast links move ~450. Between servers, ~50. Every byte that has to cross a link is slow next to
        a byte read locally.
      </p>
      <Idea>When a model doesn't fit on one chip, you split it, and pay for the conversation between the pieces. The bytes-vs-maths thinking from this course is exactly how you decide how to split.</Idea>
      <Deeper title="Why splitting GPT-2 small would make it slower">
        <p>
          Splitting GPT-2 within each layer would need 24 all-reduces per token, each only ~1.5 KB. Tiny messages are all
          latency (the cost of starting a conversation), no bandwidth. For a model this small, the talking would cost more than
          the work it saves. This is the subject of a later post in the series, run on real rented GPUs.
        </p>
      </Deeper>
    </>
  );
}
