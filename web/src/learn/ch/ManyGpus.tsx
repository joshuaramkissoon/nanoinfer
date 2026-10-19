import { useState } from "react";
import { usePalette } from "../../ui/theme";
import { chartColors, Deeper, Fig, Idea, Seg, Swatch, useFrame, useOnScreen, T, PlayPause, usePlaying } from "../kit";
import { linkTo } from "../router";

/** Ring all-reduce: N GPUs each own a vector split into N chunks; after 2(N−1) steps everyone has the sum. */
function Ring() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [n, setN] = useState(4);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = usePlaying();
  const steps = 2 * (n - 1);
  const [onScreen, vis] = useOnScreen<HTMLElement>();
  useFrame(vis && playing, (dt) => setT((v) => (v + dt * 0.9) % (steps + 1.5)));
  const step = Math.min(steps, Math.floor(t));
  const frac = t - Math.floor(t);
  const R = 100, cx = 160, cy = 130;
  const pos = (i: number) => [cx + R * Math.sin((2 * Math.PI * i) / n), cy - R * Math.cos((2 * Math.PI * i) / n)];
  // fraction of each GPU's chunks that are complete sums
  const reducePhase = step < n - 1;
  const doneChunks = step <= n - 1 ? (step === n - 1 ? 1 : 0) : 1 + (step - (n - 1));
  return (
    <Fig
      figRef={onScreen}
      n="19.1"
      title="Ring all-reduce: each GPU passes one chunk per step"
      prov="toy"
      caption={
        <>
          GPUs in a ring, each holding its own partial result split into {n} chunks (the segments). At every step each GPU
          passes one chunk to its neighbour as an <Swatch c={cc.c3}>olive dot</Swatch> over the link, and adds in the one it
          receives. A segment turns <Swatch c={cc.c1}>ember</Swatch> once it holds the complete sum. Grey segments are partial.
        </>
      }
      tryit={<>Switch between 2, 4 and 8 GPUs. Notice how the number of steps and the size of each chunk change together.</>}
    >
      <div className="lr-controls">
        <Seg options={[2, 4, 8].map((v) => ({ v, label: `${v} GPUs` }))} value={n} onChange={(v) => (setN(v), setT(0))} label="GPUs" />
        <PlayPause playing={playing} onToggle={() => setPlaying((p) => !p)} />
        <span className="lr-small">
          step {step} / {steps} · {reducePhase ? "adding up" : step < steps ? "sharing the sums" : "done: everyone has the total"}
        </span>
      </div>
      <svg viewBox="0 0 320 260" style={{ width: "100%", maxWidth: 420, display: "block", margin: "0 auto" }} role="img" aria-label={`ring all-reduce over ${n} GPUs`}>
        <circle cx={cx} cy={cy} r={R} fill="none" stroke={cc.c3} strokeOpacity={0.55} strokeWidth={1.5} />
        {Array.from({ length: n }, (_, i) => {
          const [x, y] = pos(i);
          const [x2, y2] = pos((i + 1) % n);
          const moving = step < steps && playing;
          return (
            <g key={i}>
              {moving && <circle cx={x + (x2 - x) * frac} cy={y + (y2 - y) * frac} r={5} fill={cc.c3} />}
              <rect x={x - 29} y={y - 17} width={58} height={34} rx={6} fill={pal.dark ? "#262421" : "#fff"} stroke={pal.sub} />
              {Array.from({ length: n }, (_, c) => (
                <rect
                  key={c}
                  x={x - 24 + (48 / n) * c + 0.5}
                  y={y + 3}
                  width={48 / n - 1.5}
                  height={8}
                  rx={1}
                  fill={(c + i) % n < doneChunks ? cc.c1 : pal.rule}
                />
              ))}
              <text x={x} y={y - 3} textAnchor="middle" fontSize={T.xs} fill={pal.fg}>
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
        Everything so far fits on one chip. Big models don't. A 70B model at 16 bits is 140 GB of weights, and one H100 holds
        80 GB. So the model is split across several GPUs, and that adds a new cost: the GPUs have to communicate.
      </p>
      <p>Three ways to split it:</p>
      <ul className="lr-list">
        <li>
          <b>By layer</b> (pipeline parallelism): GPU 1 runs layers 1 to 40, and GPU 2 runs 41 to 80. It is simple, but one
          token still goes through every pipeline stage in turn.
        </li>
        <li>
          <b>Within each layer</b> (tensor parallelism): every GPU holds a slice of every matrix and does a slice of every
          multiply. It is fast, but after each layer the GPUs must combine their partial results.
        </li>
        <li>
          <b>By expert</b> (expert parallelism), for mixture-of-experts models: each GPU holds a few whole experts. Each token
          goes to the GPUs that hold its chosen experts. This is key-based routing to shards, hot keys included.
        </li>
      </ul>
      <p>Combining partial results so every GPU ends up with the total is called an <b>all-reduce</b>, and the classic way to do it is a ring:</p>
      <Ring />
      <p>
        The same arithmetic from part 2 decides whether this is worth it. Inside an H100, memory moves about 3.35 TB/s.
        Between GPUs in one server, NVLink moves about 450 GB/s each way. Between servers, about 50 GB/s. Every byte that
        crosses a link is slow next to a byte read locally.
      </p>
      <p>
        The third split needs a word of explanation. A <b>mixture-of-experts</b> (MoE) model replaces each layer's
        feed-forward network with many smaller ones (experts), say 8 or 256. A small router sends each token to a few of
        them. Each token does the arithmetic of a small model, but every expert must stay in memory, and a busy batch reads
        most of them. So MoE adds weights without adding FLOPs per token, but it does add bytes. The{" "}
        <a href="/learn/field-guide#moe" onClick={linkTo("/learn/field-guide#moe")}>
          MoE lab in the field guide
        </a>{" "}
        lets you route a batch and watch what gets read.
      </p>
      <Idea>When a model doesn't fit on one chip, you split it and pay for the communication between the pieces. Bytes against arithmetic, as in the rest of the course, decides how to split.</Idea>
      <Deeper title="Why splitting GPT-2 small would make it slower">
        <p>
          Splitting GPT-2 within each layer would need 24 all-reduces per token, each only about 1.5 KB. Tiny messages cost
          latency (the fixed cost of sending anything), not bandwidth. For a model this small, the communication would cost
          more than the work it saves.
        </p>
      </Deeper>
    </>
  );
}
