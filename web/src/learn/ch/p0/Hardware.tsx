import { useState, type ReactElement } from "react";
import { usePalette } from "../../../ui/theme";
import { log } from "../../../lib/fmt";
import { useWidth } from "../../../ui/Figure";
import { chartColors, Deeper, Fig, G, Idea, Seg, Stat, Swatch, nf, useFrame, useOnScreen, T, PlayPause, usePlaying } from "../../kit";
import { linkTo } from "../../router";
import { MACHINES, SOURCES } from "./specs";

// ---- fig 0.3.1: few smart cores vs thousands of simple lanes ---------------------------------------

type ChipKey = "m4cpu" | "m4gpu" | "h100";
const CHIP: Record<ChipKey, { units: string; lanes: number; mem: string; detail: string }> = {
  m4cpu: { units: "10 cores (4 performance + 6 efficiency)", lanes: 10 * 4, mem: "LPDDR5X, shared", detail: "Each core is big and clever: deep pipelines, branch prediction, large caches, out-of-order execution. Built to make one thread fast." },
  m4gpu: { units: "10 GPU cores × 128 lanes", lanes: 1280, mem: "the same LPDDR5X pool", detail: "Each lane is simple: no branch prediction, tiny caches, runs in lockstep with 31 neighbours. Built to run the same arithmetic over lots of numbers." },
  h100: { units: "132 SMs × 128 lanes + 528 tensor cores", lanes: 16896, mem: "80 GB HBM3, stacked beside the die", detail: "Same idea, much bigger, plus tensor cores (hardware that multiplies small matrix tiles in one step) and HBM: DRAM stacks packaged millimetres from the die on a very wide bus. The 989 TFLOP/s is the tensor cores' dense bf16 rate; the 16,896 ordinary lanes manage ~67 TFLOP/s of f32, about 15× less. FP8 doubles the tensor-core rate again (~1,979 TFLOP/s)." },
};

function ChipPicture() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [k, setK] = useState<ChipKey>("m4gpu");
  const mach = MACHINES.find((m) => m.key === k)!;
  const info = CHIP[k];
  const [onScreen, seen] = useOnScreen<HTMLElement>();
  const [wref, w] = useWidth<HTMLDivElement>();
  const narrow = w > 0 && w < 520; // phones get a vertical layout with larger type, not a shrunken desktop one
  const [t, setT] = useState(0);
  const [playing, setPlaying] = usePlaying();
  useFrame(seen && playing, (dt) => setT((v) => (v + dt) % 1000));
  const panel = pal.dark ? "#262421" : "#ffffff";
  // bus: thickness and packet rate scale with log bandwidth
  const lb = Math.log10(mach.bw); // ~2 .. 3.5
  const busW = 3 + (lb - 1.9) * 9;
  const rate = 0.6 + (lb - 1.9) * 2.2;
  const pkts = Math.round(rate * 4);
  const fs = narrow ? 1.45 : 1; // label scale inside the die on phones
  // compute units, drawn in a 370 × 196 box at the origin and placed by a transform
  const units: ReactElement[] = [];
  if (k === "m4cpu") {
    for (let i = 0; i < 4; i++) {
      const x = 16 + i * 88, y = 28;
      units.push(
        <g key={`p${i}`}>
          <rect x={x} y={y} width={78} height={66} rx={6} fill={cc.c1} fillOpacity={0.16} stroke={cc.c1} />
          <rect x={x + 6} y={y + 42} width={66} height={16} rx={3} fill={cc.c2} fillOpacity={0.25} />
          <text x={x + 39} y={y + 24} textAnchor="middle" fontSize={T.xs * fs} fill={pal.fg}>
            P-core
          </text>
          <text x={x + 39} y={y + 54} textAnchor="middle" fontSize={8.5 * Math.min(fs, 1.3)} /* sized to its box; renders at ≥11px */ fill={pal.sub}>
            cache
          </text>
        </g>,
      );
    }
    for (let i = 0; i < 6; i++) {
      const x = 16 + i * 58, y = 108;
      units.push(
        <g key={`e${i}`}>
          <rect x={x} y={y} width={50} height={40} rx={5} fill={cc.c1} fillOpacity={0.1} stroke={cc.c1} strokeOpacity={0.7} />
          <text x={x + 25} y={y + 25} textAnchor="middle" fontSize={8.5 * Math.min(fs, 1.35)} /* sized to its 50-unit box; renders at ≥11px */ fill={pal.sub}>
            E-core
          </text>
        </g>,
      );
    }
    units.push(
      <g key="l2">
        <rect x={16} y={158} width={338} height={26} rx={4} fill={cc.c2} fillOpacity={0.2} />
        <text x={185} y={175 + (fs - 1) * 3} textAnchor="middle" fontSize={T.xs * fs} fill={pal.sub}>
          large shared caches (16 MB+)
        </text>
      </g>,
    );
  } else if (k === "m4gpu") {
    for (let c = 0; c < 10; c++) {
      const cx = 14 + (c % 5) * 70, cy = 26 + Math.floor(c / 5) * 84;
      units.push(<rect key={`c${c}`} x={cx} y={cy} width={64} height={76} rx={5} fill="none" stroke={pal.rule} />);
      for (let i = 0; i < 128; i++) {
        units.push(<rect key={`${c}-${i}`} x={cx + 4 + (i % 8) * 7.1} y={cy + 4 + Math.floor(i / 8) * 4.4} width={5.4} height={3.2} rx={0.6} fill={cc.c1} fillOpacity={0.75} />);
      }
    }
  } else {
    for (let s = 0; s < 132; s++) {
      const cx = 12 + (s % 12) * 29.5, cy = 24 + Math.floor(s / 12) * 15.2;
      units.push(<rect key={s} x={cx} y={cy} width={26} height={12} rx={1.5} fill={cc.c1} fillOpacity={0.6} />);
    }
  }
  const bwLabel = mach.bw >= 1000 ? `${nf(mach.bw / 1000, 2)} TB/s` : `${mach.bw} GB/s`;
  const srcLabel = mach.src === "measured" ? "measured" : "spec";
  const memH = k === "h100" ? 150 : 110;
  const desktop = (
    <svg viewBox="0 0 640 240" style={{ width: "100%" }} role="img" aria-label={`${mach.name}: ${info.units}, ${mach.bw} GB/s`}>
      <rect x={14} y={120 - memH / 2} width={130} height={memH} rx={8} fill={cc.c2} fillOpacity={0.12} stroke={cc.c2} />
      {Array.from({ length: k === "h100" ? 5 : 2 }, (_, i) => (
        <rect key={i} x={26} y={120 - memH / 2 + 12 + i * (k === "h100" ? 26 : 34)} width={106} height={k === "h100" ? 18 : 26} rx={3} fill={cc.c2} fillOpacity={0.3} />
      ))}
      <text x={79} y={120 + memH / 2 + 17} textAnchor="middle" fontSize={T.xs} fill={pal.sub}>
        {mach.cap} GB memory
      </text>
      <rect x={146} y={120 - busW / 2} width={102} height={busW} fill={cc.c2} fillOpacity={0.35} />
      {Array.from({ length: pkts }, (_, i) => {
        const p = ((t * rate + i / pkts) % 1 + 1) % 1;
        return <rect key={i} x={146 + p * 90} y={120 - Math.max(2, busW / 2 - 1)} width={8} height={Math.max(4, busW - 2)} rx={1.5} fill={cc.c2} />;
      })}
      <text x={198} y={120 - busW / 2 - 9} textAnchor="middle" fontSize={T.xs} fontWeight={700} fill={pal.fg}>
        {bwLabel}
      </text>
      <text x={198} y={120 + busW / 2 + 17} textAnchor="middle" fontSize={T.xs} fill={pal.muted}>
        {srcLabel}
      </text>
      <rect x={250} y={22} width={370} height={196} rx={12} fill={panel} stroke={pal.sub} />
      <text x={262} y={39} fontSize={T.xs} fill={pal.muted}>
        {info.units}
      </text>
      <g transform="translate(250,22)">{units}</g>
    </svg>
  );
  // phones: memory on top, the bus running down into the chip, the chip full width underneath
  const S = 328 / 370;
  const busTop = 92, busBot = 156;
  const narrowSvg = (
    <svg viewBox="0 0 340 340" style={{ width: "100%" }} role="img" aria-label={`${mach.name}: ${info.units}, ${mach.bw} GB/s`}>
      <rect x={70} y={8} width={200} height={70} rx={8} fill={cc.c2} fillOpacity={0.12} stroke={cc.c2} />
      {Array.from({ length: k === "h100" ? 5 : 2 }, (_, i) => (
        <rect key={i} x={82 + i * (k === "h100" ? 36 : 90)} y={18} width={k === "h100" ? 30 : 84} height={34} rx={3} fill={cc.c2} fillOpacity={0.3} />
      ))}
      <text x={170} y={69} textAnchor="middle" fontSize={T.sm} fill={pal.sub}>
        {mach.cap} GB memory
      </text>
      <rect x={170 - busW / 2} y={busTop - 14} width={busW} height={busBot - busTop + 14} fill={cc.c2} fillOpacity={0.35} />
      {Array.from({ length: pkts }, (_, i) => {
        const p = ((t * rate + i / pkts) % 1 + 1) % 1;
        return <rect key={i} x={170 - Math.max(2, busW / 2 - 1)} y={busTop - 12 + p * (busBot - busTop)} width={Math.max(4, busW - 2)} height={7} rx={1.5} fill={cc.c2} />;
      })}
      <text x={170 + busW / 2 + 10} y={120} fontSize={T.sm} fontWeight={700} fill={pal.fg}>
        {bwLabel}
      </text>
      <text x={170 + busW / 2 + 10} y={137} fontSize={T.xs} fill={pal.muted}>
        {srcLabel}
      </text>
      <rect x={6} y={busBot} width={328} height={196 * S + 4} rx={12} fill={panel} stroke={pal.sub} />
      <g transform={`translate(6,${busBot + 2}) scale(${S})`}>{units}</g>
    </svg>
  );
  return (
    <Fig
      figRef={onScreen}
      aside={<PlayPause playing={playing} onToggle={() => setPlaying((p) => !p)} />}
      n="0.3.1"
      title="A CPU is a few clever cores; a GPU is thousands of simple lanes"
      prov="diagram"
      caption={
        <>
          {narrow ? "Top" : "Left"}, <Swatch c={cc.c2}>blue</Swatch>: memory, and the bus to the chip (thicker = more
          bandwidth). {narrow ? "Below" : "Right"}, <Swatch c={cc.c1}>ember squares</Swatch>: compute units, drawn to count
          rather than scale. FLOPs per byte is how much arithmetic the chip can do for each byte it reads.
        </>
      }
      tryit={<>Switch between the three chips. Notice how much the bus changes compared with the compute units.</>}
      source={SOURCES}
    >
      <div className="lr-controls">
        <Seg
          options={[
            { v: "m4cpu", label: "M4 CPU" },
            { v: "m4gpu", label: "M4 GPU" },
            { v: "h100", label: "H100" },
          ]}
          value={k}
          onChange={(v) => setK(v as ChipKey)}
          label="chip"
        />
      </div>
      <div ref={wref}>
        {w > 0 && (narrow ? narrowSvg : desktop)}
        {narrow && <div className="p0-diagram-k lr-small">the chip: {info.units}</div>}
      </div>
      <div className="lr-stats">
        <Stat k="parallel lanes" v={nf(info.lanes)} unit={k === "m4cpu" ? "(4-wide SIMD per core)" : undefined} />
        <Stat k="peak arithmetic" v={mach.tflops >= 10 ? nf(mach.tflops) : nf(mach.tflops, 1)} unit={k === "h100" ? "TFLOP/s bf16, tensor cores" : "TFLOP/s f32"} />
        <Stat k="FLOPs per byte read" v={nf((mach.tflops * 1e12) / (mach.bw * 1e9))} accent />
      </div>
      <p className="lr-note">{info.detail}</p>
    </Fig>
  );
}

// ---- fig 0.3.2: where the bytes come from ------------------------------------------------------------

const LADDER: { name: string; gbs: number; kind: "mem" | "link"; src: string; cap?: string }[] = [
  { name: "H100 HBM3", gbs: 3350, kind: "mem", src: "NVIDIA datasheet", cap: "80 GB" },
  { name: "NVLink 4 (GPU ↔ GPU, each way)", gbs: 450, kind: "link", src: "NVIDIA: 900 GB/s total per GPU" },
  { name: "M4 GPU reading memory", gbs: 105, kind: "mem", src: "measured, docs/LOG.md", cap: "24 GB shared" },
  { name: "M4 CPU, all 10 cores", gbs: 98, kind: "mem", src: "measured, docs/LOG.md", cap: "24 GB shared" },
  { name: "PCIe 5 ×16 (CPU ↔ GPU, each way)", gbs: 64, kind: "link", src: "PCI-SIG spec, ~64 GB/s" },
  { name: "one M4 core", gbs: 57, kind: "mem", src: "measured, docs/LOG.md" },
  { name: "InfiniBand NDR (server ↔ server)", gbs: 50, kind: "link", src: "400 Gb/s per port" },
  { name: "100 GbE network", gbs: 12.5, kind: "link", src: "100 Gb/s" },
];

function BandwidthLadder() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [hi, setHi] = useState<number | null>(null);
  const x = log(5, 5000, 0, 100);
  return (
    <Fig
      n="0.3.2"
      title="Inside a chip, bytes are fast; between chips, they're slow"
      prov="external"
      caption={
        <>
          Bandwidth on a log scale (each gridline 10×). <Swatch c={cc.c2}>Blue bars</Swatch>: a processor reading its own
          memory. <Swatch c={cc.c3}>Olive bars</Swatch>: links between chips and machines.
        </>
      }
      tryit={<>Select a row to see its source. Notice how far the links sit below local memory.</>}
      source="Measured rows: docs/LOG.md (M4 Mac mini, 24 GB). External rows: NVIDIA H100 / NVLink datasheets, PCIe 5 and InfiniBand NDR line rates."
    >
      <div className="p0-legend lr-small">
        <span>
          <i style={{ background: cc.c2 }} />
          memory
        </span>
        <span>
          <i style={{ background: cc.c3 }} />
          link between chips
        </span>
      </div>
      <div className="p0-ladder">
        {LADDER.map((r, i) => (
          <div key={r.name} className={`p0-ladder-row${hi === i ? " hi" : ""}`} onPointerEnter={() => setHi(i)} onPointerLeave={() => setHi(null)} onClick={() => setHi(i)}>
            <span className="nm">{r.name}</span>
            <span className="track">
              {[10, 100, 1000].map((g) => (
                <i key={g} className="grid" style={{ left: `${x(g)}%` }} />
              ))}
              <i className="bar" style={{ width: `${x(r.gbs)}%`, background: r.kind === "mem" ? cc.c2 : cc.c3 }} />
            </span>
            <span className="v">{r.gbs >= 1000 ? `${nf(r.gbs / 1000, 2)} TB/s` : `${nf(r.gbs, r.gbs < 20 ? 1 : 0)} GB/s`}</span>
          </div>
        ))}
        <div className="p0-ceil-axis lr-small" style={{ marginLeft: 0 }}>
          {[10, 100, 1000].map((g) => (
            <span key={g} style={{ left: `${x(g)}%` }}>
              {g >= 1000 ? "1 TB/s" : `${g} GB/s`}
            </span>
          ))}
        </div>
      </div>
      <p className="lr-note" style={{ minHeight: "1.4em" }}>
        {hi != null ? (
          <>
            <b>{LADDER[hi].name}</b>: {LADDER[hi].src}
            {LADDER[hi].cap ? ` · capacity ${LADDER[hi].cap}` : ""}
          </>
        ) : (
          " "
        )}
      </p>
    </Fig>
  );
}

// ---- fig 0.3.3: splitting a model that doesn't fit ------------------------------------------------------

const LAYERS = 80; // a 70B model (Llama-3-70B shape)
const W70 = 140e9; // bf16
const LAT = { nvlink: 8e-6, network: 30e-6 }; // per small all-reduce, order of magnitude (estimate)

function SplitCost() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [link, setLink] = useState<"nvlink" | "network">("nvlink");
  const rows = [1, 2, 4, 8].map((n) => {
    const fits = W70 / n <= 80e9 * 0.95;
    const mem = W70 / n / 3.35e12; // each GPU reads its slice in parallel
    const talk = n === 1 ? 0 : 2 * LAYERS * LAT[link]; // two all-reduces per layer
    return { n, fits, mem, talk, total: mem + talk };
  });
  const mx = Math.max(...rows.filter((r) => r.fits).map((r) => r.total));
  return (
    <Fig
      n="0.3.3"
      title="Splitting a model buys bandwidth and capacity, and costs communication"
      prov="estimate"
      caption={
        <>
          A 70B model in bf16 (140 GB) split across H100s with tensor parallelism. Each GPU holds and reads a slice of every
          matrix (<Swatch c={cc.c2}>blue bars</Swatch>: time reading weights per token). After every attention and
          feed-forward step the GPUs all-reduce their partial results (<Swatch c={cc.c3}>olive bars</Swatch>: time
          communicating). The messages are 16 KB, so that time is almost all latency.
        </>
      }
      tryit={<>Switch the link to a network between servers. Notice what happens to the olive share as you add GPUs.</>}
      source="80 layers × 2 all-reduces per token; ~8 µs per small all-reduce over NVLink and ~30 µs over InfiniBand are order-of-magnitude figures for NCCL-style collectives (estimate). HBM 3.35 TB/s per H100 (datasheet)."
    >
      <div className="lr-controls">
        <Seg
          options={[
            { v: "nvlink", label: "NVLink (one server)" },
            { v: "network", label: "network (servers)" },
          ]}
          value={link}
          onChange={(v) => setLink(v as "nvlink" | "network")}
          label="link"
        />
      </div>
      <div className="p0-split">
        {rows.map((r) => (
          <div key={r.n} className="p0-split-row">
            <span className="nm">
              {r.n} GPU{r.n > 1 ? "s" : ""}
            </span>
            {r.fits ? (
              <>
                <span className="track">
                  <i style={{ width: `${(r.mem / mx) * 100}%`, background: cc.c2 }} />
                  {r.talk > 0 && <i style={{ width: `${(r.talk / mx) * 100}%`, background: cc.c3 }} />}
                </span>
                <span className="v">
                  {(r.total * 1000).toFixed(1)} ms · <b>{Math.round(1 / r.total)}</b> tok/s
                  <span className="lr-small"> · {Math.round((r.talk / r.total) * 100)}% talking</span>
                </span>
              </>
            ) : (
              <>
                <span className="track nofit">
                  <span className="nofit-k">
                    <span>doesn't fit: 140 GB of weights, 80 GB per GPU</span>
                  </span>
                </span>
                <span className="v">–</span>
              </>
            )}
          </div>
        ))}
      </div>
    </Fig>
  );
}

export default function Hardware() {
  return (
    <>
      <p>
        This chapter is the minimum hardware picture for inference. It covers why GPUs, why memory is the wall, and why more
        than one GPU is hard. Keep three numbers in mind: <b>bandwidth</b>, <b>capacity</b> and <b>link speed</b>.
      </p>
      <h2>Few clever cores, or thousands of simple ones</h2>
      <p>
        A CPU spends its transistors making one thread fast. A <G k="GPU" /> spends them on lanes, thousands of simple
        arithmetic units running the same instruction over different numbers. It is SIMD taken to an extreme. A matrix
        multiply suits a GPU well, because it is millions of independent multiply-adds.
      </p>
      <ChipPicture />
      <p>
        Now look at the last number. More lanes only help when there's arithmetic to do, and decode has about 1 FLOP per
        byte. What makes datacentre GPUs fast at decode is the memory beside them. <G k="HBM" /> is stacks of DRAM on a bus
        thousands of bits wide. An H100 reads about 30× faster than the M4, so the same model decodes about 30× faster.
      </p>

      <h2>Bandwidth, by where the bytes live</h2>
      <p>
        Apple silicon has <G k="unified memory" />: CPU and GPU share one pool, so nothing is copied between them. That
        is why the M4 can run both on the same weights. A datacentre GPU has its own HBM; the weights cross PCIe once at load
        time and then stay put.
      </p>
      <BandwidthLadder />
      <h2>When a CPU is the better choice</h2>
      <p>
        Look at the two M4 rows. The CPU and GPU read the same memory at 98 and 105 GB/s, so for one user's decode their
        speed limits are nearly equal. With 4-bit weights, our engine reached 497 tokens per second on the CPU and 817 on
        the GPU (pipelined, chapter 16). Most of that gap came from the GPU waiting less, not from its lanes. Prefill differs
        more, 2,415 against 6,800 tokens per second, because reading a prompt is arithmetic-heavy.
      </p>
      <p>
        The rule of thumb is <b>decode speed follows bandwidth; prefill and batch speed follow FLOPs</b>. A CPU is a fine
        choice for a small model, one user and shared memory. Examples are laptops, phones, edge boxes, or a big-RAM server
        running a model no single GPU can hold. A GPU wins once there's arithmetic to share out (long prompts, many users),
        or when it brings much faster memory, like an H100's HBM.
      </p>
      <p>
        Capacity is the other wall: 80 GB per H100, 24 GB on the M4. A 70B model in bf16 needs 140 GB before a single
        user's KV cache. When it doesn't fit, you either shrink it (quantise) or split it.
      </p>

      <h2>Why splitting across GPUs costs time</h2>
      <p>
        Splitting a model is sharding, with a twist. The shards must exchange partial results{" "}
        <b>on every layer of every token</b>, a synchronous barrier hundreds of times per token. Each exchange is an{" "}
        <G k="all-reduce" />, so the link (<G k="NVLink" /> inside a server, the network between servers) is on the
        critical path.
      </p>
      <SplitCost />
      <p>
        This is Amdahl's law. The per-token communication is a serial part that more GPUs don't shrink. So big models are
        split within one NVLink-connected server where possible. Across servers they are split by layers (pipeline) or by
        request (replicas), where the traffic is lighter. Chapter 19 animates the all-reduce.
      </p>
      <Idea>
        GPUs win at decode mainly because of their memory bandwidth (HBM); their lanes win prefill and big batches. Memory
        capacity decides what fits. Links between chips are 7–70× slower than local memory, so splitting a model buys
        capacity and bandwidth at the cost of communication time.
      </Idea>
      <Deeper title="What about TPUs, Groq, Cerebras, and phones?">
        <p>
          Same physics, different trades. TPUs are matrix engines with HBM, linked in big pods with fast custom
          interconnect. Groq and Cerebras keep weights in on-chip SRAM (far faster than HBM, far smaller), so a model spans
          many chips. Phones and laptops run small quantised models from shared DRAM at 50 to 120 GB/s. That is why a 3B to
          8B model in 4 bits suits on-device use. In every case, work out bytes per token ÷ bandwidth first.
        </p>
      </Deeper>
      <p className="p0-next-hint">
        Next: A Rosetta stone. Or see the{" "}
        <a href="/learn/many-gpus" onClick={linkTo("/learn/many-gpus")}>
          ring all-reduce
        </a>{" "}
        now.
      </p>
    </>
  );
}
