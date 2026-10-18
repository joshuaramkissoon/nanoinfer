import { useState, type ReactElement } from "react";
import { useInView } from "../../../ui/Figure";
import { usePalette } from "../../../ui/theme";
import { log } from "../../../lib/fmt";
import { chartColors, Deeper, Fig, G, Idea, Seg, Stat, Swatch, nf, useFrame } from "../../kit";
import { linkTo } from "../../router";
import { MACHINES, SOURCES } from "./specs";

// ---- fig 0.3.1: few smart cores vs thousands of simple lanes ---------------------------------------

type ChipKey = "m4cpu" | "m4gpu" | "h100";
const CHIP: Record<ChipKey, { units: string; lanes: number; mem: string; detail: string }> = {
  m4cpu: { units: "10 cores (4 performance + 6 efficiency)", lanes: 10 * 4, mem: "LPDDR5X, shared", detail: "Each core is big and clever: deep pipelines, branch prediction, large caches, out-of-order execution. Built to make one thread fast." },
  m4gpu: { units: "10 GPU cores × 128 lanes", lanes: 1280, mem: "the same LPDDR5X pool", detail: "Each lane is simple: no branch prediction, tiny caches, runs in lockstep with 31 neighbours. Built to run the same arithmetic over lots of numbers." },
  h100: { units: "132 SMs × 128 lanes + 528 tensor cores", lanes: 16896, mem: "80 GB HBM3, stacked beside the die", detail: "Same idea, much bigger, plus tensor cores (hardware that multiplies small matrix tiles in one step) and HBM: DRAM stacks packaged millimetres from the die on a very wide bus." },
};

function ChipPicture() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [k, setK] = useState<ChipKey>("m4gpu");
  const mach = MACHINES.find((m) => m.key === k)!;
  const info = CHIP[k];
  const [vref, seen] = useInView<HTMLDivElement>();
  const reduce = typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const [t, setT] = useState(0);
  useFrame(seen && !reduce, (dt) => setT((v) => (v + dt) % 1000));
  const panel = pal.dark ? "#262421" : "#ffffff";
  // bus: thickness and packet rate scale with log bandwidth
  const lb = Math.log10(mach.bw); // ~2 .. 3.5
  const busW = 3 + (lb - 1.9) * 9;
  const rate = 0.6 + (lb - 1.9) * 2.2;
  const DIE = { x: 250, y: 22, w: 370, h: 196 };
  const units: ReactElement[] = [];
  if (k === "m4cpu") {
    for (let i = 0; i < 4; i++) {
      const x = DIE.x + 16 + i * 88, y = DIE.y + 28;
      units.push(
        <g key={`p${i}`}>
          <rect x={x} y={y} width={78} height={66} rx={6} fill={cc.c1} fillOpacity={0.16} stroke={cc.c1} />
          <rect x={x + 6} y={y + 42} width={66} height={16} rx={3} fill={cc.c2} fillOpacity={0.25} />
          <text x={x + 39} y={y + 22} textAnchor="middle" fontSize={9.5} fill={pal.fg}>
            P-core
          </text>
          <text x={x + 39} y={y + 54} textAnchor="middle" fontSize={8.5} fill={pal.sub}>
            cache
          </text>
        </g>,
      );
    }
    for (let i = 0; i < 6; i++) {
      const x = DIE.x + 16 + i * 58, y = DIE.y + 108;
      units.push(
        <g key={`e${i}`}>
          <rect x={x} y={y} width={50} height={40} rx={5} fill={cc.c1} fillOpacity={0.1} stroke={cc.c1} strokeOpacity={0.7} />
          <text x={x + 25} y={y + 24} textAnchor="middle" fontSize={8.5} fill={pal.sub}>
            E-core
          </text>
        </g>,
      );
    }
    units.push(
      <g key="l2">
        <rect x={DIE.x + 16} y={DIE.y + 158} width={338} height={26} rx={4} fill={cc.c2} fillOpacity={0.2} />
        <text x={DIE.x + 185} y={DIE.y + 175} textAnchor="middle" fontSize={9} fill={pal.sub}>
          large shared caches (16 MB+)
        </text>
      </g>,
    );
  } else if (k === "m4gpu") {
    for (let c = 0; c < 10; c++) {
      const cx = DIE.x + 14 + (c % 5) * 70, cy = DIE.y + 26 + Math.floor(c / 5) * 84;
      units.push(<rect key={`c${c}`} x={cx} y={cy} width={64} height={76} rx={5} fill="none" stroke={pal.rule} />);
      for (let i = 0; i < 128; i++) {
        units.push(<rect key={`${c}-${i}`} x={cx + 4 + (i % 8) * 7.1} y={cy + 4 + Math.floor(i / 8) * 4.4} width={5.4} height={3.2} rx={0.6} fill={cc.c1} fillOpacity={0.75} />);
      }
    }
  } else {
    for (let s = 0; s < 132; s++) {
      const cx = DIE.x + 12 + (s % 12) * 29.5, cy = DIE.y + 24 + Math.floor(s / 12) * 15.2;
      units.push(<rect key={s} x={cx} y={cy} width={26} height={12} rx={1.5} fill={cc.c1} fillOpacity={0.6} />);
    }
  }
  const memH = k === "h100" ? 150 : 110;
  const pkts = Math.round(rate * 4);
  return (
    <Fig
      n="0.3.1"
      title="A CPU is a few clever cores; a GPU is thousands of simple lanes"
      prov="diagram"
      caption={
        <>
          Left, <Swatch c={cc.c2}>blue</Swatch>: memory, and the bus to the chip (thicker and busier = more bandwidth).
          Right, <Swatch c={cc.c1}>ember</Swatch>: compute units, drawn to count, not to scale. The FLOPs-per-byte number is
          how much arithmetic the chip can do per byte it reads: the gap decode can't fill.
        </>
      }
      tryit={<>Flip between the three. The M4's CPU and GPU share one memory, so their buses barely differ; the H100's is 30× wider.</>}
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
      <div ref={vref}>
        <svg viewBox="0 0 640 240" style={{ width: "100%" }} role="img" aria-label={`${mach.name}: ${info.units}, ${mach.bw} GB/s`}>
          <rect x={14} y={120 - memH / 2} width={130} height={memH} rx={8} fill={cc.c2} fillOpacity={0.12} stroke={cc.c2} />
          {Array.from({ length: k === "h100" ? 5 : 2 }, (_, i) => (
            <rect key={i} x={26} y={120 - memH / 2 + 12 + i * (k === "h100" ? 26 : 34)} width={106} height={k === "h100" ? 18 : 26} rx={3} fill={cc.c2} fillOpacity={0.3} />
          ))}
          <text x={79} y={120 + memH / 2 + 16} textAnchor="middle" fontSize={10} fill={pal.sub}>
            {mach.cap} GB memory
          </text>
          {/* bus */}
          <rect x={146} y={120 - busW / 2} width={DIE.x - 148} height={busW} fill={cc.c2} fillOpacity={0.35} />
          {Array.from({ length: pkts }, (_, i) => {
            const p = ((t * rate + i / pkts) % 1 + 1) % 1;
            return <rect key={i} x={146 + p * (DIE.x - 160)} y={120 - Math.max(2, busW / 2 - 1)} width={8} height={Math.max(4, busW - 2)} rx={1.5} fill={cc.c2} />;
          })}
          <text x={(146 + DIE.x) / 2} y={120 - busW / 2 - 8} textAnchor="middle" fontSize={10.5} fontWeight={700} fill={pal.fg}>
            {mach.bw >= 1000 ? `${nf(mach.bw / 1000, 2)} TB/s` : `${mach.bw} GB/s`}
          </text>
          <text x={(146 + DIE.x) / 2} y={120 + busW / 2 + 16} textAnchor="middle" fontSize={9.5} fill={pal.muted}>
            {mach.src === "measured" ? "measured" : "spec"}
          </text>
          {/* die */}
          <rect x={DIE.x} y={DIE.y} width={DIE.w} height={DIE.h} rx={12} fill={panel} stroke={pal.sub} />
          <text x={DIE.x + 12} y={DIE.y + 16} fontSize={9.5} fill={pal.muted}>
            {info.units}
          </text>
          {units}
        </svg>
      </div>
      <div className="lr-stats">
        <Stat k="parallel lanes" v={nf(info.lanes)} unit={k === "m4cpu" ? "(4-wide SIMD per core)" : undefined} />
        <Stat k="peak arithmetic" v={mach.tflops >= 10 ? nf(mach.tflops) : nf(mach.tflops, 1)} unit={k === "h100" ? "TFLOP/s bf16" : "TFLOP/s f32"} />
        <Stat k="FLOPs per byte read" v={nf((mach.tflops * 1e12) / (mach.bw * 1e9))} accent />
      </div>
      <p className="lr-note">{info.detail}</p>
    </Fig>
  );
}

// ---- fig 0.3.2: where the bytes come from ------------------------------------------------------------

const LADDER: { name: string; gbs: number; kind: "mem" | "link"; src: string; cap?: string }[] = [
  { name: "H100 HBM3", gbs: 3350, kind: "mem", src: "NVIDIA datasheet", cap: "80 GB" },
  { name: "NVLink 4 (GPU ↔ GPU, one way)", gbs: 450, kind: "link", src: "NVIDIA: 900 GB/s total per GPU" },
  { name: "M4 GPU reading memory", gbs: 105, kind: "mem", src: "measured, docs/LOG.md", cap: "24 GB shared" },
  { name: "M4 CPU, all 10 cores", gbs: 98, kind: "mem", src: "measured, docs/LOG.md", cap: "24 GB shared" },
  { name: "PCIe 5 ×16 (CPU ↔ GPU, one way)", gbs: 64, kind: "link", src: "PCI-SIG spec, ~64 GB/s" },
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
          Bandwidth on a log scale (each gridline 10×). <Swatch c={cc.c2}>Blue</Swatch>: a processor reading its own
          memory. <Swatch c={cc.c3}>Olive</Swatch>: links between chips and machines. A byte crossing NVLink is ~7× slower
          than a byte read locally on an H100; across the network, ~70×.
        </>
      }
      tryit={<>Hover a row for its source.</>}
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
          <div key={r.name} className={`p0-ladder-row${hi === i ? " hi" : ""}`} onPointerEnter={() => setHi(i)} onPointerLeave={() => setHi(null)}>
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
      title="Splitting a model buys bandwidth and capacity, and costs conversation"
      prov="estimate"
      caption={
        <>
          A 70B model in bf16 (140 GB) split across H100s with tensor parallelism: each GPU holds a slice of every matrix
          and reads only its slice (<Swatch c={cc.c2}>blue</Swatch>: time reading weights per token). After every attention
          and feed-forward block the GPUs all-reduce their partial results (<Swatch c={cc.c3}>olive</Swatch>: time talking).
          The messages are tiny (16 KB), so the talking is all latency and doesn't shrink as GPUs are added.
        </>
      }
      tryit={<>Switch the link to a network between servers. Eight GPUs now spend almost half of every token talking.</>}
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
                  <span className="nofit-k">doesn't fit: 140 GB of weights, 80 GB per GPU</span>
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
        You know CPUs. This chapter is the minimum hardware picture for inference: why GPUs, why memory is the wall, and why
        more than one GPU gets hard. Three figures, three numbers to keep: <b>bandwidth</b>, <b>capacity</b>, and{" "}
        <b>link speed</b>.
      </p>
      <h2>Few clever cores, or thousands of simple ones</h2>
      <p>
        A CPU spends its transistors making one thread fast. A <G k="GPU" /> spends them on lanes: thousands of simple
        arithmetic units running the same instruction over different numbers, like SIMD taken to an extreme. A matrix
        multiply is the perfect workload: millions of independent multiply-adds.
      </p>
      <ChipPicture />
      <p>
        But look at the last number. More lanes only help when there's arithmetic to do, and decode has ~1 FLOP per byte.
        What really makes datacentre GPUs fast at decode is the memory beside them: <G k="HBM" />, stacks of DRAM on a bus
        thousands of bits wide. An H100 reads ~30× faster than the M4, so the same model decodes ~30× faster.
      </p>

      <h2>Bandwidth, by where the bytes live</h2>
      <p>
        Apple silicon has <G k="unified memory" />: CPU and GPU share one pool, so nothing is copied between them (that's
        why our M4 can try both on the same weights). A datacentre GPU has its own HBM; the weights cross PCIe once at load
        time and then stay put.
      </p>
      <BandwidthLadder />
      <p>
        Capacity is the other wall: 80 GB per H100, 24 GB on our Mac. A 70B model in bf16 needs 140 GB before a single
        user's KV cache. When it doesn't fit, you either shrink it (quantise) or split it.
      </p>

      <h2>Why more GPUs is hard</h2>
      <p>
        Splitting a model is sharding, with a twist you'll recognise: the shards must exchange partial results{" "}
        <b>on every layer of every token</b>, a synchronous barrier hundreds of times per word. Each split also needs an{" "}
        <G k="all-reduce" />, so the link (<G k="NVLink" /> inside a server, the network between servers) is on the
        critical path.
      </p>
      <SplitCost />
      <p>
        Amdahl's law, in a new costume: the per-token communication is a serial fraction that doesn't shrink with more GPUs.
        That's why big models are split within one NVLink-connected server where possible, and across servers mostly by
        layers (pipeline) or by request (replicas), where the traffic is lighter. Chapter 19 animates the all-reduce.
      </p>
      <Idea>
        GPUs win at inference mainly because of their memory bandwidth (HBM), and their lanes win prefill and big batches.
        Memory capacity decides what fits; links between chips are 7–70× slower than local memory, so splitting a model is a
        trade, not a free speed-up.
      </Idea>
      <Deeper title="What about TPUs, Groq, Cerebras, and phones?">
        <p>
          Same physics, different trades. TPUs are matrix engines with HBM, linked in big pods with fast custom
          interconnect. Groq and Cerebras keep weights in on-chip SRAM (far faster than HBM, far smaller), so a model spans
          many chips. Phones and laptops run small quantised models from shared DRAM at 50–120 GB/s, which is why a 3–8B model
          in 4 bits is the on-device sweet spot. In every case, work out bytes per token ÷ bandwidth first.
        </p>
      </Deeper>
      <p className="p0-next-hint">
        Next: the Rosetta stone. Or see the{" "}
        <a href="/learn/many-gpus" onClick={linkTo("/learn/many-gpus")}>
          ring all-reduce
        </a>{" "}
        now.
      </p>
    </>
  );
}
