import { useState } from "react";
import { usePalette } from "../../../ui/theme";
import { useWidth } from "../../../ui/Figure";
import { chartColors, Deeper, Fig, G, Idea, Seg, Swatch, useFrame, useOnScreen, T, PlayPause, usePlaying } from "../../kit";
import { linkTo } from "../../router";

// ---- fig 0.1.1: write once, run forever ------------------------------------------------------------

function WriteOnce() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [t, setT] = useState(0);
  const [onScreen, seen] = useOnScreen<HTMLElement>();
  const [playing, setPlaying] = usePlaying();
  useFrame(seen && playing, (dt) => setT((v) => (v + dt) % 1000));
  const off = (t * 26) % 18;
  const panel = pal.dark ? "#262421" : "#ffffff";
  const [wref, w] = useWidth<HTMLDivElement>();
  const narrow = w > 0 && w < 520; // phones: training → weights on one row, inference underneath
  return (
    <Fig
      figRef={onScreen}
      aside={<PlayPause playing={playing} onToggle={() => setPlaying((p) => !p)} />}
      n="0.1.1"
      title="Training writes the weights once; inference reads them forever"
      prov="diagram"
      caption={
        <>
          {narrow ? "First" : "Left"}, <Swatch c={cc.c1}>ember</Swatch>: training, a one-off job that searches for good numbers.{" "}
          {narrow ? "Then" : "Middle"}, <Swatch c={cc.c2}>blue</Swatch>: the result, a file of frozen numbers.{" "}
          {narrow ? "Below" : "Right"}: inference, one small job per word per
          user, for as long as the model is in service. Each tick is one forward pass.
        </>
      }
    >
      <div ref={wref}>
      <svg viewBox={narrow ? "0 0 312 300" : "0 0 640 170"} style={{ width: "100%" }} role="img" aria-label="training produces weights once; inference runs them repeatedly">
        <g transform={narrow ? "translate(2,-26)" : undefined}>
        {/* training */}
        <rect x={8} y={36} width={150} height={92} rx={10} fill={cc.c1} fillOpacity={0.14} stroke={cc.c1} />
        <text x={83} y={62} textAnchor="middle" fontSize={T.sm} fontWeight={700} fill={pal.fg}>
          training
        </text>
        <text x={83} y={82} textAnchor="middle" fontSize={T.xs} fill={pal.sub}>
          once · weeks
        </text>
        <text x={83} y={98} textAnchor="middle" fontSize={T.xs} fill={pal.sub}>
          thousands of GPUs
        </text>
        <text x={83} y={114} textAnchor="middle" fontSize={T.xs} fill={pal.sub}>
          adjusts the numbers
        </text>
        <path d="M164,82 H200" stroke={pal.muted} markerEnd="url(#p0a)" />
        {/* weights file */}
        <g>
          <rect x={206} y={40} width={92} height={84} rx={6} fill={panel} stroke={cc.c2} />
          {Array.from({ length: 6 }, (_, r) =>
            Array.from({ length: 7 }, (_, c) => (
              <rect key={`${r}-${c}`} x={214 + c * 11.3} y={48 + r * 11.3} width={9} height={9} rx={1.5} fill={cc.c2} fillOpacity={0.18 + (((r * 7 + c) * 37) % 10) / 22} />
            )),
          )}
          <text x={252} y={142} textAnchor="middle" fontSize={T.xs} fontWeight={700} fill={pal.fg}>
            weights
          </text>
          <text x={252} y={157} textAnchor="middle" fontSize={T.xs} fill={pal.sub}>
            frozen · read-only
          </text>
        </g>
        </g>
        {narrow ? (
          <path d="M254,140 V170" stroke={pal.muted} markerEnd="url(#p0a)" />
        ) : (
          <path d="M304,82 H340" stroke={pal.muted} markerEnd="url(#p0a)" />
        )}
        {/* inference stream */}
        <g transform={narrow ? "translate(-340,150) scale(1)" : undefined}>
        <text x={346} y={30} fontSize={T.sm} fontWeight={700} fill={pal.fg}>
          inference
        </text>
        <text x={346} y={46} fontSize={T.xs} fill={pal.sub}>
          every word, every user, forever
        </text>
        {/* the stream fades in and out at its ends rather than cutting ticks in half */}
        <linearGradient id="p0fade" x1={346} x2={632} y1={0} y2={0} gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#fff" stopOpacity={0} />
          <stop offset="0.08" stopColor="#fff" stopOpacity={1} />
          <stop offset="0.92" stopColor="#fff" stopOpacity={1} />
          <stop offset="1" stopColor="#fff" stopOpacity={0} />
        </linearGradient>
        <mask id="p0clip" maskUnits="userSpaceOnUse" x={346} y={56} width={286} height={60}>
          <rect x={346} y={56} width={286} height={60} fill="url(#p0fade)" />
        </mask>
        <g mask="url(#p0clip)">
          {Array.from({ length: 3 }, (_, row) =>
            Array.from({ length: 18 }, (_, i) => {
              const x = 346 + i * 18 + ((off + row * 6) % 18) - 18;
              return <rect key={`${row}-${i}`} x={x} y={60 + row * 19} width={10} height={12} rx={2} fill={cc.c1} fillOpacity={0.35 + 0.65 * (i / 18)} />;
            }),
          )}
        </g>
        <text x={346} y={136} fontSize={T.xs} fill={pal.sub}>
          one tick = one forward pass = one token
        </text>
        </g>
        <defs>
          <marker id="p0a" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill={pal.muted} />
          </marker>
        </defs>
      </svg>
      </div>
    </Fig>
  );
}

// ---- fig 0.1.2: GPT-2, opened up -------------------------------------------------------------------

interface Part {
  key: string;
  name: string;
  shape: string;
  params: number;
  does: string;
}

const BLOCK_PARTS: Part[] = [
  { key: "qkv", name: "attention: make query, key, value", shape: "768 × 2304", params: 1_771_776, does: "Turns the token's 768 numbers into a query (what am I looking for?), a key (what do I contain?) and a value (what do I pass on?)." },
  { key: "proj", name: "attention: mix the heads", shape: "768 × 768", params: 590_592, does: "Combines what the 12 attention heads found into one update to the token's vector." },
  { key: "fc", name: "feed-forward: expand", shape: "768 × 3072", params: 2_362_368, does: "Projects up to 3072 numbers: a big bank of learned pattern detectors, each a dot product." },
  { key: "fcp", name: "feed-forward: contract", shape: "3072 × 768", params: 2_360_064, does: "Projects back down to 768: the update this layer adds to the token's vector." },
  { key: "ln", name: "two layer norms", shape: "2 × 2 × 768", params: 3_072, does: "Rescale the vector to a steady size before each half. Tiny." },
];
const BLOCK = BLOCK_PARTS.reduce((a, p) => a + p.params, 0);

const FILE: Part[] = [
  { key: "wte", name: "token embeddings", shape: "50257 × 768", params: 38_597_376, does: "One row of 768 numbers per token in the vocabulary. Step one is a table lookup; the same table is reused at the end to score every possible next token." },
  { key: "wpe", name: "position embeddings", shape: "1024 × 768", params: 786_432, does: "One row per position (0–1023), added so the model knows word order. 1024 is GPT-2's context window." },
  ...Array.from({ length: 12 }, (_, i) => ({
    key: `b${i}`,
    name: `layer ${i + 1}`,
    shape: "5 matrices",
    params: BLOCK,
    does: "One transformer block: attention (look at earlier tokens) then a feed-forward network (think about this token). All 12 have the same shape and different numbers.",
  })),
  { key: "lnf", name: "final norm", shape: "2 × 768", params: 1_536, does: "One last rescale before scoring." },
];
const TOTAL = FILE.reduce((a, p) => a + p.params, 0);

function ModelFile() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [sel, setSel] = useState("b0");
  const [sub, setSub] = useState<string | null>(null);
  const part = FILE.find((p) => p.key === sel)!;
  const isBlock = sel.startsWith("b");
  const detail = isBlock && sub ? BLOCK_PARTS.find((p) => p.key === sub)! : part;
  const mb = (p: number) => ((p * 4) / 1e6).toFixed(p * 4 > 1e7 ? 0 : 1);
  return (
    <Fig
      n="0.1.2"
      title="GPT-2, opened up: 124 million numbers in named blocks"
      prov="weights"
      caption={
        <>
          The whole file as one strip, each block's width proportional to how many numbers it holds. All of it is{" "}
          <Swatch c={cc.c2}>blue</Swatch> because all of it sits in memory and is read on every token. A third is the
          vocabulary table; the rest is twelve identical-shaped layers.
        </>
      }
      tryit={<>Click a layer, then click inside it: four matrices hold almost all its numbers.</>}
    >
      <div className="p0-strip" role="group" aria-label="the weights file">
        {FILE.map((p) => (
          <button
            key={p.key}
            className={`p0-seg${sel === p.key ? " on" : ""}`}
            style={{ flexGrow: p.params, background: cc.c2, opacity: sel === p.key ? 1 : p.key.startsWith("b") ? 0.55 : 0.75 }}
            onClick={() => (setSel(p.key), setSub(null))}
            aria-pressed={sel === p.key}
            aria-label={`${p.name}: ${(p.params / 1e6).toFixed(2)} million numbers`}
            title={p.name}
          />
        ))}
      </div>
      <div className="p0-strip-axis lr-small">
        <span>embeddings</span>
        <span>12 layers →</span>
        <span>{(TOTAL / 1e6).toFixed(1)}M numbers · {mb(TOTAL)} MB at 4 bytes each</span>
      </div>
      {isBlock && (
        <div className="p0-zoom">
          <div className="p0-zoom-k lr-small">inside {part.name}</div>
          <div className="p0-strip sub" role="group" aria-label={`inside ${part.name}`}>
            {BLOCK_PARTS.map((p) => (
              <button
                key={p.key}
                className={`p0-seg${sub === p.key ? " on" : ""}`}
                style={{ flexGrow: p.params, background: cc.c2, opacity: sub === p.key ? 1 : 0.6, minWidth: 4 }}
                onClick={() => setSub(p.key)}
                aria-pressed={sub === p.key}
                aria-label={p.name}
                title={p.name}
              />
            ))}
          </div>
        </div>
      )}
      <div className="p0-detail">
        <div className="p0-detail-h">
          <b>{detail.name}</b>
          <span className="lr-small">
            {detail.shape} · {(detail.params / 1e6).toFixed(2)}M numbers · {mb(detail.params)} MB
          </span>
        </div>
        <p>{detail.does}</p>
      </div>
    </Fig>
  );
}

// ---- fig 0.1.3: the serving stack ------------------------------------------------------------------

const JOBS = [
  {
    k: "load",
    label: "1 · load",
    title: "Load the weights",
    text: "Read the file into the accelerator's memory once, at start-up. For a 70B model that's 140 GB: minutes, not milliseconds. This is your cold start, and why replicas are expensive to add.",
    analog: "warming a cache · loading a read-only dataset into RAM",
    on: ["disk", "weights"],
  },
  {
    k: "run",
    label: "2 · run",
    title: "Run the forward pass",
    text: "Each step, push the current tokens through the fixed program: read every weight, read the request's cached state, produce scores for the next token. It's the same code for every request.",
    analog: "a pure function: (weights, tokens, state) → next-token scores",
    on: ["weights", "compute", "kv", "w2c", "kv2c"],
  },
  {
    k: "schedule",
    label: "3 · schedule",
    title: "Schedule requests",
    text: "Decide which requests share the next step (a batch), admit new ones, pause or evict others when memory runs out. Requests arrive at random and finish at unpredictable lengths.",
    analog: "a request scheduler coalescing work every tick",
    on: ["clients", "sched", "c2s", "s2c", "out"],
  },
  {
    k: "state",
    label: "4 · state",
    title: "Manage per-request state",
    text: "Every live request owns a KV cache that grows by one entry per token. The engine allocates it in pages, frees it when the request ends, and can share pages between requests with the same prefix.",
    analog: "session state with an allocator: per-connection buffers, paging, memoisation",
    on: ["kv", "kv2c"],
  },
];

function ServingStack() {
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [job, setJob] = useState("run");
  const J = JOBS.find((j) => j.k === job)!;
  const o = (id: string) => (J.on.includes(id) ? 1 : 0.22);
  const panel = pal.dark ? "#262421" : "#ffffff";
  const users = ["A", "B", "C"];
  const [wref, w] = useWidth<HTMLDivElement>();
  const narrow = w > 0 && w < 520;
  const marker = (
    <defs>
      <marker id="p0b" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill={pal.muted} />
      </marker>
    </defs>
  );
  // phones: the same picture stacked top to bottom (users → scheduler → device), at a readable type size
  const narrowSvg = (
    <svg viewBox="0 0 340 420" style={{ width: "100%", marginTop: "0.4rem" }} role="img" aria-label="clients, scheduler, and a GPU holding weights and KV caches">
      {marker}
      <g opacity={o("clients")}>
        {users.map((u, i) => (
          <g key={u}>
            <rect x={10 + i * 108} y={8} width={92} height={34} rx={8} fill={panel} stroke={pal.sub} />
            <text x={56 + i * 108} y={30} textAnchor="middle" fontSize={T.sm} fill={pal.fg}>
              user {u}
            </text>
          </g>
        ))}
      </g>
      <g opacity={o("c2s")}>
        {users.map((u, i) => (
          <path key={u} d={`M${56 + i * 108},44 C${56 + i * 108},62 255,58 255,80`} fill="none" stroke={pal.muted} markerEnd="url(#p0b)" />
        ))}
      </g>
      <g opacity={o("disk")}>
        <rect x={10} y={84} width={150} height={52} rx={6} fill={panel} stroke={pal.rule} />
        <text x={85} y={106} textAnchor="middle" fontSize={T.sm} fill={pal.sub}>
          model file
        </text>
        <text x={85} y={124} textAnchor="middle" fontSize={T.xs} fill={pal.muted}>
          on disk
        </text>
        <path d="M85,138 V184" stroke={cc.c2} markerEnd="url(#p0b)" />
      </g>
      <g opacity={o("sched")}>
        <rect x={180} y={84} width={150} height={60} rx={8} fill={panel} stroke={pal.sub} />
        <text x={255} y={104} textAnchor="middle" fontSize={T.sm} fontWeight={700} fill={pal.fg}>
          scheduler
        </text>
        {users.map((u, i) => (
          <g key={u}>
            <rect x={218 + i * 26} y={114} width={22} height={20} rx={4} fill={pal.code} stroke={pal.rule} />
            <text x={229 + i * 26} y={128} textAnchor="middle" fontSize={T.xs} fill={pal.sub}>
              {u}
            </text>
          </g>
        ))}
      </g>
      <path d="M255,146 V159" stroke={pal.muted} markerEnd="url(#p0b)" opacity={o("s2c")} />
      <rect x={4} y={160} width={318} height={232} rx={12} fill="none" stroke={pal.sub} />
      <text x={186} y={180} fontSize={T.xs} fill={pal.muted}>
        GPU · one device
      </text>
      <g opacity={o("weights")}>
        <rect x={14} y={188} width={152} height={84} rx={6} fill={panel} stroke={cc.c2} />
        {Array.from({ length: 4 }, (_, r) =>
          Array.from({ length: 12 }, (_, c) => (
            <rect key={`${r}-${c}`} x={21 + c * 11.8} y={195 + r * 11.8} width={9.5} height={9.5} rx={1.5} fill={cc.c2} fillOpacity={0.15 + (((r * 12 + c) * 29) % 10) / 24} />
          )),
        )}
        <text x={90} y={262} textAnchor="middle" fontSize={T.xs} fill={pal.fg}>
          weights · shared
        </text>
      </g>
      <g opacity={o("compute")}>
        <rect x={186} y={196} width={128} height={70} rx={8} fill={cc.c1} fillOpacity={0.14} stroke={cc.c1} />
        <text x={250} y={222} textAnchor="middle" fontSize={T.sm} fontWeight={700} fill={pal.fg}>
          forward pass
        </text>
        <text x={250} y={240} textAnchor="middle" fontSize={T.xs} fill={pal.sub}>
          the fixed program
        </text>
        <text x={250} y={256} textAnchor="middle" fontSize={T.xs} fill={pal.sub}>
          all users at once
        </text>
      </g>
      <path d="M168,230 H182" stroke={cc.c2} strokeWidth={2} markerEnd="url(#p0b)" opacity={o("w2c")} />
      <g opacity={o("kv")}>
        {users.map((u, i) => {
          const n = [5, 2, 4][i];
          return (
            <g key={u}>
              <text x={14} y={302 + i * 22} fontSize={T.xs} fill={pal.sub}>
                KV {u}
              </text>
              {Array.from({ length: n }, (_, j) => (
                <rect key={j} x={56 + j * 18} y={290 + i * 22} width={15} height={15} rx={2} fill={cc.c2} fillOpacity={0.55} />
              ))}
            </g>
          );
        })}
        <text x={14} y={377} fontSize={T.xs} fill={pal.muted}>
          per-request state · grows each token
        </text>
      </g>
      <path d="M150,312 C200,312 230,292 246,270" fill="none" stroke={cc.c2} strokeWidth={1.5} markerEnd="url(#p0b)" opacity={o("kv2c")} />
      <g opacity={o("out")}>
        <path d="M314,231 H333 V25 H322" fill="none" stroke={cc.c1} strokeDasharray="3 4" strokeWidth={1.4} markerEnd="url(#p0b)" />
        <text x={170} y={412} textAnchor="middle" fontSize={T.xs} fill={pal.sub}>
          tokens stream back up to each user, one per step
        </text>
      </g>
    </svg>
  );
  return (
    <Fig
      n="0.1.3"
      title="An inference server is a stateful service"
      prov="diagram"
      caption={
        <>
          Requests come in {narrow ? "at the top" : "on the left"} and tokens stream back. Inside the device: the weights (
          <Swatch c={cc.c2}>blue grid</Swatch>, one shared copy), each request's KV cache (<Swatch c={cc.c2}>blue pages</Swatch>
          , one stack per user), and the <Swatch c={cc.c1}>forward pass</Swatch> that reads both.
        </>
      }
      tryit={<>Step through the four jobs. The second and fourth are where almost all of this course's optimisations live.</>}
    >
      <div className="lr-controls">
        <Seg options={JOBS.map((j) => ({ v: j.k, label: j.label }))} value={job} onChange={setJob} label="engine job" />
      </div>
      <div ref={wref}>
      {narrow ? narrowSvg : (
      <svg viewBox="0 0 640 264" style={{ width: "100%", marginTop: "0.4rem" }} role="img" aria-label="clients, scheduler, and a GPU holding weights and KV caches">
        {marker}
        {/* clients */}
        <g opacity={o("clients")}>
          {users.map((u, i) => (
            <g key={u}>
              <rect x={6} y={50 + i * 56} width={64} height={34} rx={8} fill={panel} stroke={pal.sub} />
              <text x={38} y={71 + i * 56} textAnchor="middle" fontSize={T.xs} fill={pal.fg}>
                user {u}
              </text>
            </g>
          ))}
        </g>
        <g opacity={o("c2s")}>
          {users.map((u, i) => (
            <path key={u} d={`M72,${62 + i * 56} C96,${62 + i * 56} 100,118 122,118`} fill="none" stroke={pal.muted} markerEnd="url(#p0b)" />
          ))}
        </g>
        <g opacity={o("out")}>
          <path d="M560,150 V244 H38 V194" fill="none" stroke={cc.c1} strokeDasharray="3 4" strokeWidth={1.4} markerEnd="url(#p0b)" />
          <text x={300} y={258} textAnchor="middle" fontSize={T.xs} fill={pal.sub}>
            tokens stream back to each user, one per step
          </text>
        </g>
        {/* scheduler */}
        <g opacity={o("sched")}>
          <rect x={126} y={88} width={104} height={62} rx={8} fill={panel} stroke={pal.sub} />
          <text x={178} y={106} textAnchor="middle" fontSize={T.xs} fontWeight={700} fill={pal.fg}>
            scheduler
          </text>
          {users.map((u, i) => (
            <g key={u}>
              <rect x={140 + i * 26} y={116} width={22} height={18} rx={4} fill={pal.code} stroke={pal.rule} />
              <text x={151 + i * 26} y={129} textAnchor="middle" fontSize={T.xs} fill={pal.sub}>
                {u}
              </text>
            </g>
          ))}
        </g>
        <path d="M232,119 H268" stroke={pal.muted} markerEnd="url(#p0b)" opacity={o("s2c")} />
        {/* device */}
        <rect x={272} y={14} width={360} height={210} rx={12} fill="none" stroke={pal.sub} />
        <text x={286} y={32} fontSize={T.xs} fill={pal.muted}>
          GPU · one device
        </text>
        {/* disk */}
        <g opacity={o("disk")}>
          <rect x={130} y={20} width={96} height={40} rx={6} fill={panel} stroke={pal.rule} />
          <text x={178} y={38} textAnchor="middle" fontSize={T.xs} fill={pal.sub}>
            model file
          </text>
          <text x={178} y={51} textAnchor="middle" fontSize={T.xs} fill={pal.muted}>
            on disk
          </text>
          <path d="M228,40 H296" stroke={cc.c2} markerEnd="url(#p0b)" />
        </g>
        {/* weights */}
        <g opacity={o("weights")}>
          <rect x={300} y={44} width={150} height={86} rx={6} fill={panel} stroke={cc.c2} />
          {Array.from({ length: 5 }, (_, r) =>
            Array.from({ length: 12 }, (_, c) => (
              <rect key={`${r}-${c}`} x={308 + c * 11.5} y={52 + r * 11.5} width={9} height={9} rx={1.5} fill={cc.c2} fillOpacity={0.15 + (((r * 12 + c) * 29) % 10) / 24} />
            )),
          )}
          <text x={375} y={122} textAnchor="middle" fontSize={T.xs} fill={pal.fg}>
            weights · shared
          </text>
        </g>
        {/* compute */}
        <g opacity={o("compute")}>
          <rect x={478} y={78} width={136} height={72} rx={8} fill={cc.c1} fillOpacity={0.14} stroke={cc.c1} />
          <text x={546} y={108} textAnchor="middle" fontSize={T.xs} fontWeight={700} fill={pal.fg}>
            forward pass
          </text>
          <text x={546} y={124} textAnchor="middle" fontSize={T.xs} fill={pal.sub}>
            the fixed program
          </text>
          <text x={546} y={138} textAnchor="middle" fontSize={T.xs} fill={pal.sub}>
            all users at once
          </text>
        </g>
        <path d="M452,96 H474" stroke={cc.c2} strokeWidth={2} markerEnd="url(#p0b)" opacity={o("w2c")} />
        {/* kv caches */}
        <g opacity={o("kv")}>
          {users.map((u, i) => {
            const n = [5, 2, 4][i];
            return (
              <g key={u}>
                <text x={300} y={170 + i * 18} fontSize={T.xs} fill={pal.sub}>
                  KV {u}
                </text>
                {Array.from({ length: n }, (_, j) => (
                  <rect key={j} x={334 + j * 15} y={160 + i * 18} width={12} height={12} rx={2} fill={cc.c2} fillOpacity={0.55} />
                ))}
              </g>
            );
          })}
          <text x={334} y={218} fontSize={T.xs} fill={pal.muted}>
            per-request state · grows each token
          </text>
        </g>
        <path d="M430,178 C470,178 480,160 500,152" fill="none" stroke={cc.c2} strokeWidth={1.5} markerEnd="url(#p0b)" opacity={o("kv2c")} />
      </svg>
      )}
      </div>
      <div className="p0-job">
        <b>{J.title}.</b> {J.text}
        <div className="p0-analog">
          <span className="lr-cap-k">you know this as</span>
          {J.analog}
        </div>
      </div>
    </Fig>
  );
}

export default function WhatIsInference() {
  return (
    <>
      <p>
        You run services. A language model, once trained, is a service with an unusual profile: a fixed, enormous,
        read-only blob of data, one fixed program, and a request loop that touches the whole blob for every word it writes.
        That's it. The rest of the field is plumbing around that loop.
      </p>
      <h2>Two very different jobs</h2>
      <p>
        <b>Training</b> finds the numbers: it nudges billions of them, over weeks on thousands of GPUs, until the model
        predicts text well. <b>Inference</b> uses them: the numbers are frozen, and every request runs the same function
        over them. Training is like building a search index; inference is serving queries against it.
      </p>
      <WriteOnce />
      <p>
        Training happens once per model. Inference happens on every word of every answer, for every user, for as long as the
        model is in service, so at scale it's where the money goes. This course, and this engine, are only about inference.
      </p>

      <h2>What a model physically is</h2>
      <p>
        A file of numbers (the <G k="weights" />, also called <G k="parameters" />) plus a short, fixed program that does
        arithmetic with them. GPT-2 small, the model on this site, is 124 million numbers. “7B” means seven billion.
      </p>
      <ModelFile />
      <p>
        Most of those numbers are arranged as <b>matrices</b>. Think of a weight matrix as a learned transformation: a vector
        of 768 numbers goes in, and each output number is a weighted sum of all 768 inputs (a dot product), with the weights
        chosen by training. Stack a few of those, add a step where each word can look back at earlier words (
        <G k="attention" />, chapter 05), and repeat twelve times. The program is tiny:
      </p>
      <code className="lr-math">{`x = embed[token] + position[i]        # look up 768 numbers
for layer in 1..12:
    x += attention(norm(x), cache)     # read earlier tokens
    x += feed_forward(norm(x))         # 768 → 3072 → 768
scores = norm(x) · embedᵀ              # one score per vocabulary token
next = sample(softmax(scores))         # pick the next token`}</code>
      <p>
        Every line except the first and last is matrix × vector, which is why chapter 04 says multiply-and-add is ~99% of
        the work. <G k="softmax" /> just turns raw scores into percentages that add up to 100.
      </p>

      <h2>What an inference engine does</h2>
      <p>
        An engine (vLLM, TensorRT-LLM, llama.cpp, SGLang, or nanoinfer) wraps that loop in a server. If you squint, it's a
        stateful service with four jobs:
      </p>
      <ServingStack />
      <p>
        The weights are the easy part: immutable, shared, cacheable. The hard parts are the ones you'd expect from a
        distributed-systems background: the per-request state (the <G k="KV cache" />) and the scheduling of many requests
        with unknown lengths onto a resource that's expensive to leave idle.
      </p>
      <Idea>
        A model is a big read-only file of numbers plus a fixed program. Inference runs that program once per token, for
        every user. An engine is a stateful server around it: load, run, schedule, manage state.
      </Idea>
      <Deeper title="Why “frozen”? Doesn't the model learn from my chat?">
        <p>
          Not during inference. The weights don't change between requests; anything the model “remembers” within a
          conversation is in the prompt and the KV cache, which are per-request. Products that seem to learn about you store
          notes and put them back in the prompt. Changing the weights means another training run (fine-tuning), a separate,
          offline job.
        </p>
      </Deeper>
      <p className="p0-next-hint">
        Next: why running that loop is expensive. Or keep the{" "}
        <a href="/learn/field-guide" onClick={linkTo("/learn/field-guide")}>
          field guide
        </a>{" "}
        open in another tab.
      </p>
    </>
  );
}
