import { useMemo, useState } from "react";
import { decodeTokens } from "../../engine/client";
import { useWidth } from "../../ui/Figure";
import { usePalette } from "../../ui/theme";
import { Deeper, Fig, G, Idea, Seg, Stat, useFrame, useOnScreen, useRecorded, T, PlayPause, usePlaying } from "../kit";

function Replay() {
  const rec = useRecorded();
  const pal = usePalette();
  const [ref, w] = useWidth<HTMLDivElement>();
  const [t, setT] = useState(0);
  const [slow, setSlow] = useState(10);
  const [playing, setPlaying] = usePlaying();
  const plan = useMemo(() => {
    if (!rec) return null;
    const fw = rec.run.forwards;
    let acc = 0;
    const spans = fw.map((f) => {
      const s = { start: acc, ms: f.ms, phase: f.phase, n: f.nNew };
      acc += f.ms + 1.2; // + the tiny sampling step between forwards
      return s;
    });
    return { spans, total: acc };
  }, [rec]);
  const [onScreen, vis] = useOnScreen<HTMLElement>();
  useFrame(vis && playing && !!plan, (dt) =>
    setT((x) => {
      const n = x + (dt * 1000) / slow;
      return plan && n > plan.total + 600 / slow ? 0 : n;
    }),
  );
  if (!rec || !plan) return <Fig figRef={onScreen} n="8.1" title="The loop, replayed" prov="recorded"><div className="lr-empty">loading…</div></Fig>;
  const toks = rec.run.tokens;
  const nPrompt = toks.filter((x) => x.prompt).length;
  const cur = plan.spans.findIndex((s) => t >= s.start && t < s.start + s.ms + 1.2);
  const doneF = plan.spans.filter((s) => t >= s.start + s.ms).length;
  const shownGen = Math.min(toks.length - nPrompt, doneF);
  const inF = cur >= 0 ? Math.min(1, (t - plan.spans[cur].start) / plan.spans[cur].ms) : 1;
  const phase = cur < 0 ? -1 : inF < 1 ? 1 : 2; // 1 = forward, 2 = pick
  const decodes = plan.spans.filter((s) => s.phase === "decode");
  const dMed = decodes.map((s) => s.ms).sort((a, b) => a - b)[decodes.length >> 1] ?? 0;
  const pre = plan.spans[0];
  const W = Math.max(280, w);
  const sx = (ms: number) => (ms / plan.total) * W;
  const node = (x: number, label: string, sub: string, on: boolean) => (
    <g>
      <rect x={x - 62} y={8} width={124} height={44} rx={8} fill={on ? pal.accent : "transparent"} fillOpacity={on ? 0.12 : 0} stroke={on ? pal.accent : pal.rule} />
      <text x={x} y={27} textAnchor="middle" fontSize={T.xs} fontWeight={700} fill={pal.fg}>
        {label}
      </text>
      <text x={x} y={42} textAnchor="middle" fontSize={T.xs} fill={pal.muted}>
        {sub}
      </text>
    </g>
  );
  return (
    <Fig
      figRef={onScreen}
      n="8.1"
      title="The loop, replayed"
      prov="recorded"
      caption={
        <>
          A real run of the engine, replayed with its measured timings. Top: the four steps of the loop, the current one lit.
          Middle: the text. Bottom: a timeline where each bar is one <G k="forward pass" />, as wide as it took. The wide
          first bar is <G k="prefill" />; each short bar after it is one <G k="decode" /> step.
        </>
      }
      tryit={<>Switch to <b>real speed</b>, the speed it ran in V8 on one core of the M4. Notice how the first bar compares with the rest.</>}
    >
      <div className="lr-controls">
        <PlayPause playing={playing} onToggle={() => setPlaying((p) => !p)} />
        <button className="lr-btn" onClick={() => setT(0)}>
          ⟲
        </button>
        <Seg
          options={[
            { v: 1, label: "real speed" },
            { v: 10, label: "10× slower" },
            { v: 40, label: "40× slower" },
          ]}
          value={slow}
          onChange={setSlow}
          label="speed"
        />
      </div>
      <svg viewBox="0 0 560 60" style={{ width: "100%", maxWidth: 560, display: "block", margin: "0 auto 0.6rem" }}>
        {node(70, "text so far", `${nPrompt + shownGen} tokens`, phase === 1 && inF < 0.08)}
        {node(215, "the model", "12 layers", phase === 1 && inF >= 0.08)}
        {node(360, "scores", "50,257 of them", phase === 1 && inF > 0.96)}
        {node(495, "pick one", "sample", phase === 2)}
        {[132, 277, 422].map((x) => (
          <path key={x} d={`M${x + 2},30 h18`} stroke={pal.muted} markerEnd="url(#lr-arr)" />
        ))}
        <defs>
          <marker id="lr-arr" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L6,3 L0,6 z" fill={pal.muted} />
          </marker>
        </defs>
        <path d="M495,52 v4 H70 v-4" fill="none" stroke={phase === 2 ? pal.accent : pal.rule} strokeDasharray="3 3" />
      </svg>
      <div className="lr-text" style={{ minHeight: "5.5rem", fontSize: "var(--t-md)" }}>
        <span className="ctx">{decodeTokens(toks.slice(0, nPrompt).map((x) => x.id))}</span>
        <span>{decodeTokens(toks.slice(nPrompt, nPrompt + shownGen).map((x) => x.id))}</span>
        <span className="lr-cursor" />
      </div>
      <div ref={ref} style={{ marginTop: "0.8rem" }}>
        {w > 0 && (
          <svg width={W} height={52}>
            {plan.spans.map((s, i) => (
              <rect
                key={i}
                x={sx(s.start)}
                y={s.phase === "prefill" ? 6 : 14}
                width={Math.max(1.5, sx(s.ms))}
                height={s.phase === "prefill" ? 26 : 18}
                rx={2}
                fill={s.phase === "prefill" ? pal.accent : pal.heat[3]}
                opacity={t >= s.start ? 1 : 0.25}
              />
            ))}
            <line x1={sx(t)} x2={sx(t)} y1={0} y2={38} stroke={pal.fg} strokeWidth={1.5} />
            <text x={0} y={50} fontSize={T.xs} fill={pal.muted}>
              prefill
            </text>
            <text x={W} y={50} fontSize={T.xs} fill={pal.muted} textAnchor="end">
              {Math.min(t, plan.total).toFixed(0)} / {plan.total.toFixed(0)} ms of real time
            </text>
          </svg>
        )}
      </div>
      <div className="lr-stats">
        <Stat k="prefill" v={((pre.n / pre.ms) * 1000).toFixed(0)} unit={`tok/s (${pre.n} tokens in ${pre.ms.toFixed(0)} ms)`} />
        <Stat k="one decode step" v={dMed.toFixed(1)} unit="ms" />
        <Stat k="decode" v={(1000 / dMed).toFixed(0)} unit="tok/s" accent />
      </div>
    </Fig>
  );
}

export default function Loop() {
  return (
    <>
      <p>
        You now have every piece of the loop. Turn the text into tokens. Run them through the model. Get 50,257 scores. Pick
        one. Add it to the text. Run the model again.
      </p>
      <Replay />
      <p>
        The loop has two phases, and they behave very differently. <b>Prefill</b> takes the whole prompt in one pass; here 14
        tokens went through together. <b>Decode</b> produces one token per pass, because each new token depends on the one
        just chosen. There's no way to compute token 20 before you know token 19.
      </p>
      <p>
        Prefill handled its 14 tokens faster per token than decode did. Prefill reads each weight once and uses it for all
        14 prompt tokens. Every decode pass reads all 124 million weights to produce a single token. This early,
        single-core engine shows only a small gap. On the M4's GPU, prefill runs about 8× faster per token than decode
        (chapter 16). Prefill and decode do the same arithmetic per token but read very different bytes per token. Part 2
        is about that difference.
      </p>
      <Idea>Generation is a loop of full forward passes, one per token. Prefill reads the prompt in one pass; decode writes one token per pass, and each pass reads every weight in the model.</Idea>
      <Deeper title="What's the user actually waiting for?">
        <p>
          Two numbers describe how a model feels. <b>Time to first token</b> (TTFT) is how long until the first token appears,
          mostly the prefill. <b>Time per output token</b> (TPOT) is the gap between tokens after that, one decode step. A long
          document as a prompt hurts the first; a long answer hurts the second.
        </p>
        <p>
          That completes part 1. Part 2 measures <b>why each decode step takes as long as it does</b>.
        </p>
      </Deeper>
    </>
  );
}
