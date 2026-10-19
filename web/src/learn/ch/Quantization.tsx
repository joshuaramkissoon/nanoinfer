import { useMemo, useState } from "react";
import { useWidth } from "../../ui/Figure";
import { usePalette } from "../../ui/theme";
import { Climb } from "../Climb";
import { chartColors, Deeper, Fig, G, Idea, Seg, Stat, Swatch, useLearnData } from "../kit";

function Block() {
  const d = useLearnData();
  const pal = usePalette();
  const cc = chartColors(pal.dark);
  const [bits, setBits] = useState<32 | 8 | 4 | 2>(4);
  const [ref, w] = useWidth<HTMLDivElement>();
  const wts = d?.quant.block.w ?? null;
  const q = useMemo(() => {
    if (!wts) return null;
    if (bits === 32) return { rec: wts, codes: null as number[] | null, scale: 0, levels: [] as number[] };
    const qmax = 2 ** (bits - 1) - 1;
    const scale = Math.max(...wts.map(Math.abs)) / qmax;
    const codes = wts.map((v) => Math.max(-qmax - 1, Math.min(qmax, Math.round(v / scale))));
    const levels: number[] = [];
    for (let k = -qmax - 1; k <= qmax; k++) levels.push(k * scale);
    return { rec: codes.map((c) => c * scale), codes, scale, levels };
  }, [wts, bits]);
  if (!wts || !q || !d) return <Fig n="15.1" title="Squeezing 32 real weights" prov="weights"><div className="lr-empty">loading…</div></Fig>;
  const W = Math.max(300, w), H = 220, pl = 8, top = 10, bottom = 10;
  const mx = Math.max(...wts.map(Math.abs)) * 1.08;
  const y = (v: number) => top + ((mx - v) / (2 * mx)) * (H - top - bottom);
  const col = (W - pl * 2) / 32;
  const bytes = bits === 32 ? 128 : (32 * bits) / 8 + 2;
  let se = 0, ss = 0;
  wts.forEach((v, i) => {
    se += (v - q.rec[i]) ** 2;
    ss += v * v;
  });
  const rel = Math.sqrt(se / ss);
  const whole = d.quant.err.find((e) => e.bits === bits);
  return (
    <Fig
      n="15.1"
      title="Squeezing 32 real weights"
      prov="weights"
      caption={
        <>
          32 consecutive weights from GPT-2's layer 6 feed-forward network. <Swatch c={cc.c2}>Blue</Swatch> stems: the
          original 32-bit values. <Swatch c={cc.c1}>Ember</Swatch> dots: what's left after storing each in fewer bits. The
          block's biggest weight sets one shared scale, and each weight rounds to the nearest allowed level (the faint
          horizontal lines, when there are few enough to see).
        </>
      }
      tryit={<>Step from 32 to 8 to 4 to 2 bits. Notice how far the ember dots move from the stems, and the rounding error beside it.</>}
    >
      <div className="lr-controls">
        <Seg
          options={([32, 8, 4, 2] as const).map((b) => ({ v: b, label: `${b}-bit` }))}
          value={bits}
          onChange={setBits}
          label="bits per weight"
        />
      </div>
      <div className="p0-legend lr-small">
        <span>
          <i style={{ background: cc.c2 }} />
          original weight (32-bit)
        </span>
        {bits !== 32 && (
          <span>
            <i style={{ background: cc.c1, borderRadius: "50%" }} />
            what {bits} bits can store
          </span>
        )}
      </div>
      <div ref={ref}>
        {w > 0 && (
          <svg width={W} height={H}>
            {q.levels.length > 0 && q.levels.length <= 16 && q.levels.map((l, i) => <line key={i} x1={pl} x2={W - pl} y1={y(l)} y2={y(l)} stroke={pal.rule} />)}
            <line x1={pl} x2={W - pl} y1={y(0)} y2={y(0)} stroke={pal.muted} />
            {wts.map((v, i) => {
              const cx = pl + col * i + col / 2;
              return (
                <g key={i}>
                  <line x1={cx} x2={cx} y1={y(0)} y2={y(v)} stroke={cc.c2} strokeWidth={2} strokeLinecap="round" opacity={0.75} />
                  {bits !== 32 && <circle cx={cx} cy={y(q.rec[i])} r={4} fill={cc.c1} stroke={pal.dark ? "#211f1d" : "#fbfaf7"} strokeWidth={2} />}
                </g>
              );
            })}
          </svg>
        )}
      </div>
      <div className="lr-stats">
        <Stat k="bytes for these 32" v={bytes} unit={bits === 32 ? "(4 each)" : `(${((bytes / 128) * 100).toFixed(0)}% of 32-bit)`} />
        <Stat k="speed limit" v={`${(128 / bytes).toFixed(1)}×`} unit="vs 32-bit" accent />
        <Stat k="rounding error" v={whole ? `${(whole.rel * 100).toFixed(1)}%` : "0%"} unit={bits === 32 ? "none" : `whole matrix · ${(rel * 100).toFixed(1)}% here`} />
      </div>
    </Fig>
  );
}

export default function Quantization() {
  return (
    <>
      <h2 style={{ marginTop: "2.2rem" }}>The bottleneck: 495 MB per token</h2>
      <p>
        At 146 tokens per second the engine is at 60% of the speed limit. The limit itself is now the problem, because
        every token reads 495 MB of weights. Faster code can't read memory faster. The only way past the limit is to read{" "}
        <b>fewer bytes</b>.
      </p>
      <h2>The idea: fewer bits per weight</h2>
      <p>
        Store each weight in fewer bits: 16, 8 or 4 instead of 32. This is <G k="quantization">quantisation</G>. Nearby
        weights have similar sizes, so the engine groups them in blocks of 32. Each block has one scale, which sets the size
        of one step. Each weight is then a small whole number of steps.
      </p>
      <Block />
      <p>
        8 bits per weight is nearly invisible, about half a percent of error. 4 bits gives about 10% rounding error, and the
        model gets slightly worse at predicting text. 2 bits throws most of the information away. The engine converts the
        weights back to ordinary numbers inside the processor, just before multiplying. The memory traffic shrinks and the
        arithmetic stays the same.
      </p>
      <h2>The result: 497 tokens per second on the CPU</h2>
      <p>
        These are native runs on the M4 Mac mini's CPU. 16-bit weights took decode from <b>146 to 249</b> tokens per second.
        8-bit reached <b>315</b>, and <b>425</b> after a few more changes, with the same top-choice token 98% of the time.
      </p>
      <p>
        4-bit reached <span className="k">497 tokens per second</span>, the fastest CPU result. The cost is now visible. The
        top-choice token differs from the 32-bit model's about 1 time in 5, and <G k="perplexity" /> (how surprised the model
        is by real text) rises about 8%. In a browser tab, the later 8-bit and 4-bit stages are the fastest CPU decode, about
        195 tokens per second on the M4. The best 32-bit browser stage makes about 130.
      </p>
      <Climb upto={4} n="15.2" />
      <p>
        Each stage now has its own speed limit. The 4-bit stage reads about 97 MB per token instead of 495, because each
        block of 32 weights carries a scale and the LM head stays at 8 bits. So its limit is about 5× higher: 120 GB/s ÷ 97
        MB ≈ 1,240 tokens per second. 497 is 40% of that. The engine is now far below its new limit, because the fixed costs
        per token (waking threads, the small operations) don't shrink with the bytes. Chapter 16 moves to the GPU.
      </p>
      <Idea>Decode reads every weight, so smaller weights mean faster decode. Each halving of the bits buys speed and costs some accuracy.</Idea>
      <Deeper title="The one matrix you can't squeeze">
        <p>
          Quantising every matrix to 8 bits made GPT-2 noticeably worse, and at 4 bits it fell apart: perplexity went from
          37 to 516. Testing one matrix at a time found the cause. It was the LM head (the output layer that scores every
          token), which is also the embedding table from chapter 3. It holds a third of GPT-2's weights and loses the most
          when compressed. Keeping that one matrix at 16 bits fixed it.
        </p>
        <p>
          Big models have this problem much less, because in a 7B model that table is about 2% of the weights. So this fix
          matters most for small models.
        </p>
      </Deeper>
    </>
  );
}
