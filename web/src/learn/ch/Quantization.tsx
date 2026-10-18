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
          32 consecutive weights from one of GPT-2's real matrices (layer 6's feed-forward). <Swatch c={cc.c2}>Blue</Swatch>{" "}
          stems: the original 32-bit values, above or below zero. <Swatch c={cc.c1}>Ember</Swatch> dots: what's left after
          storing each one in fewer bits. The block shares one scale (its
          biggest weight sets it), and every weight is rounded to the nearest of the allowed levels (the faint horizontal
          lines, drawn when there are few enough to see).
        </>
      }
      tryit={<>Step from 32 to 8 to 4 to 2 bits. At 8 bits the dots sit on the stems; at 2 bits most weights collapse to zero.</>}
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
      <h2 style={{ marginTop: "2.2rem" }}>The bottleneck</h2>
      <p>
        At 146 tokens per second the engine is at 60% of the speed limit, and the limit itself is the problem: 495 MB of
        weights per token. Faster code can't read memory faster. The only way past the limit is to read <b>fewer bytes</b>.
      </p>
      <h2>The idea</h2>
      <p>
        Store each weight in fewer bits: 16 instead of 32, or 8, or 4. That's <G k="quantization" />. Weights in a small
        neighbourhood tend to be similar in size, so they're stored in blocks of 32 that share one scale: the scale says how
        big “1” is, and each weight is stored as a small whole number of those steps.
      </p>
      <Block />
      <p>
        8 bits per weight is nearly invisible: about half a percent of error. 4 bits is a real approximation (10%), and the
        model gets slightly worse at predicting text. 2 bits throws most of the information away. The engine converts the
        weights back to ordinary numbers inside the processor, just before multiplying: the memory traffic shrinks, the
        maths stays the same.
      </p>
      <h2>The result</h2>
      <p>
        16-bit weights: <b>146 → 249</b> tokens per second. 8-bit: <b>315</b>, and <b>425</b> after a few more tricks, with the
        same top-choice token 98% of the time. 4-bit: <span className="k">497 tokens per second</span>, the fastest the CPU
        got, and here the cost is visible: its top choice for the next word differs from the 32-bit model's about one time
        in five, and its <G k="perplexity" /> (how surprised it is by real text) rises about 8%. Speed is bought with
        accuracy, a little at a time.
      </p>
      <Climb upto={4} n="15.2" />
      <p>
        Notice the limit line no longer applies: with 4-bit weights the limit is 8× higher, around 1,200. The engine is now
        far below its new limit, because the fixed costs per token (waking threads, the small operations) don't shrink with
        the bytes. That's where the GPU comes in.
      </p>
      <Idea>Decode reads every weight, so smaller weights mean faster decode. Every halving of the bits buys speed, and costs some accuracy.</Idea>
      <Deeper title="The one matrix you can't squeeze">
        <p>
          Quantising every matrix to 8 bits made GPT-2 noticeably worse, and at 4 bits it fell apart (perplexity, a measure
          of how surprised the model is by real text, went from 37 to 516). Testing one matrix at a time found the culprit:
          the final scoring matrix, which is also the embedding table from chapter 3. It's a third of GPT-2's weights and
          the one that hates being compressed. Keeping just that matrix at 16 bits fixed it.
        </p>
        <p>
          Big models don't have this problem as badly: in a 7-billion-weight model that table is about 2% of the weights.
          Small-model lessons don't always transfer.
        </p>
      </Deeper>
    </>
  );
}
