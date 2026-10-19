import { T } from "../learn/kit";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Tk } from "../ui/Tok";
import { claim, decodes, engine, release, useEngine, useRunSource } from "../engine/client";
import type { OpRec } from "../engine/types";
import { NeedsEngine, fitCanvas, useFigureSource, useInView, useWidth } from "../ui/Figure";
import { KIND_LABEL, kindColor, rgb, usePalette, useReducedMotion } from "../ui/theme";
import { fmtBytes, fmtMs, short, sig } from "../lib/fmt";
import { softmaxAt, summariseLogits, unpackLens, type WalkData } from "../lib/recorded";

/**
 * "Inside one token": a scroll-driven walk through one real decode step. As
 * the reader scrolls, a cursor moves through the forward pass op by op; every
 * op lights up with the time, bytes and FLOPs the engine measured for it, and
 * the residual stream (the vector that *is* the token) is shown after each
 * layer. It ends on the probability distribution, which the reader can
 * re-sample at any temperature.
 */
export default function InsideToken() {
  const status = useEngine((s) => s.status);
  const run = useEngine((s) => s.run);
  const busy = useEngine((s) => s.busy);
  const rec = useEngine((s) => s.recorded);
  const src = useRunSource();
  const [viewRef, inView] = useInView<HTMLDivElement>();
  const [live, setLive] = useState<{ key: string; walk: WalkData } | null>(null);
  const ready = status === "ready";

  // Live: the last decode step of the hero run, plus its residual stream,
  // lens column and full logits (one extra analysis pass, ~0.3 s).
  const fwds = decodes(run);
  const last = fwds.at(-1);
  const key = run && last && run.done ? `${run.tokens.map((t) => t.id).join(",")}` : "";
  useEffect(() => {
    if (!ready || src !== "live" || !inView || !run?.done || !last || busy || live?.key === key) return;
    if (!claim("walk")) return;
    const pos = last.contextLen - 1;
    const toks = run.tokens.slice(0, pos + 1).map((t) => t.id);
    (async () => {
      try {
        const logits = await engine.logits(0);
        const resid = await engine.residual(toks, pos);
        const flat = await engine.lens(toks, [pos], 5);
        const C = resid.length / (flat.length / 14);
        const depth = flat.length / 14;
        const lens = unpackLens(flat, toks, [pos], 5, depth, 0).rows.map((r) => r[0]);
        const res: Float32Array[] = [];
        for (let s = 0; s < depth; s++) res.push(resid.slice(s * C, (s + 1) * C));
        setLive({ key, walk: { pos, forward: last, logits: summariseLogits(logits), resid: res, lens } });
      } finally {
        release("walk");
      }
    })();
  }, [ready, src, inView, run, last, busy, key, live?.key]);

  const walk = src === "live" ? (live?.key === key ? live.walk : null) : (rec?.walk ?? null);
  const tokens = src === "live" ? run?.tokens.map((t) => t.id) : rec?.run.tokens.map((t) => t.id);
  useFigureSource(walk ? (src === "live" ? "live" : "recorded") : "none");

  return (
    <div ref={viewRef}>
      {walk && tokens ? (
        <Walk walk={walk} tokens={tokens} />
      ) : src === "live" ? (
        <div className="nv-empty">following one token through the network…</div>
      ) : (
        <NeedsEngine>One token, followed through every operation of the network.</NeedsEngine>
      )}
    </div>
  );
}

interface Step {
  title: string;
  /** op cursor range this step scrubs through */
  from: number;
  to: number;
  /** scroll length, in viewport heights */
  vh: number;
  body: ReactNode;
}

function Walk({ walk, tokens }: { walk: WalkData; tokens: number[] }) {
  const pal = usePalette();
  // below the timer's resolution a duration is "too short to see", not zero
  const res = useEngine((s) => (s.source === "live" ? s.timerResUs : 0)) || 1;
  const fmtT = (ms: number) => (ms * 1000 < res ? `<${Math.round(res)} µs` : fmtMs(ms));
  const reduced = useReducedMotion();
  const ops = walk.forward.ops;
  const N = ops.length;
  const depth = walk.resid.length;
  const tok = tokens[walk.pos];
  const chosen = tokens[walk.pos + 1];
  const ctx = walk.pos + 1;

  // op index boundaries
  const idx = useMemo(() => {
    const firstOf = (pred: (o: OpRec) => boolean) => ops.findIndex(pred);
    const lastOf = (pred: (o: OpRec) => boolean) => ops.length - 1 - [...ops].reverse().findIndex(pred);
    const l0start = firstOf((o) => o.layer === 0);
    const l0mlp = firstOf((o) => o.layer === 0 && o.name === "ln_2");
    const l1start = firstOf((o) => o.layer === 1);
    const headStart = lastOf((o) => o.layer >= 0) + 1;
    return { l0start, l0mlp, l1start, headStart };
  }, [ops]);
  const opsBy = (name: string, layer: number) => ops.find((o) => o.name === name && o.layer === layer);
  const sum = (a: number, b: number, f: (o: OpRec) => number) => ops.slice(a, b).reduce((s, o) => s + f(o), 0);
  const qkv0 = opsBy("qkv", 0), attn0 = opsBy("attn", 0), fc0 = opsBy("fc", 0), lm = ops.find((o) => o.kind === "lm_head");
  const totalMs = sum(0, N, (o) => o.dur);
  const layerMs = sum(idx.l0start, idx.l1start, (o) => o.dur);

  const steps: Step[] = [
    {
      title: "One token walks in",
      from: 0,
      to: idx.l0start,
      vh: 85,
      body: (
        <>
          <p>
            The model has read {ctx} tokens so far. This step turns the newest one, <TokChip id={tok} />, into a guess about what
            comes next.
          </p>
          <p>
            First, <b>embed</b>: look up row {tok} of the token table (768 numbers) and add row {walk.pos} of the position table.
            That vector is the <i>residual stream</i>. Every layer reads it and adds something back. It took{" "}
            <b>{fmtT(ops[0].dur)}</b>.
          </p>
        </>
      ),
    },
    {
      title: "Attention: look back",
      from: idx.l0start,
      to: idx.l0mlp,
      vh: 95,
      body: (
        <>
          <p>
            Layer 1 normalises the vector, then one matrix multiply turns it into a query, a key and a value ({qkv0 ? `${qkv0.dims[1]}→${qkv0.dims[2]}` : "768→2304"},{" "}
            {qkv0 ? fmtBytes(qkv0.bytes) : ""} of weights). The key and value go into the KV cache for later tokens.
          </p>
          <p>
            Then the query is compared with the keys of all {ctx} tokens in the KV cache, in 12 attention heads, and their values are
            mixed. This is the only step where tokens exchange information: {attn0 ? fmtBytes(attn0.bytes) : ""} read in{" "}
            {attn0 ? fmtMs(attn0.dur) : ""}.
          </p>
        </>
      ),
    },
    {
      title: "The feed-forward network",
      from: idx.l0mlp,
      to: idx.l1start,
      vh: 85,
      body: (
        <>
          <p>
            Then the feed-forward network (FFN) widens the vector to 3,072 numbers, applies GELU and narrows it back to 768. Its two
            matrix multiplies hold two-thirds of a layer's weights ({fc0 ? fmtBytes(fc0.bytes * 2) : "19 MB"}), so they take most of its time.
          </p>
          <p>
            One whole layer: <b>{fmtMs(layerMs)}</b>, about {fmtBytes(sum(idx.l0start, idx.l1start, (o) => o.bytes))} read.
          </p>
        </>
      ),
    },
    {
      title: "Eleven more times",
      from: idx.l1start,
      to: idx.headStart,
      vh: 210,
      body: (
        <>
          <p>
            The same 11 operations repeat in layers 2 to 12, each with its own weights. The barcode is the real residual
            stream after each layer. It grows, and a handful of dimensions become very large. GPT-2 is known for these “massive
            activations”.
          </p>
          <p>
            The line under it is the <b>logit lens</b> (fig 3): what the model would say if it stopped at this depth. Early
            layers guess common tokens, and the final answer usually appears only in the last few layers.
          </p>
        </>
      ),
    },
    {
      title: "The LM head: score every token",
      from: idx.headStart,
      to: N,
      vh: 90,
      body: (
        <>
          <p>
            A final LayerNorm, then the biggest single operation: compare the vector with <b>all 50,257</b> token embeddings. GPT-2
            reuses its token table here, so this one matrix multiply reads {lm ? fmtBytes(lm.bytes) : "154 MB"}, about a third of all
            the bytes in the step.
          </p>
          <p>
            It took <b>{lm ? fmtMs(lm.dur) : ""}</b> of the step's {fmtMs(totalMs)}.
          </p>
        </>
      ),
    },
    {
      title: "Pick one",
      from: N,
      to: N,
      vh: 110,
      body: (
        <>
          <p>
            Out come 50,257 scores. A softmax turns them into probabilities. The model chose <TokChip id={chosen} accent />. Try
            the controls. <b>Temperature</b> sharpens or flattens the distribution, and <b>top-k</b> drops all but the k best
            tokens before picking. These are the real logits from this step.
          </p>
        </>
      ),
    },
  ];

  // ---- scroll → cursor ------------------------------------------------------------
  const stepRefs = useRef<(HTMLDivElement | null)[]>([]);
  const [cursor, setCursor] = useState(0);
  const [active, setActive] = useState(0);
  useEffect(() => {
    let raf = 0;
    const update = () => {
      raf = 0;
      const mid = window.innerHeight * 0.55;
      let cur = 0, act = 0;
      stepRefs.current.forEach((el, k) => {
        if (!el) return;
        const r = el.getBoundingClientRect();
        const t = Math.max(0, Math.min(1, (mid - r.top) / r.height));
        if (r.top <= mid) {
          act = k;
          cur = steps[k].from + t * (steps[k].to - steps[k].from);
        }
      });
      setCursor(cur);
      setActive(act);
    };
    const on = () => (raf ||= requestAnimationFrame(update));
    update();
    window.addEventListener("scroll", on, { passive: true });
    window.addEventListener("resize", on);
    return () => {
      window.removeEventListener("scroll", on);
      window.removeEventListener("resize", on);
      cancelAnimationFrame(raf);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [N, idx.headStart]);

  const done = Math.floor(cursor);
  const frac = cursor - done;
  const at = Math.min(N - 1, done);
  const cur = ops[at];
  const acc = (f: (o: OpRec) => number) => sum(0, Math.min(done, N), f) + (done < N ? frac * f(ops[done]) : 0);
  // depth reached: layers fully completed at the cursor (0 = just embedded)
  const layersDone = ops.slice(0, Math.min(done, N)).filter((o) => o.name === "residual_2").length;
  const dDepth = Math.min(depth - 1, cursor >= 1 ? layersDone : 0);
  const finished = cursor >= N - 0.001 && active === steps.length - 1;

  return (
    <div className="nv-scrolly">
      <div className="nv-scrolly-graphic">
        <div className="nv-walk">
          <div className="nv-walk-head">
            <TokChip id={tok} big />
            <span className="nv-walk-arrow" style={{ ["--p" as string]: Math.min(1, cursor / N) }}>
              <i />
            </span>
            <span className={`nv-walk-out${finished || cursor >= N - 0.5 ? " on" : ""}`}>
              <TokChip id={chosen} big accent />
            </span>
          </div>
          <Lanes ops={ops} cursor={cursor} reduced={reduced} />
          <div className="nv-walk-counters">
            <Counter k="time" v={fmtMs(acc((o) => o.dur))} sub={`of ${fmtMs(totalMs)}`} />
            <Counter k="bytes read" v={fmtBytes(acc((o) => o.bytes))} />
            <Counter k="FLOPs" v={short(acc((o) => o.flops))} />
            <Counter k="ops" v={`${Math.min(N, Math.ceil(cursor))}`} sub={`/ ${N}`} />
          </div>
          <div className="nv-walk-op">
            {cursor < N - 0.001 ? (
              <>
                <span style={{ color: kindColor(pal, cur.kind) }}>■</span> {cur.layer >= 0 ? `layer ${cur.layer + 1} · ` : ""}
                <b>{cur.name}</b> · {fmtT(cur.dur)} · {fmtBytes(cur.bytes)} · {short(cur.flops)} FLOP ·{" "}
                {cur.dur > 0 ? `${sig(cur.bytes / cur.dur / 1e6, 2)} GB/s` : ""}
              </>
            ) : (
              <>done: {N} ops, {fmtMs(totalMs)} · {fmtBytes(sum(0, N, (o) => o.bytes))} read</>
            )}
          </div>
          {!finished ? (
            <>
              <Barcode v={walk.resid[dDepth]} all={walk.resid} label={dDepth === 0 ? "residual stream after the embedding" : `residual stream after layer ${dDepth}`} />
              {walk.lens.length > 0 && (
                <div className="nv-walk-lens">
                  <span className="nv-muted">if it stopped here:</span>{" "}
                  {walk.lens[dDepth].top.slice(0, 4).map((t, j) => (
                    <span key={j} className={`nv-chip${t.id === chosen ? " next" : ""}`}>
                      <Tk id={t.id} /> <span className="nv-muted">{(t.p * 100).toFixed(t.p < 0.1 ? 1 : 0)}%</span>
                    </span>
                  ))}
                </div>
              )}
            </>
          ) : (
            <Sampler walk={walk} chosen={chosen} />
          )}
        </div>
      </div>
      <div className="nv-scrolly-steps">
        {steps.map((s, k) => (
          <div key={k} ref={(el) => (stepRefs.current[k] = el)} className={`nv-step${k === active ? " on" : ""}`} style={{ minHeight: `${s.vh}vh` }}>
            <div className="nv-step-card">
              <div className="nv-step-n">{String(k + 1).padStart(2, "0")}</div>
              <h3>{s.title}</h3>
              {s.body}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TokChip({ id, big, accent }: { id: number; big?: boolean; accent?: boolean }) {
  return (
    <span className={`nv-tokchip${big ? " big" : ""}${accent ? " accent" : ""}`}>
      <Tk id={id} />
      {big && <small>#{id}</small>}
    </span>
  );
}

function Counter({ k, v, sub }: { k: string; v: string; sub?: string }) {
  return (
    <div className="nv-stat">
      <span className="k">{k}</span>
      <span className="v">
        {v}
        {sub && <small>{sub}</small>}
      </span>
    </div>
  );
}

/** The forward pass as 14 lanes (embed, 12 layers, head), each op a segment
 *  whose width is its measured time. Lit up to the cursor. */
function Lanes({ ops, cursor, reduced }: { ops: OpRec[]; cursor: number; reduced: boolean }) {
  const pal = usePalette();
  const [ref, w] = useWidth<HTMLDivElement>();
  const lanes = useMemo(() => {
    const out: { label: string; ops: { o: OpRec; i: number }[] }[] = [];
    let curLabel = "";
    ops.forEach((o, i) => {
      const label = o.layer >= 0 ? `L${o.layer + 1}` : out.length === 0 ? "embed" : "head";
      if (label !== curLabel) {
        out.push({ label, ops: [] });
        curLabel = label;
      }
      out[out.length - 1].ops.push({ o, i });
    });
    return out;
  }, [ops]);
  const maxLane = Math.max(...lanes.map((l) => l.ops.reduce((s, x) => s + x.o.dur, 0)));
  const LH = 11, G = 4, LW = 42;
  const H = lanes.length * (LH + G);
  const sx = (w - LW - 4) / maxLane;
  return (
    <div ref={ref} className="nv-walk-lanes">
      {w > 0 && (
        <svg className="nv-svg" viewBox={`0 0 ${w} ${H}`} height={H} role="img" aria-label="ops of one forward pass, lit up as you scroll">
          {lanes.map((l, k) => {
            let x = LW;
            const y = k * (LH + G);
            const laneOn = l.ops[0].i < cursor;
            return (
              <g key={k}>
                <text x={LW - 6} y={y + LH - 2} textAnchor="end" style={{ fontSize: T.xs, fill: laneOn ? pal.fg : pal.muted }}>{l.label}</text>
                <rect x={LW} y={y} width={w - LW - 4} height={LH} fill={pal.code} rx={1.5} />
                {l.ops.map(({ o, i }) => {
                  const ow = Math.max(0.8, o.dur * sx);
                  const x0 = x;
                  x += ow;
                  const lit = i < Math.floor(cursor);
                  const curOp = i === Math.floor(cursor) && cursor < ops.length;
                  const part = curOp ? cursor - i : lit ? 1 : 0;
                  return (
                    <g key={i}>
                      <rect x={x0} y={y} width={ow} height={LH} fill={kindColor(pal, o.kind)} opacity={0.16} />
                      {part > 0 && <rect x={x0} y={y} width={ow * part} height={LH} fill={kindColor(pal, o.kind)} opacity={lit ? 0.9 : 1} />}
                      {curOp && <rect x={x0 - 1} y={y - 1.5} width={ow + 2} height={LH + 3} fill="none" stroke={pal.fg} strokeWidth={1.2} rx={1.5} style={reduced ? undefined : { transition: "x 0.08s" }} />}
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
      )}
      <div className="nv-legend" style={{ marginTop: "0.35rem", fontSize: "var(--t-xs)" }}>
        {[...new Set(ops.map((o) => o.kind))].map((k) => (
          <span key={k}>
            <i style={{ background: kindColor(pal, k) }} />
            {KIND_LABEL[k] ?? k}
          </span>
        ))}
        <span className="nv-muted">width = measured time</span>
      </div>
    </div>
  );
}

/** 768 numbers as 768 thin bars: warm = positive, cool = negative, on one
 *  shared scale across depths so you can see the stream grow. */
function Barcode({ v, all, label }: { v: Float32Array; all: Float32Array[]; label: string }) {
  const pal = usePalette();
  const [ref, w] = useWidth<HTMLDivElement>();
  const cv = useRef<HTMLCanvasElement>(null);
  // colour scale: this depth's 98th percentile |v|, so the few massive
  // dimensions saturate and the rest stays readable; the stream's growth is
  // shown separately by the size bar
  const scale = useMemo(() => {
    const a = Array.from(v, Math.abs).sort((x, y) => x - y);
    return a[Math.floor(a.length * 0.98)] || 1;
  }, [v]);
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  const maxNorm = useMemo(() => Math.max(...all.map((d) => Math.sqrt(d.reduce((s, x) => s + x * x, 0)))), [all]);
  const big = useMemo(() => {
    const idx = Array.from(v.keys()).sort((a, b) => Math.abs(v[b]) - Math.abs(v[a]));
    return idx.slice(0, 3);
  }, [v]);
  useEffect(() => {
    const c = cv.current;
    if (!c || !w) return;
    const H = 34;
    const ctx = fitCanvas(c, w, H);
    ctx.clearRect(0, 0, w, H);
    const warm = rgb(pal.accent), cool = rgb(pal.cool), bg = rgb(pal.code);
    const bw = w / v.length;
    for (let i = 0; i < v.length; i++) {
      const t = Math.min(1, Math.abs(v[i]) / scale);
      const c0 = v[i] >= 0 ? warm : cool;
      const k = Math.pow(t, 0.7);
      ctx.fillStyle = `rgb(${bg[0] + (c0[0] - bg[0]) * k},${bg[1] + (c0[1] - bg[1]) * k},${bg[2] + (c0[2] - bg[2]) * k})`;
      const hh = H * (0.25 + 0.75 * k);
      ctx.fillRect(i * bw, (H - hh) / 2, Math.max(1, bw), hh);
    }
  }, [v, w, pal, scale]);
  return (
    <div ref={ref} className="nv-walk-barcode">
      <div className="nv-muted" style={{ fontSize: "var(--t-xs)", display: "flex", justifyContent: "space-between", gap: "0.6rem", flexWrap: "wrap" }}>
        <span>{label} · 768 numbers</span>
        <span>
          ‖x‖ = {sig(norm, 3)} · biggest dims {big.map((i) => `#${i}`).join(" ")}
        </span>
      </div>
      <canvas ref={cv} className="nv-canvas" />
      <div className="nv-walk-size" title="size of the residual stream, relative to its size after the last layer">
        <i style={{ width: `${(100 * norm) / maxNorm}%` }} />
      </div>
    </div>
  );
}

/** The final distribution, re-sampled at the reader's temperature / top-k. */
function Sampler({ walk, chosen }: { walk: WalkData; chosen: number }) {
  const pal = usePalette();
  const [temp, setTemp] = useState(0.8);
  const [topK, setTopK] = useState(40);
  const [draws, setDraws] = useState<number[]>([]);
  const { probs, tailMass } = useMemo(() => softmaxAt(walk.logits, temp), [walk, temp]);
  // top-k: keep the k best, renormalise; k = 0 means keep everything
  const kept = topK > 0 ? probs.slice(0, topK) : probs;
  const keptMass = kept.reduce((s, p) => s + p.p, 0) + (topK > 0 ? 0 : tailMass);
  const shown = kept.slice(0, 10).map((p) => ({ ...p, q: p.p / keptMass }));
  const restQ = 1 - shown.reduce((s, p) => s + p.q, 0);
  const draw = () => {
    const out: number[] = [];
    for (let n = 0; n < 16; n++) {
      let r = Math.random() * keptMass;
      let pick = -1;
      for (const p of kept) {
        r -= p.p;
        if (r <= 0) {
          pick = p.id;
          break;
        }
      }
      out.push(pick);
    }
    setDraws(out);
  };
  return (
    <div className="nv-walk-sampler">
      <div className="nv-controls" style={{ gap: "0.6rem 1.2rem" }}>
        <label className="nv-field">
          temperature
          <input type="range" min={0} max={2} step={0.05} value={temp} onChange={(e) => setTemp(+e.target.value)} />
          <span className="nv-num" style={{ width: "2.6rem" }}>{temp === 0 ? "greedy" : temp.toFixed(2)}</span>
        </label>
        <label className="nv-field">
          top-k
          <input type="range" min={0} max={8} step={1} value={topK === 0 ? 8 : Math.round(Math.log2(topK))} onChange={(e) => setTopK(+e.target.value === 8 ? 0 : 2 ** +e.target.value)} />
          <span className="nv-num" style={{ width: "2.6rem" }}>{topK === 0 ? "all" : topK}</span>
        </label>
        <button className="nv-btn" onClick={draw} style={{ padding: "0.35rem 0.8rem" }}>sample ×16</button>
      </div>
      <div className="nv-probs" style={{ marginTop: "0.6rem" }}>
        {shown.map((d) => (
          <div key={d.id} className={`nv-prob${d.id === chosen ? " chosen" : ""}`}>
            <span className="tk"><Tk id={d.id} /></span>
            <span className="b">
              <i style={{ width: `${Math.max(0.5, d.q * 100)}%`, transition: "width 0.2s" }} />
            </span>
            <span className="pc">{(d.q * 100).toFixed(d.q < 0.1 ? 1 : 0)}%</span>
          </div>
        ))}
        <div className="nv-prob">
          <span className="tk nv-muted">{topK === 0 ? "50k others" : kept.length > 10 ? `${kept.length - 10} others` : "nothing else"}</span>
          <span className="b">
            <i style={{ width: `${Math.max(0, restQ * 100)}%`, background: pal.muted }} />
          </span>
          <span className="pc">{(Math.max(0, restQ) * 100).toFixed(1)}%</span>
        </div>
      </div>
      {draws.length > 0 && (
        <div style={{ marginTop: "0.6rem", display: "flex", flexWrap: "wrap", gap: "0.3rem" }}>
          {draws.map((id, i) => (
            <span key={i} className={`nv-chip${id === chosen ? " next" : ""}`}>{id < 0 ? "(tail)" : <Tk id={id} />}</span>
          ))}
        </div>
      )}
    </div>
  );
}
