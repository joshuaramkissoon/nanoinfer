# 03 — Explorable writeup + running nanoinfer live in the browser

Scope: how to make the nanoinfer post outstanding as an *explorable* (the reader manipulates the real engine), and how to ship the Rust engine to the browser (WASM, threads, WebGPU, weights, instrumentation), then how to embed it in `~/dev/jkmr`.

Confidence markers: **[verified]** I checked a primary source this session. **[known]** stable, well-established knowledge. **[uncertain]** needs a spike before relying on it.

---

## 1. What the best explorables actually do

### Exemplars, distilled

| Work | What makes it work | Live vs static | Steal this |
|---|---|---|---|
| **Bartosz Ciechanowski** (ciechanow.ski — e.g. /gps, /bicycle, /airfoil) | Dozens of small demos, each introducing **one** new variable. Prose sets up a question, the demo answers it, the next paragraph uses what you just felt. Hand-written WebGL/canvas, no framework. Demos only render while on screen. Almost every figure can be dragged; sliders sit directly under figures. | Everything live, but each demo is tiny and scoped. | One concept per figure. Put the control in the figure, not in a side panel. Lazily init demos on scroll. |
| **distill.pub** (e.g. "Communicating with Interactive Articles", distill.pub/2020/communicating-with-interactive-articles; "Attention and Augmented RNNs") | Figures are part of the argument rather than decoration. Margin notes, hover-linked equations and diagrams, small multiples. Explicitly treats "research debt" as something the explanation pays down. | Mixed. Often precomputed data with live re-rendering. | Hover a term in the prose to highlight it in the figure (and the other way round). Small multiples over a parameter. |
| **Brendan Bycroft LLM viz** (bbycroft.net/llm, github.com/bbycroft/llm-viz) | A 3D walkthrough of every tensor in a GPT, with a guided narrative and a free camera. **[verified]** It runs inference on a tiny minGPT "sort ABC" model. The README says GPT-2-sized weights aren't shipped because they are "100's of MBs". | Live, but only on a toy model. | Seeing shapes and magnitudes of every intermediate is compelling. **Lesson:** even Bycroft avoided shipping real GPT-2 weights. Downloading 125–250MB is a product decision we must design around (§4). |
| **Polo Club Transformer Explainer** (poloclub.github.io/transformer-explainer, arxiv.org/abs/2408.04619) | **[verified]** Live GPT-2 small through ONNX Runtime Web. Weights are split into chunks for parallel download and cached in IndexedDB. Svelte + D3. Sankey-style token-centric overview, one block and one head at a time (progressive disclosure), temperature and top-k/top-p sliders. 490k+ users; 74% of a 145-person formative study preferred it over a blog post. | Live model; the viz is computed from real activations. | Temperature and top-k re-sampling from the *same* logits gives instant feedback. Show one head at a time, with paging between heads. |
| **Jay Alammar** (jalammar.github.io/illustrated-transformer) | Static illustrations, but with **consistent visual vocabulary** (the same colour or box always means the same tensor) and very gradual build-up. | Static. | Pick a visual grammar for "token", "vector", "weight matrix" and "KV entry", then never break it. |
| **Simon Boehm — CUDA matmul** (siboehm.com/articles/22/CUDA-MMM) | **[verified]** A worklog of 10 kernels, each tied to one bottleneck, with a running table of GFLOP/s and % of cuBLAS (1.3% → 93.7%). Napkin-math lower bounds come first, then each kernel follows motivation → concept → diff-only code → measured result. Negative results are kept. | Static. | **The spine for our post.** A running results table with an explicit "% of speed-of-light" column, and diff-only code per step. |
| **Andrew Chan — yalm** (andrewkchan.dev/posts/yalm.html) | **[verified]** Frames everything against a memory-bandwidth "speed of light" (bandwidth ÷ bytes read per token). Goes from 0.6 tok/s to ~64 tok/s on Mistral-7B, compared against llama.cpp and calm. Profiler screenshots, honest regressions. Weakness: tok/s lives in prose with no running table, so the reader has to track it. | Static. | The roofline/SoL framing, and comparing against a known-good reference engine. Fix his weakness with a persistent ladder widget. |
| **Red Blob Games** (redblobgames.com, e.g. /pathfinding/a-star/introduction.html) | Every diagram is draggable. Code and diagram sit side by side, and the code is the *real* code. Amit writes about this in "making-of" posts. | Live. | Show the real Rust kernel next to the figure it drives, and highlight the line currently executing. |
| **Nicky Case** (ncase.me, "Parable of the Polygons", "Evolution of Trust") | Play first, explain after. The reader forms an intuition, *then* gets the principle. Playful, with short text between interactions. | Live sims. | Open with a "make it go" moment: the reader types a prompt and GPT-2 runs in their tab *before* any explanation. |
| **How to Scale Your Model** (jax-ml.github.io/scaling-book) | Rooflines and arithmetic intensity drive every decision. Worked problems with hidden answers. Clean, consistent diagrams of sharding and memory. | Mostly static. | Spoiler-style "predict the number, then reveal" quizzes. Arithmetic intensity as the recurring lens. |
| **Lilian Weng** (lilianweng.github.io) | Dense, rigorous, heavily cited, with a clear taxonomy. | Static. | Citations and precise definitions, which an engineering audience trusts. |

### Bret Victor, applied concretely

From "Explorable Explanations" (worrydream.com/ExplorableExplanations) and "Up and Down the Ladder of Abstraction" (worrydream.com/LadderOfAbstraction):

1. **Reactive documents.** Numbers in the prose are live. "At **f16**, each token reads **249 MB** of weights, so on your machine's ~**X GB/s** the ceiling is **Y tok/s**." Here `f16` is a draggable or clickable token and X and Y come from a real benchmark run in the reader's tab. Implement this as an inline `<LiveNumber>` component.
2. **Explorable examples.** Every claim has a "try it" next to it. "The KV cache makes decode O(n) per token instead of O(n²)" is backed by a toggle that turns the cache off and plots the measured per-token latency.
3. **Contextual information.** Hover any op in a timeline to see its shape, FLOPs, bytes and achieved GB/s, plus the Rust source line.
4. **The ladder.** Show one concrete instance (this token, layer 5, head 3), then **abstract over a parameter**: sweep a variable and show the whole family as a line or small multiples (tile size → GFLOP/s, threads → tok/s, context length → latency). Then **abstract over time**: show the whole hillclimb as one chart. Let the reader climb down again by clicking a point on the abstract chart to load that concrete configuration. This is the most important single design rule for our post.
5. **Show the data, not a depiction of it.** Bret Victor's complaint about "pictures of the idea" maps directly onto Josh's "not mockups" requirement. Every figure should be able to say where its numbers came from (run id, browser, backend).

### Practical rules (synthesised)

- **Pacing:** alternate text, then one figure and one control. Roughly 150–300 words between interactive figures. No figure should introduce more than one new control.
- **Interaction density:** first figure has zero required interaction and autoplays (reduced-motion aware). Later figures add controls. Advanced controls go behind a "more" disclosure.
- **Numbers:** always give units, always use a fixed number of significant figures, and use tabular-nums monospace (the site is already monospace). Show **ratio to the ceiling** alongside absolute values (Boehm's % of cuBLAS → our % of speed-of-light). Show provenance ("measured just now in your browser · Chrome · WASM SIMD · 1 thread").
- **Honest live data:** browsers vary hugely. Pair every live number with the **reference number from Josh's machine** (native, from `docs/LOG.md`) so the reader can compare.
- **Fail soft:** every live figure has a precomputed "recorded trace" fallback, captured from the real engine natively. It must be labelled as recorded, not live. The figure works before the weights download, on Safari without threads, and when JS is slow.

---

## 2. Live visualisation designs for the post (11)

All are driven by real engine output. "Recorded" means a trace captured from the native or WASM engine and shipped as a small binary file. "Live" means computed in the reader's tab.

1. **"Run it" hero: token stream with probabilities.** The reader types a prompt and presses Generate. Tokens stream in, each underlined by p(token), and hovering one shows the top-10 next-token distribution as bars. Temperature, top-k and top-p sliders *re-sample from the cached logits* without re-running the model, so feedback is instant. A tok/s counter sits in the corner. *Data:* `token` events (`topk`, `logprob`, `entropy`). *Tech:* DOM/SVG. *Before the weights load:* replay a recorded generation at its real timing.

2. **The Ladder: the whole hillclimb as one chart.** An x-axis of optimisation steps (naive → … → final) and a y-axis of tok/s on a log scale, with a dashed line at the memory-bandwidth speed of light. Two series: Josh's native measurement and *your browser*, which fills in as the reader clicks "benchmark this step". Clicking a step shows its diff summary and hypothesis and jumps to that section. It is pinned or sticky in a compact form throughout the post, which fixes yalm's "track it in prose" weakness. *Data:* `LOG.md` → JSON at build time, plus live `run_end` events from engine variants (§3.2 runtime knobs). *Tech:* SVG.

3. **Speed-of-light gauge (roofline for decode).** The reader picks a dtype (f32/f16/int8) and context length. The figure computes bytes per token (weights plus KV read) and multiplies by a **live-measured memory bandwidth** (a WASM `memcpy`/streaming-sum microbenchmark run in the tab) to give the ceiling, then shows a needle for the achieved tok/s. This makes clear why quantisation is the biggest lever for decode. *Data:* `bandwidth_probe` result, `mem` event, `run_end`. *Tech:* SVG.

4. **Where the time goes: per-token op flame strip.** One decode step drawn as a horizontal bar: 12 layer blocks, each subdivided into ln1, qkv matmul, attention, proj, ln2, mlp_up, gelu, mlp_down, plus the final lm_head (often the single biggest op at 50257×768). Small multiples stack one strip per optimisation step, so the reader sees matmul shrink and the "long pole" move. Hover shows shape, FLOPs, bytes and achieved GFLOP/s or GB/s. *Data:* `op` events. *Tech:* Canvas 2D (about 100 rects × N steps, with hover hit-testing). Needs fine timers (§3.4).

5. **KV cache on/off: watching O(n²) happen.** A toggle switches the KV cache on or off, and a context-length slider runs up to 1024. A live line chart plots per-token latency against position. With the cache off it grows linearly per token (quadratic total); with it on it stays nearly flat. Beside it, a grid fills cell by cell as K/V rows are written for each layer, with a memory counter (`2 × layers × T × d_model × bytes`). *Data:* `token.dur_us`, `kv` events. *Tech:* SVG chart plus Canvas grid.

6. **Attention small multiples (12 layers × 12 heads).** Show the reader's own prompt. Clicking a token draws every head's attention row as a 12×12 grid of mini-heatmaps (the distill small-multiples move). Clicking one head expands it to a full T×T matrix with token labels. This ties the KV cache to *what it's used for*. *Data:* `tensor` captures of the attention probs (pulled on demand, §3.3). *Tech:* Canvas or WebGL (144 × T² cells; at T=128 that's 2.4M cells, so use Canvas with ImageData or a single WebGL texture).

7. **Matmul loop order and tiling: the real kernel, benchmarked live.** One panel animates a small (say 16×16) matmul showing which elements of A, B and C are touched, with a simulated cache hit/miss counter, for naive i-j-k, loop-interchanged and tiled versions. A second panel runs **the actual nanoinfer matmul kernel** at 768×768 (and 768×3072) with a **tile-size slider**, and plots GFLOP/s against tile size as the reader drags. Clicking a point animates that tile configuration (climbing up and down the ladder). *Data:* `bench_matmul(m,n,k,tile,variant)` → timing. *Tech:* Canvas animation plus SVG chart.

8. **SIMD lanes.** A dot product drawn as scalar (one multiply-add at a time) versus `f32x4` (four lanes), with relaxed-SIMD FMA where supported. Live bench buttons run the real scalar and SIMD kernels and show the speed-up. Reports honestly that `simd128` is 4-wide, so the ceiling is about 4×, and where the measurement lands. *Data:* microbench exports. *Tech:* SVG.

9. **Threads scaling (Amdahl, live).** A thread-count slider from 1 to `navigator.hardwareConcurrency` runs a decode benchmark at each setting. It plots measured tok/s against threads with an ideal line and a fitted Amdahl curve, and the reader can see the "serial fraction" number. The figure is disabled with an explanation when `crossOriginIsolated` is false; a recorded trace from Josh's M-series machine is shown instead. *Data:* `run_end` per thread count. *Tech:* SVG.

10. **Quantisation: what int8 costs.** Pick a weight matrix (e.g. `h.5.mlp.c_fc`) to see its real value histogram, the int8 grid overlaid, and the per-block scale. A block-size slider (32/64/128) shows the reconstruction error. Run the same prompt in f16 and in int8 side by side, then report top-1 agreement %, KL divergence per token, and download and memory size. *Data:* `tensor` capture of the weights (or a precomputed histogram), and two runs' `token` events. *Tech:* SVG histogram plus DOM.

11. **Prefill vs decode, and CPU vs GPU race.** A prompt-length slider measures prefill throughput (batched, compute-bound) against decode (one token, memory-bound), showing why their tok/s differ by an order of magnitude. A second mode is a side-by-side race of WASM-SIMD CPU and WebGPU on the same prompt. It shows the WebGPU dispatch-overhead story: at batch 1, small GPT-2 kernels may be dominated by per-dispatch cost (see §3.5). *Data:* `run_start.config.backend`, `op`, `run_end`. *Tech:* SVG plus DOM.

Optional 12th: **wasm linear memory map.** A 1D bar of the WASM heap showing regions for weights, KV, scratch and activations, live as generation proceeds. It is cheap to build from `mem` events and makes "500MB of f32 in a 4GB address space" tangible.

**Prioritise:** 1, 2, 4, 5 and 7 are the core. 3 and 10 carry the quantisation section. 6 is the crowd-pleaser. 9 and 11 depend on threads and WebGPU working.

---

## 3. Technical architecture

```
nanoinfer/
  crates/
    core/      # model, kernels, sampler, KV cache. No I/O. Generic over Tracer.
    trace/     # event structs + compact binary encoder (shared native & wasm)
    cli/       # native bench/hillclimb harness; can dump traces to file (recorded fallbacks)
    wasm/      # wasm-bindgen facade: Engine class, load_weights, generate, bench_* fns
    gpu/       # (later) wgpu backend; same Backend trait; compiles to WebGPU on wasm32
  web/         # TS package: worker.ts, engine-client.ts (typed RPC), trace decoder
```

### 3.1 Build targets

- **Baseline build (stable Rust):** `wasm32-unknown-unknown` with `RUSTFLAGS="-C target-feature=+simd128"`, `wasm-pack build --target web`, single-threaded. Use `core::arch::wasm32` intrinsics (`f32x4_mul`, `f32x4_add`, `v128_load`). Shipped to everyone. **[known]** SIMD128 is in all current browsers.
- **Relaxed-SIMD variant:** `+relaxed-simd` gives `f32x4_relaxed_madd` (FMA). **[verified, caniuse]** Chrome 114+ and Firefox 146+ support it; Safari 26.x doesn't. Build a separate `.wasm` and pick at runtime with `wasm-feature-detect` (`relaxedSimd()`).
- **Threads variant:** wasm-bindgen-rayon. **[verified]** It requires pinned nightly, `+atomics,+bulk-memory`, `-Z build-std=panic_abort,std`, linker args `--shared-memory --import-memory --max-memory=…` (raise from the README's 1GB to 4GB for f32 weights), `--target web` only, and `await initThreadPool(n)`. Vite is supported. **Run all engine calls inside a dedicated Worker**, because the main thread can't `Atomics.wait`, and you don't want rayon's caller blocking UI anyway. Select at runtime via `crossOriginIsolated && await threads()`.
- So there are about 3–4 `.wasm` artefacts (simd / simd+relaxed / threads+simd[+relaxed]). Each is small (the engine code is likely <1MB). Weights are separate.

### 3.2 WASM API (wasm-bindgen facade)

```rust
#[wasm_bindgen] pub struct Engine { /* model, kv, tracer, cfg */ }
#[wasm_bindgen] impl Engine {
  pub fn new(cfg: JsValue) -> Engine;                      // dtype, backend, threads, kernel variant, capture spec
  pub fn load_shard(&mut self, idx: u32, bytes: &[u8]);    // called per shard as it streams in
  pub fn finish_load(&mut self) -> Result<(), JsError>;
  pub fn set_knobs(&mut self, knobs: JsValue);             // kv_cache on/off, matmul variant, tile size, threads
  pub fn prefill(&mut self, tokens: &[u32]) -> Vec<u8>;    // returns trace chunk
  pub fn step(&mut self, sampling: JsValue) -> Vec<u8>;    // one decode token -> trace chunk
  pub fn resample(&self, sampling: JsValue) -> u32;        // re-sample last logits (viz 1)
  pub fn read_tensor(&self, capture_id: u32) -> Vec<f32>;  // pull-on-demand captures (viz 6, 10)
}
#[wasm_bindgen] pub fn bench_matmul(m:u32,n:u32,k:u32,variant:u8,tile:u32,iters:u32) -> f64;
#[wasm_bindgen] pub fn bench_bandwidth(bytes:u32, iters:u32) -> f64;
```

Key design choice: **every hillclimb step stays callable at runtime.** Keep old kernels behind an enum (`MatmulVariant::{Naive, Interchanged, Tiled{t}, Simd, SimdTiled, Threaded}`, `kv_cache: bool`, `dtype`) instead of deleting them. The Ladder (viz 2) and the sliders then run *the actual historical code* in one binary, not a re-implementation. This needs discipline in the nanoinfer repo from day one: **don't overwrite kernels, add variants.** Where a step is structural (e.g. memory layout), a git tag per step and a native-only measurement is acceptable. Label those points "native only".

Tokenizer: GPT-2 BPE in Rust inside the same crate (vocab plus merges is about 1.5MB, which can be lazily fetched), so tokens and text are consistent with the engine.

### 3.3 Worker and React plumbing

- `engine.worker.ts` owns the WASM instance (and the rayon pool, which spawns sub-workers). The main thread never touches WASM.
- Typed RPC over `postMessage` (hand-rolled, ~80 lines, or Comlink at ~1KB). Trace chunks come back as **transferable `ArrayBuffer`s** (zero-copy).
- One `EngineProvider` (React context) per post, created lazily when the first live figure mounts. It exposes `useEngine()` → `{status, load(), generate(prompt, opts), bench(...)}` and a small subscribe-able store (`useSyncExternalStore`) of the decoded trace. All figures read the **same run**, so viz 1, 4, 5 and 6 can all reflect the generation the reader just triggered. This is a strong, cohesive effect.
- **Push vs pull:** cheap events (op timings, tokens, KV, mem) are pushed per token. Heavy tensors (attention probs: 12×12×T floats per token per row; full T×T matrices) are **captured lazily**. The engine keeps only what the `CaptureSpec` asks for, and figures `read_tensor()` on click. Don't stream 2MB/token you don't render.

### 3.4 Trace / event schema (proposal)

Principles: fixed-size binary records written into a `Vec<u8>` ring per token (no JS boundary crossing per event), strings interned once in `run_start`, the same encoder native and wasm (so `cli` can dump recorded traces the web decoder reads), and a zero-cost `NoopTracer` via generics (`fn forward<T: Tracer>`) so that benchmarks aren't polluted. Bench both with and without tracing and publish the overhead.

TypeScript view after decoding:

```ts
type RunId = number;
type OpKind = "embed"|"ln"|"matmul"|"attn_scores"|"softmax"|"attn_mix"|"gelu"|"residual"|"lm_head"|"sample";

type TraceEvent =
  | { t: "run_start"; run: RunId; ts_us: number;
      config: { model: "gpt2-124m"; dtype: "f32"|"f16"|"int8"; backend: "wasm"|"wasm-simd"|"wasm-relaxed"|"wasm-threads"|"webgpu"|"native";
                threads: number; step: string /* hillclimb step id, e.g. "s07-tiled-matmul" */;
                knobs: Record<string, number|string|boolean>; ua?: string; isolated: boolean; timer_res_us: number };
      op_names: string[]; prompt_tokens: number[] }
  | { t: "op"; run: RunId; tok: number; layer: number /* -1 = outside blocks */; op: number /* index into op_names */;
      kind: OpKind; start_us: number; dur_us: number; flops: number; bytes: number; shape: [number, number, number] }
  | { t: "token"; run: RunId; idx: number; phase: "prefill"|"decode"; id: number; text: string;
      logprob: number; entropy: number; topk: Array<[id: number, p: number]>; dur_us: number }
  | { t: "kv"; run: RunId; len: number; capacity: number; bytes: number }
  | { t: "mem"; run: RunId; heap_bytes: number; weights: number; kv: number; scratch: number }
  | { t: "capture"; run: RunId; id: number; tok: number; layer: number; head?: number;
      name: "attn_probs"|"residual"|"logits"|"weights"; shape: number[]; dtype: "f32" }   // payload pulled via read_tensor(id)
  | { t: "run_end"; run: RunId; n_prompt: number; n_gen: number; prefill_ms: number; decode_tok_s: number; peak_heap: number };
```

Binary `op` record ≈ 32 bytes: `u16 tok, i8 layer, u8 op, u8 kind, u32 start_us, u32 dur_us, f32 flops_g, f32 bytes_m, u16×3 shape`. GPT-2 decode is about 12×10+3 ≈ 125 ops/token, so 4KB per token. Trivial.

`flops` and `bytes` are computed analytically from shapes in Rust (not measured), so figures can derive achieved GFLOP/s and GB/s and arithmetic intensity. That's the Boehm/scaling-book lens.

**Timer resolution caveat [known]:** `performance.now()` is coarsened to about 100µs in non-isolated contexts (Chrome) and about 5µs when `crossOriginIsolated`. Firefox and Safari coarsen differently. Many GPT-2 small ops are sub-100µs, so (a) record `timer_res_us` in `run_start`, (b) when it's coarse, time ops by **aggregating over K tokens** or run a dedicated "profile mode" that repeats each op N times, and (c) say this in the figure caption. From WASM, call `performance.now()` via `web_sys`/`js_sys` (or import a tiny JS function). Each call crosses the boundary (~tens of ns), so that's fine per op, but not per inner-loop iteration.

**WebGPU timings:** CPU timestamps around `queue.submit` are meaningless for per-op GPU cost. Use the `timestamp-query` feature where available (Chrome; **[uncertain]** availability and quantisation on Safari and Firefox), otherwise only report whole-token time.

### 3.5 WebGPU (wgpu → browser)

- wgpu compiles to `wasm32` and targets the browser's WebGPU. The same WGSL kernels can run natively (Metal/Vulkan) and in-browser, which is a nice symmetry for the post. Prior art: `web-rwkv` (pure-Rust WebGPU RWKV, runs in browser), `rullama` (Rust → WASM, Gemma on WebGPU via wgpu, hand-written WGSL), Ratchet (huggingface/ratchet), Burn's wgpu backend, and candle wasm examples (phi, llama2-c, whisper, quant-qwen3, moondream; CPU WASM) at github.com/huggingface/candle/tree/main/candle-wasm-examples.
- **Availability [verified, web.dev Nov 2025]:** WebGPU ships in Chrome/Edge, Safari 26 (macOS/iOS/iPadOS 26), Firefox 141+ on Windows and 145+ on macOS ARM. **Subgroups:** Chrome/Edge 144+ only (caniuse). **shader-f16:** **[uncertain]**, so feature-detect it.
- **Dispatch overhead [verified, arxiv.org/abs/2604.02344]:** 24–36µs per dispatch on Vulkan and 32–71µs on Metal (API overhead). At batch 1, "per-operation overhead dominates regardless of kernel quality", and fusion gave +53% on Vulkan. GPT-2 small with about 125 dispatches/token could therefore hit a **~4–9ms/token floor (~110–250 tok/s) from dispatch alone**. That's a great section: fuse (ln+matmul, qkv into one matmul, gelu into mlp_up, softmax+mix), and show dispatch count per step in viz 4. **[uncertain]** whether batching all dispatches into a single command encoder and submit amortises this in-browser. Spike it.
- Reference perf: WebLLM retains "up to 80% of native" (arxiv.org/abs/2412.15803), using TVM-compiled WebGPU kernels plus WASM. That's a sensible "what good looks like" bar for the WebGPU section.
- WebGPU doesn't need cross-origin isolation, so it may beat threaded WASM as the "fast path" on Safari.

### 3.6 Cross-origin isolation (COOP/COEP) on Vercel — the important gotcha

- Threads need `SharedArrayBuffer`, which needs `crossOriginIsolated`, which needs `Cross-Origin-Opener-Policy: same-origin` plus `Cross-Origin-Embedder-Policy: require-corp` (or `credentialless`) **on the top-level document**.
- Vercel: add `headers` entries in `vercel.json` with a `source` pattern (Vercel KB: vercel.com/kb/guide/fix-shared-array-buffer-not-defined-nextjs-react). jkmr currently has only a SPA rewrite.
- **SPA trap [known, high-impact]:** isolation is decided when the *document* loads. jkmr is a React Router SPA. If a reader lands on `/` and clicks through to `/blog/nanoinfer`, there's no new document load, so the post page is **not isolated** even if that path has the headers. Options:
  - **(A, recommended) Apply COOP+COEP site-wide.** jkmr self-hosts its fonts and has no third-party scripts or images, according to `index.html` and `index.css`. All post images live under `/blog/…`. So `require-corp` costs nothing today. Costs: any future cross-origin embed (YouTube iframe, external image, analytics script) needs CORP/CORS or `credentialless`, and `COOP: same-origin` severs `window.opener` for popups (irrelevant here). Verify the OG/Twitter crawlers don't care (they don't execute COEP).
  - (B) Headers only on `/blog/nanoinfer` and use `<Link reloadDocument>` (or a plain `<a>`) to that post wherever it's linked. Lower blast radius, but easy to break.
  - Note: an iframe can't be isolated unless its parent is, so "put the demo in an isolated iframe" doesn't work.
- `credentialless` would loosen the embed constraint, but **[verified]** Safari hasn't shipped it (WebKit bug 230550 still open). Safari *does* support `require-corp`, so use `require-corp`.
- Cross-origin weight fetches are fine under `require-corp` as long as they're CORS fetches (they are with `fetch()`).
- **Design consequence:** threads are an *enhancement*. The baseline experience (single-thread SIMD, plus WebGPU where available) must be good without isolation. Viz 9 degrades to a recorded trace.

---

## 4. Weights hosting plan

GPT-2 124M ≈ 124.4M params (the `wte` of 50257×768 is about 38.6M and tied with lm_head). Sizes: **f32 ≈ 498MB, f16 ≈ 249MB, int8 ≈ 125MB + scales (~130MB)**. Never ship f32 over the wire. If a "naive f32" step must run live, upcast f16 → f32 in the worker (500MB of heap is OK on desktop under wasm32's 4GB limit, but too much for most phones).

**Plan:**
1. **Format:** a custom packed file (or safetensors) per dtype, **sharded into ~16–32MB files** plus a `manifest.json` (shard list, sha256, tensor offsets, version). Shards enable parallel download, a progress bar, and resume. Transformer Explainer does exactly this (chunks plus IndexedDB). Pre-transpose and lay out tensors in the engine's kernel layout at pack time, so load is just a memcpy into the WASM heap via `load_shard`.
2. **Host:** a Hugging Face Hub model repo (e.g. `joshuaramkissoon/nanoinfer-gpt2`). It's free, CDN-backed, and **[verified-ish]** plain GETs to `/resolve/main/...` work cross-origin via redirect to the Xet bridge. **[verified]** HTTP *Range* requests from browsers currently fail CORS preflight on the Xet hop (discuss.huggingface.co/t/cors-support-for-http-range-requests-on-dataset-files/172172), so **use whole-shard GETs, never Range**. Sharding makes Range unnecessary anyway. **Fallback:** Cloudflare R2 public bucket with CORS (`Access-Control-Allow-Origin: https://www.joshuaramkissoon.com`) and zero egress fees. **Avoid Vercel static hosting for weights:** Hobby bandwidth limits and 125–250MB per reader add up fast. **[uncertain]** exact current Vercel limits.
3. **Cache:** store fetched shards in the **Cache API** keyed by the versioned URL (`.../v3/f16/shard-07.bin`), and call `navigator.storage.persist()`. OPFS is the alternative if we want to stream-write without holding a second copy. Either way, stream `response.body` into the worker and hand each shard to `load_shard` as it completes, which keeps peak JS memory to about one shard plus the WASM heap. Show "cached ✓ 125MB" with a "clear" button (being a good citizen).
4. **UX:** **never auto-download.** The hero shows a recorded generation replaying in real time, and a clear button reads "Load GPT-2 into this tab · int8 · 130MB · runs locally, nothing sent anywhere". Default to **int8 on mobile or `deviceMemory ≤ 4`**, and f16 on desktop. After the first load, every figure on the page goes live together.
5. **Recorded traces:** the `cli` dumps binary traces (the same schema) for every figure's default state from Josh's machine. These are small (KBs to low MBs) and live under `public/blog/nanoinfer/traces/` on Vercel. They're the fallback and also the "Josh's machine" reference series.
6. **[uncertain]** iOS Safari per-tab memory limits for a ~400MB+ WASM heap. Test int8 on a real iPhone early.

---

## 5. Viz tech in React

- **SVG** for charts with fewer than ~2k marks and anything needing crisp text, hover or accessibility (ladder, gauge, scaling curves, histograms, token stream). It's themeable via CSS vars for free.
- **Canvas 2D** for dense grids and timelines (op flame strips, KV grid, attention heatmaps up to ~1M cells via `ImageData`), with manual hit-testing.
- **WebGL/WebGPU rendering** only if a figure needs more than ~1M animated cells or 3D. I don't recommend 3D (Bycroft owns that, and it fights the minimal aesthetic). If needed, use raw WebGL or regl (~25KB), not react-three-fiber plus three (~150KB+ gz).
- **Libraries, keeping it small:** use **d3 sub-packages only** (`d3-scale`, `d3-shape`, `d3-array`, `d3-format` are a few KB each), with React rendering the SVG. Skip full d3, visx (fine but unnecessary), and framer-motion (~30KB+; CSS transitions and `requestAnimationFrame` suffice, and the site already does reduced-motion in CSS and `matchMedia`, see `PDoom.tsx` and `Timeline.tsx`). Code-split all post components so the home page bundle is unchanged.
- **Reduced motion:** follow the existing pattern (`window.matchMedia("(prefers-reduced-motion: reduce)")`). Animated figures get a step button and scrubber instead of autoplay, and streaming tokens appear per chunk without the typewriter effect.
- **Theme:** jkmr's post theme is the `.minimal` CSS vars (`--m-bg`, `--m-fg`, `--m-muted`, `--m-rule`, `--m-accent` = #b8541f light / #e3915e dark, plus `--m-heat-0..4`, a ready-made 5-step sequential ramp ideal for heatmaps and attention). Canvas figures should read these with `getComputedStyle` and re-read on `data-theme` change (a MutationObserver on `.minimal`) and on the `prefers-color-scheme` change event. Note: the Tailwind `ink/bone/ember` tokens belong to the classic site, not the post theme.
- **Mobile:** the article column is `max-w-[44rem]`. Figures must work at 360px. Use small multiples that reflow (12×12 heads → 6×24 or paged), large touch targets for sliders (≥44px), hover detail available on tap, no hover-only information, and pointer events (not mouse events) for drag. Heavy live figures should show a "run" button on mobile rather than auto-running.
- **Performance:** init each figure on `IntersectionObserver` entry, pause rAF loops off-screen (the Ciechanowski pattern), and never render from inside the worker message handler. Batch into rAF.

---

## 6. Embedding live components in jkmr posts

**Current state:** `src/content/posts.ts` imports each post as `?raw` Markdown. `src/pages/Post.tsx` → `toSegments()` splits the body at <code>```timeline</code> fenced blocks, renders the Markdown parts with `marked` into `.m-prose` divs, and renders the timeline parts as `<Timeline spec>` (parser in `components/minimal/timelineSpec.ts`). **This is already a shortcode system.** Generalise it rather than adopting MDX.

**Why not MDX:** it would change the `?raw` + `marked` pipeline, add `@mdx-js/rollup`, make prose authoring JSX-sensitive (`{`, `<` in technical text about Rust and generics is painful), and re-style everything that `.m-prose` currently relies on. That's a lot of churn for one post.

**Proposal (minimal diff, consistent with the codebase):**

1. Generalise the fence regex in `toSegments` from `timeline` to a registry. A fence <code>```live &lt;name&gt;</code> takes an optional JSON or `key: value` body as props:
   ````md
   ```live kv-cache
   maxContext: 512
   defaultCache: off
   ```
   ````
   Keep <code>```timeline</code> working (register it as a component with its existing parser).
2. Create `src/components/live/registry.ts`, mapping `name → React.lazy(() => import("./KvCache"))`. Lazy imports mean the WASM, worker and d3 bits are only fetched on posts that use them, and the home page bundle is unchanged. Render inside `<Suspense fallback={<FigureSkeleton/>}>` with a fixed `min-height` per figure to avoid layout shift.
3. Add an inline reactive number for Bret Victor-style live prose. Have `marked` pass through a tiny token syntax, e.g. `{{live:tok_s}}` or an HTML element `<live-num k="tok_s"></live-num>`. Since marked passes raw HTML through and the bodies are trusted, after rendering a segment, portal React components into those placeholder nodes (`createPortal` into `querySelectorAll("live-num")`). This keeps prose as plain Markdown.
4. Add an `EngineProvider` wrapping the segments in `Post.tsx`, only when the post contains a `live` block (lazy-loaded too). It owns the worker and the shared trace store, so all figures share one engine run.
5. Add a figure frame component `<LiveFigure title caption provenance>`: monospace caption, provenance line ("live · Chrome · wasm-simd · 1 thread" vs "recorded · M3 · native"), and a reset button. It should use the `.minimal` tokens so it matches `.m-prose`. Figures should be allowed to break out wider than 44rem on desktop (e.g. `w-[min(64rem,100vw-2rem)]` centred) for the flame strip and attention grid.
6. Assets: put `.wasm`, the worker and recorded traces in `src` (Vite handles `new URL("./x.wasm", import.meta.url)` and `new Worker(new URL(...), {type: "module"})`) or under `public/blog/nanoinfer/`. The web/ package from nanoinfer can be vendored as a built artefact, or consumed as a git or file dependency. Prefer **vendoring built output** (`pkg/` from wasm-pack plus a `.d.ts`) so the jkmr Vercel build doesn't need a Rust toolchain.
7. `vercel.json`: add site-wide COOP `same-origin` + COEP `require-corp` (option A in §3.6), add `Cache-Control: public, max-age=31536000, immutable` for hashed wasm and trace files, and confirm `.wasm` is served as `application/wasm` (Vercel does this by default **[known]**; it matters for `instantiateStreaming`).
8. Reading-time: `minutes()` in `posts.ts` counts words in raw Markdown, so fence bodies inflate it slightly. It's negligible.

No changes have been made to jkmr. This is a plan only.

---

## 7. Risks and open questions (spike list)

1. **Browser perf reality.** Single-thread WASM SIMD f16→f32 decode of GPT-2 124M: expect roughly single-digit to ~20 tok/s on a laptop **[uncertain]**. Measure early, because it sets the post's tone.
2. **WebGPU dispatch floor** for batch-1 GPT-2, and whether one-encoder submission helps in-browser.
3. **iOS memory** for the int8 heap, and whether f16 is viable on iPad.
4. **COOP/COEP site-wide** with no regressions. Check the OG image and any future embeds.
5. **HF CDN** behaviour for whole-file cross-origin GETs from the production origin (test from a deployed preview, not localhost).
6. **Runtime-selectable kernel variants.** This is a repo-discipline decision Josh should make *now*, before the hillclimb overwrites history.
7. **Tracing overhead.** Prove that `NoopTracer` compiles out (bench both) so the post's numbers are trustworthy.

## Sources

- ciechanow.ski · distill.pub/2020/communicating-with-interactive-articles · bbycroft.net/llm + github.com/bbycroft/llm-viz · poloclub.github.io/transformer-explainer + arxiv.org/abs/2408.04619 · jalammar.github.io/illustrated-transformer · siboehm.com/articles/22/CUDA-MMM · andrewkchan.dev/posts/yalm.html · redblobgames.com · ncase.me · jax-ml.github.io/scaling-book · lilianweng.github.io
- worrydream.com/ExplorableExplanations · worrydream.com/LadderOfAbstraction
- github.com/RReverser/wasm-bindgen-rayon · caniuse.com/wf-wasm-simd-relaxed · caniuse.com/wf-webgpu-subgroups · web.dev/blog/webgpu-supported-major-browsers · bugs.webkit.org/show_bug.cgi?id=230550 (COEP credentialless in Safari)
- vercel.com/kb/guide/fix-shared-array-buffer-not-defined-nextjs-react · discuss.huggingface.co/t/cors-support-for-http-range-requests-on-dataset-files/172172 · github.com/huggingface/datasets/issues/7931
- github.com/huggingface/candle/tree/main/candle-wasm-examples · arxiv.org/abs/2412.15803 (WebLLM) · arxiv.org/abs/2604.02344 (WebGPU dispatch overhead) · docs.rs/web-rwkv · docs.rs/crate/rullama
