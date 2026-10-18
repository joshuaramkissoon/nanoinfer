# Writeup notes (living draft)

Running context for the blog series, written *while* building so the post writes itself.
Every agent/stage appends here: what we tried, why, the number, the "aha", and which
live visualisation shows it. Keep entries short and concrete. Real numbers only.

## The pitch (draft)

**What happens between pressing enter and the first word appearing?** And how fast can
one laptop make it go?

I wrote an LLM inference engine from scratch in Rust, started with the dumbest version
that gives the right answer, and then made it faster one measured step at a time. Every
chart in this post is the real engine, running in your browser.

## Motivations (the threads that run through every section)

1. **Intuition for inference.** A forward pass is not magic: it's ~100M multiply-adds per
   token, a few hundred MB of weights, and a loop. Readers should leave able to estimate
   tok/s on the back of an envelope for any model on any machine.
2. **How optimised can we get?** There's a hard ceiling: memory bandwidth ÷ bytes of
   weights. Every step is plotted as a % of that ceiling. The story is a climb toward a
   line we can compute before writing any code.
3. **Decode is memory-bound, prefill is compute-bound.** The single most important idea in
   inference engineering. Everything (KV cache, quantization, batching, speculative
   decoding, paged attention) is a consequence of it. Introduce early; pay it off repeatedly.
4. **Optimisation is a science.** Hypothesis → change → measure → explain. Include the
   failures and the steps that didn't help.
5. **From one chip to many.** The same arithmetic (bytes moved vs flops done) explains why
   you shard models across GPUs and what all-reduce costs. Later post(s).

## Series shape (draft)

- **Post 1 — "From 0 to the roofline on a laptop"**: naive → KV cache → memory layout →
  SIMD → threads → quantization. CPU, M4. Live: in-browser engine with stage switcher,
  tok/s staircase, roofline chart, KV cache growth, per-op flame view.
- **Post 2 — GPU**: Metal/WebGPU kernels, fusion, flash-decoding.
- **Post 3 — Serving**: batching, continuous batching, paged attention, speculative decoding.
- **Post 4 — Many devices**: tensor parallel, ring all-reduce, pipeline parallel (simulated
  on one Mac, then real rented GPUs).

## Visualisation ideas (each must be driven by real engine data)

- Stage switcher + live token stream: pick a stage, watch it generate, tok/s counter.
- Roofline: each stage is a dot; dots climb toward the bandwidth ceiling.
- Tok/s staircase across stages (from bench/results.jsonl + live browser runs).
- Per-op timeline (from the Trace hook): where does a token's time go?
- KV cache filling up token by token; recompute-vs-cache cost curves (O(n²) vs O(n)).
- Attention heatmap for the current token, per layer/head.
- Next-token probability bars (top-k) as it generates; temperature slider.
- Quantization: weight histogram f32 vs int8 vs int4, error vs speed tradeoff.

## Stage notes

<!-- Append per stage: ### Stage N — name / Hypothesis / Change / Result (numbers, % of ceiling) / Aha / Viz / Quotes-to-use -->

### Stage 0: naive (recompute everything)

- **Hypothesis:** the dumbest correct forward pass is easy to verify, and painfully slow
  at decode, because every new token re-runs the *whole* sequence.
- **Change:** scalar f32 GPT-2, one thread, no cache. Logits match HF transformers to
  <2e-4. Greedy output matches token for token.
- **Result (M4, 1 thread, prompt 64):** prefill 26.2 tok/s. Decode **0.37 tok/s**, which is
  **0.15%** of the 243 tok/s ceiling (120 GB/s ÷ 495 MB of f32 weights per token).
- **Aha:** decode speed ≈ prefill speed ÷ context length (26 / 72 ≈ 0.36). Each decode
  step *is* a full prefill, so token 1000 costs 1000× token 1, and generation is O(n²).
- **Viz:** recompute-vs-cache cost curve: rows pushed through the network per token,
  climbing linearly for naive (from `ForwardInfo.n_computed` in the trace).
- **Line:** "The first version of the engine was correct, and it took nearly three
  seconds to say each word."

### Stage 1: KV cache

- **Hypothesis:** old tokens are only ever used through their keys and values, which never
  change. Cache them, and each new token needs 1 row of compute, not `context_len`.
- **Change:** a 2 × 12 × 1024 × 768 f32 cache (72 MiB at full context). Same kernels,
  same logits. Only where attention reads K/V from changed.
- **Result:** decode **18.95 tok/s** (51× stage 0). That's **7.8% of ceiling**. Prefill
  unchanged at 27 tok/s (the prompt is one forward either way).
- **Aha 1:** decode is supposed to be memory-bound, but ours moves 10 GB/s out of 120.
  It's stuck on scalar multiply-add latency (~4.7 GFLOP/s). You have to *earn*
  memory-boundness with fast kernels. That sets up SIMD.
- **Aha 2:** prefill is faster per token than decode (27 vs 19) because decode pays the
  768 × 50257 LM head (a third of GPT-2 small's weights!) on every token.
- **Aha 3:** attention is <1% of decode time at this size. The matmuls are everything.
- **Viz:** KV cache grid filling token by token (`kv` events). Per-op flame strip showing
  lm_head + 4 matmuls = 99% (`op` events). Stage switcher: naive vs kv-cache tok/s live.
- **Line:** "One cache, 51× faster, and we're still at 8% of what the hardware allows."
- **Side note for the post:** macOS put the benchmark on an efficiency core and halved the
  numbers until we set the thread's QoS. Measure the measurement first.

### Stage 2: unrolled (16 accumulators)

- **Hypothesis:** the scalar dot product is one dependency chain, so the core waits on
  itself. Sixteen independent partial sums give it 16 chains to overlap, and since we
  pick the summation order, the compiler may vectorise them.
- **Change:** `[f32; 16]` accumulators over 16-wide chunks, then a tree reduction. Plain
  Rust, no intrinsics. The compiler turned it into 4 NEON registers (4 × `fmul.4s` +
  4 × `fadd.4s` per 16 weights).
- **Result:** decode **100.5 tok/s** (5.8×), **41% of the 243 tok/s spec ceiling**; prefill
  **185 tok/s** (7.5×). Same perplexity as f32 (51.837 vs 51.837).
- **Aha:** every matmul now moves ~54.5 GB/s, the most one core can read (our `membw`
  probe: 57 GB/s on 1 thread, 98 GB/s on all 10, so the CPU never sees the 120 GB/s spec).
  One change to a loop moved decode from "waiting on arithmetic" to "waiting on memory".
  The textbook "decode is memory-bound" just became true for us.
- **Aha 2:** the compiler won't vectorise `iter().sum()` because float addition isn't
  associative, but it *will* vectorise 16 explicit accumulators: you gave it permission
  by writing the order down.
- **Viz:** roofline: stage 1's dot sits on the compute roof far below the slope; stage 2's
  dot lands on the 1-thread bandwidth line. Bandwidth probe bar chart (1..10 threads).
- **Line:** "Sixteen variables instead of one made it 5.8× faster, and turned the
  bottleneck from the CPU into the memory."

### GPU probe: what a dispatch costs (Metal, M4)

- **Hypothesis:** "GPU kernel launches are expensive" is folklore; measure it before
  designing around it.
- **Change:** an empty kernel, dispatched N times in one command buffer vs one command
  buffer per dispatch; float4 read/copy kernels for bandwidth.
- **Result:** inside one command buffer a dispatch costs **0.72 µs** of GPU time. A
  command buffer round trip (commit, wait) costs **~150 µs**, 200× more. The GPU reads
  **~80 GB/s** at best here (CPU's 10 threads: 98).
- **Aha:** the expensive thing isn't the kernel launch, it's the CPU waiting for the GPU.
  GPT-2 has 135 ops per token; one wait per op is 20 ms/token of nothing.
- **Viz:** two bars per token: "135 waits" (20 ms) vs "1 wait" (0.15 ms) vs the GPU's
  actual kernel time.
- **Line:** "Asking the GPU to do nothing costs less than a microsecond. Asking it whether
  it's finished costs 150."

### GPU stage 1: metal-naive

- **Hypothesis:** port the CPU kernels one-to-one (one GPU thread per output, a
  command buffer + wait per op). Correct immediately, slower than the CPU at decode.
- **Change:** 7 Metal kernels mirroring `kernels/reference.rs`, weights copied once into
  shared GPU buffers (0.13 s; unified memory, so it's a memcpy).
- **Result:** decode **14.6 tok/s** (6% of ceiling), *slower than one CPU core* (19).
  Prefill **286 tok/s**, 10× the CPU, with the same naive kernels.
- **Aha 1:** it's not just the waits. A GPU thread is a slow CPU: a 768-wide layernorm on
  one GPU thread takes 165 µs; attention as 12 threads (one per head) is 34% of the
  token. CPU-shaped code uses 12 of the GPU's ~1,280 lanes.
- **Aha 2:** prefill wins anyway, because 64 tokens × 2304 outputs is enough
  independent work to fill the GPU even with dumb kernels.
- **Viz:** per-op GPU-time bar for one decode token (attn 34%, fc_proj 15%, lm_head 10%,
  layernorms 16%); the same chart for CPU stage 1 beside it, where matmuls are 99%.
- **Line:** "My first GPU version was slower than one CPU core."

### GPU stage 2: one command buffer per token

- **Hypothesis:** encode all 135 ops, commit once, wait once. Saves ~20 ms of round trips.
- **Change:** one flag (`Sync::PerForward`). Same kernels.
- **Result:** decode **41.3 tok/s** (2.8×, 17% of ceiling), prefill 403 tok/s.
- **Aha:** the GPU's *own* busy time dropped 43% (41 → 23.5 ms) with identical kernels.
  Fed tiny command buffers with gaps, the GPU pays a start-up cost per buffer and idles;
  fed one long one, it stays busy 97% of the time.
- **Viz:** timeline strip: 135 short GPU bursts with gaps vs one solid block.
- **Line:** "Same kernels, one fewer question per op: 2.8× faster."

### GPU stage 3: kernels shaped like a GPU (SIMD matvec, then parallel attention/layernorm)

- **Hypothesis:** the matvec is the bottleneck; make 32 lanes read 512 contiguous bytes
  and reduce with `simd_sum`, and decode approaches the memory wall.
- **Change (3a):** SIMD-group matvec (4 rows per SIMD-group, float4 loads). **Result:**
  decode 41 → **47 tok/s**. The matvecs got 2.4× faster, but they were only ~45% of the
  token: the 12-thread attention (41%) and 1-thread layernorm (20%) now dominate.
- **Change (3b):** a 256-thread threadgroup per layernorm row and per attention head
  (`simd_sum`/`simd_max` + threadgroup memory). **Result:** decode **137 tok/s** (2.9×),
  **56% of the 243 spec ceiling, ~83% of what this GPU can actually read** (~80 GB/s);
  75 GB/s while the GPU is busy. Attention went from 6.2 ms to 62 µs per token: 100×.
- **Aha 1:** "decode is memory-bound" only holds once *nothing else* is slow. With one
  bad kernel left, a perfect matvec bought 13%. Amdahl first, roofline second.
- **Aha 2:** the GPU now beats one CPU core (19 tok/s naive, 100 unrolled) and every
  variant of the matvec we tried lands between 75 and 90 GB/s. The kernel is done; only
  fewer bytes can go faster.
- **Viz:** stacked per-op bars for metal-onecb → simd → par: the matvec block shrinks,
  then the attention/layernorm block vanishes.
- **Line:** "The GPU got 13% faster when I fixed the obvious bottleneck, and 3× faster when
  I fixed the one I hadn't measured."

### GPU stage 4: prefill matmuls (tiled, then simdgroup_matrix)

- **Hypothesis:** prefill reuses each weight n times, so it should be compute-bound, but
  only if the kernel *actually* reuses it. The naive kernel re-reads every weight for
  every token.
- **Change:** (a) classic shared-memory tiling, 32 tokens × 32 outputs per threadgroup;
  (b) Apple's 8×8 hardware matrix multiply (`simdgroup_float8x8`), 32 × 64 tiles.
- **Result (512-token prompt):** naive 456 → tiled **2,188** → MMA **5,947 tok/s**, 13× in
  two steps and ~32× the best single CPU core. The GEMMs hit ~2 TFLOP/s, 45% of the
  GPU's f32 peak. Decode untouched (126 tok/s).
- **Aha 1:** prefill and decode want *different kernels*. The same weights, the same maths,
  but decode wants 32 lanes streaming one row; prefill wants an 8×8 tile held in
  registers and reused across tokens.
- **Aha 2:** with fast GEMMs, attention became 28% of prefill. It's O(n²) and our kernel
  re-reads every key for every query. Next bottleneck, already visible.
- **Aha 3 (unified memory cuts both ways):** a CPU benchmark running at the same time cut
  GPU decode from 140 to ~100 tok/s. CPU and GPU share one memory bus; the CPU's 55 GB/s
  came straight out of the GPU's budget.
- **Viz:** prefill tok/s vs prompt length for naive / tiled / MMA; a "memory-bound vs
  compute-bound" dial that swings as the prompt grows.
- **Line:** "The GPU has a matrix-multiply instruction. Using it made prefill 2.7× faster
  than my best hand-written tiling."

### GPU stage 5: f16, Q8_0, Q4_0 weights on the GPU

- **Hypothesis:** decode is bandwidth-bound, so halving bytes per weight halves the time.
- **Change:** matvec and matmul kernels that read f16 / 8-bit / 4-bit weights and expand
  them in registers (block-of-32 formats with a scale per block, like llama.cpp's
  Q8_0/Q4_0, but laid out as "all quants, then all scales" so loads stay aligned).
- **Result:** f16 **245 tok/s** (1.84×, same 71 GB/s), Q8_0 **365 tok/s**, Q4_0 **363–475 tok/s**
  (noisy). Each step buys less: Q4 reads 1.5× fewer bytes than Q8 for ~no speed-up.
- **Aha 1 (the quality bill):** GPT-2 small's *LM head* is fragile. 4-bit on the head
  alone wrecks the model (perplexity 35 → 266, top-1 0/4). 8-bit on the head leaves
  perplexity unchanged (35.20 vs 35.23) but changes **10% of greedy choices**.
  Perplexity said "lossless"; top-1 agreement said otherwise. Track several metrics.
- **Aha 2 (law 6 arrives):** at 2 ms per token the fixed costs (encoding 135 GPU commands,
  waiting for the GPU, waking up, picking the argmax on the CPU) are ~40% of the token.
  The small model became overhead-bound, as predicted before writing any code.
- **Viz:** quality-vs-speed scatter: one dot per (blocks dtype, head dtype), x = tok/s,
  y = KL vs f32, with the Q4-head dots falling off a cliff. Stacked bar per dtype: GPU
  matvec time shrinking, fixed overhead constant.
- **Line:** "Four-bit weights everywhere except one matrix: fine. Four-bit on that one
  matrix: the model forgets English."

### GPU stage 6: fusion vs. not waiting (argmax on GPU, fused kernels, pipelined decode)

- **Hypothesis:** at ~2 ms/token the fixed costs dominate. Cut dispatches by fusing
  ops into the matvecs (LayerNorm, bias + GELU, residual add, KV write), and move the
  argmax onto the GPU.
- **Change:** GPU argmax; two fusion designs (LayerNorm inside every matvec: 137 → 64
  dispatches; epilogues only: 89); then a GPU-resident greedy loop: each forward reads
  its input token from a GPU buffer the previous forward's argmax wrote, so all 64
  tokens' command buffers are submitted at once and the CPU waits once.
- **Result (Q4 weights):** fusion **≈ +5%, inside the noise**; full LayerNorm fusion made
  the big matvecs slower (every threadgroup re-normalised the input). Not waiting:
  **586–632 tok/s** (+30%), GPU busy 100% of the wall time. Q8: **517**, f16: **284** tok/s.
- **Aha:** the expensive thing was never the number of kernels; it was the CPU and GPU
  taking turns. Same lesson as stage 2 (one command buffer per token), one level up
  (one wait per *generation*).
- **Aha 2:** the GPU argmax only saves ~0.1 ms, but without it there is no GPU-side token
  for the next step to read: it's the enabler, not the win.
- **Viz:** CPU/GPU timeline lanes for 4 tokens: stage 5 alternates (GPU idle while the
  CPU encodes, waits, picks the argmax), stage 6d is a solid GPU bar with the CPU running
  ahead.
- **Line:** "Halving the number of GPU kernels did nothing. Not asking the GPU whether it
  was done made it 30% faster."

### GPU stage 7: flash-decoding and flash attention

- **Hypothesis:** at long context the KV cache, not the weights, is the problem. At 960
  tokens the f32 cache is 71 MB per token (more than the 89 MB of Q4 weights), attention
  is 46% of a decode token and 54% of a 960-token prefill.
- **Change:** decode: split the keys into 64-key chunks across threadgroups and merge
  the partial softmaxes (flash-decoding). Prefill: a flash-attention kernel that keeps a
  32-query × 32-key score tile in on-chip memory, with an online softmax and 8×8 hardware
  matrix multiplies, so keys are read once per 32 queries instead of once per query.
- **Result:** prefill attention at 960 tokens **113 → 14 ms (8×)**; prefill **4.8k → 8.7k
  tok/s**. Decode at context 930: **362 → ~390 tok/s**, the long-context decode is now
  bound by reading the KV cache itself.
- **Aha:** the "online softmax" trick (keep a running max and sum, rescale what you
  have when the max grows) is what lets attention be computed in tiles without ever
  writing the n × n score matrix.
- **Viz:** decode tok/s vs context length for pipelined / flash-decoding; prefill time
  stacked by op at P=960 before/after (attention block shrinking from half to an eighth).
- **Line:** "At 1,000 tokens of context the model's notes weigh more than the model."
- **Follow-up (f16 KV cache):** storing the notes at half precision: decode at context
  ~990 **375 → 475 tok/s** (+27%), nothing at short context. Both versions move the same
  ~60 GB/s; only the byte count changed.

### GPU: the whole climb in one table (for the Ladder viz)

| stage | decode tok/s | prefill (512) | the one idea |
|---|---|---|---|
| naive | 14.6 | — | CPU code on a GPU, one wait per op |
| one command buffer | 41 | — | stop asking "are you done?" 135 times |
| SIMD matvec + parallel attention | 137–140 | 456 | use 256 threads, not 1 |
| simdgroup_matrix prefill | 126–133 | 5,947 | the GPU has a matrix instruction |
| f16 / Q8 / Q4 | 245 / 365 / ~400 | ~6,300 | fewer bytes |
| pipelined (Q4) | 586–632 | — | the CPU stops waiting between tokens |
| flash attention + f16 KV (Q4) | 602–641 | 8,700–8,900 | tiles + online softmax |

- **Arc for the post:** naive GPU < one CPU core; the GPU only wins once the code is
  shaped for it, and at GPT-2 size the last 2× came from *removing waiting*, not from
  faster maths. Decode ends at ~45% of the spec ceiling (Q4) / ~59% (f16); most of the
  gap is the GPU's ~80 GB/s practical read bandwidth on this machine plus small-matrix
  latency.
### GPU stage 8a: metal-lanes (more loads in flight per lane)

- **Hypothesis:** the Q4 matvec on 768-wide rows waits on memory latency, not
  bandwidth: each busy lane has one 16-byte load in flight, and 8 of 32 lanes idle.
- **Change:** 8 lanes per row, 4 rows per SIMD-group at once; each lane walks 3
  blocks of its row with independent loads, then a 3-step shuffle reduction.
- **Result:** narrow Q4 matrices **38 → 58 GB/s** in isolation; whole-token decode
  **+5–8%** (576–592 → 599–640 tok/s, Q4, interleaved on a loaded machine).
- **Aha:** "memory-bound" has two flavours. Bandwidth-bound means the pipe is full;
  latency-bound means the pipe is empty because nobody asked for enough at once.
  The fix for the second is more requests in flight, not fewer bytes.
- **Viz:** per-matrix GB/s bars (old vs new layout), with the 80 GB/s practical line.
- **Line:** "Same bytes, same maths, 50% faster: the lanes just asked for more data
  before waiting."

### Stage 3: neon (hand-written SIMD), a null result

- **Hypothesis:** hand-written NEON with fused multiply-adds, computing 4 output rows per
  pass to share loads of `x`, beats the compiler's version.
- **Change:** tried 4-rows-per-pass first. Textbook inner loop, and decode got *slower*
  (101 → 74 tok/s). A microbenchmark showed one core reads 49 GB/s from one sequential
  stream but only 32–38 GB/s from 4 interleaved ones, and software prefetch made it
  worse. Kept the simple version: one weight row at a time, `vfmaq_f32`.
- **Result:** decode 100.3 tok/s, prefill 183 tok/s. **Identical to stage 2.**
- **Aha:** decode was already waiting on memory, so fewer instructions bought nothing. And
  prefill didn't move either, which exposes the real prefill problem: the loop streams
  the whole weight matrix again for every prompt row. That's a *reuse* problem, not an
  arithmetic one.
- **Aha 2:** "saving loads" can cost bandwidth. The memory system likes one long
  sequential read, and four interleaved ones confuse the prefetcher.
- **Viz:** small multiples: GB/s vs number of interleaved streams (1, 2, 4, 8). Stage 2 vs
  stage 3 bars side by side, equal heights, captioned "the compiler had already done this".
- **Line:** "I wrote the assembly-level version by hand, and it was exactly as fast as the
  version the compiler wrote for me."
### Stage 4: tiled (register blocking for prefill)

- **Hypothesis:** prefill multiplies 64 prompt rows by the *same* weights, but our loop
  fetches each weight once per row. Compute a 4×4 block of outputs at once, and each
  fetched weight gets used 4 times straight from a register.
- **Change:** a 4×4 NEON micro-kernel (16 accumulators, 8 loads per 16 FMAs). Weight
  blocks in the outer loop, so a 48 KB weight panel stays in L1 while every prompt row
  visits it. Decode path unchanged.
- **Result:** prefill **~450–470 tok/s** (2.5×), with matmuls at ~100 GFLOP/s = 71% of one
  core's NEON peak. Decode unchanged at ~100 tok/s.
- **Aha:** same instructions as stage 3, 3× the matmul speed, purely from reuse. Prefill
  is compute-bound now, as the textbook says.
- **Aha 2:** fast matmuls expose the "free" ops. GELU (scalar `tanh`) is now 12% of prefill
  time, more than the attention output projection.
- **Viz:** tile animation: a 4×4 block of outputs lighting up while a 4-row weight panel
  stays highlighted ("in L1") and prompt rows stream past. Prefill flame strip with GELU
  suddenly visible.
- **Line:** "The arithmetic was never the problem. Fetching the same number 64 times was."

### Browser (wasm simd128, 1 thread, Chrome on the M4)

Same Rust, compiled to `wasm32-unknown-unknown` with `+simd128`, running in a Web Worker on the demo page (`web/`).
Measured with the page's Ladder benchmark: 64-token prompt (same text as `nanoinfer bench`), greedy, median of 8 decode
steps, `performance.now()` at 5 µs resolution (the page is cross-origin isolated). Engine wasm is 270 KB (100 KB gzipped).

| stage | browser decode tok/s | native decode | browser prefill tok/s (P=64) | native prefill |
|---|---|---|---|---|
| naive | 0.29 | 0.37 | n/a | 26 |
| kv-cache | 13.8–14.0 | 18.9 | 20 | 27 |
| unrolled | 83–101 | 100.5 | 171 | 185 |
| neon (runs `unrolled` in wasm) | 85 | 100.3 | n/a | 183 |
| tiled (scalar tiles in wasm) | 93 | 101.1 | 175 | 469 |

Browser ranges are two sessions. The low end was measured while the native hillclimb was benchmarking on the same Mac,
so treat the spread as contention.

- **The wasm build gets most of the native speed.** Decode lands at 73–100% of native: `unrolled` reached 101 tok/s
  in a quiet run, level with native. The 16-accumulator kernel vectorises to simd128 just as it does to NEON, so the
  hillclimb's biggest step so far carries straight into the browser (6–7× over kv-cache in both).
- **Aha: tiling needs registers.** `tiled` prefill is 2.5× faster natively (469 vs 185) but flat in wasm (175 vs 171).
  Without NEON the 4×4 tile falls back to scalar code, and the reuse it buys is eaten by the loss of SIMD. Not every
  rung transfers to every platform, and the Ladder shows which ones do.
- **The roofs in a tab.** Streaming read from one wasm thread: 48–62 GB/s (the page takes the best of two probes; the
  first probe right after the 1 GB weight load reads ~35 GB/s). simd128 multiply-add peak: 15–19 GFLOP/s (no FMA in
  baseline wasm). So the browser's f32 speed of light is ~100–120 tok/s, and `unrolled` reaches ~75–85% of it.
- **Aha: in the browser, decode is compute-bound.** Without FMA a single wasm core tops out near 19 GFLOP/s, which puts
  the ridge at ~0.35 FLOP/byte, *below* decode's 0.5. On the roofline the `unrolled` matmul dots sit on the flat roof,
  not the slanted one. Natively the ridge is far to the right and decode is memory-bound. The next browser rung is
  relaxed-SIMD FMA (`f32x4_relaxed_madd`), which should double the roof. That's a nice "same code, different
  bottleneck" moment for the post.
- **Aha: the long pole moves.** Under `unrolled`, matmuls each move ~47 GB/s, and `gelu` (a scalar `tanh` per element)
  has grown from noise to 4% of decode time. The LM head stays ~28–30% of time and 31% of bytes on every stage.
- **The KV race, live.** naive vs kv-cache under a fair scheduler (whichever lane has used less engine time goes
  next): after ~23 s of engine time naive had written 25 tokens and kv-cache 96 (in 7 s), a 12.6× gap that keeps widening.
  Naive's per-token cost climbs linearly to 1.5 s at context 30. kv-cache stays flat at ~70 ms. Greedy outputs are
  identical token for token.
- **Main thread vs worker.** The same wasm on the main thread ran `unrolled` at 96 tok/s against ~84 in the worker, with
  kv-cache equal (14.0 vs 13.8). Partly probe noise, and the native hillclimb was benchmarking on the same machine at the
  time. Worth re-measuring on a quiet machine before quoting.
- **Loading:** 548 MB f32 safetensors streamed into wasm memory, parsed and transposed by core in ~1.5 s from localhost.
  Peak wasm heap ~1.05 GB. Production wants pre-packed f16/Q8 shards (`web/NEEDS_FROM_CORE.md`).
- **Viz:** fig. 2 speed-of-light gauge, fig. 5 roofline (matmul dots on the compute roof), fig. 6 KV race, fig. 8 ladder
  (native and browser bars side by side under their ceilings).
- **Line:** "The same Rust, in a browser tab, matches the native decode speed, and hits a different wall."

### Stage 5: threads (rayon)

- **Hypothesis:** decode is memory-bound and one core can't saturate memory (57 of ~98
  GB/s), so splitting every matmul across cores should buy up to ~1.7× decode and
  several × prefill.
- **Change:** each matmul's weight rows are cut into `threads × 4` chunks and handed to
  rayon. Each thread streams one contiguous slab. Behind a `threads` cargo feature, so
  the browser build stays single-threaded.
- **Result:** prefill **1021 tok/s** at 8 threads (2.1×). Decode only **118 tok/s** (1.3×,
  58 GB/s, 49% of the spec ceiling). E-cores alone: ~51 tok/s at most (their cluster
  reads ~25 GB/s).
- **Aha:** small matmuls got *slower* with more threads (attention projection: 43 µs on 1
  thread, 62 µs on 4). Waking sleeping worker threads costs tens of µs, and a token has
  49 matmuls. For a model this small the coordination is the bottleneck, not the math
  or the memory.
- **Viz:** thread-count slider → decode/prefill tok/s curves flattening (P-cores, then
  E-cores), with the membw curve drawn behind as "what memory could give".
- **Line:** "Ten cooks, one narrow pantry door, and a manager who takes a nap between
  every order."

### Stage 6: spin-pool (threads that never sleep)

- **Hypothesis:** if waking threads is the cost, don't let them sleep. Workers spin on an
  atomic counter and see new work in ~100 ns.
- **Change:** a ~150-line thread pool. Work is published by bumping one 64-bit atomic.
  Every thread (including the caller) grabs chunks from it, so fast cores take more.
- **Result:** decode **115–124 tok/s** (~60 GB/s, ~50% of the spec ceiling), 10–20% over
  rayon. Small matmuls 2× faster to dispatch (41 vs 99 µs). Prefill no better (rayon's
  984 vs 870).
- **Aha (failure first):** the first version spun hard and went from 93 to **36 tok/s** at
  10 threads. With other apps on the machine, spinning threads steal the CPU from the
  thread doing the last bit of work, and everyone waits for it. Spin, then yield, then
  sleep fixed it. A spin pool assumes you own the machine. A laptop never does.
- **Aha 2:** with 10 cores on the matmuls, the "small" serial ops (GELU, attention) are now
  half of prefill time. Amdahl's law, live.
- **Viz:** Gantt strip of one decode token: 10 thread lanes, colored bars per matmul chunk,
  gray gaps = dispatch/join. rayon vs spin side by side.
- **Line:** "The fastest way to hand out work is to have the workers already standing at
  the counter, as long as they step aside when someone else needs the room."

### Reference point: Apple Accelerate

- Same forward, matmuls swapped for Apple's `cblas_sgemm`/`sgemv` (runs on the M4's matrix
  coprocessor). Prefill **1480 tok/s**, beating our best NEON (1021). Decode **111 tok/s**,
  *below* our spin-pool (115–124).
- **Aha:** a matrix engine sells FLOPs. Decode needs bytes. Apple's own library can't read
  memory any faster than a NEON loop, so on decode a hand-written loop keeps up with it.

### Stage 7: fast-ops (the "small" ops)

- **Hypothesis:** with 10 cores on the matmuls, single-threaded scalar GELU and attention
  are half of prefill. Make them vectorise and run in parallel too.
- **Change:** `tanh` and `exp` as branch-free polynomials (accurate to ~1e-6, so the
  compiler can vectorise them, unlike libm calls). Attention split over (query, head)
  pairs across cores. GELU split into chunks.
- **Result:** prefill **~1400–1500 tok/s** at P=64 (+75%), **1200–1370 at P=512** (2.2×).
  Decode +5% at short context, **+36% at 550 tokens of context** (attention grows with
  context). Perplexity unchanged to 4 decimals. Prefill now ties Apple's Accelerate,
  and beats it at long prompts.
- **Aha:** GELU went 0.1% → 12% → 1.9% of the time across the stages, without changing
  once until now. Whatever you haven't optimised becomes the bottleneck.
- **Aha 2:** the first fast `exp` was wrong by 4e-6 at x = 20 because computing x·log₂e
  in f32 throws bits away. The fix is a 1960s trick (Cody–Waite): split ln 2 into a part
  that multiplies exactly and a small correction.
- **Viz:** prefill flame strip, stage 6 vs 7: the gray GELU/attention bands collapsing.
- **Line:** "The matmuls were fixed four stages ago. This stage fixed the arithmetic around
  them, which had become half the work."

## Open questions / things to verify before publishing
