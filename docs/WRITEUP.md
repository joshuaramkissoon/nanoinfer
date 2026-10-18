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

### GPU stage 9: batched decode (why batching is the whole game)

- **Hypothesis:** at batch 1 each weight byte does one multiply-add. Decode B
  sequences per forward and each byte does B: aggregate throughput grows ~B× for
  free until the arithmetic catches up with the bytes (the ridge), and the ridge
  moves left as the weights shrink.
- **Change:** a ragged, paged forward (any mix of sequences and positions per step;
  PagedAttention-style 16-token KV pages + block table), a batched matvec that
  dequantises each weight once for up to 16 tokens and reuses each activation load
  across 2 weight rows, the prefill GEMMs from 32 rows, per-row flash-decoding and
  argmax. GPU-resident loop for B streams. Bit-identical tokens alone or batched.
- **Result (agg tok/s, B = 1 → 8 → 64 → 256):** Q4 657 → 2235 → 4606 → **4986**
  (7.6×); f16 268 → 1829 → 4002 → 4892 (18×); f32 134 → 939 → 3372 → 4146 (31×).
  Per-stream speed at Q4 B=64: 72 tok/s (13.9 ms per token).
- **Aha:** the ridge is where quantisation stops paying. f32 stays memory-bound to
  B≈8–16 (7× throughput for +14% latency at B=8); Q4 is compute-bound by B≈4. At
  B ≥ 64 every dtype lands at ~4–5k tok/s on the same GEMMs: a 4-bit model and a
  32-bit model serve the same number of tokens per second.
- **Aha 2:** the first batched kernel stalled at B=4 (~250 GFLOP/s): reusing weights
  across tokens isn't enough, activations have to be reused across weight rows too.
  Register blocking again, one level up.
- **Viz:** the batching curve: x = B (log), left axis aggregate tok/s, right axis
  per-token latency, one line per dtype, shaded memory-bound region ending at each
  dtype's ridge (`bench/batching.json`). Slider on B shows "each stream gets X tok/s,
  the GPU makes Y".
- **Line:** "Serving one user at a time, the GPU spends its life waiting for weights.
  Serving 64, it finally has something to do with them."

### GPU stage 10: continuous batching (a bus that never waits at the terminal)

- **Hypothesis:** a static batch runs until its longest request finishes, so short
  requests idle in their slots and new arrivals queue behind the whole batch.
  Scheduling per iteration (retire and admit every step) keeps the GPU full.
- **Change:** a scheduler over the paged ragged forward: each iteration = one
  decode row per running request + prompt chunks of newly admitted ones under a
  token budget; finished requests leave immediately and their KV pages return to
  the pool. Synthetic workload: 128 requests, prompts 16–256, outputs 16–256.
- **Result (Q4, batch ≤ 32):** offline 1197 → **1903 tok/s (1.59×)**; Poisson 16 req/s
  948 → 1725 (1.82×) with TTFT p50 **4.4 s → 62 ms**; at 8 req/s continuous keeps up
  (TTFT 24 ms) while static queues (TTFT 2.0 s, 80× worse). Peak KV −35% offline.
- **Aha:** throughput is the smaller win. The big one is latency: under real arrivals
  a static server makes every request wait for a stranger's long answer.
- **Aha 2:** the prefill budget trades TTFT against inter-token jitter: 32 tokens per
  iteration → TTFT 283 ms, TPOT p99 15.3 ms; 512 → 56 ms, 18.9 ms.
- **Viz:** Gantt of the first 32 requests under both policies (arrival → admitted →
  first token → done bars, from `bench/serving.json` `gantt`), plus a strip of rows
  per iteration (decode vs prefill, `iters`) showing static's sawtooth of dying
  batches vs continuous's flat full line; KV pages over time beside it.
- **Viz 2 (paged KV):** the pool as a grid of 16-token pages coloured by request,
  stepping through `kv_snapshots` (`bench/serving.json`): pages appear as
  sequences grow, vanish when they finish, and get reused scattered; caption
  with `kv_stats` (3.3% internal fragmentation vs static's 810-vs-530-page
  reservation).
- **Quiet re-run numbers:** offline 1389 → 2198 tok/s; 16 req/s 1080 → 1812 with
  TTFT p50 3.47 s → 40 ms.
- **Line:** "The static server was busy 100% of the time and still made people wait
  four seconds to hear the first word."

### Machine note for every GPU number above (quiet re-run, 18 Oct)

- Stages 8a–10 were first measured at load 15–30. On a quiet machine the GPU
  reads at **~105 GB/s** (not 77–85), metal-lanes Q4 decodes at **817 tok/s**
  (not ~600), and the batching curve tops out at **6,453 tok/s** (Q4, B=256;
  part of that is the 64 × 64 GEMM from 128 rows). `bench/batching.json` holds
  the quiet run. Use these numbers in the post; the shapes of the curves did
  not change.

### GPU stage 8b: metal-kv8 (the KV cache in 8 bits)

- **Hypothesis:** at context ~1000 the f16 KV cache is ~40% on top of the Q4
  weights per decode token; int8 + one scale per (position, head) halves it.
- **Result:** −1.5% at 64 tokens of context, +4.6% at 512, **+5.4% at 960**
  (618 → 651 tok/s); KL vs an f32 cache 9e-5. A third of what the byte count
  promises.
- **Aha:** "context makes decode slower" is only partly bytes. The 0.4 ms
  that 900 extra positions add is mostly attention work (dequant, softmax
  merge, an extra dispatch), not DRAM traffic.
- **Viz:** stacked bar per token (weights / KV / other) at P = 64, 512, 960,
  f16 vs int8 KV.
- **Line:** "Halving the cache's bytes bought 5%: the cache was never just bytes."

### GPU stage 11: speculative decoding (a small model guesses, the big one checks)

- **Hypothesis:** batch-1 decode is memory-bound, so a target forward over
  k + 1 rows costs about one step. Let gpt2 (Q4, ~750 tok/s) guess k tokens and
  gpt2-medium/large (f16) check them in one forward: E = (1 − α^{k+1})/(1 − α)
  tokens per target forward, speedup E/(kc + 1). Output identical to the
  target's greedy output (asserted every run).
- **Result:** gpt2-medium 106–132 → 180–371 tok/s (**1.6–2.9×**); gpt2-large
  54–63 → 88–268 tok/s (**1.6–4.25×**): a 774M model at 268 tok/s on a base
  M4. Best k = 7 where α ≈ 1, 4–6 otherwise.
- **Aha:** tokens per forward follow the formula to within a few %; the
  speedup lands at 75–90% of it because verifying k + 1 rows isn't quite free
  and each round has three CPU↔GPU waits. And there's a cliff at k = 8: 9 rows
  round up to the 16-row kernel.
- **Aha 2:** α depends on the text more than on the models: license
  boilerplate and lists 0.99–1.0, prose 0.80–0.93, wikitext 0.80–0.84, and code
  0.88 with gpt2-medium but 0.65 with gpt2-large. A bigger target wins mainly
  because the draft gets relatively cheaper (c 0.16 → 0.08).
- **Viz:** token playback (`bench/specdec.json` `playback`, k = 4): draft tokens
  appear grey, turn green when accepted, red at the first rejection, the
  target's correction slides in; beside it measured speedup vs k with the
  formula curve per prompt, α in the legend.
- **Line:** "The big model never generates faster. It just gets to say 'yes,
  yes, yes, no' four tokens at a time."
- **Aha 3 (round 2):** "verifying k tokens costs one step" is true to 8 rows within
  27%, then the batched matvec runs out of arithmetic (16 rows = 2× a step). With
  the measured verify cost in the formula, every (prompt, k) is predicted within
  5–15%. A 12-row kernel shaved the k = 8 cliff (13.9 → 12.3 ms).
- **Caveat for the post:** greedy GPT-2 loops after ~50 tokens, inflating α on
  every prompt; say so, or use 48-token continuations for the headline.

### WebGPU: the same climb through a portable API (crates/webgpu)

- **Hypothesis:** WebGPU is how a browser tab reaches the GPU. Same rules as
  Metal (dispatches are cheap inside a pass, round trips are expensive), but
  through a translation layer and without Metal's SIMD-group intrinsics.
- **Result (native wgpu on Metal, quiet machine):** 16 → **451 tok/s** decode
  over 13 rungs (naive → one submit → workgroup matvec → parallel LN/attention
  → f16 → Q8 → GPU argmax → GPU-resident loop → tiled prefill → Q8 head),
  prefill 400 → 2,676 tok/s. **75–85% of hand-written Metal at every
  rung** (Q8 resident 451 vs 600).
- **Aha:** the gap isn't dispatch overhead (1.8 µs vs 0.76 µs per dispatch is
  4% of a token); it's the matvec kernel (59 vs 79 GB/s) without `simd_sum`.
  Kernel fusion, worth a lot on Metal, did nothing here.
- **Viz:** the Metal and WebGPU ladders side by side, same rung names
  (`bench/webgpu.json` carries the paired Metal numbers), plus the probe row:
  per-dispatch / per-pass / round-trip costs.
- **Line:** "A portable API costs you about a fifth, and almost none of it is
  where you'd expect."

### WebGPU round 2: LayerNorm into the weights, vector attention, f16 KV

- **Hypothesis:** with dispatch overhead ruled out, the WebGPU gap is in kernels:
  LayerNorm kernels (13% of a profiled token), scalar-load attention at long
  context, and an f32 KV cache.
- **Change:** (1) *lnfold*: fold LN gain/bias into the next matrix at upload,
  y = rstd·(W'x − mean·rowsum W') + b'; the matvec sums x and x² from loads it
  makes anyway, so decode has no LayerNorm kernels (89 → 65 dispatches).
  (2) *attn4*: q in registers, vec4 keys/values, softmax sum inside P·V.
  (3) *kv16*: f16 KV cache. (Dead end: flash-decoding, slower everywhere.)
- **Result (Q8, interleaved):** short context 516 → 556 tok/s (+8%); at prompt
  960 324 → 435 tok/s (+34%), prefill at 960 +20%. **~90% of Metal** at short
  context (556 vs 600).
- **Aha:** flash-decoding lost because attention at long context was already
  reading DRAM at ~90 GB/s with 12 workgroups; the problem was bytes (5.9 MB
  of f32 K/V per layer at 960 tokens), so f16 KV fixed what more parallelism
  couldn't. Check the bytes before adding threads.
- **Aha 2:** a LayerNorm is a scale and a shift; both can live in the next
  matrix. The fused-LN that failed on Metal re-normalised x in every
  threadgroup; the algebraic fold needs only two extra sums.
- **Aha 3 (the robustness tax):** browsers bounds-check every array index in a
  shader. Turning that off natively made prefill 41% faster; rewriting the
  matmul to move vec4s instead of scalars (4× fewer checked accesses) got
  +67% prefill *with* the checks on (webgpu-vec, 2,650 → 4,413 tok/s), and
  left only 6% on the table.
- **Viz:** decode tok/s vs context length (64, 512, 960) for q8head → lnfold →
  attn4 → kv16, plus the flash-decoding bar sitting below attn4.
- **Line:** "We split attention across 16× more workgroups and it got slower.
  It was never short of workers, only of bandwidth."

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

### Browser (wasm in a tab, Chrome and V8 on the M4)

Same Rust, compiled to `wasm32-unknown-unknown`, running in a Web Worker on the demo page (`web/`). Three builds of
`crates/wasm`, picked at runtime by `WebAssembly.validate` on the real binaries: **simd128** (every browser with wasm
SIMD), **relaxed** (+ relaxed-SIMD, Chrome 114+ / Firefox 145+), **threads** (+ atomics and a shared heap, needs
cross-origin isolation, built with nightly). Measured with the page's Ladder benchmark or the same loop in Node
(`web/scripts/record.mjs`, same wasm, same V8): 64-token prompt, greedy, median of the decode steps.

| stage | where | decode tok/s | prefill tok/s (P=64) | native decode / prefill |
|---|---|---|---|---|
| naive | core | 0.3 | 19 | 0.37 / 26 |
| kv-cache | core | 13.9 | 20 | 19.0 / 27 |
| unrolled | core | 96–101 | 169 | 100.5 / 185 |
| neon (runs unrolled) | core | 97 | 171 | 100.3 / 183 |
| tiled (scalar tile) | core | 96 | 164 | 101.1 / 469 |
| wasm-simd v1 (4 rows per pass) | browser | 72 | 259 | n/a |
| **wasm-simd** (1 row per pass) | browser | 97–101 | 255–260 | n/a |
| **wasm-fma** (relaxed-SIMD FMA) | browser | 98–101 | **354–370** | n/a |
| **wasm-threads** (10 workers) | browser | 87–89 | **809–818** | (threads: 118 / 1021) |
| wasm-threads (4 workers) | browser | 91 | 661 | |
| q4 (core, scalar dequant) | core | 91–125* | 150–198* | n/a yet |
| q8 (core, scalar dequant) | core | 36–39* | 168–199* | |
| f16 (core, scalar convert) | core | 12–14* | 125–158* | |

V8 numbers from Node on a lightly loaded machine; the threads and quantised rows (*) are Chrome while the native
hillclimb agents had the machine at load average ~25, so treat them as lower bounds and their ratios as the signal.

- **Explicit SIMD: prefill yes, decode no.** The simd128 4×4 register tile lifts prefill from 169 to 255–260 tok/s
  (1.5×): each weight vector loaded feeds four rows. Natively `tiled` gets 2.5× from the same idea, but core's tile is
  scalar in wasm (no NEON to lower to), which is why `tiled` was flat here. Decode is unchanged: it's the
  autovectorised `unrolled` loop in different clothes.
- **Failure, then fix: don't interleave weight rows.** The first decode kernel walked four weight rows at once to
  reuse each load of `x`. It was 25–30% *slower* than `unrolled` (72 vs 100 tok/s). Four interleaved streams 3 KB apart
  is the pattern stage 3's NEON null result already warned about; one row at a time with four accumulators matches
  `unrolled` exactly.
- **Relaxed-SIMD FMA: +40% prefill, 0% decode.** `f32x4_relaxed_madd` lowers to `fmla` on arm64. Prefill goes
  255 → 370 tok/s; decode doesn't move. That is the roofline talking: decode is memory-bound, prefill compute-bound.
- **Correction: in the browser, decode is memory-bound after all.** Round 1 reported a 15–19 GFLOP/s compute roof
  and concluded decode was compute-bound in a tab. The probe was wrong twice: its first call ran in V8's baseline
  compiler (Liftoff, before tier-up to TurboFan), and a `black_box(&acc)` per iteration forced the accumulators
  through memory. Warmed up and without the spill, one core does **52 GFLOP/s** with simd128 mul+add and **~96
  GFLOP/s** with relaxed FMA. Ridge points: 0.9 and 1.6 FLOP/byte, both right of decode's 0.5. Decode at ~100 tok/s
  moves ~50 GB/s against a measured ~57–64 GB/s single-thread stream: 80–85% of this tab's speed of light.
- **Threads: 4.8× prefill, nothing for decode.** `wasm-threads` splits wasm-fma's matmuls over a wasm-bindgen-rayon
  pool of Web Workers sharing the heap. Prefill: 818 tok/s, 2.2× wasm-fma and 4.8× `unrolled`, near native rayon
  `threads` (1021). Decode: 87–91, no gain at 4 or 10 workers. Each token is 49 matmuls of ~0.2 ms; a rayon
  fork/join through `Atomics.wait` wake-ups costs a visible fraction of that, and one wasm thread already pulls most
  of the bandwidth the worker pool can get. Same lesson as native stage 5/6 (overhead dominates small matmuls), harsher.
- **Core's threaded stages can't thread in wasm.** Stage 6+ start OS threads (`std::thread::spawn`), which
  `wasm32-unknown-unknown` doesn't have, so the wasm builds never enable core's `threads` feature. Those stages run
  single-threaded and the page labels them "wasm: 1 thread"; the browser gets its own threaded rung instead.
- **Quantised stages, unexpected order.** In wasm, q4 (91–125 tok/s) roughly ties the best f32 rungs even with scalar
  dequant, q8 is slow (36) and f16 is slowest (12): f16→f32 is a bit-twiddling scalar conversion per weight (native
  uses NEON `fcvt`), while q4's nibble unpack is cheap integer work on a third of the bytes. Natively all three
  should be ordered by bytes; in a tab the decode cost of the format matters as much as its size. (q4 also changes
  the greedy output: "to read the right, the right, ...". Fewer bytes, worse model.)
- **f16 download.** The page now fetches a pre-converted f16 safetensors (`web/scripts/to-f16.mjs`): **274 MB instead
  of 548**, widened to f32 inside wasm while it streams (`Bytes::from_f16`), so the heap peak is unchanged (~1.05 GB)
  and every kernel still sees f32. Cost: max |Δlogit| 0.03–0.05 against the f32 checkpoint; greedy output identical
  for 32/32 tokens on three prompts. Load from localhost: 1.6 s.
- **The roofs in a tab:** stream read 50–64 GB/s single thread (noisy, contention-sensitive); compute 52 (mul+add) /
  96 (FMA) GFLOP/s after warm-up. f32 speed of light ≈ 100–125 tok/s.
- **The KV race, live** (unchanged from round 1): naive vs kv-cache under a fair scheduler; naive's per-token cost
  climbs linearly, kv-cache stays flat at ~70 ms; greedy outputs identical.
- **Loading:** f16 274 MB (or f32 548 MB with `?weights=f32`) streamed into wasm memory, parsed and transposed by core
  in ~1.5 s from localhost, cached in OPFS. Engine wasm 400–510 KB per build (140–175 KB gzipped).
- **Viz:** fig. 2 inside one token (scroll-driven, real per-op timings, residual stream, re-sampleable logits), fig. 3
  logit lens, fig. 7 roofline (decode matmuls now on the slanted roof), fig. 10 every stage racing, fig. 11 ladder
  (decode/prefill toggle, per-build browser bars, recorded bars dashed).
- **Line:** "The same Rust, in a browser tab, matches the native decode speed. Then FMA and threads make prefill 5×
  faster and decode not at all, because decode was never waiting on the arithmetic."

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

### Stage 8: f16 weights

- **Hypothesis:** decode reads every weight per token, so half the bytes ≈ double the speed.
  f16 keeps 11 significant bits, so nobody should notice.
- **Change:** weights converted to half precision once when the session starts. The kernel
  widens 8 halves to floats with one instruction pair (`fcvtl`, via inline assembly
  because Rust's f16 intrinsics are still unstable) and multiplies in f32.
- **Result:** decode **~180 tok/s** (1.5–1.75× f32). Perplexity +0.03%, 99.8% same top token.
- **Aha:** we got 1.6×, not 2×. Halving the bytes exposed costs that don't shrink with the
  bytes: thread dispatch, attention, the serial bits. Memory-bound is a *moving* target.
- **Viz:** speed-of-light gauge: switch dtype f32 → f16 and watch the ceiling needle
  double while the achieved needle moves less.

### Stage 9: Q8_0 (8-bit), and the matrix you can't compress

- **Hypothesis:** 8-bit blocks (32 weights share one scale) are "near-lossless", and int8 dot
  products (`sdot`: 16 multiply-adds per instruction) are fast.
- **First try:** everything 8-bit. Fast (~268 tok/s), but perplexity +4.7% and only 84% of
  top tokens matched. At 4 bits the same recipe gave perplexity 516 (from 37).
- **Detective work:** quantising one matrix at a time showed the culprit: the **LM head**,
  the 50257 × 768 embedding table GPT-2 reuses to produce logits. At 4 bits that one matrix
  alone multiplies perplexity 6.6×. Every other matrix costs 1–4%. Its *input* is also hard
  to quantise: GPT-2's final LayerNorm boosts a few channels 17×, and those outliers wreck
  8-bit rounding for their neighbours.
- **Result:** Q8_0 blocks + an f16 LM head: perplexity −0.1% (noise), 98% top-1, decode
  **~230–240 tok/s**. A clever fold of the LayerNorm gain into the head helped but not
  enough.
- **Aha:** a third of GPT-2 small is one matrix, and it's the one that hates quantisation.
  On a 7B model the embedding is ~2% of the weights. Small-model lessons don't transfer.
- **Viz:** "quantisation microscope": a bar per matrix, KL when only that matrix is
  quantised, with the LM head's bar off the chart. Plus the ln_f gain plot with its
  17× spikes.
- **Line:** "The fastest matrix to compress was the one we couldn't afford to."

### Stage 10: Q4_0 (4-bit)

- **Change:** 4-bit blocks, unpacked to int8 in registers, same `sdot`. LM head stays f16.
- **Result:** decode **~240–263 tok/s**, only 5–10% over 8-bit. Perplexity +7.3%.
- **Aha:** the f16 LM head is now 59% of the bytes, and per-token overhead (~2.5 ms of a
  ~4 ms token) doesn't shrink with the weights. We cut the bytes 3.8× since f32 and decode
  only doubled. The next win isn't smaller numbers, it's less waiting.

### Stage 11: q8-fast (stop paying for coordination)

- **Hypothesis:** a decode token is now mostly *overhead*: ~49 thread dispatches at up to
  8 µs each, plus attention running on one core.
- **Change:** a second thread pool, where claiming work is one atomic add (not a
  compare-and-swap loop) and threads report completion once each, not per chunk.
  Attention split across cores even for short contexts. Same weights, same answers,
  bit for bit.
- **Result:** empty dispatch 8.3 → 2.6 µs. Decode **+4–15%**: Q8 **360 tok/s** at 6 threads,
  Q4 **372** at 4.
- **Aha:** the best thread count for decode is 4–6, not 10. The efficiency cores add almost
  no memory bandwidth, and every step waits for the slowest thread.
- **Aha 2 (method):** the machine's background load drifted ±20% within minutes, so this
  stage was measured A/B interleaved. Benchmarks taken five minutes apart were lying to us.
- **Viz:** per-op flame strip of one token at stage 10 vs 11, with the LM head now the
  widest bar.

### Stage 12: q8-rescore (an approximate LM head, corrected where it matters)

- **Hypothesis:** of the LM head's 50257 logits, only those near the top decide what we
  say. Score everything cheaply in 8-bit, then recompute the top few hundred exactly.
- **First try:** the top token was always right (100%), but the probability distribution
  was still off (KL 1e-2). GPT-2 spreads real probability over thousands of tokens, so
  the "unimportant" tail isn't negligible. A clever statistical tail correction did
  nothing.
- **Fix:** the error was again in the *activations*: 8-bit can't hold the LayerNorm's
  17× outlier channels. Store each activation as two 8-bit numbers (value + residual).
  That doubles cheap integer math on a memory-bound pass, and the error drops 10×.
- **Result:** decode **394 tok/s (Q8, +13%)** and **480 tok/s (Q4, +17%)**. Q8 quality: +0.2%
  perplexity, 98% same top token as f32.
- **Aha:** the trick isn't "use fewer bits", it's "spend the precision where the probability
  mass is". And when a bug-hunt and an optimisation keep pointing at the same 6 channels
  of one LayerNorm, believe them.
- **Viz:** logit bar chart: approximate (8-bit) bars in gray, rescored top candidates
  snapping to their exact values in color, with the margin as a dashed line below the max.

### Stage 13: q8-attn (attention in registers)

- **Change:** a hand-written attention kernel for 64-wide heads that keeps the query and
  the output in NEON registers the whole time.
- **Result:** decode **404 tok/s** at short context (+3%), **+6.5% at 512 tokens**.
- **Aha:** at 600 tokens of context, attention is still 37% of the time, and it's not the
  arithmetic, it's the memory layout. Each head's keys are scattered in 256-byte pieces,
  and at 512 tokens the cache is 38 MB per token, more than a quarter of the (8-bit)
  weights. The KV cache is now a weight you have to read.

### Stage 14: q8-kv16 (the KV cache, compressed and reordered)

- **Change:** store keys/values per head contiguously, in half precision.
- **Result:** decode **+7% at short context, +17% at 512 tokens, +60% at 900**. Quality vs
  f32 identical (KL 1.84e-3 either way).
- **Aha:** compared against the 8-bit stage it replaced, the KL looked bad. Compared
  against the f32 truth, nothing changed. Two lossy versions disagree with each other
  more than either disagrees with the truth, because every tiny change flips int8
  rounding somewhere. Always measure against the reference.
- **Aha 2:** halfway through, memory bandwidth on this machine halved (another
  workload on the GPU). Only interleaved A/B measurements survived it.

### Stage 15: q8-prefill (integer math for the prompt)

- **Change:** the prompt's activations are quantised to 8-bit too, and the prefill
  matmul uses `sdot` (32 multiply-adds per 2 instructions) instead of float FMAs.
- **Result:** prefill **2415 tok/s** at 64 tokens (+52%) and **2143** at 512, now **1.6–2.4×
  Apple's Accelerate**. Quality unchanged within noise.
- **Aha:** decode wanted fewer *bytes*; prefill wants fewer *instructions*. The same
  int8 trick pays both bills, for different reasons.

## Open questions / things to verify before publishing

## The learner path (web: /, /learn, /lab)

The demo page was a wall of expert figures. It's now a course with the engine as its evidence (plan in `web/IA.md`):
a landing page asking *what happens when a model writes the next word, and how fast can one computer do it?*, then 19
one-idea chapters in three parts, then the Lab (the old page, curated).

- **Part 1, how a model writes text** (8 chapters): next-token guessing on a real run, a live GPT-2 BPE tokenizer in
  ~80 lines of TS (cross-checked ID-for-ID against the engine's Rust tokenizer), real embedding neighbours (" Monday" →
  the other six days at cosine 0.80–0.89), a real 8×8 corner of a GPT-2 matrix times a real embedding, toy attention
  then the real 144 heads, the residual stream + logit lens, sampling on real logits, the loop replayed at its measured
  timings (prefill 14 tokens in 82 ms; decode ~10 ms/token in V8).
- **Part 2, why it's slow**: the truck and the factory (0.5 FLOP/B vs the tab's 1.6 ridge), the speed-limit
  calculator (bandwidth ÷ bytes; 243 tok/s for f32 GPT-2 on a 120 GB/s M4), and the starting line (0.37 tok/s, 0.15% of
  the limit, because naive recomputes the whole text: 26 tok/s prefill ÷ 72 tokens ≈ 0.36).
- **Part 3, making it fast**: the hillclimb regrouped into six milestones, each "bottleneck → idea → result", with one
  climb chart gaining a bar per chapter: KV cache 19 → faster arithmetic 100 → many cores 146 → smaller numbers 497 →
  the GPU 817 tok/s (quiet machine; 641 under load) → batching 6,453 tok/s total (256 users, 25 each). Ending on a ring all-reduce teaser.
- Every figure says what you're looking at and what to try, carries a provenance tag (toy / diagram / real weights /
  recorded engine run / measured / live), and the chart colours are a validated 3-hue palette.
- Select any text to ask about it: a Tangent-style tooltip and streamed side panel, proxied server-side (Together,
  DeepSeek V4.1 Flash; the `-1017` release is preferred automatically once Together serves it).
- **Line:** "One idea per page, and every number on it is one we measured."

### New in the course: batching, serving, and guessing ahead (18 Oct)

- **Numbers come from the bench files now, not from the prose.** `web/scripts/learn-data.py` reads
  `bench/results.jsonl`, `batching.json`, `serving.json`, `specdec.json` and `webgpu.json` into
  `public/learn/data.json`; the climb chart, the GPU chapter, the speed-limit calculator, batching and the landing
  claim all read that. The quiet re-measurements moved three headlines: single-stream GPU decode 641 → **817 tok/s**,
  GPU read speed 80 → **105 GB/s**, and the landing claim 1,700× → **2,200× faster for one user** (817 / 0.37; the
  batching total is a different measure and never enters it).
- **Chapter 17, Batching**, is now three figures, one idea each: total vs per-user speed as the batch grows (4-bit:
  788 → 6,453 tok/s total at 256 users, while each user drops to 25); the same 24 requests served by static and by
  continuous batching on one time axis (median first word **3.5 s → 40 ms** at 16 requests a second, 1.7× the
  throughput); and, folded, the paged KV cache's real block tables (request 0's 18 pages scattered across the pool by
  step 250; 3.3% of handed-out memory unused; peak 493 pages vs static's 799).
- **Chapter 18, Guess ahead** (speculative decoding): intuition first (decode is memory-bound, so checking 5 tokens
  costs about what writing 1 does; a model 10× smaller guesses the easy ones), then a playback of a real recorded run
  showing which guesses the big model kept, how many it threw away, and its own token per check, then speed-up by kind
  of text. GPT-2 large with a 4-bit GPT-2 small drafter: **63 → 268 tok/s (4.25×)** on licence text, faster than GPT-2
  medium alone; acceptance ~1.0 for licences and lists, 0.80–0.93 for prose, 0.65–0.88 for code. A fold says plainly
  that greedy GPT-2 loops, which flatters every acceptance rate.
- **Lab picker**: the new WebGPU stages have plain names (folded LayerNorm, 4-wide attention, 16-bit KV cache, wider
  loads; flash decoding tagged "tried, slower, kept for honesty"), the column headers stay pinned while the list scrolls,
  and "this browser" is filled from `public/recorded/webgpu.json`: the page's own benchmark run in Chrome 155 on the
  M4 (load ~3, nothing else benchmarking). In the tab: best decode **558 tok/s** (webgpu-kv16), prefill up to **5,463
  tok/s** (webgpu-vec), on a par with native wgpu (556 / 4,413) and ~90% of hand-written Metal's matching rung (600).
- **Line:** "A small model can do much of a big model's typing, without changing a word, as long as the text is
  predictable."

