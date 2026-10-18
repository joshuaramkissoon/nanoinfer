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

## Open questions / things to verify before publishing
