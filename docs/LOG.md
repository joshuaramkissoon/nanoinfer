# Hillclimb log

The engineering journal. Every step: hypothesis, change, measured numbers, how far
we are from the ceiling, and what surprised us. Numbers come from
`cargo run --release -p nanoinfer-cli -- bench ...` on the machine below and are
also in `bench/results.jsonl` (one JSON line per run, with git sha).

**Machine:** Apple M4 (base, 4P + 6E cores), 24 GB unified memory, macOS.
**Model:** GPT-2 124M (`openai-community/gpt2`), f32 weights, 1 thread.
**Bench defaults:** prompt 64 tokens, 1 warmup + 5 timed runs, median ± stddev.
The bench thread asks for QoS `USER_INTERACTIVE` so macOS keeps it on a
performance core (see the Stage 1 surprises).

## The ceiling

Decoding one token at batch size 1 has to stream every weight from memory once.
Nothing can be faster than:

```
decode ceiling = memory bandwidth / bytes of weights read per token
               = 120 GB/s / 494.6 MB  ≈ 243 tok/s   (f32)
```

494.6 MB = all 12 blocks' matrices and biases (85.0M params) plus the full
50257 × 768 token embedding, which doubles as the LM head (38.6M params), at 4
bytes each. The position table is excluded (only one row is read per token).
120 GB/s is Apple's spec figure for the base M4. Stream benchmarks typically
reach 85–90% of spec, so the *practical* ceiling is closer to ~210 tok/s. We
plot against the spec number and say so.

**Measured ceiling (`nanoinfer membw`, stage 2 session).** Each of N threads sums
its own slice of a 512 MB buffer (u64 lanes, so the loop is load-bound). Best of
10 passes, with Backdrop.app using ~2 cores in the background:

| threads | 1 | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| read GB/s | 58 | 70 | 92 | 96 | 98 | 98 |

So the CPU cluster tops out at **~98 GB/s, 82% of the 120 GB/s spec**, and that
needs ≥6 threads. One P-core alone can stream **~57 GB/s**. Realistic f32 decode
ceilings: **~198 tok/s** (all cores) and **~115 tok/s on one thread**. We keep
the 120 GB/s spec number as the headline denominator (so % figures stay
comparable across the series) and quote the measured ones alongside.

(First version of the probe reported 431 GB/s at 10 threads. The coordinating
thread took the timestamp after the start barrier, and with every core busy it
was descheduled until the workers had nearly finished. Each worker now times
itself and we take max(end) − min(start).)

This ceiling only falls if we read fewer bytes per token: smaller dtypes (f16
halves it, int8 quarters it) or batching (several sequences share one read
of the weights).

| Stage | Prefill tok/s | Decode tok/s | % of decode ceiling | Speed-up vs prev |
|---|---|---|---|---|
| 0 naive | 26.16 ± 0.31 | 0.368 ± 0.003 | 0.15% | — |
| 1 kv-cache | 27.06 ± 0.43 | 18.95 ± 0.16 | 7.8% | 51× decode |
| 2 unrolled (16 accumulators) | 184.8 ± 3.4 | 100.5 ± 2.1 | 41.4% | 5.8× decode, 7.5× prefill |
| 3 neon (explicit `fmla`) | 182.6 ± 1.8 | 100.3 ± 1.4 | 41.4% | 1.00× (null result) |
| 4 tiled (4×4 register blocking) | 469 ± 119 (noisy; ~450 in quiet runs) | 101.1 ± 16 | 41.7% | 2.5× prefill |
| 5 threads (rayon, 8 threads) | 1021 ± 13 | 118.3 ± 1.7 | 48.8% | 2.1× prefill, 1.2× decode |
| 6 spin-pool (8 threads) | 833 (rayon's 984 better) | 115.0 ± 3.5 (best seen 124) | 47–51% | 1.1–1.2× decode vs rayon |
| 7 fast-ops (10 threads) | ~1400–1500 (interleaved; 1327 ± 301 recorded) | ~119 (interleaved) | ~49% | 1.75× prefill, 2.2× at P=512 |
| 8 f16 weights (4–10 threads) | ~1240–1320 | ~180 (4t) | 37% of 485 / 74% of 243 | 1.5× decode |
| 9 q8_0 + f16 head | ~1240–1330 | ~230–240 | 34% of 693 | 1.3× decode vs f16 |
| 10 q4_0 + f16 head | ~1215–1230 | ~240–263 | 28% of 918 | 1.05–1.1× vs q8 |
| 11 q8-fast / q4-fast (low-latency pool) | ~1415 | **360 (q8, 6t) / 372 (q4, 4t)** per-token, interleaved | 52% of 693 / 41% of 918 | 1.04–1.15× |
| 12 q8-rescore / q4-rescore | ~1400 | **394 (q8) / 480 (q4)**, 6 threads | 46% of 860 / 39% of 1237 | 1.13–1.17× |
| 13 q8-attn (NEON attention) | ~1425 | **404** (ctx 64–128), 292 (ctx 512+) | 47% of 860 | 1.03× short / 1.065× long ctx |
| 14 q8-kv16 (head-major f16 KV) | ~1415 | +7% short ctx / **+60% at 900 ctx** (A/B; bandwidth-starved period) | | |
| 15 q8-prefill (int8 sdot prefill) | **2415** (P=64), 2143 (P=512) | ~400 (6t, per-token) | 58% of 693 | 1.5× prefill |

(prompt 64 tokens, 16 decode steps for both rows, so they're directly comparable.)

---

## Stage 0: naive

**Hypothesis.** The simplest correct forward pass, recomputing the whole sequence
for every new token, will be correct but terribly slow at decode, and its decode
speed will fall as the context grows, because each token costs a full forward over
`context_len` rows.

**What it is.** `crates/core/src/stages/naive.rs`. Scalar f32 kernels
(`kernels/reference.rs`): a triple-loop matmul with one accumulator per output, a
straightforward layernorm, tanh-GELU, and causal attention that materialises scores
per (query, head). No caching, no SIMD, no threads, default release profile (no LTO,
no `target-cpu=native`). Only the last position's logits are computed: an
inference engine never needs the others.

**Correctness.** Last-position logits match HF transformers (f32, eager attention)
on 4 prompts with max |diff| between 7.6e-5 and 1.8e-4 (test tolerance 1e-3). Greedy
generation matches HF token for token.

**Numbers** (`bench --stage naive --prompt-len 64 --tokens 16 --runs 3`):

| | tok/s |
|---|---|
| prefill (64 tokens, one forward) | 26.16 ± 0.31 |
| decode (context 64 → 80) | 0.368 ± 0.003 |
| % of 243 tok/s ceiling | 0.15% |

**Reading the numbers.** Decode ≈ prefill ÷ average context: 26.2 / ~72 ≈ 0.36.
Each "decode" step *is* a prefill of the whole context, so generating token 80 costs
80× generating token 1. Generating N tokens is O(N²) total.

**What surprised me.**
- Prefill isn't faster per token than a single forward. In theory prefill reuses each
  weight across 64 rows, so it should be compute-dense and fast. Our matmul loop
  order (for each row → for every weight row → dot product) streams the whole weight
  matrix again for every input row, so it gets no reuse at all. Blocking or tiling the
  matmul to reuse weights across rows is a future stage, and it's a *prefill*
  optimisation, not a decode one.
- The scalar matmul runs at about 4.6 GFLOP/s. That's more than one FMA latency chain
  should allow (~2.2 GFLOP/s at ~4 cycles per dependent FMA, 4.4 GHz). The M4's
  out-of-order core overlaps the *next* output's independent dot product with the
  current one. That's free ILP, not compiler vectorisation: LLVM can't reorder a
  float reduction without fast-math.

**Learner notes.**
- The "stupid" version is worth writing. It's the oracle every later stage is tested
  against, and its cost curve (quadratic) is the motivation for the KV cache.
- The causal mask is not a matrix of -inf. It's just a loop bound: query `p` looks at
  keys `0..=p`.
- HF GPT-2 stores linear weights as `[in, out]` (`Conv1D`). We transpose to `[out, in]`
  at load so every dot product walks memory contiguously.

---

## Stage 1: KV cache

**Hypothesis.** Old positions are only ever used through their keys and values in
attention, and those never change (causal attention). If each layer caches K and V,
a decode step only pushes 1 row through the matmuls instead of `context_len` rows.
Decode should speed up by roughly the average context length (~70× here), minus the
LM head, which both stages pay once per token anyway.

**What changed.** `crates/core/src/stages/kv_cache.rs`: a `[n_layer, max_seq, C]` cache
for K and V (72 MiB at the full 1024 context). The forward runs only the new tokens,
appends their K/V rows to the cache, and attends over the cache. The kernels are
unchanged (same `Reference` set). The `attention` kernel was written from day one to
take K/V as `(slice, stride)`, so the same function reads from the fused QKV buffer
(naive) or from the cache (kv-cache).

**Correctness.** Same logits as HF (identical max |diff| to naive: the arithmetic per row
is the same). A test feeds prompts one token at a time through the cache, and another
checks naive and kv-cache agree on every step of a 6-token greedy decode.

**Numbers:**

| run | prefill tok/s | decode tok/s | % ceiling |
|---|---|---|---|
| prompt 64, 16 decode steps | 27.06 ± 0.43 | **18.95 ± 0.16** | 7.8% |
| prompt 64, 64 decode steps | 26.92 ± 0.30 | 18.83 ± 0.24 | 7.8% |
| same, `--with-trace` (Recorder + `Instant` per op) | 26.78 ± 0.24 | 18.59 ± 0.12 | 7.7% |
| prompt 512, 256 decode steps | 24.78 ± 0.96 | 17.38 ± 0.54 | 7.2% |

**51× decode speed-up** at context ~64–80. Prefill is unchanged, as expected: the
prompt is one forward either way.

Where a decode token's time goes (`generate --profile`, kv-cache, short context):

| op | share | GFLOP/s | GB/s |
|---|---|---|---|
| lm_head (768 → 50257) | 29% | 5.0 | 10.0 |
| fc_proj (3072 → 768) | 26% | 4.1 | 8.1 |
| fc (768 → 3072) | 22% | 4.9 | 9.9 |
| qkv (768 → 2304) | 16% | 4.9 | 9.9 |
| attn_proj (768 → 768) | 5.5% | 4.9 | 9.8 |
| attention, layernorms, gelu, residuals, kv writes | <1% | | |

**What surprised me.**
- **Decode isn't memory-bound yet.** The textbook line is "decode is memory-bound". Ours
  moves ~10 GB/s out of 120. It's bound by scalar FMA latency, at the same ~4.7
  GFLOP/s as prefill. Memory-boundness is something you *earn* with fast enough
  kernels. Until then, FLOPs are the bottleneck everywhere. This is the setup for the
  SIMD stage: the roofline dot for stage 1 sits far under the compute roof, not
  against the bandwidth slope.
- **Prefill is faster per token than decode** (27 vs 19 tok/s) even though prefill
  gets no weight reuse. The difference is the LM head: decode pays the full
  768×50257 projection (31% of all weights) for every token, prefill pays it once per
  prompt. On a 124M model the embedding table is a third of the model. On a 7B model
  it's ~2%.
- **Attention is invisible at this size.** <1% of decode time at short context, and only
  ~8% slower decode at context 512–768. The O(n) attention term only matters at much
  longer contexts or once the matmuls are fast.
- **fc_proj is ~20% slower than fc** despite identical FLOPs. fc_proj does 768 dot
  products of length 3072. fc does 3072 of length 768. Hypothesis: shorter
  independent chains give the out-of-order core more overlap at chain boundaries. To
  verify when we touch the matmul.
- **macOS scheduled the benchmark on an efficiency core.** Before setting thread QoS,
  consecutive runs gave 19.2, 9.8, 6.5, 15.9, 17.9 tok/s (stddev 4.6) on a loaded
  machine. With `pthread_set_qos_class_self_np(USER_INTERACTIVE)` it was 18.7 ± 0.6
  under the same load. Benchmark hygiene first, optimisation second.
- **Tracing is ~free.** With a timing `Recorder` attached (≈135 `Instant::now()` pairs
  per token), decode was 18.59 vs 18.83 tok/s (−1.3%, about one stddev). With
  `NoTrace` the trace calls are compiled out entirely (`T::ENABLED` is a `const`).

**Learner notes.**
- The KV cache trades memory for compute: 2 × layers × context × d_model floats. For
  GPT-2 small at full context that's 72 MiB, about 15% of the weights. For a 70B model
  with long contexts and many users, KV-cache memory, not weights, limits batch size.
  Hence paged attention, GQA/MQA, KV quantisation (later chapters).
- With the cache, decode does *exactly* one row of matmul per layer per token, so
  weight bytes per token are fixed. That's what makes the bandwidth ceiling a
  meaningful target from here on.

**Next.** We're at 7.8% of the ceiling, compute-latency-bound. The next steps attack FLOP
throughput: multiple accumulators / SIMD in the matmul (NEON, 4 lanes × several
accumulators), then threads, then reduce bytes (f16/int8) once we are actually
bandwidth-bound.

---

## Stage 2: unrolled (multiple accumulators)

**Hypothesis.** Stage 1's matmul is one long chain of dependent adds: each
`acc += a*b` waits ~4 cycles for the previous one. If each dot product keeps 16
independent partial sums, the out-of-order core can overlap 16 chains, and,
because we chose the summation order explicitly, LLVM is allowed to vectorise
them (no fast-math needed: nothing is reordered behind our back). Expect several×
on both prefill and decode, until decode hits the single-core bandwidth wall.

**What changed.** `kernels/unrolled.rs`: `dot()` over `as_chunks::<16>()` with an
`[f32; 16]` accumulator array, tree-reduced at the end, scalar remainder. New
kernel set `Unrolled` (only `matmul` overridden), stage `unrolled` =
`kv_cache::forward::<Unrolled, _>`. Plain portable Rust, no intrinsics.

**Disassembly** (`objdump -d target/release/nanoinfer`, `unrolled::matmul`): the
inner loop is 4 × `ldp q` (2 for x, 2 for w), 4 × `fmul.4s`, 4 × `fadd.4s`. So the
16 scalar accumulators became 4 NEON registers, as hoped. It is **not** fused:
Rust never contracts `a*b + c` into `fmla` (that changes rounding), so we pay
two instructions and two roundings per lane. Explicit NEON (`vfmaq_f32`) fixes that.

**Correctness.** Logits vs HF within 1e-3 (same test as every stage). WikiText-2
(255 tokens, `ppl` command): ppl 51.8370 vs 51.8369 for kv-cache, mean KL 6e-10
nats, top-1 agreement 100%, max |Δlogit| 0.001. Summation order only.

**Numbers** (prompt 64, 64 decode steps, 5 runs; kv-cache re-measured in the same
session because the machine was loaded):

| stage | prefill tok/s | prefill GFLOP/s | decode tok/s | decode GB/s | % of 243 (spec) | % of 115 (1-thread measured) |
|---|---|---|---|---|---|---|
| kv-cache (same session) | 24.7 ± 0.7 | 4.2 | 17.3 ± 0.3 | 8.5 | 7.1% | 15% |
| **unrolled** | **184.8 ± 3.4** | 31.6 | **100.5 ± 2.1** | **49.7** | **41.4%** | **87%** |

Per-op decode profile (`generate --profile`, 64 tokens):

| op | share | GFLOP/s | GB/s |
|---|---|---|---|
| lm_head | 29% | 27.4 | 54.9 |
| fc_proj | 22% | 27.2 | 54.5 |
| fc | 22% | 27.2 | 54.6 |
| qkv | 16% | 27.2 | 54.4 |
| attn_proj | 5% | 27.3 | 54.8 |
| gelu | 2.8% | | |
| attn | 2.3% | | |

**What surprised me.**
- **One code change flipped decode from compute-bound to memory-bound.** Every
  matmul now runs at ~54.5 GB/s, flat across shapes, which is the measured
  single-core read bandwidth (57 GB/s). Stage 1 was waiting on arithmetic. Stage 2
  is waiting on DRAM. Faster arithmetic on one core can now only buy the last
  ~13% of decode; it still matters for prefill.
- **The fc vs fc_proj gap from stage 1 vanished** (both 54.5 GB/s). It was a
  latency-chain effect, gone once the chains are short and independent.
- **GELU is now 2.8% of decode** (scalar `tanh`, 3072 per layer) and attention
  2.3%: things that were invisible at 19 tok/s start to show at 100.
- Prefill went 7.5× (25 → 185) and decode 5.8×. Prefill at 31.6 GFLOP/s is still
  compute-bound (`fmul`+`fadd`, no FMA, one weight row in flight at a time).

---

## GPU (Metal)

The GPU path lives in `crates/metal` (`nanoinfer-metal`, macOS only; an empty
crate elsewhere). It reuses core's loader (`Model`, already `[out, in]`),
tokenizer and config, and runs its own forward pass
(`crates/metal/src/forward.rs`). Each stage is a `MetalStage` → `Recipe`
(sync mode, dtype, matvec/matmul kernel, norm/attention style, fusion, GPU
argmax). Kernels are only added, never edited, so every stage stays runnable:
`nanoinfer bench --stage metal-<name>` (the CLI routes `metal-*` stages to
`crates/cli/src/metal_bench.rs`). Tests (`crates/metal/tests/reference.rs`)
check every Metal stage against the HF fixtures, both as one prefill and fed
token by token, plus greedy equality for f32/f16 stages.

Bindings: `objc2-metal` 0.3 (the `metal` crate is deprecated). Shaders are MSL
source compiled at startup (`newLibraryWithSource`, `MTLMathMode::Safe`, i.e.
no fast-math). All buffers are `StorageModeShared`: on unified memory the CPU
writes tokens straight into a GPU buffer and reads logits straight out of one.

Bench extras for GPU stages: wall ms/token, **GPU-busy ms/token** (sum of
`GPUEndTime − GPUStartTime` over the token's command buffers), dispatches and
command buffers per token, GB/s on wall time and while the GPU is busy.

**Machine note.** During these runs a desktop app (Backdrop, a video-wallpaper
renderer) was using ~200% CPU and some GPU, and the CPU hillclimb agent was
benchmarking at the same time. Numbers are medians of 5 runs but carry that
noise floor (a few %).

### The rules of the game: dispatch overhead and GPU bandwidth (`examples/probe.rs`)

`cargo run --release -p nanoinfer-metal --example probe`

**Dispatch overhead** (empty kernel, 32 threads):

| what | cost |
|---|---|
| encode one dispatch (CPU side, setPipeline + setBuffer×k + dispatch) | 0.075 µs + ~0.013 µs per buffer bound |
| one dispatch on the GPU, inside a serial encoder (back-to-back) | **0.72 µs** (flat from 100 to 10k dispatches) |
| a command buffer with 1 dispatch: commit → waitUntilCompleted, wall | **~150 µs** (136–160) |
| 135 dispatches (one GPT-2 token's worth) in one command buffer: GPU | 99 µs |

So there are two very different overheads. Inside one command buffer a
dispatch costs ~0.7 µs of GPU time (the serial encoder drains the pipe between
dependent dispatches). But every CPU↔GPU round trip (commit + wait) costs
~150 µs. GPT-2 has 135 ops per token: one round trip per op is ~20 ms per token
of pure synchronisation, a hard cap of ~50 tok/s before any maths happens.

**Bandwidth** (float4 grid-stride read / copy kernels over 32–512 MB shared
buffers, GPU timestamps): **~77 GB/s median, ~80–85 GB/s best** for reads,
similar for copies (counting read+write). Thread count (16k–262k) and 4-way
unrolling made no difference. That is ~68% of the 120 GB/s spec and below the
98 GB/s the CPU's 10 threads measured (`membw`). Possibly contention from the
wallpaper app's GPU work; to re-check on a quiet machine. Practical f32 GPU
ceiling: 82 GB/s ÷ 494.6 MB ≈ **165 tok/s** (vs 243 at spec).

### Metal stage 1: metal-naive (one kernel per op, commit + wait per op)

**Hypothesis.** A literal port of the CPU reference kernels (one thread per
output element, serial loops) with a command buffer + wait per op will be
correct immediately and *slower than the CPU* at decode: 135 round trips ×
~150 µs ≈ 20 ms/token of sync alone, plus kernels that don't use the GPU's
parallelism (layernorm is one thread per row; attention is one thread per
(query, head) = 12 threads for a decode step). Prefill should be much better
than CPU, since it has n × d_out independent dot products.

**What it is.** `shaders/naive.metal`: `embed`, `layernorm_naive` (1 thread per
row, two-pass), `matmul_naive` (1 thread per output; neighbouring threads read
neighbouring *rows* of W, i.e. 3–12 KB apart: uncoalesced), `gelu_naive`,
`residual_add`, `kv_write`, `attention_naive` (1 thread per (query, head),
scores in a scratch buffer, two-pass softmax). Weights are copied once into
shared `MTLBuffer`s at load (0.13 s); the LM head reuses the wte buffer.

**Correctness.** Max |Δlogit| vs HF 6.1e-5 … 2.3e-4 on the 4 fixture prompts
(same as CPU), top-1 equal, token-by-token decode equal, greedy 24-token
continuations equal.

**Numbers** (prompt 64, 64 decode steps, 5 runs):

| | value |
|---|---|
| prefill | 286 ± 26 tok/s (49 GFLOP/s), **10.6× the CPU's 27** |
| decode | **14.6 ± 0.5 tok/s**, 6.0% of 243: *slower than CPU stage 1 (19)* |
| wall / GPU-busy per token | 68.3 ms / 41.1 ms (GPU idle 40% of the time) |
| dispatches / command buffers per token | 135 / 135 |

GPU time per op for one decode step (`--profile`, context 65):

| op | GPU µs (12 layers) | share |
|---|---|---|
| attn (12 threads per layer!) | 8818 | 34% |
| fc_proj | 3888 | 15% |
| lm_head | 2483 | 10% |
| qkv | 2211 | 9% |
| ln_1 / ln_2 (1 thread) | 2004 / 1937 | 8% / 8% |
| fc | 1952 | 8% |
| kv_write | 1108 | 4% |
| attn_proj | 1019 | 4% |
| gelu, residuals, embed | ~30 each | <0.5% |

**What surprised me.**
- **The GPU lost to one CPU core at decode**, as predicted, but *not mainly*
  because of sync: the kernels themselves take 41 ms of GPU time. A single
  GPU thread is slow (~1 GHz, in-order-ish, long memory latency): one thread
  doing a 768-wide layernorm takes ~165 µs; one thread per (query, head)
  attention takes 735 µs per layer. The CPU-shaped kernels use 12 of the GPU's
  ~10×128 lanes.
- **The uncoalesced matvec is fine on the biggest matrix**: lm_head reads
  154 MB in 2.48 ms = 62 GB/s. Each thread walks its own row, so a SIMD-group
  touches 32 different cache lines per step, but with 50k rows there are
  enough threads in flight to hide the latency, and each line is fully used by
  its thread over the next 16 iterations. The small matrices (768 rows for attn_proj / fc_proj) are
  latency-bound: fc_proj has only 768 threads each doing a 3072-long serial dot
  product → 324 µs per layer for 9.4 MB (29 GB/s).
- `kv_write` takes 92 µs per layer for copying 2 × 3 KB. Probably per-command-
  buffer fixed cost on the GPU when it touches the big KV buffers (37 MB each);
  gelu/residual dispatches that touch small buffers take 2.5 µs. To confirm.
- Prefill is 10× the CPU *with the same naive kernels*, because 64 × 2304
  independent dot products finally fill the GPU.

### Metal stage 2: metal-onecb (one command buffer per forward)

**Hypothesis.** Same kernels, but encode all 135 dispatches into one command
buffer and commit + wait once per token. Removes ~134 × 150 µs ≈ 20 ms of round
trips, so wall time should drop to about the 41 ms of GPU-busy time: ~24 tok/s.

**What changed.** `Recipe.sync = PerForward` (the `Runner` only commits in
`flush`). Serial compute encoders order dependent dispatches, so no explicit
barriers are needed.

**Numbers** (prompt 64, 64 decode steps, 5 runs):

| | metal-naive | **metal-onecb** |
|---|---|---|
| prefill tok/s | 286 ± 26 | **403 ± 9** |
| decode tok/s | 14.6 ± 0.5 | **41.3 ± 0.2** (17.0% of 243) |
| wall ms/token | 68.3 | 24.2 |
| GPU-busy ms/token | 41.1 | **23.5** |
| command buffers / token | 135 | 1 |

**2.8× decode, better than predicted.** The surprise: **the GPU's own busy time
fell 43%** (41 → 23.5 ms) though the kernels are byte-for-byte identical. With
a wait after every op the GPU idles between tiny command buffers and pays a
per-command-buffer start cost (and likely clocks down); fed one long command
buffer it stays busy (97% of wall). The `kv_write` anomaly above is consistent
with that: per-cb fixed cost, not the copy.

Now 97% of the wall time is GPU kernel time, so the next lever is the kernels:
the attention and layernorm kernels that use a dozen threads, and a matvec
that coalesces.

**Profiling note.** `bench --profile` now times every op *alone*: 20
back-to-back copies of the dispatch in their own command buffer, GPU time ÷ 20
(`gpu::PROFILE_REPEAT`). That removes the per-command-buffer start cost that
inflated metal-naive's per-op numbers (kv_write: 92 µs per-op-synced vs 1.2 µs
steady state), but small matrices then hit in cache on repeats (attn_proj
2.4 MB "reads" at 110–140 GB/s), so trust the GB/s column only for the 28 MB+
matrices (fc, fc_proj, lm_head). `examples/matvec_sweep.rs` measures matvec
bandwidth with the cache defeated (≥ 256 MB of distinct matrix copies,
round-robin).

### Metal stage 3a: metal-simd (SIMD-group matvec for decode)

**Hypothesis.** Decode's matvec should be the bottleneck: make it coalesced and
it approaches the ~80 GB/s probe figure. Each SIMD-group computes R = 4 rows;
lane l reads float4s l, l+32, ... of each row (512 contiguous bytes per
SIMD-group per step), the x float4 is loaded once and reused for the 4 rows,
`simd_sum` reduces. 4 SIMD-groups per threadgroup = 16 rows. Prefill keeps the
naive matmul, so only decode changes.

**Result.** Decode **46.7 ± 0.05 tok/s** (+13%), prefill unchanged (403).
Hypothesis wrong about *what the bottleneck was*: profile (steady state):

| op | metal-onecb µs | metal-simd µs |
|---|---|---|
| attn (naive) | 6204 | 6166 (41%) |
| ln_1 + ln_2 (naive) | 2972 | 3055 (20%) |
| fc_proj | 3203 (35 GB/s) | 1316 (86 GB/s) |
| fc | 1579 (72 GB/s) | 1293 (88 GB/s) |
| lm_head | 2439 (63 GB/s) | 1984 (78 GB/s) |
| qkv + attn_proj | 1911 | 1030 |

The matvecs went 2.4× faster in aggregate (fc_proj 2.4×: with 768 rows of 3072,
the naive kernel had only 768 threads in flight; now 3072 SIMD lanes share
each row). But they were only ~45% of the token. Now 61% of the time is the
12-thread attention and the single-thread layernorm.

**matvec_sweep** (GB/s, cache defeated): naive 29–76 (worst on the short-and-
wide fc_proj/attn_proj), every SIMD variant (R ∈ {1,2,4,8} × {1,2,4,8}
SIMD-groups) 60–90 GB/s with run-to-run noise of ±10 GB/s on this loaded
machine: no variant is clearly better. R=4 × 4 SIMD-groups (the stage's
choice) measured 76–86. The kernel is at the memory wall; the remaining
differences are noise.

### Metal stage 3b: metal-par (threadgroup layernorm + attention)

**What changed.** `layernorm_simd`: one 256-thread threadgroup per row, two
reductions through `simd_sum` + 32 floats of threadgroup memory (same two-pass
maths). `attention_simd`: one 256-thread threadgroup per (query, head); each
SIMD-group takes every 8th key, each lane multiplies 2 of the 64 dims (float2)
and `simd_sum` gives q·k; scores live in threadgroup memory; threadgroup
max/sum for the softmax; then 4 groups of 64 threads accumulate disjoint key
subsets of P·V and are summed through threadgroup memory.

**Numbers** (prompt 64, 64 decode steps, 5 runs):

| | metal-simd | **metal-par** |
|---|---|---|
| decode tok/s | 46.7 | **137.0 ± 2.9** (56.5% of 243; ~83% of the measured-80-GB/s ceiling) |
| wall / GPU-busy ms per token | 21.4 / 20.6 | 7.30 / 6.63 |
| GB/s (wall / while GPU busy) | 23 / 24 | **67.7 / 74.7** |
| prefill tok/s | 403 | 426 |

Steady-state profile: lm_head 35%, fc 24%, fc_proj 21%, qkv 12%, attn_proj 3%,
attention 1.0% (62 µs for 12 layers, was 6166), layernorms 0.9% (55 µs, was
3055). **Attention went 100× faster** by using 256 threads per head instead
of 1. Decode is now ~97% weight streaming.

**Where the remaining time goes** (7.30 ms wall per token):
- 5.8 ms: the ideal (494.6 MB at 85 GB/s).
- ~0.8 ms: GPU busy above ideal: matvecs at 70–80 not 85 GB/s, ~135 × 0.7 µs
  dispatch gaps (0.1 ms), small-op kernels.
- ~0.7 ms: wall above GPU-busy: encoding 135 dispatches on the CPU (~30 µs),
  commit → GPU start → completion signal latency, and the CPU argmax over 50257
  logits.

Next for decode: fewer bytes (f16 halves them), and cut the per-token fixed
costs (fusion, GPU argmax). For prefill: real matmuls (stage 4).

**Contention surprise (unified memory cuts both ways).** While the CPU agent
ran a 1-thread decode bench, metal-par decode measured **96–102 tok/s**;
uncontended, **140**. One CPU core streaming weights at ~55 GB/s takes that
bandwidth out of the same DRAM the GPU reads. From here on, Metal benches are
recorded through a wrapper that waits until no other `nanoinfer` process is
running and flags runs where one started mid-bench. Even uncontended, decode
drifted 118–140 tok/s across the session for unchanged kernels (the desktop
wallpaper app); compare stages measured back to back, not across hours.

### Metal stage 4: prefill matmuls (metal-tiled, metal-mma)

Up to now prefill used `matmul_naive`: one thread per output, re-reading
weights from DRAM/L2 for every token. Prefill is compute-bound in principle
(each weight is reused n times) but only if the kernel actually reuses it.
Decode is unchanged in both stages (n = 1 still takes the SIMD matvec).

**4a, metal-tiled.** Textbook shared-memory tiling: a 256-thread threadgroup
computes a 32 × 32 output tile (32 tokens × 32 outputs), each thread a 1 × 4
strip; K is walked in steps of 32 with X[32×32] and W[32×32] staged in
threadgroup memory (padded +1 column against bank conflicts). Each weight is
fetched from memory once per 32 tokens instead of once per token.

**4b, metal-mma.** `simdgroup_float8x8` hardware matrix multiply-accumulate. A
128-thread threadgroup (4 SIMD-groups) computes 32 tokens × 64 outputs; each
SIMD-group owns a 32 × 16 slab = 4 × 2 accumulator matrices. X[32×32] and
W[64×32] tiles are staged with float4 loads; W is read with
`simdgroup_load(..., transpose = true)` so the MMA sees X · Wᵀ directly. The
result goes through threadgroup memory so ragged edges (n not a multiple of
32: fixture prompts are 5–27 tokens) are handled by a bounds-checked copy.

**Correctness.** Both match HF to ≤ 5.0e-4 (f32 accumulate in a different
order), same top-1, greedy equal.

**Numbers** (5 runs each, uncontended):

| stage | prefill P=64 | prefill P=512 | GFLOP/s (P=512) | decode (P=64, 64 steps) |
|---|---|---|---|---|
| metal-par (naive matmul) | 436 | 456 | 78 | 140 / 126 (drift) |
| **metal-tiled** | 1964 ± 51 | 2188 ± 14 | 372 | 122 |
| **metal-mma** | 3292 ± 296 | **5947 ± 133** | **1011** | 126 |

(CPU best at this point: ~185 tok/s prefill on one core.)

Prefill profile at P = 512 (`--profile`, each op timed alone):

| op | metal-tiled | metal-mma |
|---|---|---|
| qkv | 49.9 ms, 436 GFLOP/s | 11.1 ms, **1961 GFLOP/s** |
| fc | 66.3 ms, 437 | 14.8 ms, 1958 |
| fc_proj | 65.8 ms, 441 | 20.5 ms, 1413 |
| attn_proj | 16.7 ms, 435 | 4.0 ms, 1825 |
| attention (`attention_simd`) | 21.5 ms | 21.7 ms (**28%**) |
| sum of ops | 225 ms | 76.8 ms |

**What surprised me.**
- **Hardware MMA is 4.5× the hand-tiled kernel** on the same tile sizes, and
  reaches ~2 TFLOP/s, 45% of the ~4.3 TFLOPS f32 peak, with no tuning
  (no double buffering, no register-resident X fragments across K steps).
- fc_proj (K = 3072, only 768 outputs) is the slowest GEMM per FLOP: 12 × 16 =
  192 threadgroups for 10 cores is too few to hide the K loop's latency. A
  split-K or smaller-N tile would fix it.
- **Attention is now 28% of prefill**: `attention_simd` is a decode kernel (one
  threadgroup per (query, head), re-reading all previous keys for every
  query: O(n²) memory traffic, no reuse). Flash-attention-style tiling is the
  next prefill lever.
- P = 64 prefill is much worse than P = 512 per token (3.3k vs 5.9k): 64
  tokens = 2 row tiles, so most GEMMs launch only 2 × 36 threadgroups.

### Metal stage 5: fewer bytes (metal-f16, metal-q8, metal-q4)

**Hypothesis.** Decode is at ~75 GB/s while the GPU is busy, so tok/s should
scale with 1 / bytes per weight: f16 ≈ 2×, Q8_0 ≈ 3.8×, Q4_0 ≈ 7× on the GPU
part, until fixed per-token costs (dispatch gaps, commit/wait, CPU argmax)
start to dominate.

**What changed** (`shaders/quant.metal`, `weights.rs`; all on the metal-mma recipe):
- `matvec_f16<R>`: half weights, each lane reads 2 × half4 = 16 B per step
  (512 B per SIMD-group), converts in registers, f32 accumulate.
- `matvec_q8_0<R>` / `matvec_q4_0<R>`: block-of-32 formats with an f16 scale,
  **structure-of-arrays** (all quants, then all scales) instead of GGML's
  interleaved 34/18-byte blocks, so every load is 4/16-byte aligned. Same
  bytes as GGML. Q8 lanes read 16 int8 (half a block); Q4 lanes read one whole
  block as a `uint4`, unpacking low nibbles (elements 0–15) and high nibbles
  (16–31) GGML-style.
- Prefill: one `matmul_mma_q<Loader>` template whose W-tile loader dequantises
  to f32 into threadgroup memory; the MMA loop is unchanged.
- Quantisation on the CPU at load (Q8_0: d = max|w|/127; Q4_0: GGML rounding,
  d = signed-max / −8). 0.1–0.25 s.
- `tests/kernels.rs`: every matvec/matmul kernel in every dtype vs a CPU dot
  product over the *same dequantised* weights (≤ 1e-3), so layout bugs and
  quantisation error are tested separately.
- `Recipe.head_dtype`: the tied LM head (wte, 31% of the bytes) gets its own dtype.

**Quality first** (`examples/quant_quality.rs`: KL(HF || ours) and top-1 on
the 4 fixture prompts; WikiText-2 ppl / KL / top-1 vs our f32 over 2 × 512
tokens, decoded token by token):

| blocks | LM head | fixture KL (mean) | top-1 | wiki ppl | wiki KL | wiki top-1 vs f32 |
|---|---|---|---|---|---|---|
| f32 | f32 | 3.7e-10 | 4/4 | 35.230 | 0 | 100% |
| f16 | f16 | 7.8e-6 | 4/4 | 35.233 | 1.1e-5 | 99.7% |
| q8_0 | f16 | 8.0e-5 | 4/4 | 35.198 | 2.8e-4 | 98.3% |
| q8_0 | **q8_0** | 4.7e-3 | 4/4 | 35.196 | 9.4e-3 | **90.1%** |
| q4_0 | f16 | 1.5e-2 | 4/4 | 37.699 | 6.8e-2 | 82.5% |
| q4_0 | q8_0 | 1.9e-2 | 4/4 | 37.600 | 7.4e-2 | 80.5% |
| q4_0 | **q4_0** | 1.76 | **0/4** | **282** | 2.17 | 31.7% |
| f16 | **q4_0** | 1.98 | 0/4 | 266 | 2.12 | 32.8% |

**The LM head is the fragile part of GPT-2 small.** Q4_0 on the head alone
(f16 everywhere else) destroys the model (ppl 35 → 266); Q4_0 on every block
matrix with a Q8 head costs ppl +7%. Even Q8_0 on the head raises KL 34× over
Q8_0 blocks with an f16 head, and flips 10% of greedy choices, **while
perplexity doesn't move (35.196 vs 35.230)**. Perplexity alone would have
called all-Q8 lossless. Hypothesis (not yet verified): ln_f's output has a few
huge outlier dimensions, so logit = Σ x_i w_i amplifies the quantisation error
of exactly those weights, and block scales set by outliers starve the rest of
the block. So: **metal-q8 = all Q8_0** (top-1 4/4, ppl unchanged, 90% greedy
agreement), **metal-q4 = Q4_0 blocks + Q8_0 head** (the Q4 head is unusable).
Correctness tests: f16 max |Δlogit| ≤ 0.10 (tol 0.25) and greedy equal to HF;
quantised stages are tested by KL (Q8 < 0.01, Q4 < 0.25) and top-1.

**Numbers** (prompt 64, 64 decode steps, 5 runs; ceiling = 120 GB/s ÷ that
stage's bytes per token):

| stage | MB / token | decode tok/s | % of its ceiling | GPU-busy ms | wall ms | GB/s busy | prefill P=512 |
|---|---|---|---|---|---|---|---|
| metal-mma (f32) | 494.6 | 133 ± 2 | 54.8% of 243 | 6.78 | 7.52 | 72.9 | 5947 |
| **metal-f16** | 247.4 | **245 ± 8** | 50.5% of 485 | 3.48 | 4.08 | 71.1 | 6166 |
| **metal-q8** | 131.7 | **365 ± 10** | 40.1% of 911 | 2.26 | 2.74 | 58.4 | 6452 |
| **metal-q4** | 89.3 | 363–475 | 27–35% of 1344 | 1.72–2.21 | 2.1–2.76 | 40–52 | 6465 |

(metal-q4 measured 475 ± 27 in one quiet session and 315–363 in three later
ones, while Spotlight's `mdworker` processes were using ~4 cores. At ~2 ms per
token, CPU wake-up latency on the commit → wait path shows up directly.)

**What surprised me.**
- **f16 is a clean 1.84× decode** at the same ~71 GB/s: bandwidth-bound,
  exactly as the roofline says. Prefill barely changes (compute-bound; the
  loader converts to f32 anyway).
- **Q8 and Q4 stop scaling.** Q8 reads 1.9× fewer bytes than f16 but is 1.5×
  faster; Q4 reads 1.5× fewer than Q8 and is about the same speed. GB/s while
  busy falls 71 → 58 → 40–52. Two reasons, both visible in the q4 profile
  (sum 1.78 ms): (1) small matrices become latency-bound: attn_proj at Q4 is
  0.33 MB, takes 16 µs per layer = 20 GB/s; (2) a fixed per-token floor:
  ~0.5 ms of wall time outside the GPU (encode 135 dispatches, commit, wait,
  wake up, CPU argmax over 50257 logits), plus ~0.3 ms of non-matvec kernels
  and ~135 × 0.7 µs dispatch drains. At 2 ms/token that floor is ~40%.
- In `matvec_sweep`, Q4 on 768-wide rows reaches 37–60 GB/s vs 80 for 3072-wide:
  a row is 24 blocks, so 8 of 32 lanes idle and each lane has one 16-byte load
  in flight. Wider per-lane work or more rows per SIMD-group would help; the
  bigger win is cutting the fixed costs (next stage).

This is DISTILLED law 6 arriving on schedule: at the quantised tiers, GPT-2
small is overhead-bound, not bandwidth-bound.

### Metal stage 6: attacking the fixed costs (metal-argmax, metal-fused-ln, metal-fused, metal-pipelined)

All four build on metal-q4 (Q4_0 blocks, Q8_0 head), where the fixed per-token
costs are largest relative to the token. Each stage keeps the previous one's
changes.

**6a, metal-argmax.** `argmax_partial` (32 threadgroups × 1024 threads, strided,
`simd_shuffle_down` reductions on (value, index), ties → lowest index like
`sampler::argmax`) + `argmax_final` (one SIMD-group). The CPU reads 4 bytes
instead of scanning 50257 floats. +2 dispatches.

**6b, metal-fused-ln (dead end, kept runnable).** Fuse everything into the
matvecs: LayerNorm in the prologue (every threadgroup stages its input row in
threadgroup memory and normalises it, recomputing mean/variance), and
epilogues for bias + GELU, `x += …` residuals, and writing K/V rows straight
into the cache. 11 dispatches per layer → 5; 137 → 64 per token. Per-op
profile showed why it doesn't pay: lm_head went 530 → 658 µs (1571
threadgroups each re-normalising the same 768 floats before their first
weight load) and fc_proj 171 → 282 µs (32 rows per threadgroup → only 24
threadgroups for 768 rows, and each stages 12 KB of input first). It saved
~115 µs of small kernels and lost ~240 µs in the matvecs.

**6c, metal-fused.** Epilogue fusion only (`matvec_epi<W, R, EPI>`: x read from
device memory exactly like the unfused matvec, same shapes), LayerNorm kept as
its own 2.5 µs kernel: 7 dispatches per layer, 89 per token. Removes the
residual, GELU and kv_write kernels and their dispatch drains.

**6d, metal-pipelined.** The CPU stops waiting between tokens.
`MetalSession::decode_greedy(first, n)` encodes token i's forward to read its
input id from `gen[pos]` (`embed_gen`) and its argmax to write `gen[pos + 1]`,
then commits all n command buffers up front and waits once at the end. The
GPU runs the forwards back to back; the CPU's encode time, the commit → start
latency and the completion wake-up all overlap GPU work. Same tokens as the
step-by-step loop (test `gpu_greedy_paths_agree_with_cpu_argmax`), and the
fused + pipelined path reproduces HF's greedy continuation exactly in f32 and
f16 (`fused_pipelined_matches_hf_greedy_in_every_dtype`). Real streaming would
keep 2–3 command buffers in flight and read tokens from completion handlers;
the throughput is the same.

**Numbers** (prompt 64, 64 decode steps, **10 runs** each, back to back,
uncontended; run-to-run noise on this machine is ±5–10%):

| stage | dispatches / token | GPU-busy ms | wall ms | GPU % of wall | decode tok/s | % of ceiling |
|---|---|---|---|---|---|---|
| metal-q4 (stage 5) | 135 | 2.03–2.18 | 2.43–2.83 | 77–83% | 353–412 | 26–31% of 1344 |
| metal-argmax | 137 | 1.82 | 2.29 | 79% | 437 ± 34 | 32.5% |
| metal-fused-ln | 64 | 1.73 | 2.10 | 82% | 477 ± 35 | 35.5% |
| metal-fused | 89 | 1.76 | 2.17 | 81% | 460 ± 28 | 34.2% |
| **metal-pipelined** | 89 | **1.70** | **1.71** | **100%** | **586 ± 25** (614–632 in other sessions) | **43.6%** |
| metal-pipelined-q8 | 89 | 2.00 | 2.01 | 100% | 498 ± 37 (517) | 54.7% of 911 |
| metal-pipelined-f16 | 89 | 3.61 | 3.62 | 100% | 277 ± 14 (284) | 57.1% of 485 |

**What surprised me.**
- **Fusion was a wash; not waiting was the win.** Halving the dispatch count
  (137 → 64) moved GPU time by ~5%, inside the noise. The GPU cost of a small
  dependent dispatch is ~1–3 µs, so removing ~70 of them saves ~0.1–0.2 ms, and
  the LN-prologue version paid most of that back in slower matvecs. Taking the
  CPU out of the loop removed ~0.45 ms per token (wall 2.17 → 1.71) *and* the
  GPU's busy time dropped (1.76 → 1.70): with no gaps, the GPU never idles
  long enough to clock down. Same effect as stage 2, one level up.
- **The argmax step's gain is mostly noise-sized** (≈ 0.1 ms: 50257 compares
  on the CPU + reading 200 KB of logits), but it's what *enables* pipelining:
  without a GPU-side argmax there's no token for the next forward to read.
- With pipelining, GPU busy = wall (100%). From here the only levers are GPU
  kernel time: f16 and Q8 pipelined reach 68.5 / 65.8 GB/s (57% / 55% of their
  spec ceilings) while Q4 sits at 52.6 GB/s: the Q4 matvec on 768-wide rows is
  latency-bound (one 16-byte load per lane per row), and attention (12
  threadgroups, one per head) is now ~9% of a Q4 token.
- Quality bill reminder: metal-pipelined is Q4_0 (ppl +7%, 80% top-1 vs f32);
  metal-pipelined-q8 keeps ppl but agrees with f32 on 90% of greedy picks;
  metal-pipelined-f16 is effectively lossless (99.7%).

### Metal stage 7: attention that scales with context (metal-flashdec, metal-flash)

At prompt 64 attention is ~9% of a Q4 token. At context ~960 (`--profile`,
metal-fused) it is **46% of a decode token** (1233 µs) and **54% of a
960-token prefill** (113 ms): the f32 KV cache at 960 positions is 70.8 MB per
token, *more than the Q4 weights* (89 MB), and `attention_simd` streams it with
only 12 threadgroups (one per head) at decode, while at prefill every query's
threadgroup re-reads all earlier keys.

**7a, metal-flashdec (flash-decoding).** `attn_decode_split`: grid (head,
64-key chunk); each 256-thread threadgroup computes a partial softmax over its
chunk (local max m, sum l, unnormalised o[64]) into a scratch buffer;
`attn_decode_merge` combines the chunks per head (M = max m_s, L = Σ l_s
e^(m_s−M), out = Σ o_s e^(m_s−M) / L). 12 → up to 192 threadgroups at decode;
+12 dispatches per token. Prefill still uses `attention_simd`.

**7b, metal-flash (flash-attention prefill).** `attn_prefill_flash`: a
128-thread threadgroup owns 32 queries of one head and walks the keys in
blocks of 32, staging each K/V block in threadgroup memory once for all 32
queries. Per block each SIMD-group (8 queries): S = Q·Kᵀ via 8×8 MMAs
(transposed K loads), online softmax on S in threadgroup memory (lane = one
row × 8 columns, `simd_shuffle_xor` across the row's 4 lanes, causal mask),
rescale O by diag(α) with one more MMA per 8×8 tile, O += P·V. Divide by l at
the end. Scores never touch device memory.

**Correctness.** Fixture prompts are shorter than one chunk, so a new test
compares SplitK and Flash against `attention_simd` at context 300 (5 chunks;
prefill split as 100 + 150 tokens so the flash kernel runs with pos0 > 0):
max |Δlogit| 9.2e-5 (prefill) and 6.1e-5 / 9.2e-5 (decode), f32.

**Numbers** (Q4 pipelined base, 64 decode steps, 5 runs):

| stage | prefill P=512 | prefill P=960 | decode @ ctx ~100 | decode @ ctx ~550 | decode @ ctx ~930 |
|---|---|---|---|---|---|
| metal-pipelined | ~6000 | 4763 | 586–632 | — | 362 |
| metal-flashdec | 6562 | 4437–4900 | 641 | 470 | 352–406 |
| **metal-flash** | **8901** | **8496–8696** | 575–626 | 469 | 386–391 |

Prefill attention at P = 960: **113 ms → 13.9 ms (8.2×)**; prefill is now
GEMM-bound again (GEMMs ~1.8–2.0 TFLOP/s, 87% of the 107 ms). Decode attention
at context 961: 1233 → 673 µs. Long-context decode improved only ~10%
(362 → ~390): at context 930 a Q4 token reads 89 MB of weights + 69 MB of f32
KV cache, so the remaining lever is fewer KV bytes (f16 KV), not more
parallelism.

**Surprise.** The flash prefill kernel was right first time, and the cheapest
trick in it is the diag(α) rescale: a simdgroup matrix has no per-row scale
op, but multiplying by an 8×8 diagonal matrix is one more MMA per tile.

**7c, metal-kv16 (f16 KV cache).** K/V stored as half: the qkv epilogue
(`mve_qkv_h_*`) and `kv_write_h` convert on write; `attn_decode_split<half>` and
`attn_prefill_flash<half>` convert on load (the kernels became templates over
the KV element type; the float instantiations keep their names). Halves the
KV bytes: 37 KB per position instead of 74 KB. Max |Δlogit| vs f32 KV at
context 300 (f32 weights): 3.9e-3 prefill, 4.1e-3 decode; fixture KL and top-1
unchanged from metal-flash.

| prompt | metal-flash decode | **metal-kv16 decode** | bytes/token (weights + KV) | effective GB/s |
|---|---|---|---|---|
| 64 | 575–579 | 577–602 | 89 + 7 MB → 89 + 3.5 MB | ~55 |
| 512 | 434–469 | **526–549** | 89 + 44 → 89 + 22 MB | ~60 |
| 960 | 375–386 | **439–475** | 89 + 73 → 89 + 37 MB | ~60 |

(decode runs 64 steps after the prompt, so "960" averages context ~992.)
+21–27% at long context, nothing at short context, as the byte count predicts:
both variants move ~60 GB/s of weights + cache. Long-context decode is
bandwidth-bound on the KV cache now; next levers would be Q8 KV or a smaller
cache (GQA doesn't apply to GPT-2).

### GPU summary (all stages, M4 10-core GPU)

Decode: prompt 64 + 64 greedy steps, median of 5–10 runs. Ceilings use the
120 GB/s spec ÷ that stage's bytes per token (practical GPU read bandwidth
measured ~80 GB/s, i.e. ~67% of spec).

| stage | what changed | dtype | prefill P=64 | prefill P=512 | decode tok/s | % spec ceiling |
|---|---|---|---|---|---|---|
| metal-naive | CPU kernels ported, wait per op | f32 | 286 | — | 14.6 | 6.0% |
| metal-onecb | one command buffer per token | f32 | 403 | — | 41.3 | 17.0% |
| metal-simd | SIMD-group matvec | f32 | 403 | — | 46.7 | 19.2% |
| metal-par | parallel layernorm + attention | f32 | 426–436 | 456 | 137–140 | 56–58% |
| metal-tiled | tiled prefill matmul | f32 | 1964 | 2188 | 122* | 50% |
| metal-mma | simdgroup_matrix prefill | f32 | 3292 | 5947 | 126–133* | 52–55% |
| metal-f16 | f16 weights | f16 | 4060 | 6166 | 245 | 50.5% of 485 |
| metal-q8 | Q8_0 weights | q8_0 | 3279 | 6452 | 365 | 40.1% of 911 |
| metal-q4 | Q4_0 blocks, Q8_0 head | q4_0 | 3563 | 6465 | 363–475 | 27–35% of 1344 |
| metal-argmax | GPU argmax | q4_0 | 4253 | — | 437 | 32.5% |
| metal-fused-ln | LN-prologue fusion (dead end) | q4_0 | 4801 | — | 477 | 35.5% |
| metal-fused | epilogue fusion | q4_0 | 4181 | — | 460 | 34.2% |
| metal-pipelined | no CPU wait between tokens | q4_0 | 4528 | — | 586–632 | 43.6–47% |
| metal-pipelined-q8 | same, Q8_0 | q8_0 | 4541 | — | 498–517 | 55–57% of 911 |
| metal-pipelined-f16 | same, f16 | f16 | 3848 | — | 277–284 | 57–59% of 485 |
| metal-flashdec | split-K decode attention | q4_0 | 4900 | 6562 | 641 | 47.7% |
| metal-flash | flash-attention prefill | q4_0 | 5541 | 8901 | 575–626 | 43–47% |
| **metal-kv16** | f16 KV cache | q4_0 | 6269 | 8707 | **602** (549 @ ctx 550, 439–475 @ ctx 990) | 44.8% |

\* same decode kernels as metal-par; the difference is machine drift.

Best numbers: **decode 641 tok/s** (Q4 + flash-decoding, short context;
~72% of the ~890 tok/s practical Q4 ceiling at 80 GB/s), **284 tok/s at f16**
(88% of the practical f16 ceiling), **140 tok/s f32** (85%). **Prefill 8.9k
tok/s** at P = 512 (GEMMs at 1.8–2.0 TFLOP/s, ~45% of f32 peak), 33× the
CPU's 4×4-tiled single core.

**What to try next (GPU).**
- Q4 matvec on 768-wide rows is latency-bound (one 16 B load per lane per row):
  give each lane 2–4 rows in flight, or split rows across half-SIMD-groups.
- Prefill GEMMs: double-buffer the K loop, keep X fragments in registers,
  try `simdgroup_half8x8` (if FP16 MMA throughput > FP32 on this GPU), and
  split-K for fc_proj (K = 3072, only 768 outputs).
- Fuse ln_f + LM head + argmax (the head is 30% of a Q4 token and the logits
  never need to leave the GPU in greedy mode).
- Q8 KV cache; batch > 1 decode (the matvec turns into a skinny GEMM, and
  the weights are read once for the whole batch).
- Re-measure on a quiet machine: GPU read bandwidth (80 GB/s here) and the
  run-to-run spread were both affected by a desktop video app and Spotlight.

## GPU round 2: single-stream leftovers, batching, serving, speculative decoding

**Machine note (round 2).** Load average 18–30 for most of this round:
Spotlight's `mds_stores` at ~340% CPU indexing the 4.7 GB of new weights
(gpt2-medium/large; `weights/` and `target/` now carry `.metadata_never_index`),
Backdrop ~190%, a headless Chromium (web agent) ~100%, and the CPU agent's
10-thread runs at times. Every bench waits until no other nanoinfer/cargo
process is running (`ps` check) and records the load average in its note.
Comparisons are interleaved A/B in one session.

### Metal stage 8a: metal-lanes (8 lanes per row in the decode matvecs)

**Hypothesis.** Round 1 found the Q4 matvec on 768-wide rows latency-bound: a
row is 24 blocks of 16 B, `matvec_epi` gives it a whole SIMD-group and walks
the SIMD-group's rows one after another, so 8 lanes idle and each busy lane
has one 16 B load in flight before a `simd_sum`. Give each row 8 lanes and run
4 rows per SIMD-group at once: every lane walks 3 blocks of its row (12 for
fc_proj), all loads independent, then a 3-step `simd_shuffle_xor` reduction
inside the 8-lane group. Expect the narrow Q4 matrices to approach the wide
ones (~60–70 GB/s).

**What changed.** `shaders/lanes.metal`: `matvec_lanes<W, R, EPI>` with
unit functors (Q4_0 block, Q8_0 half-block, f16 8-wide), the same epilogues as
`matvec_epi` (QKV + f16 KV write, GELU, residual, plain). `Recipe.lanes`;
stage `metal-lanes` = metal-kv16 + lanes for every decode matvec including
the Q8 head. Shape sweep: `examples/lanes_sweep.rs` (cache defeated, median
of 7):

| GB/s | qkv | attn_proj | fc | fc_proj | lm_head |
|---|---|---|---|---|---|
| mve_q4_0 (round 1) | 38.1 | 39.2 | 53.1 | 61.2 | 68.1 |
| **mvl_q4_0 R=1, 4 SGs** | **57.5** | **49.5** | **61.1** | 61.7 | 75.2 |
| mvl_q4_0 R=2, 4 SGs | 61.0 | 47.7 | 60.7 | 57.2 | 71.7 |
| mve_q8_0 | 70.5 | 58.5 | 67.8 | 69.0 | 73.7 |
| mvl_q8_0 R=1, 4 SGs | 61.5 | 59.5 | 74.9 | 75.3 | 82.5 |
| mve_f16 | 76.1 | 68.9 | 72.5 | 76.6 | 75.7 |
| mvl_f16 R=1, 4 SGs | 82.4 | 75.9 | 74.6 | 85.3 | 89.6 |

Q4 on 768-wide rows: **+50% (38 → 58 GB/s)**; Q8/f16 move within ±10% (noise
on this machine). R=2 (8 rows per SIMD-group, 6 loads in flight per lane) is
no better than R=1: 3 loads in flight is enough to cover the latency, and
fewer SIMD-groups per matrix hurts the small ones. Kept R=1, 4 SIMD-groups
(`NANOINFER_LANES=R,NSG` overrides it for sweeps).

**Numbers** (prompt 64 + 64, 5 runs, interleaved, load 25):

| | kv16 | lanes |
|---|---|---|
| pair 1 | 576 | 614 |
| pair 2 | 584 | 640 |
| pair 3 | 592 | 599 |
| recorded (7 runs, load 30) | | 592 ± 47 (P=64), 584 ± 21 (P=512, kv16 was 549) |

**+5–8% decode.** Less than the kernel sweep promises because the narrow
Q4 matrices (qkv + attn_proj) were only ~25% of the token, and the Q4 kernel
is still at ~58 GB/s on them: per 16 B of weights a lane also loads 128 B of
x (from L1), dequantises 32 nibbles and does 32 FMAs, so it is no longer
waiting on DRAM alone. Q4 decode now sits at ~45% of the 120 GB/s spec
ceiling, ~67% of the 80 GB/s practical one.

### Metal stage 9: batched decode (metal-batch, B = 1 … 256)

**Hypothesis.** At batch 1 every weight byte read from DRAM is used for one
multiply-add (0.5–3.5 FLOP/byte, depending on dtype). With B sequences per
forward the matvec becomes a skinny GEMM: the weights are read once per
*step*, so step time should stay flat (memory-bound) while aggregate tok/s
grows ~B×, until the arithmetic catches up with the bytes (the ridge). The
ridge should move left as the weights shrink: f32 has 8× the bytes of Q4 per
FLOP, so it should stay memory-bound to a much bigger B.

**What changed** (new, nothing earlier edited):
- `crates/metal/src/batch.rs` + `shaders/batch.metal`: a ragged, paged
  forward. `Batch::step(work)` takes any list of `(sequence, tokens)`; rows of
  different sequences sit side by side, rows of one sequence are consecutive
  positions. The KV cache is paged (PagedAttention): per layer a pool of
  16-position f16 pages and a block table per sequence, allocated on demand
  and returned when a sequence ends. Token ids live in a GPU table
  `toks[seq, pos]`; the per-row argmax writes `toks[seq, pos + 1]`, which is
  what `Batch::decode_resident` uses to run many steps with the CPU waiting
  once (the round-1 pipelining, for B sequences).
- `matvec_batch` / `matvec_batch_r`: 8-lane-per-row matvec that dequantises a
  weight unit into registers once and dots it with NB ≤ 16 activation rows;
  the `_r` variant gives each lane R weight rows so every x load serves R rows.
  From 32 rows on, the forward switches to the round-1 simdgroup-matrix
  prefill GEMMs (+ GELU / residual / paged KV-write kernels).
- Attention: `b_attn_split` / `b_attn_merge`, flash-decoding per row over its
  own sequence's pages (grid head × chunk × row).
- LM head only for rows that need logits (`b_ln_gather` + head matvec/GEMM),
  `b_argmax_rows` per row.
- **Batch invariance by construction**: every kernel used below 32 rows
  accumulates each token's dot products in the same order whatever NB and R
  are (unit sums formed identically, same 8-lane shuffle tree). A sequence's
  tokens are bit-identical alone or in a batch (`tests/batch.rs`), which is
  what later makes speculative decoding provably equal to greedy.
- Tests (`tests/batch.rs`): f32 batch path reproduces HF greedy (matvec-only
  and GEMM-from-4-rows); Q4 and f16: 4 prompts prefilled in one ragged
  forward + 10 CPU-loop steps + 10 GPU-resident steps == each prompt alone;
  page reuse after freeing sequences; all-row logits + truncate.

**First attempt: `matvec_batch` (R = 1) is cache-bound at ~250 GFLOP/s.**
Q4, matvec only: B=1 631, B=2 952, B=4 992, B=8 1039, B=16 924 tok/s: flat
from B=4. Each weight row re-reads the activation slice: at B=8 that is
32 B of x (from L2) per 0.56 B of Q4 weight, ~4 GB of L2 reads per step.
Rows per lane (the `_r` kernels) cut that R×. Sweep (`NANOINFER_MVR`, Q4,
agg tok/s):

| R by NB (1,2,4,8,16) | B=1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 0,2,4,4,2 | 631 | 1151 | 1646 | 1554 | 2558 | 2808 |
| 0,4,4,4,4 | 579 | 1051 | 1606 | 1503 | 1379 | 1743 |
| 2,2,2,2,2 | 570 | 1183 | 1731 | **2165** | 2494 | 2648 |

R=4 with 8–16 tokens spills registers (R × NB accumulators twice over). Kept
R=2 for NB ≥ 2 and the R=1 kernel for single rows (B=1 is latency-bound and
prefers more SIMD-groups). Results are identical bit for bit across all of
these, only the speed changes.

**Numbers** (prompt 64, 64 GPU-resident decode steps, median of 3, load
14–16; GEMM from 32 rows; `bench/batching.json`):

| B | Q4 agg tok/s | Q4 step ms | Q8 agg | f16 agg | f16 step ms | f32 agg | f32 step ms |
|---|---|---|---|---|---|---|---|
| 1 | 657 | 1.52 | 488 | 268 | 3.73 | 134 | 7.48 |
| 2 | 1194 | 1.68 | 923 | 574 | 3.49 | 249 | 8.02 |
| 4 | 1819 | 2.20 | 1594 | 1047 | 3.82 | 490 | 8.16 |
| 8 | 2235 | 3.58 | 2139 | 1829 | 4.37 | 939 | 8.52 |
| 16 | 2585 | 6.19 | 2499 | 2538 | 6.30 | 1320 | 12.1 |
| 32 | 3704 | 8.64 | 3458 | 2977 | 10.8 | 2241 | 14.3 |
| 64 | 4606 | 13.9 | 4533 | 4002 | 16.0 | 3372 | 19.0 |
| 128 | 4786 | 26.7 | 4807 | 4069 | 31.5 | 3838 | 33.3 |
| 256 | **4986** | 51.3 | 4916 | 4892 | 52.3 | 4146 | 61.7 |

Speed-up over B=1: Q4 7.6×, Q8 10×, f16 18×, f32 31×. Matvec-only Q4 (no
GEMM switch): 2646 at B=32, 2577 at B=64, so the GEMMs win from B≈32.

**What surprised me.**
- **The ridge moves with the dtype, exactly as the roofline says.** Step time
  (the latency every stream sees) is flat while memory-bound: f32 stays at
  7.5–8.5 ms up to B=8 (7× the throughput for +14% latency), f16 to B≈4–8,
  Q4 only to B≈2 (1.52 → 1.68 ms) and is 2.4× slower per step by B=8. With
  ~600 GFLOP/s of batched matvec, the crossover B* ≈ (step time at B=1) ×
  FLOP/s ÷ 2·params: Q4 1.5 ms → B* ≈ 4, f16 3.7 ms → ≈ 9, f32 7.5 ms → ≈ 18.
  Quantisation makes single-stream decode faster and makes batching pay
  off *less*: at B ≥ 64 every dtype ends at ~4–5k tok/s, compute-bound on
  the same GEMMs (1.0–1.2 TFLOP/s, ~25–30% of f32 peak), and Q4's byte
  savings are worth almost nothing (4986 vs 4892 tok/s for f16 at B=256).
- **Per-stream speed is the price.** Q4 B=64: 4606 tok/s aggregate but 72
  tok/s per stream (13.9 ms per token) vs 657 alone. That's the
  throughput/latency frontier a server picks a point on.
- **The first batched kernel was compute-starved, not bandwidth-starved.**
  Reusing weights across tokens is half the job; the activations have to be
  reused across weight rows too (R rows per lane), or the L2 becomes the
  wall. Same lesson as CPU stage 4 (register blocking), on the GPU.
- B=1 through the batched path (657 tok/s, CPU waits once per 64 steps) is
  as fast as the round-1 single-stream best: the ragged/paged machinery
  costs nothing measurable.

### Metal stage 10: continuous batching vs static batching (metal-serve)

**Hypothesis.** A static batch runs until its *longest* request finishes:
short requests hold a slot (and their KV reservation) doing nothing, and
requests that arrive meanwhile wait for the whole batch. Iteration-level
scheduling (Orca) retires and admits sequences every step, so the GPU always
has ~max_batch live rows. Expect higher throughput when output lengths vary,
and far lower time-to-first-token (TTFT) when requests arrive over time.

**What changed.** `crates/cli/src/metal_serve.rs` (`nanoinfer metal-serve`).
Synthetic workload: prompts uniform 16–256 tokens, outputs uniform 16–256
(fixed per request, no EOS), arrivals all at t = 0 or Poisson. Both
schedulers drive the same `Batch::step` (CPU in the loop every iteration):
- static: take up to 32 arrived requests, prefill them, decode until the
  longest is done (finished rows keep computing), free, repeat. KV reserved
  per slot = prompt + the batch's max output.
- continuous: each iteration admits waiting requests while a slot is free
  and the pool has pages for the prompt; builds one forward with a decode row
  per running sequence plus prompt chunks under a 256-token prefill budget
  (chunked prefill, Sarathi-style); retires finished sequences at once. KV
  pages are allocated as sequences grow.
The clock is the measured wall time of each iteration (idle gaps waiting for
the next arrival are skipped, not slept).

**Numbers** (Q4 weights, f16 paged KV, 128 requests, max batch 32, load
8–11; `bench/serving.json`):

| arrivals | policy | tok/s | makespan | TTFT p50 / p99 | TPOT p50 / p99 | KV pages peak |
|---|---|---|---|---|---|---|
| all at t=0 | static | 1197 | 14.7 s | 7.86 / 11.5 s | 11.4 / 13.9 ms | 810 (478 MB) |
| all at t=0 | continuous | **1903 (1.59×)** | 9.2 s | 3.65 / 7.59 s | 15.3 / 18.2 ms | 530 (313 MB) |
| Poisson 8/s | static | 850 | 19.2 s | 1.97 / 4.47 s | 11.8 / 13.7 ms | 740 |
| Poisson 8/s | continuous | 1042 (1.23×) | 15.7 s | **24 / 56 ms** | 4.8 / 10.5 ms | 203 |
| Poisson 16/s | static | 948 | 17.2 s | 4.37 / 8.01 s | 11.5 / 14.5 ms | 773 |
| Poisson 16/s | continuous | **1725 (1.82×)** | 9.5 s | 62 / 545 ms | 15.1 / 19.5 ms | 500 |

Prefill budget at 16 req/s (continuous): 32 tokens → TTFT p50 283 ms, TPOT
p99 15.3 ms; 256 → 62 ms / 19.5 ms; 512 → 56 ms / 18.9 ms.

**What surprised me.**
- **At 8 req/s continuous batching simply keeps up** (makespan ≈ the 16 s
  arrival window, TTFT p50 24 ms), while static batching queues: TTFT 2 s,
  i.e. **80× worse**, at only 1.23× lower throughput. The throughput number
  undersells it; TTFT is where static batching hurts.
- **Static batches are mostly padding.** With outputs uniform in 16–256 the
  batch runs ~max_out steps while the mean request needs half of that: in
  the offline run static computes 1052 iterations for 17.5k tokens,
  continuous 683.
- **KV memory**: paging + retiring cuts peak KV from 810 to 530 pages offline
  (−35%), and to 203 at 8 req/s; static reserves for the longest request in
  the batch whether or not it is ever used (the vLLM "fragmentation" waste).
- Continuous batching's TPOT is *higher* (15 vs 11 ms) when saturated: its
  iterations carry prompt chunks alongside the decodes, and the batch is
  always full, so each running request shares the GPU with more work. The
  prefill budget is the knob between TTFT and TPOT (Sarathi's point): 32
  tokens per iteration halves the p99 TPOT jitter but quintuples TTFT.


### metal-serve, re-run on a quiet machine + paged-KV snapshots for the viz

`metal-serve` now snapshots the continuous scheduler's block tables every 10
iterations (`kv_snapshots` in `bench/serving.json`, ≤ 20 per run: for each
live request its id, length and page ids in position order) and summarises
them (`kv_stats`). `Batch::block_table(s)` exposes a sequence's pages.

| arrivals | static tok/s | continuous tok/s | × | TTFT p50 static → cont. | KV peak pages static → cont. |
|---|---|---|---|---|---|
| all at t=0 | 1389 | **2198** | 1.58 | 6.88 s → 3.16 s | 810 → 530 |
| Poisson 8/s | 885 | 1051 | 1.19 | 1.39 s → **23 ms** | 729 → 195 |
| Poisson 16/s | 1080 | **1812** | 1.68 | 3.47 s → **40 ms** | 799 → 493 |
| 16/s, budget 32 | 1081 | 1780 | 1.65 | → 150 ms (TPOT p99 13.4 ms) | → 472 |
| 16/s, budget 512 | 1075 | 1828 | 1.70 | → 40 ms (TPOT p99 17.4 ms) | → 491 |

(16–20% faster than the load-8–11 run above; same ratios.)

**Paged KV in numbers** (continuous): live sequences average 26 (offline),
17 (16/s), 3.4 (8/s); each holds ~14 pages (≈ 224 positions); internal
fragmentation, the unused tail of each sequence's last page, is **3.2–3.3%**
of allocated KV. Static batching instead reserves prompt + the batch's
longest output per slot: its 810-page peak against continuous's 530 is the
waste paging removes. After a few hundred iterations block tables are
scattered across the pool (request 0 at iteration 250: pages 0–3, 15, 30, 4,
6, 8, 10, 32, 35, 50, 81, …): pages freed by finished requests are reused
immediately, which is exactly what a contiguous per-sequence cache can't do.

### Salvage note (18 Oct, evening)

The Metal round-2 and WebGPU agents were stopped mid-work by a kernel
watchdog panic (too many concurrent benches/builds from parallel agents).
Their uncommitted code built and passed every test as left (`cargo test`
per crate, `--test-threads=2`). Everything below was re-measured afterwards
on a **quiet machine** (load 2–5, one heavy process at a time), so the
numbers are higher than the round-2 entries above, which ran at load 15–30.
Same session re-check of the round-1 probe: GPU read bandwidth is
**~105 GB/s median** (32–512 MB, float4 grid-stride), not the 77–85 GB/s
measured under load; a dispatch inside one command buffer costs 0.76 µs,
commit + wait ~130 µs. So the practical decode ceiling is ~105 GB/s, ~88% of
the 120 GB/s spec, and the earlier "most of the gap is the GPU's 80 GB/s"
was contention, not the hardware.

Quiet re-run of the single-stream best (prompt 64 + 64, 7 runs): metal-lanes
Q4 **817 tok/s** decode (was 592 ± 47 at load 30), 6,800 tok/s prefill.

### Metal stage 8b: metal-kv8 (Q8 KV cache)

**Hypothesis.** At context ~1000 the f16 KV cache is 12 layers × 1000 × 768
× 2 (K, V) × 2 B ≈ 37 MB per decode token, ~40% on top of the 89 MB of Q4
weights. Store each (position, head) slice of 64 values as int8 + one f32
scale (68 B instead of 128 B): ~47% fewer KV bytes, so expect up to ~15% at
long context and nothing at short context.

**What changed.** `shaders/kv8.metal`: `kv_quant8` (one threadgroup per
(token, head, K|V): amax via simd_max, scale = amax/127, round), and copies of
the f16-KV flash attention kernels (`attn_prefill_flash_q8`,
`attn_decode_split_q8`) that dequantise each key/value on load with its
scale. `Recipe.kv_q8`; stage `metal-kv8` = metal-lanes with the Q8 cache. The
qkv matvec loses its fused f16 KV-write epilogue (it writes f32 K/V into the
qkv buffer, then `kv_quant8` runs): +12 dispatches per token (113).

**Quality** (`tests/reference.rs`, f32 weights so only the cache differs,
context 300): KL vs the f32 cache 9.0e-5 after a 150-token prefill chunk,
8.8e-8 on the decode step; same top-1.

**Numbers** (Q4 weights, 64 decode tokens, 7 runs, interleaved, load 2–4):

| prompt | metal-lanes (f16 KV) | metal-kv8 | Δ |
|---|---|---|---|
| 64 | 817 ± 10 | 804 ± 9 | −1.5% |
| 512 | 697 ± 11 | 729 ± 7 | +4.6% |
| 960 | 618 ± 4 | 651 ± 2 | +5.4% |

**What surprised me.** Half the KV bytes buys a third of what bandwidth
predicts (+5% at 960 vs ~15%). Decode at P=960 is 1.62 ms vs 1.22 ms at
P=64: the 0.4 ms that context adds is mostly *not* bytes. At 37 MB / 105 GB/s
the f16 cache costs ~0.35 ms of pure transfer, but the split-K attention
kernels also pay for the dequant, the scale loads and the online-softmax
merge; the int8 cache removes ~0.17 ms of transfer and adds a dispatch per
layer. At short context the extra dispatch makes it a net loss. Kept as a
stage (it is the right move for long contexts and for batching, where KV
dominates memory), not as the default.

### GEMM sweep: 64 × 64 tiles for big batches (`examples/gemm_sweep.rs`)

**Hypothesis.** The round-1 prefill GEMM (`matmul_mma_*`, 32 × 64 output
tile, weights staged as f32) is what batched decode uses from 32 rows. At
128+ rows a 64 × 64 tile with the weight tile kept as half
(`matmul_mma2_*`) halves the activation traffic per FLOP and the threadgroup
memory per weight.

**Numbers** (GFLOP/s, old / new, Q4; Q8 and f16 alike; max |diff| ≤ 2e-4,
f16 bit-identical):

| shape | n=32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| qkv 2304×768 | 929 / 547 | 966 / 1049 | 1048 / 2169 | 2204 / 2025 | 1761 / 2530 |
| fc 3072×768 | 1718 / 1062 | 1993 / 1962 | 2110 / 2291 | 2218 / 2423 | 2254 / 2535 |
| fc_proj 768×3072 | 919 / 553 | 1493 / 1053 | 1704 / 1715 | 2004 / 1883 | 2049 / 2214 |
| lm_head 50257×768 | 2089 / 1312 | 2175 / 2501 | 2231 / 2575 | 2249 / 2616 | 2273 / 2596 |

Below 64 rows half of the bigger tile is padding (−40%); from 128 rows it is
+5–15% on most shapes. `Batch::gemm` switches to `matmul_mma2` at
`GEMM2_MIN_ROWS = 128` (not F32). The single-stream prefill path still uses
the old kernel.

### Batching curve, re-run (metal-batch, quiet machine, mma2 from 128 rows)

Prompt 64, 64 GPU-resident decode steps, median of 3, load 2–4
(`bench/batching.json`, which the web page reads):

| B | Q4 agg | Q4 step ms | Q4 per stream | Q8 agg | f16 agg | f16 step ms | f32 agg | f32 step ms |
|---|---|---|---|---|---|---|---|---|
| 1 | 788 | 1.27 | 788 | 629 | 372 | 2.69 | 198 | 5.06 |
| 2 | 1412 | 1.42 | 706 | 1173 | 720 | 2.78 | 390 | 5.13 |
| 4 | 2188 | 1.83 | 547 | 2045 | 1268 | 3.15 | 733 | 5.46 |
| 8 | 2448 | 3.27 | 306 | 2349 | 2205 | 3.63 | 1380 | 5.80 |
| 16 | 2858 | 5.60 | 179 | 2811 | 2922 | 5.48 | 1815 | 8.81 |
| 32 | 4005 | 7.99 | 125 | 3983 | 3398 | 9.42 | 3157 | 10.1 |
| 64 | 5111 | 12.5 | 80 | 5161 | 4637 | 13.8 | 4610 | 13.9 |
| 128 | 6042 | 21.2 | 47 | 6048 | 5927 | 21.6 | 4995 | 25.6 |
| 256 | **6453** | 39.7 | 25 | 6402 | 6178 | 41.4 | 5607 | 45.7 |

vs the loaded run above: B=1 +20% (788 vs 657), B=256 +29% (6453 vs 4986; Q8
B=128 +26%). The quiet machine accounts for most of the small-B gain; the
B ≥ 128 gain is mostly the 64 × 64 GEMM (1.6 TFLOP/s at B=256, ~37% of f32
peak). The ridge is unchanged: f32 step time flat to B=8 (+15%), f16 to B≈4–8,
Q4/Q8 to B≈2–4. At B ≥ 64 the dtypes converge again (4.6–5.2k tok/s at 64,
5.6–6.5k at 256).

### Metal stage 11: speculative decoding (metal-spec)

**Hypothesis.** A small draft model guesses k tokens; the target checks all
of them in one forward over k + 1 rows. Batch-1 decode is memory-bound, so
verifying k + 1 rows costs about one target step. With per-token acceptance
α and draft cost c (draft step / target step), greedy speculative decoding
emits E = (1 − α^{k+1}) / (1 − α) tokens per target forward and the speedup
is E / (k·c + 1) (Leviathan et al.). Output is exactly the target's greedy
output, by construction (the batched forward is batch-invariant, stage 9).

**What changed.** `crates/cli/src/metal_spec.rs` (`nanoinfer metal-spec`),
on two `Batch` engines (target f16, draft gpt2-124M Q4). Per round: one
draft forward over the not-yet-seen tokens, then k − 1 draft tokens on the
GPU-resident loop (one wait); one target forward over [g, d1..dk] with
logits for every row; accept the longest matching prefix plus the target's
own next token; roll both KV caches back with `truncate` (paged, so rollback
is a length change). Baseline: the target alone on its GPU-resident loop.
Output is checked equal to the target's greedy output on every run. Six
prompts (code, list, license text, prose, story, Q&A, wikitext), 128 tokens,
median of 3 (2 for gpt2-large), k = 1..8 (`bench/specdec.json`, which also
carries per-token accept/reject playback for k = 4).

**Numbers, gpt2-medium f16 target** (target alone 106–132 tok/s, draft
alone ~700–800 tok/s, c ≈ 0.16–0.17; speedup measured / formula E/(kc+1)):

| prompt | α | k=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | best |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 0.88 | 1.37/1.62 | 1.55/1.98 | 1.81/2.23 | 1.84/2.39 | 1.82/2.41 | 1.84/2.50 | 1.78/2.47 | 1.54/2.47 | 230 tok/s |
| list | 1.00 | 1.45/1.71 | 1.80/2.25 | 2.06/2.66 | 2.26/2.99 | 2.49/3.26 | 2.64/3.49 | **2.93**/3.68 | 2.61/3.84 | 371 |
| license | 0.99 | 1.46/1.71 | 1.85/2.24 | 2.19/2.66 | 2.41/2.99 | 2.45/3.25 | 2.61/3.47 | 2.88/3.65 | 2.57/3.80 | 367 |
| prose | 0.80 | 1.26/1.53 | 1.41/1.86 | 1.44/1.94 | 1.59/2.01 | 1.42/2.07 | 1.44/1.99 | 1.35/1.95 | 1.19/2.01 | 210 |
| story | 0.84 | 1.41/1.57 | 1.59/2.02 | 1.74/2.18 | 1.75/2.17 | 1.80/2.37 | 1.68/2.32 | 1.73/2.40 | 1.58/2.40 | 210 |
| qa | 0.99 | 1.52/1.73 | 1.89/2.28 | 2.12/2.67 | 2.52/3.00 | 2.63/3.34 | 2.91/3.49 | 3.01/3.67 | 2.81/3.96 | 325 |
| wikitext | 0.80 | 1.41/1.54 | 1.62/1.86 | 1.70/1.99 | 1.70/2.05 | 1.68/2.06 | 1.69/2.12 | 1.68/2.05 | 1.52/1.99 | 180 |

**gpt2-large f16 target** (target alone 54–63 tok/s, c ≈ 0.075–0.08):

| prompt | α | k=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | best |
|---|---|---|---|---|---|---|---|---|---|---|
| code | 0.65 | 1.49/1.46 | 1.48/1.71 | 1.64/1.88 | 1.64/1.93 | 1.64/1.92 | 1.58/1.96 | 1.56/1.87 | 1.30/1.84 | 88 tok/s |
| list | 1.00 | 1.70/1.86 | 2.19/2.60 | 2.78/3.25 | 3.14/3.83 | 3.44/4.34 | 3.79/4.79 | 4.12/5.21 | 3.70/5.58 | 249 |
| license | 0.99 | 1.60/1.84 | 2.07/2.56 | 2.71/3.18 | 3.09/3.71 | 3.42/4.19 | 3.73/4.60 | **4.25**/4.97 | 3.66/5.30 | 268 |
| prose | 0.93 | 1.66/1.80 | 2.03/2.40 | 2.43/2.92 | 2.85/3.33 | 3.01/3.60 | 3.26/3.80 | 3.24/4.01 | 2.68/4.16 | 200 |
| story | 0.95 | 1.67/1.81 | 1.94/2.42 | 2.60/3.04 | 2.67/3.38 | 2.99/3.75 | 3.27/4.15 | 3.62/4.41 | 3.03/4.51 | 226 |
| qa | 0.98 | 1.64/1.86 | 2.14/2.60 | 2.59/3.15 | 2.97/3.68 | 3.46/4.33 | 3.80/4.53 | 3.78/4.89 | 3.69/5.57 | 238 |
| wikitext | 0.84 | 1.56/1.68 | 1.91/2.26 | 2.20/2.64 | 2.21/2.76 | 2.38/2.92 | 2.47/3.12 | 2.44/3.12 | 2.05/3.12 | 153 |

**What surprised me.**
- **Tokens per forward match the formula almost exactly** (e.g. license k=7:
  7.94 measured vs 7.75 predicted; prose k=4 3.26 vs 3.34), but the *speedup*
  lands at 75–90% of E/(kc+1). Two costs the formula leaves out: verifying
  k + 1 rows is not free (gpt2-medium: 8.6–10 ms at k=1 vs 7.9 ms for one
  target step, 11 ms at k=7), and each round has three CPU↔GPU waits (draft
  first step, draft resident loop, verify), ~0.13 ms each.
- **A cliff at k = 8.** Verify jumps from ~11 ms (8 rows) to ~14 ms (9 rows)
  for gpt2-medium, 20 → 25.5 ms for gpt2-large. The batched matvec rounds the
  row count up to a power of two (`nb = n.next_power_of_two()`), so 9 rows run
  the NB = 16 kernel with 7 rows of padding and twice the accumulators.
  Best k is 7 wherever α ≈ 1, 4–6 elsewhere: right at the edge of the
  8-row kernel.
- **The bigger the target, the bigger the win**: gpt2-large reaches **4.25×**
  (63 → 268 tok/s) because c halves (0.08 vs 0.16), exactly the formula's
  lever. A 774M model decoding at 268 tok/s on a base M4 is faster than
  gpt2-medium alone (128).
- **α is a property of the text, not of the models.** Boilerplate (license,
  month lists, Q&A format) gets α ≈ 0.99–1.0 and 2.9–4.3×; natural prose and
  wikitext 0.80–0.84 and 1.6–2.5×. Code is the odd one: 0.88 with
  gpt2-medium but **0.65 with gpt2-large** (large writes different code than
  small does; nothing to do with how predictable code is). Caveat: greedy
  GPT-2 falls into repetition loops after ~50 tokens, which inflates α on
  every prompt (the list and Q&A prompts loop almost immediately).
- k = 1 already gives 1.3–1.7×: one free draft token per target step.

### Speculative decoding, round 2: what verification really costs

**Verify cost vs rows** (gpt2-medium f16, one forward over B rows through the
batched matvec, `metal-batch --weights weights/gpt2-medium --dtype f16`):

| rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 13 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| step ms | 7.85 | 8.17 | 8.83 | 8.90 | 9.57 | 9.56 | 9.80 | 10.00 | 11.87 | 12.03 | 12.82 | 14.11 | 15.77 |
| × one row | 1.00 | 1.04 | 1.12 | 1.13 | 1.22 | 1.22 | 1.25 | **1.27** | 1.51 | 1.53 | 1.63 | 1.80 | 2.01 |

So "checking k + 1 tokens costs one step" holds to 8 rows within 27%, then
the batched matvec turns compute-bound (~570–720 GFLOP/s, ~15% of peak: the
kernel does plain FMAs, no simdgroup_matrix). The GEMM path is no rescue
below 32 rows: with `--gemm-min-rows 4`, 16 rows take 21 ms (39 GB/s).

**The k = 8 cliff, fixed halfway.** The batched matvec rounded rows up to a
power of two, so 9 rows ran the 16-row kernel. Added a 12-row chunk
(`mvb_*_b12`, `mvr_*_b12_r{1,2,4}`; same per-token summation order, so
batch invariance holds). Verify at k = 8: 13.8–14.8 → **12.2–12.4 ms**;
k = 8 speedup 2.57–2.61 → 2.69–2.72 on the α ≈ 1 prompts. Still not better
than k = 7, because the draft keeps getting longer (1.4 ms per draft token).

**The model, corrected.** With the measured verify cost v(k+1) in place of
1, speedup ≈ E / (k·c + v(k+1)) predicts every (prompt, k) to within 5–15%
(license k=7: 3.23 predicted, 2.94 measured; naive formula 3.65). The rest is
the three CPU↔GPU waits per round (~0.4 ms of a ~20 ms round) and the draft's
first, non-resident step.

Re-run (`bench/specdec.json`, gpt2-medium pair, load 3): best 394–397 tok/s
(list, license, k=7: 2.93–2.94× over 132–135 tok/s).

### Single-stream Metal: what's left (analysis, one dead end)

Profile of metal-lanes (Q4 blocks, Q8 head, f16 KV), one decode step at
context 65, quiet machine (1.22 ms): lm_head 387 µs (32%, **106 GB/s**:
at the bandwidth roof), fc 214 µs (74 GB/s), fc_proj 186 µs (86 GB/s), qkv
182 µs (66 GB/s), attention 150 µs (12%, 24 dispatches: split + merge),
attn_proj 51 µs, ln_1 + ln_2 51 µs.

- **A cheaper head with exact rescoring (dead end).** The CPU's stage 12
  rescored a Q8 approximate head against f16. On the GPU the head is already
  Q8, so the next step down is Q4 + rescore with Q8 (`examples/head_err.rs`,
  same body, only the head differs, 1024 WikiText positions): Q4 vs Q8 head
  logit error **p50 10.6, p99 22.6** (max over the vocabulary per position),
  argmax agreement **26%**. A margin that guarantees the Q8 argmax is a
  candidate would have to be ~45 logits: thousands of candidates. The
  ln_f-gain outlier channels make 4-bit head weights unusable, as round 1
  found for plain Q4.
- **Fused ln_f + LM head + argmax (not done).** The head runs at 106 GB/s,
  so fusion can only save the 200 KB logits write, the argmax_partial pass
  and two dispatches: ~5–10 µs of 1,220 (<1%).
- **Q4 multi-row**: tried in stage 8a (R = 2 rows per lane): no better than
  R = 1.
- Left on the table: the narrow matvecs (qkv at 66 GB/s), short-context
  attention (12.5 µs per layer for 65 positions: two dependent dispatches),
  and the LayerNorms (4%). The WebGPU fold (`webgpu-lnfold`) would remove
  the last one here too, but the lanes kernel has no spare issue slots for
  the extra sums; not tried.

### Failures and dead ends (round 2)

- None of the salvaged code had to be fixed; the crash left it building.
- The first spec-decoding numbers (in `bench/specdec.json` before this
  re-run, load 5.7) looked noisy (speedup non-monotonic in k by ±0.4×); the
  quiet re-run is smooth up to the k = 8 cliff, so the noise was the
  machine.

## WebGPU (crates/webgpu: wgpu 30, WGSL; native on wgpu's Metal backend)

**Why.** The browser page runs the CPU engine as wasm; WebGPU is how a tab
gets the GPU. `crates/webgpu` is the same Rust host code and the same WGSL
kernels for both targets: natively through wgpu's Metal backend (this bench)
and in a tab through wgpu's WebGPU backend compiled to wasm
(`crates/wasm/src/webgpu.rs`, `--features webgpu` in `crates/wasm/build.sh`).
Everything GPU-facing is `async` (a tab cannot block on the GPU). Stages are
recipes, like Metal's. `tests/reference.rs`: every stage matches HF.

**Probe** (`nanoinfer webgpu-probe`, quiet machine, vs the Metal probe in
the same session):

| | wgpu → Metal | Metal direct |
|---|---|---|
| GPU time per dispatch inside one pass (1000 dispatches, timestamps) | **1.8 µs** | 0.76 µs |
| wall per dispatch, 1000 in one pass (CPU encode included) | 2.3 µs | — |
| CPU encode per dispatch (set pipeline + bind groups + dispatch) | 0.15–0.22 µs | 0.08–0.18 µs |
| one pass per dispatch (1000 passes, one submit) | 28 µs per pass | — |
| submit 1 dispatch + wait (onSubmittedWorkDone) | 165 µs | ~130 µs (commit + wait) |
| submit + copy + mapAsync 4 bytes | 196 µs | — |
| read bandwidth (256 MB) | 102 GB/s | 105 GB/s |

So the same rules as Metal, with every fixed cost ~2× larger: a dispatch
inside a pass is 1.8 µs, a round trip ~165 µs, and a *compute pass* is
expensive (28 µs each), so the whole forward goes in one pass. Bandwidth
reaches the same ~100 GB/s: wgpu's overhead is per dispatch, not per byte.

**Stage table** (GPT-2 124M, prompt 64 + 64 greedy, 5 runs, load ~2;
Metal rung doing the same thing measured in the same session;
`bench/webgpu.json`):

| stage | what changed | decode tok/s | prefill tok/s | disp/token | Metal equivalent |
|---|---|---|---|---|---|
| webgpu-naive | WGSL port of the CPU kernels, submit + wait per op | 16.2 | 400 | 137 | metal-naive 19.4 |
| webgpu-onesubmit | one encoder + one submit per token | 33.9 | 490 | 137 | metal-onecb 44.6 |
| webgpu-matvec | 64 threads per row, vec4 loads, workgroup reduction | 41.3 | 502 | 137 | |
| webgpu-par | 256-thread LayerNorm and attention | 149 | 513 | 137 | metal-par 171 |
| webgpu-subgroup | matvec reduced with subgroupAdd (native only) | 168 | 555 | 137 | |
| webgpu-fused | bias+GELU, residual, KV write in matmul epilogues | 145 | 542 | 89 | |
| webgpu-f16 | f16 weights packed in u32 | 243 | 595 | 87 | metal-f16 307 |
| webgpu-q8 | Q8_0 weights, f16 head | 345 | 402 | 87 | metal-q8 470 |
| webgpu-argmax | argmax on the GPU | 349 | 403 | 89 | |
| webgpu-resident | 16 tokens per submit, one readback | 383 | 397 | 89 | |
| webgpu-tiled | 64 × 64 tiled prefill matmul | 384 | 2,584 | 89 | |
| webgpu-ln64 | LayerNorm on 64 threads, row in registers | 386 | 2,598 | 89 | |
| webgpu-q8head | Q8_0 LM head too | **451** | 2,676 | 89 | metal-pipelined-q8 600 |

(The original run of these stages, at load 92 during the crash, recorded
11–299 tok/s; the quiet numbers above replace them.)

**What surprised me.**
- **WebGPU is 75–85% of hand-written Metal at every rung** (f32 par 149 vs
  171, f16 243 vs 307, Q8 resident 451 vs 600), and the gap is *not* the
  dispatch overhead: 89 dispatches × (1.8 − 0.76) µs ≈ 0.09 ms of a 2.2 ms
  token (4%). It is the matvec: webgpu-q8head streams 59 GB/s while
  metal-pipelined-q8 streams 79 GB/s. Metal has `simd_sum` and
  `simdgroup_matrix`; the portable kernels reduce through workgroup memory.
- **Fusion did nothing** (137 → 89 dispatches, 149 → 145 tok/s): at 1.8 µs
  per dispatch, 48 fewer dispatches are 86 µs of a 6.7 ms f32 token.
- **Subgroups are worth +13% (149 → 168) on f32** but only natively; Chrome
  exposes `subgroups` behind a flag, so the browser stages build without it.
- **The GPU-resident loop is worth +10%** (349 → 383): the ~165 µs round trip
  is 6% of a 2.9 ms token, plus the CPU's encode time no longer sits between
  tokens.
- Prefill needs the tiled matmul (400 → 2,584 tok/s, 6.5×) but is still 2.5×
  behind Metal's simdgroup_matrix GEMM (6,800), which WGSL has no
  equivalent for.
- GPU pass timestamps on wgpu-Metal are unreliable for small passes (they
  report more GPU time than wall time); the table uses wall time only.

### WebGPU round 2: closing the gap to Metal (webgpu-lnfold, -attn4, -flashdec, -kv16)

**Machine note.** From ~19:30 macOS's `mediaanalysisd` (photo analysis) and
VideoToolbox encode/decode sessions shared the GPU (IOKit: GPU 91% busy with
no bench running), so absolute numbers drift by up to 15% between minutes.
Every comparison below is interleaved A/B in one loop; `bench/webgpu.json`
reports the best of the 2–4 quiet-session runs per stage.

**Where the time goes** (`bench --stage webgpu-q8head --profile`, each op
×20 back to back, prompt 64): fc 18.5%, lm_head 17%, fc_proj 17%, qkv 14%,
attention 9%, attn_proj 8.6%, **ln_1 + ln_2 13% (13 µs per LayerNorm)**.
Per-matrix: lm_head 95 GB/s, fc_proj 89, fc 72, qkv 66, attn_proj 39.

**Threads per row** (`NANOINFER_WGPU_TPR`, webgpu-q8head): 8 → 482,
16 → 513, **32 → 521 (default)**, 64 → 487 tok/s. Already at the optimum.

**webgpu-lnfold: LayerNorm folded into the weights.** Metal's fused-LN
(stage 6b) failed because every threadgroup re-normalised x before its
first weight load. Instead, fold the algebra into the weights at upload:
W·(g ⊙ x̂ + b_ln) + b = (W diag g)·x̂ + (b + W·b_ln), and
(W')·x̂ = rstd · (W'·x − mean · rowsum(W')). So the decode matvec reads the
raw residual x; the threads of each workgroup's first row also sum x and x²
from the same vec4s and reduce them in the same barrier steps; the epilogue
applies rstd and the mean correction (row sums of the *stored, quantised* W'
are kept after the biases in the bias buffer). No LayerNorm kernels in
decode: 89 → 65 dispatches. Prefill keeps a LayerNorm with unit gain and
zero bias in front of the folded matrices. Same HF KL as q8head (5–7e-3),
same greedy agreement.
Result (interleaved, 3 pairs): 509–519 → **536–542 tok/s (+4%)**. The
profile's 13 µs per LayerNorm overstated it: the real saving is ~3 µs per
removed dispatch.

**webgpu-attn4: attention with vector loads.** At prompt 960 attention is
29% of the profile (65 µs per layer). `attn_par` computed each score with 64
scalar loads of q and k and P·V with one dim per thread (L/4 dependent scalar
loads), plus 20 barriers. `attn_par4`: q in registers as 16 vec4s, keys as
vec4s, P·V with 4 dims × 1/16 of the keys per thread, and the softmax
denominator summed inside the P·V pass (12 barriers). Decode only (prefill
was 5% slower on it).

| prompt | lnfold | attn4 | Δ |
|---|---|---|---|
| 64 | 534–541 | 545–551 | +2% |
| 512 | 407 | 463 | **+14%** |
| 960 | 327 | 400 | **+22%** |

**webgpu-flashdec (dead end, kept runnable).** Decode attention split over
64-key chunks (grid chunks × heads, 64 threads each) plus a merge kernel per
layer. Correct (max |diff| 6e-5 vs attn4 at context 300) and *slower
everywhere*: 565 → 535 (P=64), 457 → 437 (512), 393 → 340 (960). Profile at
960: split 95 µs + merge 13.5 µs per layer vs attn4's 65 µs. The reason is
the cache size, not parallelism: at 960 positions one layer's f32 K + V is
12 heads × 960 × 64 × 4 B × 2 = 5.9 MB, i.e. attn4's 65 µs is already
~90 GB/s. Twelve workgroups were enough to saturate DRAM; 192 smaller ones
only added a merge and partial writes. (Metal's flash-decoding paid because
Metal's single-workgroup attention was latency-bound at the time.)

**webgpu-kv16: f16 KV cache.** Halve those bytes: K/V packed two halves per
u32. The qkv matvec writes plain f32 rows; `kv_write16` packs K and V
(+12 dispatches); attn4 dequantises on load (`kv_f16.wgsl`), also used for
prefill. KL vs the f32 cache at context 300: 6e-11, same top-1.

| prompt | attn4 | kv16 | Δ | prefill attn4 → kv16 |
|---|---|---|---|---|
| 64 | 536 | 539 | +0.5% | 2929 → 2799 |
| 512 | 452 | 476 | +5% | 3672 → 3752 |
| 960 | 388 | 437 | **+13%** | 2831 → 3383 (+20%) |

**Threads per row, again, and wider Q8 loads** (`NANOINFER_WGPU_TPR_NARROW`
/ `_WIDE` for 768- vs 3072-wide rows, `NANOINFER_WGPU_Q8X16=1` for a Q8
matvec that loads 16 int8 per vec4<u32> with one scale, on webgpu-kv16):
TPR narrow/wide 32/32 518, 16/32 519, 16/64 530, 32/64 521, 64/64 471,
16/128 526: noise except 64 being worse. 16-wide Q8 loads: 513–517 →
523–537 (+2–4%, with 16 threads per narrow row).

**The robustness tax.** wgpu (like every browser) injects bounds checks on
every dynamic index into storage, workgroup and local arrays.
`NANOINFER_WGPU_UNCHECKED=1` (native-only diagnostic:
`create_shader_module_trusted` with `ShaderRuntimeChecks::unchecked()`)
removes them. webgpu-kv16, prompt 64: decode 513 → **545 (+6%)**, prefill
2,650 → **3,747 (+41%)**. The scalar tiled matmul does 8 workgroup loads
per 16 FMAs, each with a clamp. A browser can't turn this off, so the fix
has to be fewer, wider accesses.

**webgpu-vec: vector traffic.** (1) `matmul_tiled4`: same 64×64 tile and 4×4
register blocks, but staging is one vec4 of X and one `wget4` (4 weights)
per thread per 16-deep K step, the tiles are row-major vec4s over k (padded
to 5 vec4s per row against bank conflicts), and the inner loop does 16 vec4
dots per 8 vec4 loads. (First try stored [k][row] vec4s by component from
four threads: a race, since stores are whole-vector; KL 9.2.) (2) the 16-wide
Q8 matvec loads with 16 threads per 768-wide row (`Recipe.wide_q8`).

| prompt | kv16 prefill | vec prefill | kv16 decode | vec decode |
|---|---|---|---|---|
| 64 | 2,650 | **4,413 (+67%)** | 512 | 538 (+5%) |
| 512 | 3,617 | **4,944 (+37%)** | 461 | 478 (+4%) |
| 960 | 3,264 | 4,286 (+31%) | 420 | 433 (+3%) |

Unchecked, webgpu-vec gains only 6% more at prompt 512 (4,960 → 5,239), vs
33% for the scalar kernel: most of the tax is gone. A K step of 32 instead
of 16 (fewer barriers) changed nothing (4,431 / 4,898). Prefill is now ~49% of
Metal's simdgroup_matrix GEMM at prompt 512 (4,944 vs 10,100), was 36%.

**Where WebGPU ends** (best quiet runs, `bench/webgpu.json`): Q8 decode
**556 tok/s at prompt 64** (attn4; kv16 539, vec 538), 478 at 512 (vec), 435
at 960 (kv16); prefill 4,413 tok/s at 64 and 4,944 at 512 (vec);
metal-pipelined-q8 (the matching Metal rung: Q8 weights + Q8 head, GPU loop)
600 → **~90%** of hand-written Metal at short context (was 75–85% after
round 1). The remaining gap is the narrow matvecs (qkv 66 GB/s, attn_proj
39 GB/s vs Metal's lanes kernel at 58–80) and prefill (4,944 vs ~10,100 tok/s
at 512: no simdgroup_matrix in WGSL).

**Browser-relevant costs.** Natively, a dispatch inside a pass costs 1.8 µs
of GPU time and a pass 28 µs; `onSubmittedWorkDone` round trip 165 µs,
`mapAsync` of 4 bytes 196 µs. In a tab every one of these crosses the GPU
process IPC as well (Chrome's Dawn wire), so the GPU-resident loop (16
tokens per submit, one `mapAsync`) and one pass per forward matter more there,
and fusion that removes dispatches matters less than it seems. The wasm
build (`crates/wasm`, `--features webgpu`) exposes the same probe
(`webgpu_probe`); the in-browser numbers belong to the web page's run.

---

## Stage 3: neon (explicit SIMD), a null result

**Hypothesis.** Stage 2's vectorised loop uses `fmul` + `fadd` (Rust won't fuse
them) and loads 2 vectors per multiply-add. Hand-written NEON can use `vfmaq_f32`
(fused, half the instructions) and compute **4 output rows per pass**, sharing each
load of `x` across 4 weight rows (1.25 loads per FMA). Expect prefill to gain
noticeably; decode only up to the ~13% of single-core bandwidth left on the table.

**Attempt 1: 4 rows per pass (failed).** Inner loop exactly as intended
(`objdump`: 16 × `fmla.4s`, 10 × `ldp q` per 16 columns, no spills). Decode went
*down*: **74.5 vs 100.9 tok/s** (two interleaved runs each), all matmuls dropping
from ~54 to ~25–30 GB/s, fc_proj (12 KB between rows) worst at 24.7 GB/s.

Microbenchmark (`/tmp`, not committed): sum a 512 MB buffer as rows of length d,
k rows at a time, one P-core:

| d | 1 stream | 2 | 4 | 8 |
|---|---|---|---|---|
| 768 | 49.2 GB/s | 46.0 | 38.4 | 24.5 |
| 3072 | 48.3 | 45.3 | 31.7 | 34.5 |

So it's the memory system, not the kernel: a P-core streams ~49 GB/s from one
sequential stream and ~32–38 GB/s from 4 interleaved ones. Software prefetch
(`prfm pldl1strm` 64–1024 floats ahead in each stream) made it *worse*
(29–37 GB/s). At batch 1 the extra loads of `x` (always in L1) were never the
bottleneck. DRAM is, and 4 streams are worse for the prefetcher than one.
(Repacking 4 rows into one interleaved stream would fix it. We'll revisit that
only if a later stage needs the x reuse at batch 1.)

**Attempt 2 (kept): one row at a time, fused.** `kernels/neon.rs`: `dot1` = 4 ×
`float32x4` accumulators, `vfmaq_f32`, `vaddvq_f32` reduction. Also exposes
`matvec(out, x, w, bias, .., o0)` over a row range, for the threading stage.
Non-aarch64 targets (wasm32) fall back to the stage 2 kernel.

**Numbers** (prompt 64, 64 decode, 5 runs, interleaved with unrolled):

| stage | prefill tok/s | decode tok/s | decode GB/s |
|---|---|---|---|
| unrolled | 184.2 ± 1.4 / 185.1 ± 16.7 | 100.2 ± 0.5 / 100.0 ± 7.2 | 49.6 |
| neon | 183.9 ± 1.0 / 185.4 ± 1.2 | 101.0 ± 2.9 / 101.3 ± 0.6 | 50.1 |
| neon (recorded) | 182.6 ± 1.8 | 100.3 ± 1.4 | 49.6 |

Identical within noise. Correctness: within 1e-3 of HF (FMA rounding only).

**Why nothing moved, and why that's the interesting part.**
- **Decode** was already at 87% of single-core DRAM bandwidth. Halving the
  arithmetic instructions changes nothing when the core waits on memory.
- **Prefill didn't move either**, which says prefill isn't FMA-bound. It's
  loop-order-bound: "for each prompt row → stream the whole weight matrix"
  re-reads every matrix 64 times. They're 2.4–9.4 MB, so they sit in the 16 MB L2,
  and prefill runs at L2 bandwidth (31.6 GFLOP/s ≈ 63 GB/s of weights). The fix
  is reuse (stage 4), not faster arithmetic.
- The compiler had already done the SIMD part. Writing intrinsics by hand only
  pays when you know something the compiler can't (here: nothing yet).

---

## Stage 4: tiled (register-blocked prefill)

**Hypothesis.** Stage 3 showed prefill re-streams each weight matrix once per
prompt row (from L2). Computing a 4×4 output tile at a time (4 prompt rows × 4
weight rows, 16 NEON accumulators) uses every loaded weight 4 times from a
register (0.5 loads per FMA, was 2). With weight-block-outer loop order, a 4-row
weight panel (≤ 48 KB) stays in L1 while all prompt rows pass over it. Expect
prefill to become FMA-bound, at a few × stage 3. Decode unchanged.

**What changed.** `kernels/tiled.rs`: `tile_4x4` micro-kernel (`vfmaq_f32`, 4 `x`
loads + 4 `w` loads + 16 FMAs per 4 columns), `matmul_cols(.., o_begin, o_end)`
computing a column range for all rows (written so threads can take disjoint
column ranges next), ragged edges (n % 4, d_out % 4 = the LM head's 50257, k tail)
via the stage 2 dot. n < 4 (decode) calls stage 3's `matvec`. Unit test vs the
reference kernel on a 7 × 13 × 10 problem with all three ragged edges (exact match
on integer data).

**Numbers** (prompt 64, 64 decode, 7 runs; neon in the same session: prefill
183 ± 24, decode 100.2 ± 3.0). The machine was busy (other agents' builds and GPU
benches plus Backdrop), so stddevs are large and medians are what to read:

| | prefill tok/s | prefill GFLOP/s | decode tok/s |
|---|---|---|---|
| tiled, P=64 | **469 ± 119** (quiet runs: 441, 451) | 80 | 101.1 ± 16 |
| tiled, P=512 | 355 ± 74 | 60 | |
| neon, P=512 | 139 ± 17 | 24 | |

Prefill profile (`generate --profile`, 109-token prompt):

| op | share | GFLOP/s |
|---|---|---|
| fc | 26% | 98 |
| fc_proj | 25% | 101 |
| qkv | 19% | 102 |
| **gelu** | **12%** | 2.1 |
| attn | 7% | 13.6 |
| attn_proj | 7% | 96 |
| lm_head (1 row) | 2.5% | 12.7 |

**What surprised me.**
- **Matmuls hit ~100 GFLOP/s, 71% of one P-core's NEON peak** (4 FMA pipes × 4
  lanes × 2 FLOP × 4.4 GHz ≈ 141 GFLOP/s). Prefill is compute-bound now, as the
  textbook says. Same arithmetic as stage 3, 3.2× the matmul throughput, purely
  from reuse.
- **GELU became 12% of prefill.** Scalar `tanhf` on 3072 × n values per layer is
  now slower than a 768×768 matmul. Attention (scalar, O(n²)) is 7% at n = 109 and
  the reason P=512 prefill drops to 355 tok/s. The "<1% elementwise" picture from
  stage 1 is gone once the matmuls are fast.
- Decode is untouched (same kernel), still ~100 tok/s at the 1-thread memory wall.

---

## Stage 5: threads (rayon)

**Hypothesis.** One core reads ~57 GB/s; all cores ~98 GB/s (membw). Decode is
memory-bound, so splitting each matmul's weight rows across threads should
scale decode toward ~1.7× (the bandwidth ratio), and prefill (compute-bound)
toward the core count, weighted by P (fast) vs E (slow) cores.

**What changed.** `kernels/threaded.rs`: each matmul's output columns are cut
into `threads × 4` chunks (multiples of 16 columns) and run with rayon's
`into_par_iter`. A decode chunk is one contiguous slab of weight rows, so one
sequential stream per thread (stage 3's lesson). Prefill chunks run stage 4's
tile kernel. Threads come from a new `threads` cargo feature on core
(`rayon` optional dep); the CLI enables it, and the wasm build doesn't, so there
the stage runs single-threaded. CLI: global `--threads N` and `--qos
interactive|utility|background`. macOS has no core pinning, but a thread's QoS
class steers it: `background` threads stay on E-cores.

**Numbers** (prompt 64, 64 decode, 7 runs; P-cores preferred; Backdrop.app using
~2 cores throughout, and other agents compiling and benchmarking at times):

| threads | prefill tok/s | prefill GFLOP/s | decode tok/s | decode GB/s |
|---|---|---|---|---|
| 1 | 476 ± 23 | 81 | 91.6 ± 4.9 | 45 |
| 2 | 651 ± 23 | 111 | 95.5 ± 2.5 | 47 |
| 3 | 771 ± 28 | 132 | 96.5 ± 3.9 | 48 |
| 4 | 826 ± 25 | 141 | 101.2 ± 5.4 | 50 |
| 5 | 888 ± 47 | 152 | 115.2 ± 1.8 | 57 |
| 6 | 947 ± 23 | 162 | 115.8 ± 1.7 | 57 |
| 8 | **1021 ± 13** | 175 | **118.3 ± 1.7** | 58.5 |
| 10 | 1013 ± 22 | 173 | 116.4 ± 1.6 | 58 |

E-cores only (`--qos background`):

| threads | prefill | decode | decode GB/s |
|---|---|---|---|
| 1 | 97 | 27.2 | 13.4 |
| 2 | 158 | 43.1 | 21.3 |
| 4 | 234 | 52.9 | 26.2 |
| 6 | 243 | 50.6 | 25.0 |

**What surprised me.**
- **Decode barely scales: 92 → 118 tok/s (1.3×) for 8 threads**, at 58 GB/s
  against the ~90–98 GB/s membw reaches with the same thread counts. Per-op
  profile at 4 threads (`generate --profile`): the 154 MB LM head (one matmul)
  gets 75 GB/s, but the 2.4 MB `attn_proj` takes **62 µs on 4 threads vs 43 µs
  on 1**. Small matmuls are dominated by fork/join: rayon parks idle workers and
  waking them costs tens of µs on macOS, × 49 matmuls per token. That's DISTILLED
  law 6 (overhead dominates small models) showing up for real.
- **Prefill scales 2.1× (476 → 1021 tok/s)** and peaks at 8 threads. Beyond the
  4 P-cores, each E-core adds about a fifth of a P-core.
- **The E-core cluster tops out at ~25 GB/s** (decode ~51 tok/s on 4–6 E-cores),
  a quarter of the P-cluster. A static even split across P and E cores would run
  at the E-cores' pace, so we make 4 chunks per thread and let rayon's work
  stealing rebalance.
- One run showed "1 thread: 57 tok/s" against 100 for the identical stage 4 code
  path. Re-running gave 100.6. That was another process hogging a P-core.
  Lesson: on a shared machine, always re-run the previous stage in the same
  session before believing a delta.

---

## Stage 6: spin-pool (persistent spinning thread pool)

**Hypothesis.** Stage 5's decode is limited by fork/join: rayon parks idle
workers, and a token has 49 matmuls of 40–150 µs each. A pool whose workers
busy-wait on an atomic epoch sees new work in ~100 ns, so small matmuls should
finally scale. Expect most of the gap between 58 GB/s and membw's ~90 GB/s to close.

**What changed.** `core/src/pool.rs` (behind `threads`, not on wasm): N−1 worker
threads plus the calling thread. Dispatch publishes a job pointer and bumps an
epoch packed with the chunk count into one `AtomicU64` (`epoch<<32 | n<<16 |
next`). Every thread claims chunk indices by CAS on that word, so P-cores just
take more chunks than E-cores, and a stale worker can never claim a chunk of the
wrong job. The join spins on a completed-chunk counter. `kernels/spin.rs` = stage
5's chunking on this pool. Test: 2000 dispatches of 1–37 chunks, each chunk runs
exactly once.

**Attempt 1: pure spinning (failed badly).** Workers spun up to 2²⁰ iterations
(~ms) before sleeping, and the joiner spun unconditionally:

| threads | rayon decode | pure-spin decode | pure-spin prefill |
|---|---|---|---|
| 4 | 103 | 106 | 645 |
| 8 | 110 | 84 | 629 |
| 10 | 93 | **36** | 349 |

At 10 threads on 10 cores, with Backdrop.app holding ~2 cores and Spotlight ~1,
we're oversubscribed. The OS can't tell a spinning thread from a working one, so
spinners steal time slices from the thread that holds the last chunk, and
everyone waits for it. **A spin pool assumes you own the machine.**

A race found while writing it: the first version kept `n_chunks` in its own
atomic. A worker could read the *next* job's chunk count while still seeing the
old epoch and claim a phantom chunk. Packing epoch, count and cursor into one
word makes the claim a single CAS that validates all three.

**Attempt 2 (kept): spin, then yield, then sleep.** Spin 2¹² iterations, then
`yield_now()` up to 2¹⁴ times (hands the core to any runnable thread, including
a preempted one of ours), then a condvar sleep. The joiner spins 2¹², then yields.

Per-op decode profile at 4 threads (same session, heavy background load):

| op (per call) | tiled 1t | rayon 4t | spin 4t |
|---|---|---|---|
| attn_proj (2.4 MB) | 60 µs | 99 µs | **41 µs** |
| qkv (7 MB) | 181 µs | 202 µs | **108 µs** |
| lm_head (154 MB) | 3.9 ms | 2.6 ms | **2.1 ms** (73 GB/s) |

Interleaved sweep (7 runs each, recorded; Backdrop ~200% + Spotlight ~100% CPU
the whole time, so 1-thread decode was 86 here vs ~100 when quieter):

| threads | rayon prefill | rayon decode | spin prefill | spin decode |
|---|---|---|---|---|
| 1 | 475 | 86.1 | 478 | 87.0 |
| 2 | 628 | 92.7 | 647 | 99.6 |
| 4 | 818 | 99.1 | 740 | 101.7 |
| 6 | 868 | 101.9 | 687 ± 149 | 91.3 ± 14 |
| 8 | 684 ± 70 | 58.5 ± 8 | 833 | **115.0 ± 3.5** |
| 10 | **984** | 103.3 | 870 | 111.9 ± 8 |

Two earlier unrecorded rounds: spin 6t 107 / 124, 10t 90 / 123 decode vs
rayon 60 / 101 and 81 / 110. membw under the same load: 4t 69, 6t 74, 10t 86 GB/s
(median), down from 90 / 96 / 98 when quieter.

**Reading it.**
- Spin wins decode by ~10–20% (best **115–124 tok/s, ~57–61 GB/s**), because
  small-matmul dispatch got 2× cheaper. Rayon still wins prefill at 10 threads
  (984 vs 870): prefill matmuls are milliseconds long, so dispatch cost doesn't
  matter, and during prefill's long *serial* stretches (GELU, attention on the
  main thread) our yielding workers compete with the main thread for cores.
- **The serial ops are now the wall.** At 1 thread, prefill spends 12% in scalar
  GELU and 7% in attention. With 10 threads on the matmuls, that same ~44 ms of
  serial work is about half of prefill. Decode spends ~0.5 ms/token in serial
  ops (≈ 7% at 120 tok/s). Next stage: vectorise and parallelise them.
- The measurement environment is the hardest part of this stage. Multithreaded
  numbers on a shared machine have ±10–15% run-to-run noise. Every comparison
  here is interleaved in one session, and repeated.

---

## Reference point: Apple Accelerate (CLI only, not a stage)

`nanoinfer accel` (`crates/cli/src/accel.rs`): a CLI-side `Kernels` set whose
matmul is `cblas_sgemv` (decode) / `cblas_sgemm` (prefill) from Accelerate,
which runs them on the M4's matrix coprocessor (AMX/SME). It's plugged into
core's ordinary `kv_cache::forward`, so layernorm, GELU and attention are still
the scalar reference kernels. It isn't a `Stage`: Accelerate is macOS-only, and
core must build for wasm.

| | prefill P=64 | decode | decode GB/s | prefill P=512 | decode @ ~520 ctx |
|---|---|---|---|---|---|
| Accelerate (default threads) | 1459 ± 400 | 108.8 ± 16.8 | 54 | 891 | 85.9 |
| Accelerate, `VECLIB_MAXIMUM_THREADS=1` | 1481 ± 26 | 111.2 ± 1.3 | 55 | | |
| ours, stage 5 (rayon, 10t), same session | | | | 675 | 86.9 |

- **Prefill: Apple wins, 1480 vs our best 1021.** That's a matrix unit, not NEON,
  and it does it with scalar GELU/attention around it. Its single-thread setting
  makes no difference: the AMX/SME unit is shared per cluster, and one core
  drives it.
- **Decode: we beat Accelerate** (spin-pool 115–124 vs 111). sgemv on AMX reads ~55
  GB/s, about what one P-core's NEON loads get. The matrix unit adds FLOPs, and
  decode needs bytes.
- At P=512 both fall off (891 vs 675) for the same reason: the scalar O(n²)
  attention, the same code in both.

---

## Stage 7: fast-ops (vectorised + parallel GELU, softmax, attention, layernorm)

**Hypothesis.** After stage 6, prefill at 10 threads spends about half its time in
ops that run on one thread and don't vectorise: GELU calls libm `tanhf` per
element, softmax calls `expf`, attention's dot products are `iter().sum()`
chains. Make them branch-free polynomials that LLVM vectorises, and spread
GELU (by chunk) and attention (by (query, head) unit) over the same spin pool.
Expect prefill up to ~2× (Amdahl), decode a few % (its serial share is ~7%),
and decode at long context more (attention grows with context).

**What changed.** `kernels/fastops.rs`:
- `tanh`: Eigen's rational approximation (odd degree-13 / even degree-6, clamp
  ±7.9). Max abs error vs `f32::tanh` < 2e-6 on [-20, 20] (test).
- `exp`: n = round(x·log2 e), r = x − n·ln2_hi − n·ln2_lo (Cody–Waite), degree-6
  Taylor for e^r, 2^n built in the exponent bits. First version reduced via
  `t = x·log2e; f = t − round(t)`, which loses bits as |x| grows (rel. error
  3.7e-6 at |x| = 20, test failed). Cody–Waite gets < 2e-6 on [-86, 20].
- GELU parallel in chunks above 32K elements (prefill only; decode's 3072 run
  serially, vectorised). `objdump`: the GELU loop is `fmul.4s`/`fadd.4s` with one
  `fdiv.4s`, no calls.
- Attention: per (query, head) unit, stage 2's dot for scores, fast exp softmax,
  axpy for the value sum (vectorises: elementwise, no reassociation). Units are
  chunked over the pool when units × keys × head_dim > 128K. Each chunk keeps its
  own scores on the stack (≤ 1024 keys, a Vec beyond). Tested against the
  reference attention (max diff < 1e-5).
- LayerNorm: multi-accumulator mean/variance.
- `spin::par_for`: spin pool natively, rayon in a wasm threads build, serial otherwise.

**Correctness.** All HF tests pass. WikiText-2, 2 × 512 tokens: ppl 35.2302 vs
35.2302 (unrolled), mean KL 1.6e-11, top-1 agreement 100%, max |Δlogit| 0.001.

**Numbers** (10 threads unless noted; interleaved A/B pairs, 5 runs each):

| | prefill P=64 | decode (ctx 64–128) | prefill P=512 | decode (ctx 512–576) |
|---|---|---|---|---|
| spin-pool (stage 6) | 811 / 808 | 113.7 / 106.6 | 549 / 556 | 75.4 / 68.9 |
| **fast-ops** | **1392 / 1500** | **119.2 / 119.9** | **1203 / 1369** | **102.6 / 119.0** |

Recorded single runs were very noisy (P=64 prefill 1126 ± 323 and 1327 ± 301,
decode 105 ± 8 and 95 ± 18) while other agents were benchmarking. The
interleaved pairs above are the trustworthy comparison. 4 threads: prefill
1010, decode 111.9.

Prefill profile (10 threads, 109-token prompt, 68 ms total): fc_proj 30% / fc
28% / qkv 20% / attn_proj 7% at **300–330 GFLOP/s**, attention 4.9%, lm_head 3.4%,
GELU 1.9% (was 12%).

**Takeaways.**
- **Prefill +75% at P=64, +2.2× at P=512, decode at ~550 context +36%** (75 → 103),
  from ops that were "<1%" at stage 1. Prefill now matches Apple's Accelerate
  (1480 at P=64), which uses a matrix coprocessor for its matmuls, and beats it
  at P=512 (1203–1369 vs 891), where Accelerate is stuck with scalar attention.
- The lesson stays the same at every level: whatever you didn't optimise
  becomes the bottleneck. At stage 1 GELU was 0.1% of decode, at stage 4 it was
  12% of prefill, now it's 1.9%.

## Browser (WASM)

Everything here runs `crates/wasm` (a facade over core) in a Web Worker on the demo page, or the same wasm in Node
(V8) via `web/scripts/record.mjs`. Ladder shape: 64-token prompt, greedy, median of the decode steps; prefill = 64
tokens / prompt forward time. Machine note: most Chrome runs below happened while the native agents were
benchmarking (load average 20–26 on 10 cores), so treat absolute numbers as lower bounds; the V8/Node numbers were
taken when it was quieter.

### Browser stage 0 (round 1 recap): simd128, one thread

`unrolled` ≈ native decode (83–101 vs 100.5 tok/s); `tiled` prefill flat (175 vs native 469) because core's 4×4 tile
has no NEON to lower to and becomes scalar.

### Probe fix: the compute roof was wrong

**Symptom.** Decode at ~100 tok/s does ~25 GFLOP/s, *above* the 15–23 GFLOP/s "peak" the page measured.
**Cause.** (1) The probe's first call runs in Liftoff, V8's baseline compiler; TurboFan tier-up happens during or
after it. (2) `black_box(&a)` inside the loop forced the 8 accumulators to memory every iteration.
**Fix.** 12 register chains, `black_box` only on the constants, one warm-up call before measuring.
**Result.** simd128 mul+add: 22 → **52 GFLOP/s**; relaxed FMA (new): 34 → **~96 GFLOP/s** (first call vs warmed).
Round 1's "decode is compute-bound in the browser" is retracted: ridge 0.9 (mul+add) / 1.6 (FMA) FLOP/B, decode is
0.5, so it's memory-bound, at ~80–85% of a ~57–64 GB/s single-thread stream.

### Browser stage 1: wasm-simd (explicit simd128, browser-only rung)

**Hypothesis.** A 4×4 register tile with real simd128 (16 vector accumulators, 8 loads feed 16 mul+adds) gets
prefill the reuse core's scalar tile can't in wasm. For decode, interleaving four weight rows should reuse each load
of `x` four times.
**Change.** `crates/wasm/src/simd.rs`, `Simd<Plain>` kernel set plugged into core's `kv_cache::forward` (core's
`Kernels` trait; no core change).
**Result (V8).**

| kernel | decode tok/s | prefill tok/s |
|---|---|---|
| unrolled (reference point) | 100.6 | 169 |
| wasm-simd v1: decode walks 4 rows at once | 71.8 | 259 |
| wasm-simd v2: decode walks 1 row, 4 accumulators | 97–101 | 255–260 |

**Failure.** v1 decode was 28% slower. Four interleaved weight streams 3 KB apart defeat the prefetcher (same lesson
as native stage 3). v2 keeps one sequential stream and matches `unrolled`, which LLVM already vectorises to the same
instructions.
**Prefill:** +53% from the tile.

### Browser stage 2: wasm-fma (relaxed-SIMD)

**Change.** Same loops with `f32x4_relaxed_madd` (one fused instruction, `fmla` on arm64). Separate build
(`+relaxed-simd`), since a module with relaxed opcodes doesn't validate without support; the worker validates each
binary and picks the best.
**Result (V8).** decode 98–101 (no change: memory-bound), **prefill 354–370** (+40% over wasm-simd, 2.2× unrolled).
Chrome threads build under load: 95 / 381.

### Browser stage 3: wasm-threads (wasm-bindgen-rayon)

**Change.** Nightly build with `+atomics,+bulk-memory,+mutable-globals` and `-Z build-std`, shared memory
(`--shared-memory --import-memory --max-memory=4GiB` and the TLS exports had to be passed explicitly: without them
the module exported a non-shared memory and `initThreadPool` failed with "Memory could not be cloned").
`Par<Relaxed>` splits each matmul's output columns over the rayon pool, 4 chunks per worker, 16-column multiples.
**First attempt (failed): enable core's `threads` feature.** Core's stage 6+ (`spin-pool`, `fast-ops`, quantised)
start OS threads with `std::thread::spawn`, which panics on wasm32. So the wasm builds never enable it; core's
threaded stages run single-threaded and the browser gets its own threaded rung.
**Bug on the way.** React StrictMode ran the boot effect twice in dev, so `initThreadPool` ran twice and rayon's
`build_global()` threw (`unwrap_throw`). Boot is now once per page.
**Result (Chrome, load ~25).**

| workers | decode tok/s | prefill tok/s |
|---|---|---|
| 1 (wasm-fma) | 95 | 381 |
| 4 | 91 | 661 |
| 10 | 87–89 | 809–818 |

Prefill 2.1× wasm-fma, 4.8× unrolled, near native rayon (1021). Decode: no gain. 49 matmuls per token at ~0.2 ms
each; fork/join via `Atomics.wait` wake-ups eats the split, and one wasm thread already gets most of the bandwidth.

### f16 download (no kernel change)

`web/scripts/to-f16.mjs` writes `model-f16.safetensors` (274 MB, from 548). `Bytes::from_f16` rewrites the header to
F32 and widens each half as it streams in, so core's loader is untouched and the heap peak is unchanged. Max
|Δlogit| vs the f32 file 0.03–0.05; greedy identical 32/32 tokens on 3 prompts. The page downloads f16 by default
(`?weights=f32` for the original).

### Core's quantised stages in wasm (as they landed)

Chrome, simd128 build, under load: q4 106 / 150, q8 36 / 168, f16 12.4 / 125 (decode / prefill). In wasm the
format's decode cost dominates: f16→f32 is scalar bit-twiddling per weight (no NEON `fcvt`), q4's nibble unpack is
cheap integer work on 0.56 bytes per weight. q4 ties or beats the f32 rungs in the browser (V8, round-robin: 91 vs 94–100), and it also changes the
greedy text ("to read the right, the right, ...").

### Recorded fallback

`web/scripts/record.mjs` (Node, relaxed build) records the hero run, attention maps (u8 log-quantised), logit lens,
the walkthrough's residual stream and logits, per-stage Ladder numbers and race lanes: ~240 KB total in
`web/public/recorded/`. Op traces are packed as interned op defs + per-forward gaps/durations in 0.1 µs, with shape
overrides only where an op differs from its decode shape (215 KB → 116 KB for the run).

---

## Stage 8: f16 weights

**Hypothesis.** Decode streams every weight once per token. Halving the bytes
should come close to doubling decode (ceiling 243 → 485 tok/s at the spec
120 GB/s; ~400 at the measured ~98). Prefill shouldn't move (compute-bound, f32
FMAs either way). f16 has an 11-bit significand (relative rounding ≤ 2⁻¹¹), so
quality should be indistinguishable.

**What changed.** The first stage whose weights differ from `Model`:
- `quant.rs`: f32 ↔ f16 conversion with round-to-nearest-even, subnormals,
  inf/NaN (test: every one of the 65536 f16 bit patterns round-trips).
- `kernels/packed.rs`: a `PackedMatrix` trait (`pack`, `bytes`, `matmul`), the
  shared driver (columns split on the spin pool, as in stage 6), and the prefill
  path: dequantise a 4-row weight panel to f32 *once*, then run a stride-general
  4×4 FMA tile over all prompt rows.
- `kernels/f16.rs`: decode dot product widening 8 halves at a time with
  `fcvtl`/`fcvtl2`. Rust's f16 NEON intrinsics (`vcvt_f32_f16`, the `f16`
  type) are still unstable, so it's two lines of stable inline `asm!`.
- `stages/packed.rs`: the KV-cache forward over a `PackedModel<M, H>` (blocks in
  format M, LM head as H). `Session::with_max_seq` builds the packed copy once
  (so a stage switch costs a repack, not a reload). The public API is unchanged.
- `Stage::weight_format()` / `head_format()` / `decode_bytes()` give each stage
  its own ceiling. Bench now reports it per stage.

**Correctness.** HF logits: max |diff| 0.024–0.10 over the 4 prompts (f32 stages:
< 2e-4). The reference test's tolerance is now per format (f16: 0.25).
WikiText-2, 4 × 512 tokens: **ppl 37.388 vs 37.378 (+0.03%), mean KL 1.3e-5
nats, top-1 agreement 99.80%**, max |Δlogit| 0.35.

**Numbers** (stages 8–10 session, interleaved; 7 runs per cell, two rounds at
10 threads, then a 5-run thread sweep; heavy background load, so ±10–15%):

| threads | fast-ops (f32) decode | f16 decode | f16 prefill |
|---|---|---|---|
| 4 | | **180 ± 10** | 946 |
| 6 | | 169 ± 6 | 1240 |
| 8 | | 147 ± 13 | 1226 |
| 10 (round 1 / 2) | 84 ± 11 / 103 ± 7 | 160 ± 24 / 181 ± 6 | 1218 / 1320 |

**Reading it.** Decode **~180 tok/s (1.5–1.75× f32)** at ~45 GB/s, *less*
bandwidth than f32 fast-ops reached (~60). We're no longer bandwidth-bound.
At 180 tok/s a token is 5.5 ms, of which the bytes need ~2.7 ms at 90 GB/s. The
rest is per-token overhead that doesn't shrink with the bytes (next stages).
And more threads stop helping beyond 4: the extra threads only add dispatch
cost and E-core stragglers.

---

## Stage 9: Q8_0 (int8 blocks, sdot), and why the LM head stays f16

**Hypothesis.** ggml's Q8_0, blocks of 32 int8 plus one scale (we keep an f32
scale: 36 bytes per 32 weights), is "near-lossless" folklore. Quantise the
activations the same way per matmul and each block's dot product becomes exact
int32 arithmetic with `sdot` (`vdotq_s32`, stable: 16 int8 multiply-adds per
instruction). Expect ~1.4× over f16 on bytes, with Δppl ≲ 0.1%.

**What changed.** `quant.rs`: `BlockQ8`/`BlockQ4`, ggml's quantisation rules,
dequantisers, error-bound tests. `kernels/q8.rs`: activation quantiser (once per
matmul, into reusable `Scratch`, so no allocation while decoding), `dot_q8` (two
blocks per iteration, `vdotq_s32` ×2 → `vcvtq_f32_s32` → FMA by `d_w·d_x`), and a
scalar fallback for wasm. Prefill dequantises panels (stage 8's path).

**Attempt 1: everything Q8, including the tied LM head (failed on quality).**
Decode ~268 tok/s, but WikiText-2 (4 × 512): **ppl +4.7%, mean KL 0.046, top-1
agreement 84.2%**, max |Δlogit| 55. Not near-lossless at all. Q4 the same way was
a disaster: ppl 37.4 → **516**.

**Finding the cause.** New diagnostic `ppl --fake-quant q8|q4|f16 --only
<matrices> [--layers ..]`: round-trip chosen f32 matrices through a format, then
run an exact f32 stage. That isolates weight error from kernel and activation
effects (1 × 512 tokens, ppl 28.76 baseline):

| fake-quantised | Δppl | mean KL | top-1 |
|---|---|---|---|
| **q4: lm_head (wte) only** | **+558%** | 2.05 | 34% |
| q4: qkv only | +4.3% | 0.030 | 90% |
| q4: fc only | +2.2% | 0.020 | 90% |
| q4: fc_proj only | +0.9% | 0.012 | 94% |
| q4: attn_proj only | +0.7% | 0.005 | 97% |
| q8: lm_head only | +0.18% | 0.0086 | 91% |
| q8: all 48 block matrices | −0.13% | 0.00027 | 99% |
| f16: everything | +0.01% | 0.00001 | 100% |

Then which *activations*? A temporary switch gave chosen matmuls f32 inputs
(decode through the dequantise-to-f32 path) with the weights still Q8:
- f32 activations for the LM head only: KL 0.038 → **0.0092**;
- f32 activations for every block matmul instead: KL stays 0.038.

So the LM head is the problem twice over: its weights (shared with the
embedding) are the most sensitive matrix in the model, and its *input* can't be
quantised. A throwaway probe printed why. GPT-2's `ln_f` gain has outlier
channels (17.4 on dim 496, 17.0 on 430, 15.3 on 314, vs rms 2.05), so the LM
head's input has a few huge channels. Quantised per block of 32, those blocks
keep ~3 bits of precision for their other 31 values.

**Tried: fold the gain into the head.** `W·(g⊙n + b) = (W·diag g)·n + W·b`. Store
`W·diag(g)` quantised and `W·b` as an f32 per-token bias, then quantise `n`
(normalised, no gain outliers). Exact in f32 (`stages/packed.rs::FoldedHead`).
It helped (KL 0.039 → 0.0103, top-1 86% → 93%) but nowhere near enough.

**Kept: Q8_0 blocks + f16 LM head** (`PlainHead<F16Matrix>`), what llama.cpp does
too (it keeps `output.weight` at higher precision).

| Q8 blocks + head as | Δppl | mean KL | top-1 |
|---|---|---|---|
| Q8 (attempt 1) | +2.9% | 0.039 | 85.6% |
| Q8, ln_f gain folded | +1.6% | 0.010 | 93.0% |
| **f16** | **−0.1%** | **0.00086** | **97.9%** |

Bytes per token: 95.6 MB blocks + 77.2 MB head = 173 MB (spec ceiling 693 tok/s).

**Numbers** (same session as stage 8): decode **~230–240 tok/s** (4–8 threads:
232 ± 15, 231 ± 7, 239 ± 4; 10 threads 230 ± 36 / 203 ± 21), prefill ~1240–1330
(the f32 tile path after dequantising). ~40 GB/s, 34% of its spec ceiling. The
f16 head is 45% of the bytes now.

---

## Stage 10: Q4_0 (4-bit blocks)

**Hypothesis.** Q4_0 halves the block bytes again (20 bytes per 32 weights).
Decode unpacks nibbles in registers (`and 0x0f` / `shr 4` / `sub 8`) into int8
and reuses stage 9's `sdot`. With the f16 head kept (stage 9's lesson), the
quality cost should be a few % ppl.

**What changed.** `kernels/q4.rs` (+ scalar fallback). Test: the integer decode
path equals "dequantise W, quantise x, f32 dot" to 1e-3, which separates
kernel bugs from quantisation error. LM head: f16 (a folded-gain Q8 head was
tried and was no better here: KL 0.075 vs 0.069).

**Correctness / quality** (2 × 512 tokens): **Δppl +7.3%, mean KL 0.069, top-1
agreement 83%**. That's what 4-bit round-to-nearest with one scale per 32
weights costs a 124M model whose qkv and fc matrices are sensitive (see the
fake-quant table). Better 4-bit schemes (per-block min, k-quants, importance
weighting) are where llama.cpp went next.

**Numbers:** decode **~240–263 tok/s** (4t 238 ± 16, 6t 263 ± 13, 8t 241 ± 61, 10t
220 ± 33), prefill ~1215–1230. Only 5–10% faster than Q8, at ~30–35 GB/s and 24–29%
of its 918 tok/s spec ceiling.

**Why Q4 barely beats Q8:**
- **The f16 head is 59% of Q4's bytes** (77 of 131 MB). Halving the blocks
  only removes 42 MB of 173.
- **Per-token overhead dominates.** Profile at 10 threads (320 tok/s run): lm_head
  0.93 ms/token at 83 GB/s (30%), attention 0.51 ms (16.5%: still serial at
  short context, below the parallel threshold), and qkv/fc/fc_proj at ~35 GB/s
  each. A 885 KB Q4 qkv matmul takes ~32 µs when its bytes need ~10 µs.
- Microbenchmark of the spin pool (empty job): dispatch+join is 1.9 µs with 10
  chunks on 10 threads but **8.6 µs p50 / 48 µs p99 with 40 chunks**. Every chunk
  claim is a CAS on one cache line, and every completion an atomic add on
  another, bouncing between the P and E clusters. × 49 matmuls ≈ 0.4 ms/token
  before any load imbalance.

So stages 8–10 cut the bytes 3.8× (495 → 131 MB) and decode only rose 2× (~120 →
~250). The remaining gap is fixed per-token overhead.

---

## Stage 11: q8-fast / q4-fast (low-latency pool)

**Hypothesis.** After stage 10, ~2.5 ms of a ~4 ms token is fixed cost. Two
measured pieces: (1) dispatch, where an *empty* 40-chunk job on stage 6's pool
costs 8.6 µs p50 / 48 µs p99, × 49 matmuls; (2) attention, single-threaded at short
context (stage 7's parallel threshold), ~0.5 ms per token. Make both cheaper and
the same weights should decode noticeably faster, bit-identical.

**What changed.**
- `core/src/fastpool.rs`, a new pool (stage 6's stays as measured). Chunk claim =
  one `fetch_add` on the packed `epoch|n|next` word (no CAS retry loop; a claim
  past the end is harmless). Completions are counted per *thread* (one add when
  it runs dry), not per chunk. The two hot atomics sit on separate 128-byte lines.
  The claim is still safe against stale workers: a claimed chunk's job can't
  finish until that chunk is counted, so the job pointer read *after* a
  successful claim is that job's. Test: 5000 random dispatches, each chunk exactly
  once, plus 4 concurrent dispatchers.
- Microbenchmark (empty jobs, p50 / p99): 10 threads × 40 chunks **8.25 / 55 µs →
  2.62 / 5.9 µs**; × 10 chunks 1.71 / 9.2 → 0.88 / 7.6 µs.
- `kernels/lowlat.rs`: `FastPar` (the packed matmuls' runtime is now a type
  parameter `P: Par`, so stages 8–10 keep stage 6's pool) and a `LowLat` kernel
  set whose attention runs (query, head) units on the fast pool whenever there's
  more than one thread (decode: 12 heads), and GELU in parallel above 16K elements.
- Stages `q8-fast` and `q4-fast`: the stage 9 / 10 weights on this runtime.
- **Bug found on the way:** `cargo test` hung. Unit tests run on parallel threads,
  and several now dispatch on the global pools at once, which the one-dispatcher
  protocol doesn't allow. Both pools now take a `try_lock` dispatch mutex, and a
  concurrent caller runs its job serially (separate commit; it changes nothing on
  the single-dispatcher path stage 6 was measured on).

**Tuning that didn't pan out.** Fewer, bigger chunks, since claims were cheaper
now. Decode chunks/thread at 6 threads: 1 → 227, 2 → 269, 3 → 280, 4 → 282, 6 → 290
tok/s. At 10 threads: 223 / 250 / 244 / 226 / 240. Fewer chunks make E-core
stragglers worse than the claims save. Kept 4.

**Measurement method change.** The machine's load drifted by ±20% over minutes
(Spotlight at 100–340% CPU, Backdrop ~200%), so whole-run medians of separate
benches weren't comparable. Two changes: bench now also reports the **median
per-token step time** over all timed tokens (robust to preemption spikes,
`decode_step_median_tok_s` in results.jsonl), and A/B comparisons use an
interleaving script (alternating 2-run benches of each stage, 5–6 rounds,
median of the per-token metric).

**Numbers** (interleaved A/B, per-token decode median, tok/s):

| threads | q8 | **q8-fast** | q4 | **q4-fast** |
|---|---|---|---|---|
| 4 | 299 | **335** (+12%) | 348 | **372** (+7%) |
| 6 | 313 | **360** (+15%) | | |
| 10 | 267 | 278 (+4%) | | |

Prefill unchanged (1415 vs 1416 at 6t). Recorded runs (6 threads, 7 runs):
q8-fast 292.6 ± 16.6 tok/s (per-token 288.5), q4-fast 338.4 ± 5.2 (per-token
338.8), prefill 1430 / 1381. Output bit-identical to stage 9 (ppl tool: KL 0,
max |Δlogit| 0).

**Where a token goes now** (q8-fast, 4 threads, 256 tokens, ~296 tok/s, `generate
--profile`): lm_head **1.03 ms (30%, f16, 75 GB/s)**, fc_proj 17%, fc 16%,
**attention 15%** (42 µs per layer at context 64–320: it streams f32 K/V), qkv 13%,
attn_proj 5%. The LM head is now the single biggest cost.

**Lessons.** E-cores still hurt decode: the best thread count is 4–6, not 10
(membw: 92 GB/s on 4 threads vs 98 on 10, so the E-cores add almost no bandwidth,
and every join waits for the slowest). Performance work on a shared machine needs
an A/B harness, not two benchmarks run five minutes apart.

---

## Stage 12: q8-rescore / q4-rescore (approximate LM head + exact rescoring)

**Hypothesis.** The f16 LM head is 1.03 ms of a ~3.4 ms token (30%, 77 MB). Most
of its 50257 logits only matter in aggregate. If a cheap approximate head scores
every token, and only those near the top get recomputed exactly with the f16
weights, the probability mass (everything greedy or top-k sampling can pick)
should keep f16-head logits while the head streams Q8 bytes (43 MB).

**What changed.** `stages/packed.rs::RescoredHead<A>`: (1) approximate logits for
all tokens from `A`; (2) candidates = tokens within `MARGIN` of the approximate
max, at most `CAP` (the top by `select_nth` if more); (3) candidates rescored in
parallel with `F16Matrix::dot_row` (f16 weights, f32 activations: exactly
stage 9's head). Stages `q8-rescore` / `q4-rescore` = stage 11 + this head.

**Attempt 1: Q8_0 approximate head (W8A8).** Top-1 agreement with the f16-head
stage was **100%** at every margin. The top is always exact. But KL wouldn't
drop below ~9e-3 (WikiText-2, vs q8-fast):

| margin / cap | KL vs f16-head stage | top-1 |
|---|---|---|
| 2 / 256 | 2.4e-2 | 100% |
| 6 / 256 | 1.0e-2 | 100% |
| 10 / 256 | 8.6e-3 | 100% |
| 10 / 2000 | 3.0e-3 | 100% |
| 15 / 8000 | 5.9e-4 | 100% |
| everything (sanity) | 0 | 100% |

GPT-2's next-token distributions have lots of entropy. Thousands of tokens sit
within 10–15 nats of the top and together carry real probability, so the
approximate tail's error matters.

**Tried: tail bias correction (failed).** If the tail logits carried independent
zero-mean noise ε, the tail mass would be inflated by E[e^ε] = e^{σ²/2}
(Jensen). Estimate μ and σ² of (exact − approx) from the rescored candidates
and shift the tail by μ − σ²/2. Result: KL 1.13e-2 (margin 6), 8.9e-3 (margin
10), no better. The candidates' errors aren't representative of the tail's,
or the error isn't independent of the logit.

**Diagnosis: it's the activations again.** Computing the approximate pass with f32
activations (a quality-only hack, through the prefill path) gave KL **1.4e-3**
(6 / 256) and **4.1e-4** (10 / 2000): 7× better. Stage 9's finding (ln_f gain
outliers wreck int8 activations) applies to the approximate pass too.

**Kept: Q8 weights × two-level int8 activations** (`kernels/q8hilo.rs`):
`x ≈ d·q_hi + (d/254)·q_lo`, where q_lo quantises the residual of q_hi (≤ d/2),
giving ~16-bit activation precision. Two `sdot` chains per block over the *same*
weight bytes: twice the integer arithmetic on a pass that is bandwidth-bound
anyway. Test: with a 17× outlier channel, hi/lo matches dequantised-W · exact-x
to 2e-3. Quality matches the f32-activation hack:

| margin / cap | KL vs f16-head stage | top-1 |
|---|---|---|
| 6 / 256 | 1.5e-3 | 100% |
| **8 / 512 (kept)** | **1.0e-3** | **100%** |
| 10 / 2000 | 4.5e-4 | 100% |

**Speed** (interleaved A/B, 6 threads, per-token decode median):

| | q8-fast | q8-rescore | q4-fast | q4-rescore |
|---|---|---|---|---|
| margin 6 / 256 | 350 | **404** (+15%) | | |
| margin 8 / 512 | 350 | **394** (+13%) | 403 | **470** (+17%) |

(A margin 10 / 2000 pair was ruined by a load spike: 295 vs 300 with min/max
152–365.) Recorded (6 threads, 7 runs): **q8-rescore 394.5 ± 29 tok/s** (per-token
394.6, 55 GB/s, 46% of its 860 tok/s spec ceiling), **q4-rescore 479.7 ± 30**
(per-token 482.3, 39% of 1237). Prefill unchanged (~1400).

**Quality vs f32** (WikiText-2, 4 × 512):

| stage | Δppl | mean KL | top-1 |
|---|---|---|---|
| q8-fast (f16 head) | −0.1% | 8.6e-4 | 97.9% |
| **q8-rescore** | **+0.21%** | **2.0e-3** | **98.1%** |
| q4-fast | +7.3% | 0.069 | 83% |
| q4-rescore | +8.0% | 0.081 | 82% |

(The two bench lines first recorded `threads: 1`: `Stage::multithreaded` was an
explicit list that rustfmt had reflowed, so my edit missed it. Fixed by hand in
results.jsonl, and the method is now "every stage after 4".)

---

## Stage 13: q8-attn / q4-attn (register-resident NEON attention)

**Hypothesis.** Attention was ~15% of a short-context decode token (34 µs per
layer) and grows with context. Stage 7's unit calls the generic 16-accumulator
dot per key (a 16-lane tree reduction for 64 products) and re-loads / re-stores
the output for every key. A head_dim-64 kernel can keep q in 16 NEON registers
for the score pass (two keys per iteration, 8 accumulators) and the 64-float
output in 16 registers for the value pass (16 independent FMA chains, one store).

**What changed.** `kernels/attn_neon.rs` (`attend64` + the stage 11 unit split on
the fast pool; other head sizes / non-aarch64 → stage 7's unit). Kernel set
`AttnNeon`. Stages `q8-attn` / `q4-attn` = stage 12 + this. Test vs the reference
attention for (n_q, pos0) ∈ {(1,0), (1,37), (3,10)}: max diff < 1e-5.

**Numbers** (6 threads; interleaved A/B, per-token decode median):

| context | q8-rescore | q8-attn |
|---|---|---|
| 64–112 | 389 | **401** (+3%) |
| 512–560 | 275 | **293** (+6.5%), prefill 1295 → 1397 |

Recorded (5 runs): prompt 64: **404.4 ± 14.8 tok/s** (per-token 407.3, 47% of the
860 spec ceiling), prefill 1425. Prompt 512: 291.6 ± 7.7, prefill 1410.

**Smaller than hoped, and the profile says why.** At 541–668 tokens of context,
attention is still **37% of decode**, 109 µs per layer, moving only ~34 GB/s of
K/V. The kernel isn't compute-bound anymore (~10 µs of FMAs per layer across 6
threads). It's the **cache layout**: K/V are stored `[position][768]`, so one
head's keys are 600 separate 256-byte pieces 3 KB apart. That's the strided
multi-stream pattern stage 3 measured as bad for the prefetcher. And at this
length the f32 cache is **38 MB per token** (2 × 12 × 512 × 768 × 4 B), 27% on
top of the 139 MB of weights. Law 5 in action. Next: a head-major f16 cache.

---

## Stage 14: q8-kv16 / q4-kv16 (head-major f16 KV cache)

**Hypothesis.** Stage 13's long-context profile blamed the cache, not the
kernel: `[layer][pos][C]` f32 means each head's keys are 256-byte pieces 3 KB
apart, and at 512 positions the cache is 38 MB per token. A head-major
`[layer][head][pos][64]` cache makes each head's K and V one contiguous stream,
and f16 halves the bytes. Expect little change at short context, a big one at
long context.

**What changed.** `stages/kv16.rs`: `KvCache16` (u16 bits, head-major), a new
forward (stage 12/13's with this cache), K/V narrowed with NEON `fcvtn` on write
and widened with `fcvtl` inside a head_dim-64 attention kernel (q and the output
in registers, as in stage 13), with a portable scalar fallback. The `Session`
keeps a length-only `KvCache` for these stages, so `kv_cache().len` and traces
still work.

**Quality.** Against q8-attn directly the KL looks alarming (4e-4 at 32-token
windows, 8e-4 at 1024, top-1 95–98%). Against the **f32 truth** the two are
identical: KL **1.84e-3 vs 1.84e-3**, Δppl −0.20% vs −0.27%, top-1 97.8% vs 97.4%.
So the f16 cache costs nothing measurable. The variant-to-variant difference is
int8 activation rounding: a tiny perturbation flips rounding boundaries in
every Q8 activation quantisation downstream, and comparing two lossy stages to
each other mostly measures that chaos. Lesson: compare lossy stages to the f32
reference, not to each other.

**Numbers** (6 threads, interleaved A/B, per-token decode median):

| context | q8-attn | q8-kv16 | |
|---|---|---|---|
| 64–112 | 271 | 290 | +7% |
| 512–560 | 216 | 253 | **+17%** |
| 900–948 | 178 | 284 | **+60%**, prefill 977 → 1074 |

**Measurement caveat.** During this stage the machine's memory bandwidth
collapsed: `membw` read **26 / 42 / 44 / 59 GB/s** at 1 / 4 / 6 / 10 threads
(median), against 57 / 91 / 96 / 98 earlier, with normal CPU load. Something on
the GPU side (other agents' Metal/WebGPU benches and a Chrome renderer at 98%)
was saturating memory. The A/B ratios above are valid; absolute numbers from
this period (recorded: q8-kv16 271 / 240 / 206 tok/s at prompts 64 / 512 / 900,
q4-kv16 278) are depressed ~2× and don't compare with earlier stages.

---

## Stage 15: q8-prefill (int8 `sdot` prefill tile)

**Hypothesis.** Prefill (85% matmuls) is compute-bound, and since stage 8 it has
dequantised each weight panel to f32 and run f32 FMAs (~2.7 multiply-adds per
instruction once loads are counted). With the prompt rows quantised to Q8_0
too (as decode already does), a 4 × 4 output tile can use `sdot`: per block of 32
and per (row, weight-row) pair, 2 `sdot` (32 multiply-adds) + 1 convert + 1 FMA by
the scales, ~8 multiply-adds per instruction. Expect prefill ~1.5–2×.

**What changed.** `kernels/q8gemm.rs`: `Q8GemmMatrix` (Q8_0 storage; decode =
stage 9's `dot_q8`; prefill = activation quantisation + `tile_4x4`: 4 weight rows'
two 16-byte halves kept in 8 registers per block, 16 int32→f32 accumulators).
Ragged rows/columns fall back to `dot_q8`. Test: tile output equals the per-row
integer dot to 1e-4 relative. Stage `q8-prefill` = stage 14 with this matrix.

**Numbers** (interleaved A/B, 10 threads):

| | q8-kv16 prefill | **q8-prefill** prefill | decode (per-token) |
|---|---|---|---|
| P=64 | 1657 | **2510** (+52%) | 346 → 359 |
| P=512 | 1523 | **2277** (+50%) | 282 → 268 (noise) |

Recorded: **P=64 2415 ± 145 tok/s (413 GFLOP/s), P=512 2143 ± 174 (364 GFLOP/s)**
at 10 threads; 6 threads: prefill 2015, decode **401 tok/s per-token** (69.5 GB/s,
58% of its 693 tok/s spec ceiling). membw had recovered by then (51 / 82 / 86
GB/s at 1 / 4 / 10 threads).

**Quality.** HF prompts (which go through prefill) give KL vs HF 3.0e-3 / 1.5e-3 /
4.5e-4 / 4.7e-5, against q8-kv16's 2.7e-3 / 1.2e-3 / 3.6e-4 / 4.9e-5. Int8
activations in prefill add very little on top of the rescored head's tail error.

**vs Apple Accelerate:** prefill **2415 vs 1480 tok/s at P=64 (1.6×)** and 2143 vs
891 at P=512 (2.4×). Accelerate's AMX/SME does f32 GEMM; we do int8 on NEON
and every other op in parallel.

---

## CPU climb: where it ended (one interleaved sweep, same session)

Every stage benchmarked back to back (6 threads for the multithreaded ones,
prompt 64, 48 decode tokens, 2 interleaved rounds, per-token decode median;
membw at the time ≈ 51 / 82 / 86 GB/s at 1 / 4 / 10 threads). Ceiling = 120 GB/s
spec ÷ the stage's bytes per token.

| stage | decode tok/s | prefill tok/s | bytes/token | spec ceiling | % |
|---|---|---|---|---|---|
| 1 kv-cache | 17.3 | 24.5 | 495 MB | 243 | 7% |
| 2 unrolled | 101.6 | 186 | 495 MB | 243 | 42% |
| 4 tiled | 95.5 | 438 | 495 MB | 243 | 39% |
| 5 threads (rayon) | 121.8 | 914 | 495 MB | 243 | 50% |
| 6 spin-pool | 138.6 | 914 | 495 MB | 243 | 57% |
| 7 fast-ops | 146.2 | 1456 | 495 MB | 243 | 60% |
| 8 f16 | 248.7 | 1408 | 247 MB | 485 | 51% |
| 9 q8 (f16 head) | 315.1 | 1386 | 173 MB | 693 | 45% |
| 10 q4 (f16 head) | 355.9 | 1417 | 131 MB | 918 | 39% |
| 11 q8-fast / q4-fast | 355.7 / 432.9 | 1443 / 1407 | | | 51% / 47% |
| 12 q8-rescore / q4-rescore | 405.0 / 495.2 | 1441 / 1387 | 139 / 97 MB | 860 / 1237 | 47% / 40% |
| 13 q8-attn | 411.4 | 1464 | 139 MB | 860 | 48% |
| 14 q8-kv16 / q4-kv16 | 425.5 / **497.4** | 1457 / 1389 | | | 49% / 40% |
| 15 q8-prefill | 405.8 | 2045 (6t); **2415 at 10t** | 139 MB | 860 | 47% |
| Apple Accelerate (ref.) | 111 | 1480 | 495 MB | 243 | 46% |

Best decode: **q4-kv16 ~497 tok/s** (29× stage 1; quality +8% ppl). Best decode
at Q8 quality (+0.2% ppl, 98% top-1): **q8-kv16 ~425 tok/s** (25×). Best
prefill: **q8-prefill 2415 tok/s at 10 threads** (99× stage 1, 1.6× Accelerate).
At ~900 tokens of context the f16 KV cache matters most (stage 14: +60%).

Achieved bandwidth tops out around 50–60 GB/s for the quantised stages, against
~82–90 GB/s that 4–10 threads can read. The remaining gap is per-token fixed
cost (~49 parallel sections + attention + LM-head bookkeeping at ~2.4 ms/token),
E-core stragglers (best decode thread count is 4–6, not 10), and a measured
decline in single-core bandwidth when four streams interleave.

## Head-to-head: nanoinfer vs llama.cpp, MLX, PyTorch MPS (18 Oct, evening)

**Question.** Every number above is nanoinfer against its own ceiling. How does
it compare with engines people actually use, on the same model, machine, and
workload?

**Method** (`scripts/headtohead/`, results in `bench/headtohead.json`).

- Model: GPT-2 124M from our own `weights/gpt2`. llama.cpp gets it through
  `convert_hf_to_gguf.py` (F32/F16), then `llama-quantize` (Q8_0, Q4_0). MLX
  gets it through `mlx_lm.convert` (f32, f16, 8- and 4-bit, group 32).
  (The converter refuses the `h.*.attn.bias` mask buffers in our safetensors;
  `strip_hf.py` drops them.)
- Workload, same everywhere: prompt 64 tokens (one forward), then 64 greedy
  decode tokens at context 64 → 128, batch 1. llama-bench: `-p 64 -n 0` for
  prefill and `-p 0 -n 64 -d 64` for decode (depth 64 means the same context).
  MLX: `mlx_lm.stream_generate` on our exact prompt tokens (its usual
  async-eval loop). PyTorch: HF `GPT2LMHeadModel` on MPS with the KV cache.
- Repetitions: 1 warmup + 7 timed, median, per config. Three rounds of the
  whole suite, strictly one engine at a time, waiting for 1-min load ≤ 8 before
  every bench (recorded per row; max 8.7); the best round is reported, for
  every engine. llama.cpp's Metal decode turned out **bimodal** (Q4_0: ~450
  or ~625 tok/s across rounds, same settings); after heating the GPU with any
  other Metal job it lands in the high mode, so `warm_check.sh` re-measures it
  that way (15 reps) and llama.cpp gets the higher of the two. nanoinfer's
  Metal decode moved < 2% across rounds.
- Threads: CPU runs at 4, 6 and 10 for both engines (all rows in the JSON).
  llama.cpp CPU: two builds. Homebrew's (build 11429) routes prefill matmuls to
  Accelerate, i.e. the AMX unit, and we pass `-nopo 1` so `-ngl 0` really
  keeps everything off the GPU; plus a source build at ff5a233 with
  `GGML_BLAS=OFF`, `GGML_METAL=OFF`: pure NEON, the like-for-like CPU number.
- Formats, matched where they can be: **Q8** = llama.cpp Q8_0 (head Q8_0 too),
  our metal-pipelined-q8 / webgpu-vec (all Q8_0, f16 scales, identical 131.7 MB
  per token) and MLX 8-bit g32 (affine: f16 scale + bias per 32, 9 bpw).
  **Q4** = Q4_0 blocks + Q8_0 head, ours (metal-lanes) and llama.cpp with
  `--output-tensor-type q8_0 --token-embedding-type q8_0` (89.3 MB both).
  llama.cpp's default Q4_0 puts the tied head in **Q6_K** (79.9 MB), measured
  too; MLX 4-bit quantises the head to 4 bits unless told otherwise, so we also
  quantise it in memory as 4-bit blocks + 8-bit head. Our CPU stages differ:
  their blocks carry an f32 scale (9 / 5 bpw, ~6–9% more bytes than ggml's
  8.5 / 4.5), and q8-kv16 / q4-kv16 / q8-prefill decode with a Q8 approximate
  head rescored in f16 (stage 12).
- Ceiling: 120 GB/s spec ÷ each format's own bytes per decoded token (weights,
  tied head counted once, position table excluded). llama.cpp bytes are read
  out of the GGUF (`gguf_bytes.py`). (`Stage::head_format` misses q8-prefill,
  whose bench line reports 173 MB; it decodes exactly like q8-kv16, 139.5 MB,
  which is what the JSON uses.)

**GPU (Metal), decode and prefill in tok/s:**

| format | engine | decode | % spec ceiling | prefill P=64 |
|---|---|---|---|---|
| f32 (495 MB) | llama.cpp | **192** | 79% | **8,271** |
| | MLX | 178 | 73% | 3,700 |
| | nanoinfer metal-mma | 176 | 72% | 4,509 |
| | PyTorch MPS | 156 | 64% | 5,026 |
| f16 (247 MB) | **nanoinfer metal-pipelined-f16** | **364** | 75% | 5,208 |
| | MLX | 342 | 70% | 5,612 |
| | llama.cpp | 333 | 69% | **10,283** |
| | PyTorch MPS | 252 | 52% | 6,286 |
| Q8 (132 MB) | **nanoinfer metal-pipelined-q8** | **612** | 67% | 5,856 |
| | nanoinfer webgpu-vec (wgpu) | 569 | 62% | 4,360 |
| | MLX 8-bit g32 (139 MB) | 534 | 62% | 6,608 |
| | llama.cpp Q8_0 | 502 | 55% | **10,706** |
| Q4 + Q8 head (89 MB) | **nanoinfer metal-lanes** | **787** | 59% | 6,725 |
| | MLX 4-bit + 8-bit head (97 MB) | 694 | 56% | 7,624 |
| | llama.cpp Q4_0 + Q8_0 head | 622 | 46% | **10,935** |
| other 4-bit | llama.cpp Q4_0 default (Q6_K head, 80 MB) | 654 | 44% | 11,064 |
| | MLX 4-bit, 4-bit head (78 MB) | 750 | 48% | 7,844 |

**CPU, best thread count per engine** (decode at 4 / 6 / 10 threads in brackets):

| format | engine | decode | prefill P=64 (best) |
|---|---|---|---|
| f32 | llama.cpp NEON | **197** @6t (142 / 197 / 165) | 1,249 (Accelerate build: **2,724** @4t) |
| | nanoinfer fast-ops | 165 @6t (137 / 165 / 164) | 2,242 |
| f16 | llama.cpp NEON | **358** @6t (273 / 358 / 281) | **2,357** |
| | nanoinfer f16 | 328 @6t (255 / 328 / 308) | 2,237 |
| Q8 | llama.cpp Q8_0 NEON | **593** @6t (488 / 593 / 483) | **5,423** |
| | nanoinfer q8-prefill | 585 @10t (472 / 580 / 585) | 3,548 |
| Q4 + Q8 head | llama.cpp Q4_0 NEON | **753** @6t (710 / 753 / 638) | **5,683** |
| | nanoinfer q4-kv16 | 742 @10t (638 / 703 / 742) | 2,274 |
| Q4 (Q6_K head) | llama.cpp Q4_0 default | 800 @6t | 5,697 |

The Accelerate build is slower than pure NEON everywhere except f32 prefill
(f16 prefill drops to 540–1,360: it converts to f32 for sgemm) and its decode
is 2–10% lower; it is in the JSON, not in this table.

**Quality** (WikiText-2 test). Each engine against its *own* f32 model, because
the tools score different windows (absolute ppl isn't comparable across them):
nanoinfer `ppl` and `ppl_mlx.py` use 4 × 512 windows, every position (MLX f32
ppl 37.3775 vs ours 37.3776, a good sign the two agree); llama-perplexity
scores the second half of 8 × 512 chunks; the Metal formats come from
`examples/quant_quality.rs` (2 × 512, on the GPU kernels).

| format | engine | Δppl vs own f32 | mean KL | top-1 agreement |
|---|---|---|---|---|
| f16 | nanoinfer CPU / Metal | +0.03% / +0.01% | 1.3e-5 / 1.1e-5 | 99.8% / 99.7% |
| | llama.cpp | +0.02% | 3e-6 | 99.8% |
| | MLX | +0.06% | 1.6e-4 | 98.8% |
| Q8 (Q8 head) | nanoinfer Metal | −0.1% | 9.4e-3 | 90.1% |
| | llama.cpp Q8_0 | +0.4% | 1.3e-2 | 90.3% |
| | MLX 8-bit | +1.2% | 1.2e-2 | 89.8% |
| Q8 (rescored head) | nanoinfer CPU q8-kv16 | +0.04% | 2.0e-3 | 98.1% |
| Q4 + Q8 head | nanoinfer Metal | +6.7% | 7.4e-2 | 80.5% |
| | llama.cpp | +8.3% | 8.8e-2 | 79.7% |
| | MLX 4-bit + 8-bit head | +5.5% | 7.6e-2 | 81.4% |
| Q4 (rescored head) | nanoinfer CPU q4-kv16 | +7.9% | 8.1e-2 | 83.1% |
| Q4, Q6_K head | llama.cpp default Q4_0 | +14.2% | 0.13 | 75.2% |
| Q4, 4-bit head | MLX default 4-bit | **+733% (ppl 311)** | 2.2 | 37.4% |

**Findings.**

- **GPU decode: nanoinfer is fastest at every format below f32.** Q4 + Q8 head
  787 vs MLX 694 (+13%) vs llama.cpp 622 (+27%); Q8 612 vs 534 / 502 (+15% /
  +22%); f16 364 vs 342 / 333. Our WebGPU path (portable WGSL through wgpu)
  beats both native engines at Q8 too (569). The reason is the one the GPU
  climb kept finding: per-token fixed cost. One command buffer per token, GPU
  argmax, no CPU wait between tokens (metal-pipelined) and 8 lanes per row on
  the narrow matrices (metal-lanes) leave nothing between kernels; llama.cpp
  waits for every token on the CPU (llama-bench syncs per decode), and on a
  model this small that gap is a large part of a 1.3–1.6 ms token. It also
  makes llama.cpp's decode depend on GPU clock state (the bimodality above).
- **MLX's headline 4-bit number is a broken model.** Its default 4-bit
  conversion quantises GPT-2's tied embedding / LM head to 4 bits: 750 tok/s,
  ppl 37 → 311, top-1 37%. The same lesson as our stage 5 (the head is the
  fragile matrix). With an 8-bit head it is 694 tok/s, below metal-lanes.
  llama.cpp's default Q4_0 sidesteps this with a Q6_K head (+14% ppl) and
  is 5% faster than with Q8_0, still 17% behind metal-lanes.
- **Where we lose on the GPU: prefill, by 1.6–1.9×.** llama.cpp runs a
  64-token prompt at 10.3–11.1k tok/s in every format; we do 4.5–6.7k (MLX
  3.7–7.8k). Its `mul_mm` kernels stage both operands as half
  (`simdgroup_half8x8`, f32 accumulators), even for F32 weights, so its "f32"
  prefill is really f16 math; ours run f32 `simdgroup_float8x8` at ~1.2
  TFLOP/s on this shape
  (the "try simdgroup_half8x8, double-buffer K" item from the GPU summary is
  still open). At P = 64, prefill is ~6 ms for everyone, so it hardly matters
  for chat, but it is a clear, known gap.
- **Where we lose on the GPU: f32 decode, by 8%** (176 vs 192). metal-mma is
  the only f32 Metal stage, and it predates pipelining and lanes (every
  later trick went into the quantised path).
- **CPU decode: llama.cpp wins at its best thread count; we are 1–16% slower.** Q8 593
  vs 585 (a tie), Q4 + Q8 head 753 vs 742 (−1.5%), f16 358 vs 328 (−8%), f32
  197 vs 165 (−16%). The quantised gap is about the bytes: our CPU blocks keep
  an f32 scale (9 / 5 bpw), ggml's an f16 one, so we stream 6–9% more per
  token. f32 and f16 are older stages that never moved onto the low-latency
  pool (stage 11) that the quantised stages use. At 10 threads the order
  flips: llama.cpp drops with the E-cores (Q4 638, Q8 483) while our pool holds
  (742, 585).
- **CPU prefill: llama.cpp wins by 1.5–2.5× on quantised formats** (Q8 5.4k vs
  our 3.5k, Q4 5.7k vs 2.3k). It repacks Q8_0/Q4_0 weights into interleaved
  `q8_0_4x8` tiles at load time and runs them through the M4's int8 matrix
  instructions (i8mm `smmla`, 2× the MACs of `sdot`); our q8-prefill uses
  `sdot`, and q4-kv16 dequantises to f32 for prefill. Ours wins f32 prefill
  against pure-NEON llama.cpp (2.2k vs 1.2k, at 10 threads), not against the
  Accelerate build (2.7k on 4 threads, through AMX).
- **Quality is a wash when the formats match.** Q8 with a Q8 head flips ~10%
  of greedy picks in all three engines; Q4 + Q8 head costs +5.5–8.3% ppl and
  ~80% top-1 in all three. Our CPU rescored head is the best Q8 on the board
  (+0.04%, 98% top-1) at ~6% more bytes than all-Q8.
- **PyTorch MPS** (eager HF transformers) is the slowest decode at f16 / f32
  (252 / 156): it's the reference for "no inference engine at all".
- **candle: not run.** candle-transformers has no GPT-2 (the closest,
  `bigcode`, is GPT-BigCode with multi-query attention, a different model), so
  there is no like-for-like candle number without writing the model ourselves.

**Verdict.** For one user decoding on this M4's GPU, the from-scratch engine
is the fastest of the four at f16, Q8 and Q4, at the same bytes and the same
quality, and uses the most of the memory bandwidth (59–75% of spec vs
llama.cpp's 46–69%). It is slower to read a prompt on both GPU and CPU, slower
at f32 on the GPU, and slightly slower than llama.cpp's NEON kernels at CPU
decode. All of this is GPT-2 small, where fixed per-token costs dominate, which
favours our design; on a 7B model the matvecs would be bandwidth-bound for
everyone and the gaps would shrink.

To reproduce: `scripts/headtohead/setup.sh` (downloads, converts, builds
llama.cpp CPU-only into `~/.cache/nanoinfer-h2h`), then `run.sh`
(`ROUNDS=3`), `warm_check.sh`, `ppl.sh`, and
`aggregate.py ~/.cache/nanoinfer-h2h bench/headtohead.json`.

## Correction: q8-prefill's bytes per token (19 Oct)

`Stage::head_format` fell through to f16 for q8-prefill, so its bench lines
reported 173 MB per token and a 693 tok/s spec ceiling. q8-prefill uses the
same rescored head as q8-kv16 (Q8_0 approximate, f16 only for candidates), so
it streams **139.5 MB** and its spec ceiling is **860 tok/s**. The stage 15
entry's "58% of 693" is really ~47% of 860 (the CPU-climb table already used
139 MB, and `bench/headtohead.json` was corrected by hand). The measured tok/s
were never affected, only the ceiling and percentages derived from it.

Fix: every CPU stage's properties now come from one table (`Stage::spec`), and
the weight and head formats are derived from the packed weights the session
actually builds. A unit test checks `decode_bytes` against the packed matrices'
real `bytes()` for every stage, so this can't drift silently again.
`bench/results.jsonl` lines recorded before the fix keep the old figure.

## Metal f32 on the pipelined path, and what streaming costs (19 Oct)

Two things from the engine review (`docs/review/04-engine.md` §2.5 item 8, §3.2e).

**Conditions.** Clean engine tree (records `2e20ebc`/`de3b77a`, `git_dirty: false`), load 1.4–2.9, AC power, a
video encoder (screen sharing) running in the background. metal-lanes ran at 755–760 tok/s pipelined, against 787
in the head-to-head, so absolute numbers here are ~3–4% low; compare within this session. Every row is the best of
2–3 rounds of 7 runs (+2 warmup), stages strictly one at a time.

### metal-pipelined-f32

- **Hypothesis:** metal-mma (the f32 row of the head-to-head, 176 vs llama.cpp's 192) predates GPU argmax, epilogue
  fusion and pipelining. The f32 fused-epilogue kernels already exist, so f32 on the pipelined path should gain
  what Q4 gained there.
- **Change:** one recipe line, `Recipe { dtype: F32, head_dtype: F32, attn: Flash, ..pipelined }` (f32 KV; lanes
  don't apply to f32). Covered by `MetalStage::ALL` in every Metal reference test.
- **Result** (interleaved A/B, 3 rounds): metal-mma **167.7–168.5** tok/s (69% of its 243 tok/s ceiling, 135
  dispatches per token, GPU busy 93% of the wall); metal-pipelined-f32 **185.0–185.8** (76%, 101 dispatches, GPU
  busy 100%). **+10%.** Prefill also moves a little (4.5k → 4.7k) from flash attention.
- **Against llama.cpp:** its 192 was measured in the head-to-head session, where metal-mma made 176. Scaling by
  today's deficit (176 / 168.5) puts metal-pipelined-f32 at ~194 in those conditions (est.), level with llama.cpp.
  The randomised head-to-head re-run will settle it.

### Streaming decode: pipelined isn't what a chat server sees

Every metal-* decode number so far is **pipelined**: all 64 command buffers are committed up front and each forward
reads the token the previous argmax wrote on the GPU, so the CPU never waits. That is a fair "what the GPU can do",
but a server has to read every token back (to stream it, detokenise it, check for a stop) before it knows what comes
next. `MetalSession::decode_stream(first, n, depth, on_token)` now does that, and `bench` reports it for pipelined
stages:

- **streaming** (depth 1): commit token t, wait, read the 4-byte argmax back, detokenise, then encode t+1. Per-token
  sync, as llama-bench does (per the review; not verified in its source).
- **lookahead** (depth 2): commit t+1 before reading t back, so the GPU never idles while the CPU reads and encodes.
  MLX's async eval does the equivalent. A server can do this; a stop is noticed one token late and the in-flight
  forward is rolled back (tested: stopping after 5 tokens and resuming reproduces the greedy sequence).

| stage | pipelined | streaming (depth 1) | lookahead (depth 2) | MLX | llama.cpp |
|---|---|---|---|---|---|
| metal-pipelined-f32 | 185.8 | 173.5 (93%) | 184.5 (99%) | 178 | 192 |
| metal-pipelined-f16 | 347.2 | 304.8 (88%) | 339.0 (98%) | 342 | 333 |
| metal-pipelined-q8 | 589.0 | 485.8 (82%) | 576.9 (98%) | 534 | 502 |
| metal-lanes (Q4 + Q8 head) | 759.8 | 593.2 (78%) | 759.8 (100%) | 694 | 622 |

(tok/s; MLX and llama.cpp columns are the 18 Oct head-to-head, not this session.)

- **Per-token sync costs a flat ~0.37 ms per token** at every format (1/593 − 1/760 = 0.37 ms; f32: 0.38 ms): the
  CPU encoding ~100 dispatches after the readback while the GPU sits idle, plus commit-to-completion latency. The
  review guessed ~130 µs and ~715 tok/s; it's nearly three times that, so the hit is 22% at Q4.
- **Honest consequence:** with per-token sync this engine is *behind* llama.cpp at every format (593 vs 622 at Q4,
  486 vs 502 at Q8, 305 vs 333 at f16) and behind MLX. Its decode lead over both holds only for a loop that keeps
  one token in flight, and with that loop it holds in full: lookahead keeps 98–100% of pipelined.
- **So the server design is settled:** depth-2 lookahead, not a sync loop. And llama.cpp encodes a token's work
  faster than we do, which is worth a look (it splits encoding across threads; unverified here).

### Browser recording re-made on a quiet machine (19 Oct)

`node web/scripts/record.mjs` (relaxed build of the current engine, Node 25.5 / V8 14.1, best of 4 interleaved
rounds, load 1.9–3.0, nothing else benchmarking). Roofs: 71 GB/s, 110 GFLOP/s (18 Oct, shared machine: 59, 97).
Decode / prefill tok/s: unrolled 111.7 / 179, fast-ops 118.9 / 201, wasm-simd 111.4 / 285, wasm-fma 113.1 / 422, f16
14.4 / 165, **q8 25.7 / 196, q4 34.6 / 190** (q8-kv16 25.6, q4-kv16 34.6, q8-prefill 25.6 / 56).

The quantised rungs are a **regression in wasm** against the "as they landed" numbers above (q8 36–38, q4 91–106).
Decode on stages 9+ is int8 activations × int8 weights, and off aarch64 that dot is the scalar fallback
(`dot_block_scalar` in `kernels/q4.rs`, likewise `q8.rs`), so wasm pays a scalar multiply-add per weight. The older
numbers came from an earlier build of those stages; the cause is not bisected. Fix: a simd128 int8 dot
(`i16x8_extmul` + `i32x4_dot_i16x8`) in the `cfg(not(target_arch = "aarch64"))` paths. Not done here.
