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
