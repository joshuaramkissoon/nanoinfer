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
