# DISTILLED: the nanoinfer reference

Distilled from [01](01-engines-and-kernels.md), [02](02-serving-and-distributed.md), [03](03-explorable-writeup-and-wasm.md) and [WRITEUP](../docs/WRITEUP.md). Target: GPT-2 124M on a Mac mini M4 (base: 120 GB/s, 4P+6E, 24 GB, no throttling). "est." means napkin math.

## 1. The 10 laws

1. **Decode tok/s ≤ bandwidth ÷ bytes per token.** At batch 1 each weight is read once for ~2 FLOPs, so an f32 matvec runs at 0.5 FLOP/B. The M4 ridges are far higher (GPU ~36, SME ~17, NEON ~6 FLOP/B). Decode is memory-bound everywhere.
2. **GPT-2 ceilings at 120 GB/s** (weights only, est.): **f32 498 MB → 241 tok/s · f16 249 MB → 482 · Q8_0 132 MB → ~900 · Q4_0 70 MB → ~1700.** Expect ~75–80% of these: llama.cpp gets ~92 GB/s on this chip ([#4167](https://github.com/ggml-org/llama.cpp/discussions/4167)). The ceiling's denominator is our measured STREAM figure.
3. **Prefill is compute-bound.** P tokens share one weight read, so intensity ≈ 2P/bytes_per_weight, which crosses the GPU ridge at P ≳ 36–72. Ceiling = FLOPs/(2·params): ~17k tok/s at 4.3 TFLOPS, ~8k SME, ~2.5k NEON (est.).
4. **The tied LM head is 31% of the bytes.** It reads all of wte (50257×768) every token. Pad the vocab to 50304.
5. **The KV cache turns O(n²) into O(n).** It costs 2·L·C·bytes = **73.7 KB/token f32 (75 MB at 1024), 37 KB f16.** That is more than the Q4 weights at full context. f32 at pos 1024 reads ~573 MB/token, so the ceiling drops to ~209 tok/s.
6. **The small-model twist: overhead dominates.** At 1000 tok/s a token gets <1 ms for ~125 ops, ~7 µs each. Fork/join, Metal commit/wait and WebGPU dispatch (24–71 µs each, [one paper](https://arxiv.org/abs/2604.02344)) dominate. Fusion pays at batch 1 because it cuts dispatches; the bytes it saves barely matter.
7. **Every speed-up has a quality bill.** Track logit diff, top-1 agreement, greedy equality, KL and Δppl. Andrew Chan's atomicAdd flushed subnormals and raised ppl ~5× ([yalm](https://andrewkchan.dev/posts/yalm.html)).
8. **Batching raises matmul intensity ≈ batch size** (H100 crosses at ~280–300). Decode attention stays memory-bound because each sequence has its own KV. So serving is limited by KV memory, not FLOPs ([scaling book](https://jax-ml.github.io/scaling-book/inference/)).
9. **Speculative decoding exploits memory-boundness.** The target verifies γ+1 tokens for about the cost of one. E[tokens] = (1−α^{γ+1})/(1−α); speedup = that/(γc+1). α=0.8, γ=4, c=0.05 gives ~2.8×. Accepting with min(1,p/q) keeps the output exactly p. Batch-1 M4 is ideal for it.
10. **Bandwidth-bound collective time is independent of N.** Ring: T = 2(N−1)α + 2(N−1)/N·S/B. Rings win for large messages, trees for small ones. GPT-2 TP decode does 24 × 1.5 KB all-reduces per token, which is pure latency, so **TP makes GPT-2 small slower**. HBM : NVLink : PCIe/IB ≈ 3350 : 450 : 50 GB/s.

## 2. Master hillclimb ladder

Where: **M** Mac, **B** browser (B\* = needs isolation for threads), **R** rented GPU. Effects are est.

| # | Step | What changes | Expected effect | Where | Viz | Crib |
|---|---|---|---|---|---|---|
| 0 | Harness | HF fp32 layer dumps, logit/KL/greedy tests, WikiText-2 ppl, STREAM probe, llama.cpp GPT-2 baseline | defines correct + ceiling | M | V2 V3 | [llm.c](https://github.com/karpathy/llm.c) test_gpt2.c |
| 1 | Naive f32, no cache | mmap safetensors, preallocated state, `iter().sum()`, recompute context | baseline; slows with position | M B | V5 | [llama2.c](https://github.com/karpathy/llama2.c) |
| 2 | KV cache | append K/V, attend over cache | flat per-token cost | M B | V5 | llama2.c |
| 3 | Layout + compiler | Conv1D→(out,in), `chunks_exact`, 8 accumulators, target-cpu, LTO | 2–4× | M B | V6 | [Boehm CPU](https://siboehm.com/articles/22/Fast-MMM-on-CPU) |
| 4 | Explicit SIMD | NEON `vfmaq_f32`; wasm simd128 (+relaxed FMA) | 1.2–2× | M B | V6 | `std::arch` |
| 5 | Threads | rayon rows + heads → persistent spin pool, P-core QoS | 2–3×, wall at ~241 | M B\* | V7 | ggml |
| 6 | Fusion/housekeeping | bias+GELU, bias+residual, zero allocs, padded LM head | 1.05–1.2× | M B | V4 | llm.c |
| 7 | f16 weights | `half`, fp16 FMA, f32 accumulate | ~1.8× (300–400) | M B | V3 V8 | yalm |
| 8 | Q8_0 + Q8 activations | block-32 int8, `vdotq_s32` | 1.6–1.9× (500–700) | M B | V8 | [ggml-common.h](https://github.com/ggml-org/llama.cpp), runq.c |
| 9 | Q4_0 → Q4_K | nibble unpack; LM head at Q6/Q8 | 1.3–1.6× (700–1100); overhead-bound | M | V8 V4 | [candle quantized](https://github.com/huggingface/candle) |
| 10 | CPU prefill GEMM | batched prompt, register tiling; Accelerate sgemm line | prefill 10–50× | M B | V9 | llm.c, Boehm |
| 11 | Metal naive | thread per row | slower than CPU (lesson) | M | V4 V9 | yalm |
| 12 | Metal SIMD-group matvec | coalesced, `simd_sum` | ~18× (Chan) | M | V3 | llama.cpp `mul_mv` |
| 13 | One cmd buffer, zero-copy | `bytesNoCopy` mmap, single encoder | large at quant tiers | M | V4 | llama.cpp Metal |
| 14 | GPU dtypes | f16/Q8/Q4 dequant in registers | 600–1300 | M | V3 | llama.cpp Metal |
| 15 | GPU prefill | `simdgroup_matrix` GEMM, flash prefill | 30–50% of peak | M | V9 | [FA2](https://arxiv.org/abs/2307.08691) |
| 16 | Long context | f16 KV, split-KV flash-decoding | flat vs position | M | V10 | [Flash-Decoding](https://pytorch.org/blog/flash-decoding/) |
| 17 | WebGPU | wgpu + WGSL, u32-packed quant, fuse dispatches | ~80% native ([WebLLM](https://arxiv.org/abs/2412.15803)) | B M | V9 V4 | web-rwkv, ratchet |
| 18 | Serving harness | TTFT/TPOT/ITL/MBU, Poisson loadgen | judge for serving | M | V11 | [Databricks](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) |
| 19 | Continuous batching | static [B,S] → ragged Orca batch + SSE | ~10× throughput | M | V11 | [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) |
| 20 | Paged KV + prefix cache | 16-token blocks, refcount, CoW; hashed blocks → radix | waste <4%; TTFT drop | M | V12 | [vLLM](https://arxiv.org/abs/2309.06180), [SGLang](https://arxiv.org/abs/2312.07104) |
| 21 | Chunked prefill | token budget per iteration | ITL p99 down | M | V11 | [Sarathi](https://arxiv.org/abs/2403.02310) |
| 22 | Speculative | small drafts large/XL; n-gram | 2–3× | M | V13 | [Leviathan](https://arxiv.org/abs/2211.17192) |
| 23 | Collectives | `Communicator` trait (shm/TCP), ring/tree/all_to_all, link emulator | reproduce 2(N−1)/N | M | V14 | [nccl-tests](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md), [MLX](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) |
| 24 | TP, PP | Megatron col/row, KV by head; layer split | TP hurts small model | M | V14 | [Megatron](https://arxiv.org/abs/1909.08053), [candle multiprocess](https://github.com/huggingface/candle/tree/main/candle-examples/examples/llama_multiprocess) |
| 25 | Toy disaggregation | prefill/decode procs, KV over TCP | KV cost × prompt | M | — | [DistServe](https://arxiv.org/abs/2401.09670) |
| R1 | ~$5, 2–4× 4090 | cudarc NCCL, nccl-tests, TP=2 GPT-2 XL | PCIe-bound TP | R | V14 | [cudarc](https://github.com/coreylowman/cudarc) |
| R2 | ~$30–60, 8×H100 | NVLink curves, TP 1–8 on ~7B, batching sweeps | TP helps | R | V14 V11 | [Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) |
| opt | SME / ZeRO-1 | inline-asm FMOPA; CPU DP training | ~2 TFLOPS | M | — | [m4-sme](https://github.com/tzakharko/m4-sme-exploration) |

**Reconciliations:**
- **KV cache is its own step (2).** 01 folded it into step 1, but WRITEUP and V5 need a cache-off baseline, so the cache stays a runtime knob.
- **Dispatch cost.** 01 assumes one command buffer fixes overhead; 03 cites 32–71 µs per WebGPU dispatch. I treat them as different regimes (native Metal cheap, WebGPU floor ~4–9 ms/token) until measured.
- **Web dtype.** 01 says Q8/Q4, 03 says f16 on desktop. I chose **Q8_0 (~130 MB) by default, f16 opt-in on desktop.** It halves the download, sits under the 128 MiB binding limit and keeps the quant story.
- **Ledger.** 01, 03 and WRITEUP each name a different file (BENCH.md, docs/LOG.md, bench/results.jsonl). Use **`bench/results.jsonl` + `docs/WRITEUP.md`**.
- **Chip.** It is the base M4, not the M4 Pro (02 asked). 03's "recorded · M3" should read M4.
- **Order.** Serving comes after GPU because it needs batched kernels. Spec decoding can move earlier.

## 3. Builder checklist

**Benchmarking**
- [ ] The ceiling is measured STREAM bandwidth at 1, 4 and 10 threads. Report achieved GB/s and % of ceiling (MBU).
- [ ] Report prefill (P=128/512), decode (pos ~1 and ~900) and TTFT separately.
- [ ] Warm up, then N≥10: median with p10/p90. Fixed prompt, greedy. Log commit and flags.
- [ ] Bench the whole model: a single matrix can fit in cache and flatter the bandwidth.
- [ ] E-cores slow static partitions. Use QoS `USER_INTERACTIVE`, work-stealing or uneven chunks.
- [ ] Use a generic `NoopTracer` and publish the tracing overhead.

**Correctness**
- [ ] f32 max-abs logit diff <1e-4, f16 ~1e-2, quant judged by KL. Check top-1 and 64-token greedy equality.
- [ ] Δppl vs **our f32** on WikiText-2 (not the paper's 29.41). Small models are quant-sensitive.
- [ ] Use deterministic reduction order in tests. Spec decoding gets a 1e5-sample histogram test.

**GPT-2 specifics**
- [ ] LayerNorm with bias, eps 1e-5, f32 two-pass stats. **tanh GELU**, not erf. Learned wpe, fused QKV of width 3C, tied head.
- [ ] HF `Conv1D` weights are **(in,out)** (`c_attn` is 768×2304). Transpose at pack time.
- [ ] Apply the −inf mask before the max, subtract the max, f32 softmax.
- [ ] Q4_0 nibbles: low = 0–15, high = 16–31. Use int32 within a block, then an f32 scale.

**Rust perf**
- [ ] `iter().zip().map().sum()` **won't vectorise** (no fast-math). Use multiple accumulators and `chunks_exact`, which also removes bounds checks.
- [ ] No hot-loop allocation. NEON intrinsics are stable; `std::simd` is nightly (`pulp` is the stable alternative). Use `objc2-metal`, not the deprecated `metal`.
- [ ] **Add kernel variants, never overwrite them** (`MatmulVariant`, `kv_cache`, dtype), so the Ladder runs historical code. Structural steps get git tags and are labelled "native only".

**WASM / COOP / COEP**
- [ ] Ship three builds selected at runtime: simd128 single-thread (stable), relaxed-SIMD (Chrome 114+/FF 146+), and threads via wasm-bindgen-rayon (nightly, atomics, `--max-memory` up to 4 GB).
- [ ] Run the engine in a Worker and pass traces as transferable ArrayBuffers.
- [ ] jkmr is a SPA, so set **COOP `same-origin` + COEP `require-corp` site-wide** (`credentialless` isn't in Safari). Threads are an enhancement only.
- [ ] `performance.now()` resolution is ~100 µs non-isolated and ~5 µs isolated. Record `timer_res_us` and aggregate over tokens.

**WebGPU**
- [ ] `maxStorageBufferBindingSize` defaults to 128 MiB, and f32 wte (154 MB) exceeds it.
- [ ] Pack int8/int4 into u32. Feature-detect `shader-f16`, subgroups and `timestamp-query`. Count dispatches.

**Weights**
- [ ] Use 16–32 MB pre-transposed shards plus a `manifest.json` (sha256, offsets). Never ship f32.
- [ ] Host on HF Hub with whole-shard GETs (**Range fails CORS**); R2 is the fallback, Vercel static is not. Use the Cache API with `storage.persist()`.
- [ ] **Never auto-download.** Replay a recorded trace until "Load GPT-2 · int8 · 130 MB · runs locally".

## 4. Learner-copy bank

- **[Intro]** Each token re-reads the whole 500 MB book of weights to say one word. Speed = reading speed ÷ book thickness.
- **[Intro]** A speed-limit sign before any code: "This car can't exceed 241 tok/s. Let's see how close we get."
- **[Roofline]** Every kernel is waiting either on the truck (memory) or on the factory (compute). Optimising the wrong one does nothing.
- **[KV cache]** Notes in the margin so you don't re-read earlier chapters. At 1024 tokens the notes (75 MB) outweigh the compressed book (70 MB).
- **[Layout]** Reading along the lines versus down the columns. Same words, but one way you turn a page per letter.
- **[Accumulators]** Checkout lanes. Rust won't open extra lanes itself, because adding floats in another order changes the answer.
- **[SIMD]** A paint roller instead of a brush.
- **[Threads]** Ten cooks, one narrow pantry door.
- **[Quantisation]** A compressed book you decompress while reading. Block scales give each paragraph its own volume knob.
- **[Quality]** Every compression comes with an invoice, and a fast kernel can quietly get dumber.
- **[GPU naive]** A thousand workers each fetching from a different shelf. Coalescing means fetching neighbours in one trip.
- **[Unified memory]** No trucking weights to the GPU's warehouse: you just hand over the address.
- **[Dispatch]** A manager who spends longer assigning tasks than the tasks take.
- **[Prefill vs decode]** Prefill reads the book once for 500 questions; decode re-reads it for each one.
- **[Flash attention]** Keep a running leaderboard instead of every score. Flash-decoding splits it across judges, then merges.
- **[Continuous batching]** A bus that never waits at the terminal.
- **[PagedAttention]** Hand out 16-token pages on demand instead of a whole notebook per guest. The block table is the index card.
- **[Spec decoding]** An intern drafts and a senior signs off. Checking five words costs the same as writing one.
- **[Ring all-reduce]** Passing dishes round a table. No link is idle, so time barely depends on table size.
- **[TP vs PP]** Chefs syncing on one dish (they need an intercom) versus an assembly line that takes time to fill.

## 5. Visualisation shortlist

All figures share one engine run, fall back to a labelled recorded trace and show provenance.

| # | Viz | Reader does | Real data | Priority |
|---|---|---|---|---|
| V1 | **Run it**: token stream + probabilities | types a prompt, hovers for top-10; temperature/top-k re-sample cached logits | `token` (topk, logprob), `resample()` | **hero** |
| V2 | **The Ladder** (sticky) | adds their browser's dot per step beside Josh's series under the speed-of-light line; clicks to jump | `results.jsonl` + live `run_end` | **hero** |
| V3 | **Speed-of-light gauge** | picks dtype/context; ceiling from live bandwidth probe; needle = achieved | `bench_bandwidth`, `run_end` | core |
| V4 | **Where a token's time goes** | hovers ops in per-step flame strips; watches overhead become the long pole | `op` events | core |
| V5 | **KV cache on/off** | toggles cache, drags context; latency line plus filling KV grid | `token.dur_us`, `kv` | core |
| V6 | **Loop order / tiling / SIMD** | access animation with miss counter; tile slider on the real kernel | `bench_matmul` | core |
| V7 | **Thread wall** | thread slider; tok/s flattens (recorded if not isolated) | `run_end` per count | core |
| V8 | **Quantisation microscope** | real weight histogram, int8 grid, block-size slider; f16 vs int8 KL | `read_tensor`, two runs | core |
| V9 | **Prefill vs decode / race** | prompt-length slider; WASM vs WebGPU, step 1 vs N race | `op`, `run_end` | core |
| V10 | **Attention heads** 12×12 | clicks a token; expands a head | `attn_probs` capture | nice |
| V11 | **Batching Gantt + frontier** | static vs continuous toggle; goodput box | serving traces | nice (post 3) |
| V12 | **Paged KV block table** | watches allocate/share/CoW | block events | nice (post 3) |
| V13 | **Spec playback** | γ slider; green/red tokens; measured vs formula | spec traces | nice (post 3) |
| V14 | **Ring all-reduce + TP** | steps phases; size slider shows ring/tree crossover | `Communicator` timings, nccl-tests | nice (post 4) |

Post 1 needs V1, V2, V4, V5 and V6. In jkmr, add `live` fences to `toSegments` plus `<live-num>` portals rather than adopting MDX.

## 6. Facts to verify before publishing

- Measured M4 CPU and GPU bandwidth; GPU ~4.3 TFLOPS; FP16 rate = FP32 on Apple GPUs; NEON ~140 GFLOPS/core; SME ~2 TFLOPS shared per cluster.
- Every ceiling and ballpark in §1–2 (est.); the llama.cpp #4167 figures (2023 build). Rerun llama-bench on GPT-2.
- WebGPU dispatch 24–71 µs ([arXiv 2604.02344](https://arxiv.org/abs/2604.02344)) and whether one submit amortises it; WebLLM's 80%.
- Browser support for WebGPU, subgroups, relaxed-SIMD, `shader-f16` and `timestamp-query`; single-thread WASM at ~single digits to 20 tok/s; iOS memory limits.
- HF cross-origin GETs from production, the Range/CORS failure, Vercel limits, the jkmr domain for R2 CORS, and that site-wide COEP doesn't break OG images.
- That `std::simd` is still nightly, that rustc accepts `apple-m4`, and that `NoopTracer` has zero overhead.
- All [R] paper figures: Orca 36.9×, vLLM 20–38%/<4%/2–4×, SGLang 6.4×, DistServe 7.4×, Splitwise, Medusa, EAGLE, EAGLE-3 (6.5× vs 1.5–2× independent), and the Chan and Boehm ladders.
- Link bandwidths, NCCL small-message latency, H100 295 FLOP/B; llm.c $20/$672, nanochat $100, GPU prices, Modal credit; the Playbook table (recalled; it didn't render).
- Quant lore (Q8_0 near-lossless, Q4_K_M close): measure on GPT-2.
