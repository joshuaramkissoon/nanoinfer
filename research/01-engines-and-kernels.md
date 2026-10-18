# 01: Single-node inference engines and kernels

Research notes for nanoinfer: a GPT-2 124M forward pass in Rust, hillclimbed from naive f32 on an M4 (24 GB) to a fast CPU, Metal and WebGPU engine.
Compiled 2026-10-18. Numbers marked **(est.)** are my own back-of-envelope figures, not measurements. Numbers marked **(unverified)** came from secondary sources I could not confirm.

---

## 0. The one idea that organises everything

At batch size 1, generating a token means **reading every weight once** and doing about 2 FLOPs per weight. A matrix-vector product does 2 FLOPs per weight it loads, so in f32 that is 2 FLOPs per 4 bytes, an arithmetic intensity of 0.5 FLOP/byte. Every modern chip can do 10 to 100 times more arithmetic than that per byte of memory traffic. **Decode is memory-bound.** Its speed limit is:

```
max decode tok/s ≈ memory bandwidth (B/s) / bytes read per token (weights + KV cache)
```

Prefill (processing the prompt) handles P tokens per weight read. Its intensity is about 2P/bytes_per_weight, so for P in the hundreds it is **compute-bound**. Almost every optimisation in this document either moves fewer bytes per token (quantisation, fusion, better layouts) or gets closer to peak bandwidth or peak FLOPs (SIMD, threads, GPU, better kernels). Andrew Chan, Simon Boehm and llama.cpp all work from this frame.

---

## 1. Resources, distilled

### 1.1 Karpathy, llama2.c (`run.c`, `runq.c`)
https://github.com/karpathy/llama2.c (I read the source.)

- **Shape of `run.c`.** About 970 lines and one file. It contains `Config`, a `TransformerWeights` struct of raw pointers into one mmap'd blob, a `RunState` with every scratch buffer preallocated (`x, xb, xb2, hb, hb2, q, k, v, att, logits, key_cache, value_cache`), plus `rmsnorm`, `softmax` and `matmul`. `forward(transformer, token, pos)` runs **one token at a time**, so prefill is just decode in a loop. A sampler and a BPE tokenizer complete the file. This is the right skeleton for nanoinfer step 1. There are **no allocations in the hot loop**, and every tensor is a slice into one buffer.
- **mmap weights.** `read_checkpoint` calls `mmap(..., PROT_READ, MAP_PRIVATE)` and points the weight structs straight into the mapping. Load time is close to zero, and the OS page cache handles residency. Borrow this with `memmap2`.
- **matmul is a matvec**: `W (d,n) @ x (n,) -> xout (d,)`, row-major, with `#pragma omp parallel for` over output rows. Parallelising over output rows means threads never share an accumulator, so no reduction step is needed.
- **Numbers.** stories15M runs at about 110 tok/s on an M1 Air. Llama-2 7B runs at about 30 s/token on M1 single-thread fp32. On a 64-thread box, 7B ran at **4.6 tok/s in fp32 and 14 tok/s with int8 `runq.c`**: about 3x faster and 26 GB→6.7 GB, roughly 4x smaller. The speedup tracks the byte reduction, which is the memory-bound thesis in action.
- **`runq.c` is "Q8_0" with a configurable group size.** `export.py` defaults to `group_size=64` and halves it until it divides `dim`. Weights are symmetric int8 in [-127,127] with one f32 scale per group. **Activations are quantised dynamically** before each matmul, so the inner loop is int8×int8 accumulated in int32 per group, then multiplied by `w.s * x.s`. Norm weights stay f32. The classifier/embedding is quantised too.
- **Thread advice.** Use physical cores, not SMT. More threads is not always faster once bandwidth saturates.
- **Why it is the ideal baseline.** You can read the whole engine in one sitting, every buffer is visible, and each later optimisation is a diff against it.

### 1.2 Karpathy, llm.c (GPT-2)
https://github.com/karpathy/llm.c (`train_gpt2.c`, `test_gpt2.c`)

- **This is GPT-2, which matters for us.** GPT-2 uses **LayerNorm with bias (eps 1e-5)**, **learned absolute position embeddings (wpe)**, **GELU with the tanh approximation** (`0.5x(1+tanh(√(2/π)(x+0.044715x³)))`), biases on every linear layer, a fused **QKV projection of width 3C**, and a **tied LM head** where logits = `lnf(x) @ wteᵀ`. It has no RoPE, no RMSNorm and no SwiGLU. Those only arrive if we move to a Llama-family model later.
- **Parameter layout.** All 16 parameter tensors sit in one contiguous float buffer, with per-layer tensors stacked along a leading L dimension (`qkvw: L×3C×C`). Activations get a second single buffer.
- **Vocab padding.** 50257 is padded to **Vp = 50304** (a multiple of 128) for alignment, and softmax runs over real V only. That is worth copying: padding the LM-head rows makes SIMD and GPU tiling clean.
- **`matmul_forward` trick.** It collapses B·T into one dimension, steps by `LOOP_UNROLL=8`, and keeps 8 accumulators so each weight load is reused across 8 tokens. That is register tiling in its simplest form. It is a prefill (GEMM) optimisation, not a decode one.
- **Numerics gotcha.** `-Ofast` broke GeLU (issue #168), so they wrapped it in precise float pragmas. Fast-math flags can silently change results. Always diff against the reference.
- **The test harness is the thing to steal.** `test_gpt2.c` loads a "debug state" file that PyTorch exported (inputs, expected logits, loss) and checks with tolerances. We should export per-layer activations from HF GPT-2 once and diff every engine variant against them.

**Gotcha for loading HF GPT-2 weights.** HF's GPT-2 uses `Conv1D`, which stores weights as **(in, out)**, the transpose of `nn.Linear`. `c_attn.weight` is `768×2304`. Decide early whether to transpose at load time so the matvec reads contiguous rows. Read contiguous rows: for `y = Wx`, store W as `(out, in)` row-major so each output is a dot product over a contiguous row.

### 1.3 llama.cpp / ggml
https://github.com/ggml-org/llama.cpp (I read `ggml/src/ggml-common.h` for the block structs.)

**Quant formats.** They are blockwise, and the scales live inline with the data so one memory stream feeds the kernel:

| Format | Block | Layout | Bits/weight | Dequant |
|---|---|---|---|---|
| Q8_0 | 32 | `f16 d; int8 qs[32]` = 34 B | 8.5 | `x = d·q` |
| Q4_0 | 32 | `f16 d; u8 qs[16]` = 18 B | 4.5 | `x = d·(q−8)` |
| Q4_1 | 32 | `f16 d, m; u8 qs[16]` | 5.0 | `x = d·q + m` |
| Q4_K | 256 super-block = 8×32 | `f16 d, dmin; u8 scales[12]` (6-bit scale + 6-bit min per sub-block); `u8 qs[128]` = 144 B | 4.5 | `x = d·sc_j·q − dmin·m_j` |
| Q6_K | 256 = 16×16 | `ql[128], qh[64], int8 scales[16], f16 d` | 6.5625 | `x = d·sc_j·q` |
| Q8_K | 256 | `f32 d; int8 qs[256]; int16 bsums[16]` | (activations only) | `bsums` let the kernel apply the K-quant "min" term with one multiply per sub-block |

- **K-quants put a quantised scale on each sub-block.** Two levels of scales give fine granularity at about 0.5 bits/weight of overhead. "Q4_K_M" is a *mix*: sensitive tensors such as attn_v, ffn_down and the output layer get Q6_K or Q5_K, and the rest get Q4_K.
- **Nibble-order gotcha in Q4_0.** Low nibbles hold elements 0..15 and high nibbles hold 16..31. They are not interleaved pairs. Get this wrong and you get garbage that still looks plausible.
- **The real CPU trick is quantising activations as well**, to Q8_0 or Q8_K, per matmul call. The dot product then becomes int8·int8→int32 using NEON `sdot` (`vdotq_s32`, 16 MACs per instruction) or `smmla` (i8mm). Unpacking nibbles takes `vandq_u8(q, 0x0F)` and `vshrq_n_u8(q, 4)`, then subtracting 8.
- **mmap by default.** The GGUF tensor data is aligned so a mapped file can be used in place. On Metal, the mapped pages are wrapped as `MTLBuffer`s with `newBufferWithBytesNoCopy`, so CPU and GPU share the same physical pages with zero copies. This is the unified-memory payoff.
- **Threading model.** ggml splits each op across N threads by rows (`ith/nth`), with a **spin-wait barrier between ops**. The threads persist, so there is no per-op spawning. That matters because an op on a small model takes microseconds. See section 3: rayon fork/join per op may cost a noticeable fraction of our budget.
- **Metal kernels.** There are two families. `mul_mv_*` handles decode: matrix-vector, one or more SIMD-groups per row block, a `simd_sum` reduction, dequant in registers. `mul_mm_*` handles prefill: tiled GEMM using `simdgroup_float8x8` / `simdgroup_half8x8` and `simdgroup_multiply_accumulate`, staging tiles in threadgroup memory. The dispatcher switches to `mul_mm` above a small batch threshold.
- **Reference numbers for our exact chip** come from the official Apple Silicon thread (https://github.com/ggml-org/llama.cpp/discussions/4167). That is llama-bench, Llama-2 7B, `-p 512 -n 128`, an old Nov-2023 build. On M4 with a 10-core GPU and 120 GB/s:

  | | F16 | Q8_0 | Q4_0 |
  |---|---|---|---|
  | prefill pp512 (t/s) | 230 | 224 | 221 |
  | decode tg128 (t/s) | 7.4 | 13.5 | 24.1 |

  Read across a row and the pattern is clear. **Decode scales with bytes**: F16→Q4_0 is 3.2x faster. **Prefill barely changes** because it is compute-bound. Q4_0 7B is about 3.8 GB, so 24.1 tok/s ≈ **92 GB/s effective, about 77% of the 120 GB/s peak**. That gives a realistic ceiling fraction for a well-tuned memory-bound kernel on this chip.
- **Use llama.cpp as "the boss".** It can run GPT-2 GGUF. Benchmark it on the same M4 with `llama-bench -m gpt2-f32.gguf/q8_0/q4_0` so every nanoinfer step has a real target line.

### 1.4 Andrew Chan, "Fast LLM Inference From Scratch" (yalm)
https://andrewkchan.dev/posts/yalm.html (Dec 2024. It is the only inference post on his blog. I found no follow-up beyond his reference to zeux's `calm`.)

- **Format.** He states the theoretical max first, then climbs. The setup is Mistral-7B with FP16 weights, a 15.02 GB read per token, an EPYC 7702P at 204.8 GB/s (**13.6 tok/s ceiling**) and an RTX 4090 at 1008 GB/s (**67.1 tok/s ceiling**). Each step is reported against that ceiling.
- **CPU ladder.** Naive single-thread fp32 ran at 0.6 tok/s. OpenMP matmul brought it to 4.2. Parallel attention over heads gave 4.4. FP16 weights with hand-written F16C conversion reached **8.2–8.4**, against llama.cpp's 8.7. Threading delivered the biggest gain, because GCC had already auto-vectorised the fp32 loop. FP16 needed **explicit** SIMD conversion because the compiler would not do it. FP32 accumulation preserved perplexity.
- **GPU ladder.** One thread per row ran at **2.9 tok/s, slower than the CPU** because of uncoalesced reads and idle lanes. One warp per row with a warp reduction jumped to **51.7**. Fusing residual and GLU gave 54.1. Coalesced writes gave 56.1. Shared-memory attention accumulation gave 63.7. An FP16 KV cache with manual unroll and prefetch reached **63.8**, beating llama.cpp's 61.0. Matmuls were about 94% of kernel time, so fusion gains have a ceiling.
- **Gotchas worth retelling.** (1) Global `atomicAdd` in attention **flushed subnormals to zero**, which raised perplexity about 5x on a 4k-token text. Speed changes must come with quality checks. (2) The FP16 KV cache first *regressed* to 53.6, because 16-bit loads only filled 64 B per warp against a 128 B transaction. Naive conversion also stopped the compiler from unrolling and prefetching. Manual unroll with batched register prefetch fixed it, and `#pragma unroll` alone did not.
- **What to borrow.** Open with the roofline ceiling, keep a table of every step as tok/s and % of ceiling, include the regressions, and run a perplexity check alongside each speed number.

### 1.5 Simon Boehm, CUDA matmul and CPU matmul posts
CUDA: https://siboehm.com/articles/22/CUDA-MMM. CPU: https://siboehm.com/articles/22/Fast-MMM-on-CPU

- **CUDA ladder** (A6000, fp32 SGEMM, as % of cuBLAS): naive 1.3%, GMEM coalescing 8.5%, SMEM caching 12.8%, 1D block-tiling 36.5%, 2D block-tiling 68.7%, vectorised loads 78.4%, autotuning 84.8%, warptiling 93.7%. The largest single jumps come from **coalescing** and **register blocking**, where each thread computes many outputs so loaded data is reused.
- **CPU ladder** (i7-6700, N=1024). Naive took 4481 ms. `-O3 -march=native -ffast-math` brought that to 1621 ms. A register accumulator gave 1512. **Loop reorder (i-k-j) gave 89 ms, about 16x and the biggest win, because it made B access sequential and let the compiler vectorise.** L1 tiling gave 70. Multithreading over disjoint output blocks gave 16. MKL took 8.
- **Lessons.** (1) Memory access order beats everything. (2) Flags don't fix a cache-hostile loop. (3) Optimal tile sizes are empirical: his best (~16) was far from the textbook estimate (~3.5). (4) Parallelise over *disjoint outputs* so no reduction is needed.
- **Post format to emulate.** Open with a summary table and napkin math (FLOPs, bytes, lower-bound time), then give one section per kernel: one idea, a code snippet, an access-pattern diagram, the measured number and profiler evidence. Keep negative results in and end with "what I'd try next". The diagrams of which thread touches which memory are what readers remember, and they map directly onto our live-viz plan.

### 1.6 Rust prior art and crates

- **srush/llama2.rs** (https://github.com/srush/llama2.rs). This is the most "performance-Rust" port. It uses 4-bit GPTQ with group size 64 or 128 selected at build time, `portable_simd` (which needs **nightly**), rayon, and memmap2 (a 70B model "loads instantly"). Model dimensions are compile-time cargo features, so loops have constant bounds and the compiler can unroll them and drop bounds checks. Reported speeds on an Intel i9 desktop are about 9 tok/s for 7B quantised and about 1 tok/s for 70B. **Borrow** compile-time shapes, through const generics or features, for the hot kernels.
- **gaxler/llama2.rs** (https://github.com/gaxler/llama2.rs). A straight port: single file, no mmap, optional rayon (`parallel` feature). On a 16-core VM, 110M runs at ~10 tok/s single-thread and ~65 with rayon. That is a useful sanity number for our naive GPT-2 124M step, which is a similar size.
- **candle** (https://github.com/huggingface/candle). HF's Rust tensor library with CPU, CUDA and Metal backends. `candle-core/src/quantized/` is a **Rust port of the GGML k-quant block formats with NEON dot kernels** (`k_quants.rs`, `neon.rs`), so it is the best Rust reference for bit-exact Q4_K and Q8_K packing. CPU GEMM goes through the `gemm` crate. Its Metal backend uses Metal kernels adapted from MLX and llama.cpp. Read it as a reference, but don't depend on it, because the point of nanoinfer is to write the kernels ourselves.
- **mistral.rs** (https://github.com/EricLBuehler/mistral.rs). Built on candle, with serving features: PagedAttention, continuous batching, in-situ quantisation (ISQ) and many model families. It is relevant later, for serving, not for kernels.
- **burn + CubeCL** (https://github.com/tracel-ai/burn, https://github.com/tracel-ai/cubecl). A framework whose GPU kernels are written in Rust (CubeCL) and compiled to WGSL, CUDA, Metal and others. CubeCL is worth a look as an option for "one kernel source for Metal and WebGPU", though hand-written WGSL is more instructive.
- **Crates to use:**
  - `safetensors`: header-parse plus zero-copy views. The data section starts 8-byte aligned, so it is safe to `bytemuck::cast_slice::<u8, f32>`.
  - `memmap2`: mmap the weights.
  - `bytemuck`: zero-cost casts.
  - `half`: f16/bf16, with hardware conversion on aarch64.
  - `rayon` to start, then possibly a hand-rolled spin pool.
  - `std::arch::aarch64`: stable NEON intrinsics such as `vfmaq_f32`, `vld1q_f32`, `vdotq_s32` and `vcvt_f32_f16`. Use `#[target_feature(enable = "neon,dotprod,fp16")]` and `is_aarch64_feature_detected!`.
  - `std::simd`: still nightly-only `portable_simd` as far as I know, so check before committing. `pulp` is a stable safe-SIMD alternative.
  - `tiktoken-rs` or `tokenizers` for GPT-2 BPE.
  - `criterion` for kernel microbenchmarks.
  - `wgpu` for WebGPU and native.
  - **`objc2-metal` for Metal.** The `metal` crate (gfx-rs/metal-rs) is **deprecated**, and its README points to objc2/objc2-metal.
  - For WASM: `wasm-bindgen` and `wasm-bindgen-rayon`.
- **Rust-specific perf gotchas:**
  1. **f32 sum reductions will not auto-vectorise** without fast-math, because FP addition isn't associative and Rust has no `-ffast-math`. A naive `iter().zip().map().sum()` dot product stays scalar. Fix it by hand with 4 to 8 independent accumulators over `chunks_exact(16)`, or with explicit NEON. This is probably the single most surprising step for learners.
  2. Bounds checks in hot loops: `chunks_exact` and slicing to a known length upfront let the compiler prove indices safe.
  3. Build with `-C target-cpu=native`, or `apple-m4` if your rustc knows it. Also set `lto = "fat"`, `codegen-units = 1` and `panic = "abort"` in the release profile.
  4. Avoid `Vec` allocation per token and preallocate a `RunState`, as llama2.c does.

---

## 2. Roofline, the M4, and GPT-2 124M ceilings

### 2.1 M4 (base) numbers

| Resource | Figure | Confidence |
|---|---|---|
| Unified memory bandwidth | **120 GB/s** (LPDDR5X-7500, 128-bit) | High (Apple spec) |
| GPU (10-core) FP32 peak | **~4.3 TFLOPS** (1280 ALUs × 2 × ~1.6–1.7 GHz) | Medium. Sources disagree on clock |
| GPU FP16 | Reportedly the same FLOP rate as FP32 on Apple GPUs. The benefit is bandwidth and registers, not 2x math | Medium (unverified) |
| CPU | 4 P-cores + 6 E-cores (10-core M4) | High |
| NEON FP32 per P-core | ~4 × 128-bit FMA pipes ≈ 32 FLOP/cycle × ~4.4 GHz ≈ **~140 GFLOPS/core** | Low (est.) |
| All-core NEON FP32 | **~0.6–0.8 TFLOPS** | Low (est.) |
| SME (P-cluster matrix unit) | **~2 TFLOPS FP32** measured (LIBXSMM 512² GEMM: 1755–1833 GFLOPS; a microbenchmark paper says >2.3). **One unit shared per cluster**, so it does not scale with core count | Medium |
| Neural Engine | 38 TOPS, not directly programmable. Out of scope | n/a |

Sources: https://en.wikipedia.org/wiki/Apple_M4, https://tnzr.org/sme (Jena SME work), https://github.com/tzakharko/m4-sme-exploration.

**Ridge points**, the FLOP/byte above which a kernel is compute-bound: GPU ≈ 4300/120 ≈ **36 FLOP/B**, NEON ≈ 700/120 ≈ **6 FLOP/B**, SME ≈ **17 FLOP/B**. A decode matvec in f32 sits at 0.5 FLOP/B, f16 at 1, Q8 at ~2 and Q4 at ~3.5. All of them sit far left of every ridge, so **decode is memory-bound on every unit of the M4.** Prefill with P tokens multiplies intensity by about P. On the GPU it crosses the ridge at roughly P ≳ 36–72 tokens.

**Realistic bandwidth.** llama.cpp Q4_0 7B decode reaches about 92 GB/s, roughly 77% of peak. **We don't know how much of the 120 GB/s the CPU cores alone can pull.** Measure it first with a STREAM-style read benchmark in Rust, one thread and then all threads, P-cores versus all cores. Make that number the CPU ceiling.

### 2.2 GPT-2 124M byte budget

- Params ≈ 124.4M. wte is 50257×768 = **38.6M, about 31% of the model, and the tied LM head reads all of it every token**. wpe is 0.8M. The 12 blocks hold about 85M (~7.1M each: QKV 1.77M, proj 0.59M, fc 2.36M, fc_proj 2.36M, plus norms and biases).
- FLOPs per token ≈ 2 × 124M ≈ **0.25 GFLOP**, plus attention: 2 × 2 × L × C × ctx ≈ 37k FLOP per context position, which is negligible at short context.
- **KV cache** = 2 × 12 × 768 × 4 B = **73.7 KB per token in f32**, so 75 MB at a full 1024 context. That is *more than the Q4 weights* (~70 MB). At long context, KV reads matter, which motivates an f16/q8 KV cache and flash-decoding.

**Decode ceilings at 120 GB/s**, weights only, short context, **(est.)**:

| Weights | Bytes/token | Ceiling @120 GB/s | @~92 GB/s (llama.cpp-achieved fraction) |
|---|---|---|---|
| f32 | 498 MB | **241 tok/s** | ~185 |
| f16 | 249 MB | **482** | ~370 |
| Q8_0 (8.5 bpw) | 132 MB | **~900** | ~700 |
| Q4_0 (4.5 bpw) | 70 MB | **~1700** | ~1300 |

**Prefill ceiling**, compute-bound, ~0.25 GFLOP/token: GPU at 4.3 TFLOPS ≈ **~17k tok/s** ideal and maybe 5–10k realistic. SME ≈ ~8k ideal. All-core NEON ≈ ~2.5k ideal.

**The small-model twist.** At 1000+ tok/s the per-token budget is **under 1 ms**. GPT-2 runs about 12 × ~10 ops ≈ 120–150 ops per token, so **each op gets about 7 µs**. Per-op overhead then dominates: rayon fork/join, Metal command-buffer commit and wait, or kernel dispatch. A 7B model is bandwidth-bound, but GPT-2 at the quantised tiers is likely to be **latency- and overhead-bound**. This is a useful plot twist for the post, and it is why llama.cpp uses spinning thread pools and why GPU engines encode the whole forward pass into one command buffer, and eventually use graphs or persistent kernels.

### 2.3 How to compute the ceiling for any model (copy for the post)

```
bytes/token = Σ_tensors numel × bits_per_weight / 8   (+ KV: 2·L·d_kv·ctx·bytes)
decode ceiling = BW / bytes_per_token
prefill ceiling = peak_FLOPs / (2 · params)            (per token, short prompts)
```

---

## 3. Kernel-level techniques

- **Kernel fusion.** Fusing elementwise ops into a neighbouring matmul saves a full write and read of the activation, but activations are tiny at batch 1. **For decode the real win of fusion is fewer dispatches or barriers** (see 2.2), not bytes. Candidates for GPT-2: bias+GELU into fc, bias+residual into proj and fc_proj, LayerNorm into the next matvec's input load, and Q, K and V already fused as c_attn. Andrew Chan's gain from fusion was about 5%, because matmuls were 94% of the time.
- **FlashAttention v1** (https://arxiv.org/abs/2205.14135). Tile Q, K and V into SRAM and use **online softmax**: keep a running max m and denominator ℓ, rescale the accumulated output when m changes. This avoids materialising the N×N score matrix in HBM. Its benefit is IO, not FLOPs. **v2** (https://arxiv.org/abs/2307.08691) cuts the non-matmul FLOPs (rescaling happens once at the end), parallelises over sequence length as well as batch and heads, and partitions work between warps better. For us it only matters for **prefill**, and only at long contexts. At GPT-2's 1024 max, a naive attention is fine at first.
- **Flash-decoding** (https://pytorch.org/blog/flash-decoding/). At decode q-len is 1, so FlashAttention parallelises only over batch×heads and uses **<1% of an A100 at batch 1**. The fix is to **split the KV sequence into chunks** and compute each chunk's partial attention in parallel, along with its log-sum-exp. A second small kernel then rescales and sums the partials. The result is up to 8x end-to-end at very long context, and attention up to 50x faster than FA2, nearly constant up to 32k. For GPT-2 with 12 heads and a 10-core GPU, a split-KV decode attention is the natural kernel once context exceeds a few hundred tokens. On CPU the same idea is "parallelise attention over (head, kv-chunk)".
- **Numerics:**
  - **Softmax.** Subtract the row max before `exp`. Use online softmax for streaming. Accumulate in f32 even with f16 inputs. Mask future positions with -inf *before* the max.
  - **LayerNorm/RMSNorm.** Compute mean and variance in f32. A two-pass or Welford algorithm avoids catastrophic cancellation in `E[x²]−E[x]²`. eps is 1e-5 for GPT-2.
  - **GELU.** GPT-2 was trained with the tanh approximation, so use the same one. An exact erf GELU gives small logit diffs that look like bugs.
  - **RoPE** (Llama only). Precompute cos/sin tables. Computing `pos × θ` in f32 loses precision at large positions, so use f64 for table generation. Pairing convention matters: adjacent pairs (llama2.c) versus half-split (HF `rotate_half`). It is the #1 porting bug.
  - **Quantised accumulation.** int8·int8→int32 per block, then scale in f32. Never accumulate across blocks in f16.
  - **Subnormals.** Andrew Chan's atomicAdd flushed them to zero. Fast-math or FTZ modes can matter for attention accumulators.

---

## 4. Apple Silicon specifics

- **NEON** has 32 × 128-bit registers: 4 f32, 8 f16 or 16 i8 lanes. The useful M4 extensions are `fp16` (native f16 FMA on 8 lanes), `dotprod` (`sdot`/`udot`: 4-way int8 dot into int32), `i8mm` (`smmla`: a 2×8·8×2 int8 tile per instruction) and `bf16`. All of them are reachable from stable Rust `std::arch::aarch64` with `target_feature`. Use 4 to 8 independent accumulators to hide FMA latency, which is about 4 cycles with several pipes, so you need enough independent chains in flight.
- **AMX / SME.** Pre-M4 chips had undocumented AMX, reachable only through Accelerate. **M4 implements standard Arm SME**, with a 512-bit streaming vector length. An FP32 `FMOPA` does a 16×16 outer product, which is 512 FLOPs per instruction. There is one SME unit per cluster, shared by its cores, and it reaches ~2 TFLOPS FP32. Code has to enter streaming mode (`smstart`/`smstop`), and NEON is unavailable inside it. **Rust has no stable SME intrinsics**, so access would be inline `asm!`. That makes it an optional, advanced step. **It is great for prefill GEMM and pointless for decode matvec**, because it doesn't increase bandwidth. Easy access: `cblas_sgemm` from **Accelerate**, which uses AMX or SME, works as the "what does Apple's library get?" calibration line.
- **Metal:**
  - SIMD-group width is 32, and `simd_sum` / `simd_shuffle_down` handle reductions.
  - Threadgroup memory is 32 KB.
  - `simdgroup_float8x8` / `simdgroup_half8x8` with `simdgroup_load` and `simdgroup_multiply_accumulate` provide the matrix units for GEMM. Use them for prefill, not decode.
  - Decode matvec: assign **one SIMD-group, or a few, per output-row block**, read the weights coalesced (consecutive lanes read consecutive addresses), and reduce with `simd_sum`. Start with the naive one-thread-per-row version and use its slowness as a lesson, as Andrew Chan did.
- **Unified memory.** Create `MTLBuffer`s over the mmap'd, page-aligned weight file with `newBufferWithBytesNoCopy` + `StorageModeShared`, and there is no upload. Keep activations and the KV cache in shared buffers too, so the CPU can read logits directly for sampling. **Encode the whole forward pass into one command buffer** with a single commit and wait per token. Per-dispatch overhead is the likely bottleneck at GPT-2 scale.
- **WebGPU and WASM, for the live blog:**
  - WGSL has no int8 or int4 types, so pack weights in `u32` and unpack with shifts.
  - f16 needs the `shader-f16` feature. Subgroup ops are now shipping in Chrome but are not universal, so keep a fallback.
  - **The default `maxStorageBufferBindingSize` is 128 MiB.** GPT-2's f32 wte is ~154 MB, so it **exceeds the default**. Request higher adapter limits or split the tensor. With Q8 or Q4 weights this mostly goes away.
  - wasm32 has `simd128` (no FMA in baseline; relaxed-simd adds `relaxed_madd`). Its memory is capped at 4 GB.
  - Threads need SharedArrayBuffer, which needs COOP/COEP headers on the host. Plan the Vercel config for that.

---

## 5. Benchmarking methodology

1. **Correctness before speed, at every step:**
   - Dump the HF `GPT2LMHeadModel` fp32 reference: per-layer residual streams plus final logits for a few fixed prompts.
   - Gate CI on max-abs logit diff. Expect < 1e-4 for f32 at different summation orders, ~1e-2 for f16, and larger for quant.
   - Also check top-1 token agreement and **greedy-decode sequence equality** for about 64 tokens.
   - Report KL divergence of the softmax against the f32 reference. It is more meaningful than raw logit diff for quantised variants.
2. **Perplexity for quantisation.** Use a fixed eval: WikiText-2 test, GPT-2 tokenizer, 1024 context, a fixed stride. Report Δppl against *our own* f32 baseline. The GPT-2 paper's 29.41 used special detokenisation, so absolute numbers won't match. Expect small models to be **more sensitive** to quantisation than 7B models. As rough llama.cpp lore for 7B, Q8_0 is near-lossless, Q4_K_M sits a few hundredths of a ppl above f16, and Q4_0 is worse (unverified). Measure GPT-2 rather than assume.
3. **Separate the metrics:**
   - **Prefill tok/s** at fixed P, for example 128 and 512.
   - **Decode tok/s** at fixed context positions, for example 1 to 128 and also at pos 900, so KV cost shows.
   - **TTFT.**
   - **Achieved GB/s** = bytes_per_token × tok/s, and **% of measured ceiling**. This is the headline metric of the post.
4. **Statistics:**
   - Warm up with a few full tokens, so pages fault in, caches fill and the GPU clocks ramp.
   - Then run N ≥ 10 trials and report the median and p10/p90, or min-of-N for microbenchmarks.
   - Fix the prompt and use greedy sampling.
   - Record the commit hash, rustc version, power source and machine model. A MacBook Air throttles under sustained load and a Mac mini doesn't. Check which one Josh has.
   - macOS can't pin threads to cores. Use QoS `USER_INTERACTIVE` so threads go to P-cores, and test thread counts 1, 4 (P only), 6, 8 and 10. E-cores can *slow* a statically partitioned matvec, because the slowest thread sets the pace, so use work-stealing or uneven chunks.
5. **Profile:**
   - Use Instruments: Time Profiler for CPU, Metal System Trace for GPU timeline gaps and dispatch overhead.
   - Count per-op time in-engine with a `--trace` flag. The same data can drive the "where does a token's time go" visualisation.
6. **Keep a `BENCH.md` ledger.** For each step record the date, commit, tok/s for prefill and decode, GB/s, % of ceiling, Δppl and a note. This is the raw material for the post.

---

## 6. Proposed hillclimb ladder (GPT-2 124M, M4)

The speedup figures are **(est.)** expectations from the sources above, not promises. "Ceiling" means 120 GB/s for decode.

| # | Step | What changes | Expected effect | Decode ballpark |
|---|---|---|---|---|
| 0 | Harness | HF reference dumps, logit/KL diff tests, perplexity script, STREAM bandwidth probe, llama.cpp GPT-2 baseline | defines "correct" and "ceiling" | n/a |
| 1 | Naive f32, 1 thread | llama2.c-shaped: mmap safetensors, preallocated RunState, idiomatic `iter().sum()` matvec | baseline. Scalar because reductions don't vectorise | ~10–30 tok/s |
| 2 | Layout + compiler | Transpose Conv1D weights to (out,in) rows, `chunks_exact`, 8 accumulators, target-cpu, LTO | 2–4x (auto-vectorisation unlocked) | ~40–80 |
| 3 | Explicit NEON | `vfmaq_f32`, unrolled, prefetch-friendly. One core should approach its own bandwidth limit | 1.2–2x | ~60–120 |
| 4 | Threads | rayon over output-row chunks, attention over heads. Then a persistent spin-barrier pool | 2–3x, capped by bandwidth (~241 f32 ceiling) | ~150–200 |
| 5 | Housekeeping fusion | bias+GELU, bias+residual fused. No per-token allocs. Padded vocab LM head | 1.05–1.2x. It matters more as tokens get faster | ~170–210 |
| 6 | f16 weights | `half` + NEON `fcvtl` or native fp16 FMA, f32 accumulate | ~1.8x (halves bytes) | ~300–400 |
| 7 | Q8_0 + Q8 activations | block-32 int8, `vdotq_s32`. Δppl check | ~1.6–1.9x | ~500–700 |
| 8 | Q4_0 → Q4_K | nibble unpack, mixed precision for sensitive tensors (LM head at Q6/Q8) | ~1.3–1.6x. Now overhead-bound | ~700–1100 |
| 9 | Prefill GEMM | batch the prompt and tile (llm.c 8-row reuse → 2D register blocking à la Boehm), compare with Accelerate `sgemm` (SME) | prefill 10–50x over token-at-a-time | prefill ~1–3k tok/s |
| 10 | Metal naive | one thread per row matvec. Teachable regression, possibly slower than CPU | ↓ | maybe < CPU |
| 11 | Metal SIMD-group matvec | SIMD-group per row, coalesced, `simd_sum` | big (Andrew Chan saw 18x) | near f32 ceiling ~180–220 |
| 12 | One command buffer, zero-copy | `bytesNoCopy` mmap, all dispatches in one encoder, CPU reads logits from shared memory | large at quantised tiers (removes per-op sync) | n/a |
| 13 | GPU dtypes | f16, Q8_0, Q4 dequant-in-register kernels | as CPU 6–8, larger headroom | ~600–1300 |
| 14 | GPU prefill | `simdgroup_matrix` 8×8 tiled GEMM. Flash-attention-style prefill attention | compute-bound. Aim for 30–50% of 4.3 TFLOPS | prefill ~5–8k tok/s |
| 15 | Long context | f16 KV cache, split-KV flash-decoding | flat decode tok/s vs position | n/a |
| 16 | WebGPU/WASM port | wgpu + WGSL (u32-packed quant), wasm simd128 CPU fallback | powers the live blog | browser-dependent |
| opt | SME by hand | inline-asm `FMOPA` GEMM for CPU prefill | ~2 TFLOPS. Hard | n/a |

**Theoretical ceilings on M4:** decode is **241 tok/s in f32, 482 in f16, ~900 in Q8_0 and ~1700 in Q4_0** (realistically ~75–80% of those). Prefill is **~17k tok/s** at GPU peak FP32. Expect the quantised tiers to hit dispatch and sync overhead before they hit bandwidth, and present that as a discovery.

---

## 7. Learner-copy ideas (intuitions and analogies)

- **Decode is reading a book to say one word.** To produce each token you must re-read the entire 500 MB "book" of weights. The arithmetic is trivial, the reading is the job. So speed = how fast you can read ÷ how thick the book is.
- **The roofline is a speed limit sign.** Before any code: "This car cannot go faster than 241 tok/s. Let's see how close we get." Every step reports "% of the speed limit".
- **Loop order and layout: reading a book along the lines vs down the columns.** Same words, but one way you turn a page for every letter, which is a cache miss.
- **Multiple accumulators are checkout lanes.** One cashier (accumulator) has to wait for each FMA to finish. Eight lanes keep the pipes full. And why Rust won't do it for you: adding floats in a different order gives a different answer, so the compiler refuses to reorder without permission.
- **SIMD: one instruction, four (or sixteen) numbers**. A paint roller instead of a brush.
- **Threads hit a wall.** Ten cooks, one narrow pantry door. After about 4 cooks the door (memory bandwidth) is the bottleneck, not the cooks.
- **Quantisation is shipping a compressed book.** You decompress in your head, in registers, while reading. Block scales are "each paragraph has its own volume knob", so an outlier only ruins its own 32 numbers. The bpw table maps straight onto tok/s.
- **The quality bill.** Every compression step has an invoice: perplexity and KL. Show both on the same chart: speed up, quality (almost) flat. Andrew Chan's subnormal bug shows a fast kernel can quietly get dumber.
- **GPU naive is slower than CPU.** A thousand workers who each insist on fetching from a different shelf. Coalescing means the workers fetch neighbouring items in one trip.
- **Unified memory.** On a PC the GPU has its own warehouse and you truck weights over. On the M4 both share one warehouse, so you just hand over the address.
- **Dispatch overhead.** When a token takes 1 ms, telling the GPU "now do LayerNorm" costs as much as doing it. It's a manager who spends more time assigning tasks than the tasks take. The fix is to hand over the whole day's to-do list at once (one command buffer) and merge small tasks (fusion).
- **Prefill vs decode.** Prefill reads the book once and answers 500 questions in one pass, so it's compute-bound. Decode reads the whole book per question.
- **Online softmax / flash attention.** Keep a running "best so far" and rescale when a bigger one shows up, like updating a leaderboard without storing every score. Flash-decoding is splitting the leaderboard across judges, then merging their summaries.
- **KV cache.** Notes in the margin so you don't re-read the earlier chapters. But at 1024 tokens the notes (75 MB) are bigger than the compressed book (70 MB).

---

## 8. Steps that make great live visualisations

The engine runs in-browser (WASM or WebGPU), so these can be **real measurements on the reader's own machine**.

1. **Roofline playground.** Plot the reader's device ceiling, measured with an in-browser bandwidth probe. Each ladder step is a dot that moves up as the reader toggles optimisations. Users pick dtype and watch the ceiling line move.
2. **Byte budget treemap.** GPT-2's weights as area by tensor, with wte/LM head at 31% as the surprise. Switch f32→f16→Q8→Q4 and watch the treemap shrink, with the "max tok/s" readout updating live.
3. **Matvec access-pattern animation** (after Boehm). Show memory cells lighting up as naive row-wise vs column-wise vs SIMD-chunked vs multi-threaded kernels read them. A cache-miss counter ticks. Show thread or SIMD-group ownership as colours.
4. **Race mode.** The same prompt generated side by side by step-1 and step-N engines in the reader's browser, streaming tokens with live tok/s. This is the most visceral one.
5. **Quantisation microscope.** Pick a real weight row, show the f32 values, the Q4 block (scale + nibbles) and the reconstruction error. Drag a slider for block size and watch error and outlier damage change. Pair it with live Δ-logit / KL on a prompt.
6. **Where does a token's time go.** A flame or stacked bar from the engine's `--trace` per-op timings, per step. It shows matmuls dominating early and overhead dominating late, which is the plot twist from section 2.2.
7. **Online softmax stepper.** Animate the running max and denominator over KV chunks, then the flash-decoding merge across splits.
8. **Prefill vs decode dial.** A slider for batch or prompt length P shows arithmetic intensity crossing the ridge point, with the measured tok/s curve flattening into the compute roof.
9. **Thread scaling curve.** In WASM with threads: run 1, 2, 4 and 8 workers and plot tok/s, which bends flat at the bandwidth wall.

Practical notes:
- (5), (7) and (2) need only CPU WASM and work everywhere.
- (4), (6) and (9) need WASM threads (COOP/COEP) or WebGPU.
- Keep Q8 or Q4 weights for the web demo, both for download size and the 128 MiB binding limit.

---

### Key links
- llama2.c: https://github.com/karpathy/llama2.c
- llm.c: https://github.com/karpathy/llm.c
- llama.cpp: https://github.com/ggml-org/llama.cpp
  - Apple Silicon benchmarks: https://github.com/ggml-org/llama.cpp/discussions/4167
- Andrew Chan, yalm: https://andrewkchan.dev/posts/yalm.html
- Boehm, CUDA matmul: https://siboehm.com/articles/22/CUDA-MMM
- Boehm, CPU matmul: https://siboehm.com/articles/22/Fast-MMM-on-CPU
- llama2.rs (srush): https://github.com/srush/llama2.rs
- llama2.rs (gaxler): https://github.com/gaxler/llama2.rs
- candle: https://github.com/huggingface/candle
- mistral.rs: https://github.com/EricLBuehler/mistral.rs
- burn / CubeCL: https://github.com/tracel-ai/burn, https://github.com/tracel-ai/cubecl
- metal-rs deprecation: https://github.com/gfx-rs/metal-rs
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- Flash-Decoding: https://pytorch.org/blog/flash-decoding/
- M4 SME: https://tnzr.org/sme, https://github.com/tzakharko/m4-sme-exploration
- Apple M4 spec: https://en.wikipedia.org/wiki/Apple_M4
