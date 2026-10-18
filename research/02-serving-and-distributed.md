# 02: Serving systems and distributed inference/training

Scope: what an inference engineer needs on serving (batching, KV memory, scheduling, speculative decoding, disaggregation) and on distribution (collectives, DP/ZeRO/TP/PP/SP/CP/EP), written for nanoinfer: a Rust engine on an M4 that will later use 2 to 8 rented GPUs for a few hours.

Confidence key: **[V]** means I checked it against the source while writing this. **[R]** means it is recalled from the paper or docs; it is very likely right, but re-check the figure before it goes in the blog. **[?]** means uncertain or the figure moves over time.

---

## 0. The one mental model everything hangs on: rooflines

Every step of an engine is bound by compute, memory bandwidth or communication. The skill is predicting which one before you measure.

- **Arithmetic intensity** = FLOPs / bytes moved. A kernel is compute-bound when its intensity is above the hardware ratio `C / W_hbm`.
  - H100 SXM: ~989 TFLOP/s bf16 dense / 3.35 TB/s ≈ **~295 FLOPs/byte** [R].
  - The scaling book puts H100's critical batch at **~280 tokens** in bf16, and TPU v5e's at ~240 [V].
  - M4: check which chip Josh has [?]. Base M4 is ~120 GB/s; M4 Pro is ~273 GB/s and ships in 24 GB configs. Measure it with a STREAM-style copy.
- **Matmul in decode.** One token times a weight matrix gives ~2 FLOPs per param per token, while each param (2 bytes in bf16) is read once per step. Intensity ≈ batch size, so **decode is memory-bound until batch reaches about 300** on an H100 [V: scaling book].
- **Attention in decode is always memory-bound.** Intensity is ≈ ST/(S+T) ≈ 1 for T=1. Batching does not help, because every sequence has its own KV cache [V].
- **Decode step lower bound:** `t_step ≈ (B·KV_bytes_per_seq + param_bytes) / W_hbm`. When the weights are the cost, `max(2·B·P / FLOPs, P_bytes / W_hbm)` [V].
- **KV cache size:** `bytes = 2 (K and V) × L × n_kv_heads × d_head × T × bytes_per_elem` [V].
  - GPT-2 small (L=12, 12 heads × 64, fp16): 2·12·768·2 = **36 KB/token**, so 1024 tokens ≈ 37 MB per sequence.
  - Llama-2-13B at 8k tokens is ~6.7 GB per sequence [V]. That is why KV memory, not FLOPs, limits serving.
- **Nanoinfer baseline:** GPT-2 124M in fp16 is ~250 MB of weights. At 120 GB/s the floor is ~2 ms/token, i.e. ~500 tok/s at batch 1. At 273 GB/s it is ~0.9 ms. Every optimisation should be reported as a fraction of this floor (MBU, see section 9).

Link: https://jax-ml.github.io/scaling-book/roofline/ , https://jax-ml.github.io/scaling-book/inference/

---

## 1. DeepMind, "How to Scale Your Model" (the scaling book)

https://jax-ml.github.io/scaling-book/

1. **Sharding notation** [V].
   - `A[I_X, J]` means dimension I is split across mesh axis X and J is replicated.
   - `A[I_XY, J]` shards over the flattened X·Y axes.
   - An axis can shard only one dimension, so `A[I_X, J_X]` is illegal.
   - `C{U_X}` marks a result that is "unreduced", i.e. it holds partial sums over X.
   - Adopt this notation in the docs and the blog. It makes TP and DP layouts one-liners.
2. **The four sharded-matmul cases** [V]. With contraction over J:
   1. J is unsharded: no communication.
   2. One operand is sharded on J: AllGather it first.
   3. Both operands are sharded on J: multiply locally to get `{U_X}`, then AllReduce, or ReduceScatter if the output can stay sharded.
   4. Both operands shard the same non-contracting axis: illegal, so AllGather one of them.
   All of Megatron TP (section 3) is cases 1 and 3.
3. **Collective costs (bandwidth-bound)** [V]. V is the full array in bytes and W is the link bandwidth.

   | Collective | Time |
   |---|---|
   | AllGather | ≈ V/W |
   | ReduceScatter | ≈ V/W |
   | AllReduce | ≈ 2V/W (it is ReduceScatter + AllGather) |
   | AllToAll | ≈ V/(4W) on a bidirectional ring |

   The key surprise: **bandwidth-bound collective time does not depend on device count.** AllGather and ReduceScatter are transposes of each other, so one in the forward pass becomes the other in the backward pass.
4. **Latency floor** [V]. Each hop costs at least about 1 µs, and small collectives are latency-bound. Decode-time TP all-reduces (a few KB) sit squarely in this regime (see section 4).
5. **Inference sharding rules** [V].
   - **Prefill** behaves like training: use model parallelism up to about 4 to 8 ways, then sequence parallelism.
   - **Decode:** avoid FSDP, because streaming weights over the interconnect is far slower than over HBM. Avoid pure DP inside a replica (add more replicas instead). Avoid sequence sharding. Use model/tensor parallelism, and shard the KV cache by heads, then by batch.
6. **Throughput vs latency in decode** [V]. Throughput rises with batch until KV memory traffic is about equal to weight traffic, and then returns diminish. Smaller KV (GQA/MQA, int8 KV, sliding window) means bigger batches.
7. **Serving layouts** [V].
   - Batched prefill, then generate: poor TTFT and padding waste.
   - Interleaved: prefill stalls decode.
   - Disaggregated: separate prefill and decode pools, with the KV cache shipped over the network.
8. **Speculative decoding** [V]. It does not make a step faster; it accepts more than one token per step. This is cheap exactly because decode is memory-bound.

---

## 2. Hugging Face "Ultra-Scale Playbook" (nanotron)

https://huggingface.co/spaces/nanotron/ultrascale-playbook. The Space did not render through my fetcher, so these figures come from my reading of it [R]. The formulas are standard (ZeRO paper, Megatron papers).

**Memory accounting.** Ψ = parameter count. Mixed-precision Adam costs 2Ψ (bf16 params) + 2Ψ (bf16 grads) + 12Ψ (fp32 master params plus Adam m and v) = **16Ψ bytes**. Activations come on top and scale with batch × seq × layers. That is why 7B does not fit on an 80 GB GPU for training without sharding.

| Strategy | Per-GPU memory (N = DP degree) | Comms per step |
|---|---|---|
| DDP | 16Ψ | AllReduce of grads (≈2Ψ bytes·2(N-1)/N), overlapped with backward via bucketing |
| ZeRO-1 | 4Ψ + 12Ψ/N | ReduceScatter grads + AllGather params: same volume as DDP |
| ZeRO-2 | 2Ψ + 14Ψ/N | same as ZeRO-1 |
| ZeRO-3 / FSDP | 16Ψ/N | an extra AllGather of params in backward: ~1.5× DDP volume; needs prefetch/overlap |

**Insights:**

1. **DP scales until the global batch gets too large or communication stops overlapping.** Gradient accumulation trades time for a bigger effective batch. ZeRO-1 is "free" sharding, and llm.c uses it.
2. **ZeRO shards state, not compute.** Every GPU still runs the full model on its own micro-batch, and activations are not reduced. That is why TP, SP and PP exist.
3. **TP (Megatron) shards the matmuls.** It needs about 2 AllReduces per layer in the forward pass and 2 in the backward, and they sit on the critical path. **Keep TP inside an NVLink domain (≤8 GPUs).** Across nodes on InfiniBand it collapses.
4. **SP (sequence parallelism, used together with TP)** shards the LayerNorm/dropout regions along the sequence. It turns each AllReduce into ReduceScatter + AllGather: the same bytes, but less activation memory.
5. **CP (context parallelism / Ring Attention)** shards very long sequences across GPUs. K and V blocks rotate around a ring while each GPU keeps its Q block, overlapping compute with P2P sends. It needs load balancing under a causal mask (zig-zag chunk assignment).
6. **PP (pipeline)** places layers on different GPUs and only sends activations at stage boundaries, which is cheap and fine across nodes.
   - The cost is the bubble: **bubble fraction ≈ (p−1)/m** for p stages and m micro-batches, for both GPipe (all-forward-all-backward) and 1F1B.
   - 1F1B's win is memory: only p micro-batches of activations are live, not m.
   - Interleaved stages (v chunks per GPU) give ≈ (p−1)/(v·m), at the cost of more communication.
   - Zero-bubble and DualPipe (DeepSeek-V3) split backward into weight-grad and input-grad passes to fill the bubbles.
7. **EP (expert parallelism)** places MoE experts on different GPUs. Tokens move by **AllToAll** (dispatch, then combine). It is cheap in compute and heavy in communication, and load imbalance is the main problem.
8. **Rule of thumb for choosing** [R]:
   1. If the model fits, use DP/ZeRO-1.
   2. If it does not fit, add TP within the node, then PP across nodes, or use ZeRO-3 when the interconnect is good.
   3. For long context, add CP. For MoE, add EP.
   The final config is always found by benchmarking. The playbook ran thousands of configs on up to 512 GPUs.

---

## 3. Megatron-LM tensor parallelism and pipeline parallelism

Papers: Shoeybi et al. 2019 https://arxiv.org/abs/1909.08053 ; Narayanan et al. 2021 (PTD-P, interleaved 1F1B) https://arxiv.org/abs/2104.04473 ; GPipe https://arxiv.org/abs/1811.06965 ; Korthikanti et al. 2022 (SP) https://arxiv.org/abs/2205.05198

**MLP, `Y = GeLU(X·A)·B`** [R]:
- **Column-parallel A.** Split A by columns, `A = [A1 | A2]`. Each GPU computes `GeLU(X·Ai)` locally. GeLU is elementwise, so no communication is needed. A row split would force a sync before the GeLU, which is why it uses columns.
- **Row-parallel B.** Split B by rows. Each GPU produces a partial `Y_i`, then **one AllReduce** gives the full Y.
- **The f/g conjugate operators.** f is identity in the forward pass and AllReduce in the backward; g is the reverse. That gives one AllReduce per MLP in each direction.
- In the notation above: `X[B,D] · A[D, F_Y] → H[B, F_Y]` (case 1), then `H[B,F_Y] · B[F_Y, D] → Y{U_Y}` followed by AllReduce (case 3).

**Attention** [R]:
- QKV projections are column-parallel **by head**: each GPU owns n_heads/TP heads and its own slice of the KV cache.
- Attention for a head is entirely local.
- The output projection is row-parallel and ends in one AllReduce.

**Totals and constraints:**
- 2 AllReduces per transformer layer in the forward pass. Each one is `B·S·D` elements.
- n_heads and the FFN width must be divisible by TP. Embedding and LM head: shard the vocabulary, then handle the softmax/argmax across shards.

**Decode reality check:**
- For GPT-2 small at batch 1 in fp16, each AllReduce is 768 × 2 B = **1.5 KB**, with 24 of them per token.
- This is purely latency-bound: 24 × (several µs to tens of µs) per token. With NCCL on NVLink that is roughly 5 to 20 µs each [?].
- TP on a tiny model makes a single sequence *slower*. That is a great, honest blog result. TP pays off when the weights do not fit on one device, or when per-GPU HBM bandwidth is the bottleneck and the model is large.

**Pipeline parallelism** [R]:
- **GPipe** runs all forward micro-batches, then all backward, with synchronous flushes. Bubble (p−1)/m, memory ∝ m.
- **1F1B (PipeDream-Flush)** has a warm-up of p−1 forwards, then alternates one forward and one backward. Same bubble, memory ∝ p.
- **Interleaved 1F1B** splits the bubble by v.
- **PP for inference** has no bubble problem in steady state if you keep p batches in flight. It adds latency per token (p hops) but raises throughput. It works well over slow links, which makes it ideal for the simulated multi-process setup on the Mac.

---

## 4. Collectives: ring all-reduce, NCCL, and building your own

**Ring all-reduce derivation** [R, standard; Patarasuk & Yuan 2009, Baidu's 2017 ring-allreduce blog]:
- **Setup:** N workers in a ring, each holding a vector of S bytes split into N chunks.
- **Phase 1, ReduceScatter:** N−1 steps. At each step a worker sends one chunk (S/N) to its right neighbour and adds the chunk it receives from the left. Afterwards worker i owns the fully reduced chunk i.
- **Phase 2, AllGather:** N−1 more steps that circulate the reduced chunks.
- **Cost:**

  `T_ring = 2(N−1)·α + 2·(N−1)/N · S/B`

  where α is per-message latency and B is per-link bandwidth.
- **What the formula says:**
  - The bandwidth term tends to 2S/B as N grows. It is bandwidth-optimal and independent of N.
  - The latency term grows linearly in N.
- **Tree / double-binary-tree** (NCCL ≥2.4): the latency term is ~2·log2(N)·α, and the bandwidth is close to optimal when pipelined.
  - **Small messages favour trees; large messages favour rings.** NCCL picks per message size and topology.
  - NCCL also has NVLS (NVSwitch in-network reduction on H100 systems) and CollNet/SHARP [R].
- **NCCL benchmark vocabulary** (nccl-tests): `algbw = S / t` and `busbw = algbw · 2(N−1)/N` for all-reduce. busbw is comparable to link bandwidth across different N, so **always report busbw.** https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md

**Hardware numbers worth memorising** [R]:

| Link | Bandwidth |
|---|---|
| H100 NVLink4 | 900 GB/s bidirectional (≈450 GB/s each way) per GPU |
| A100 NVLink3 | 600 GB/s bidirectional |
| PCIe Gen4 x16 | ~32 GB/s per direction |
| PCIe Gen5 x16 | ~64 GB/s per direction |
| InfiniBand NDR | 400 Gb/s ≈ 50 GB/s per NIC (usually 8 NICs per node) |

The ratio HBM : NVLink : PCIe/IB of about 3350 : 450 : 50 explains every parallelism rule above.

**Simulating N devices on one Mac (the core local plan):**
- **Ranks are OS processes.** Spawn N worker processes (rank 0..N−1) from a launcher, like `mpirun` or `torchrun`. Each one loads only its shard of the weights.
- **Transport options:**
  1. **TCP loopback with tokio.** This is the realistic version: the same code runs across machines and is easy to throttle.
  2. **Unix domain sockets:** faster locally.
  3. **Shared memory:** `memmap2` on a file in /dev/shm-equivalent, or the `shared_memory` / `raw_sync` crates, with an atomic sequence number per slot as a barrier. This is fastest and models NVLink-like P2P.
- **Build the primitives in order:** `send/recv`, then `barrier`, then naive `allreduce` (gather to rank 0 and broadcast), then ring `reduce_scatter` + `all_gather` = ring `allreduce`, then tree allreduce, then `all_to_all`. Unit-test each against a single-process sum.
- **Fake the link speeds.** Add a configurable bandwidth/latency shaper (token bucket plus a sleep per message) so the same run can emulate "NVLink" (α=2 µs, 450 GB/s) or "PCIe/IB" (α=10 µs, 25 GB/s). That makes the ring vs tree crossover and the TP vs PP trade-off *visible* on a laptop. Label clearly in the blog that these are emulated.
- **Gotcha: the processes are not independent devices.** On the M4 every rank shares one GPU, one memory bus and one set of cores. Compute time per rank will not shrink like real devices, and Metal command queues from several processes contend with each other. To isolate communication from compute, either run ranks on the CPU backend with pinned thread counts, or report communication time separately (timestamp each collective).
- **Gotcha: sum in a fixed order.** Floating-point reduction order differs between ring and naive, so results differ in the last bits. Compare with a tolerance, and pick a deterministic order for reproducible tests.
- **Prior art to read:** MLX's `mlx.distributed` ring backend over TCP/Thunderbolt is a production example of exactly this on Macs [R]. https://ml-explore.github.io/mlx/build/html/usage/distributed.html

**Rust on real GPUs:**
- **`cudarc`** has an `nccl` module with safe wrappers for comm init (`Comm::from_rank`), all_reduce, broadcast and friends [R]. https://github.com/coreylowman/cudarc. Candle's `llama_multiprocess` example uses it for TP inference and is the closest Rust reference. https://github.com/huggingface/candle/tree/main/candle-examples/examples/llama_multiprocess
- **NCCL init needs a `ncclUniqueId` passed out-of-band.** Use your own TCP rendezvous (rank 0 serves it), as llm.c's TCP mode does. Your Mac collective layer can then double as the bootstrap.
- **Other options:** `mpi` (rsmpi) works but is heavy. tokio is right for the control plane (scheduler, HTTP/SSE API, rendezvous) and wrong for the hot data path.
- **Keep one trait, two backends.** A `Communicator` trait with `MacShm`/`Tcp` and `Nccl` implementations means the engine code is identical locally and on rented boxes. That is the single most important design decision for cheap GPU time.

---

## 5. Serving systems

### Orca: continuous (iteration-level) batching (OSDI '22)
https://www.usenix.org/conference/osdi22/presentation/yu [R]
1. **The problem.** Request-level batching waits for the longest sequence in the batch, so finished slots sit idle and new requests queue.
2. **Iteration-level scheduling.** The scheduler re-forms the batch every decode step: finished requests leave, new ones join.
3. **Selective batching.** Non-attention ops (QKV, MLP) run on the flattened `[total_tokens, D]` tensor regardless of each sequence's length, and attention runs per sequence. This idea is why modern engines use "ragged"/varlen batches rather than padded `[B, S]` batches.
4. **Result.** The paper reports up to ~36.9× throughput over FasterTransformer on GPT-3 175B at the same latency. The baseline is weak, so cite it as "an order of magnitude".

### vLLM / PagedAttention (SOSP '23)
https://arxiv.org/abs/2309.06180 [R]
1. **The problem.** Pre-allocating a contiguous KV buffer of `max_seq_len` per request wastes memory in three ways: reservation, internal fragmentation and external fragmentation. The paper measures existing systems using only **20 to 38%** of KV memory for actual tokens. vLLM gets waste **under 4%**.
2. **The fix is OS virtual memory for the KV cache.** KV lives in fixed-size **blocks** (default 16 tokens). Each sequence has a **block table** that maps logical block to physical block. Blocks are allocated on demand from a free list.
3. **The attention kernel gathers through the block table.** That is an indirection cost, which is why kernels take `block_table` and `seq_lens` arguments.
4. **Sharing via reference counts plus copy-on-write.** Parallel sampling, beam search and shared prefixes point at the same physical blocks, and the last partial block is copied on write.
5. **Preemption when out of blocks:** either *swap* KV to CPU memory or *recompute* it later (re-prefill). vLLM prefers recompute for whole sequence groups.
6. **Result.** 2 to 4× throughput vs FasterTransformer/Orca at the same latency. The gain is larger with long sequences and beam search.
7. **For nanoinfer:** the block manager is ~200 lines of pure Rust (free list, ref counts, tables) and can be tested without any kernel. Build it before writing the paged kernel.

### SGLang / RadixAttention (2023–24)
https://arxiv.org/abs/2312.07104 [R]
1. **Keep KV after a request finishes**, in a **radix tree** keyed by token sequences, with LRU eviction of leaves when memory runs out.
2. New requests match their longest cached prefix and skip prefill for it. The hits are system prompts, few-shot examples, multi-turn chat and agent loops.
3. **Cache-aware scheduling.** Ordering the queue by longest-shared-prefix raises the hit rate, but can starve some requests, so it needs a fairness knob.
4. **Result.** The paper reports up to ~5 to 6.4× throughput on prefix-heavy workloads.
5. vLLM's "automatic prefix caching" does the same thing with hashed full blocks instead of a radix tree. That is simpler and a good first version for nanoinfer: hash(block tokens + parent hash) → physical block.

### Chunked prefill (Sarathi / Sarathi-Serve, OSDI '24)
https://arxiv.org/abs/2403.02310 [R]
1. **The problem.** A long prompt's prefill monopolises an iteration, and every decoding user sees a latency spike ("generation stall").
2. **The fix.** Split prefills into chunks and fill each iteration up to a **token budget**: all running decodes (one token each) plus a prefill chunk.
3. **Why it works.** Decode is memory-bound, so prefill tokens "piggyback" on compute the decodes leave idle.
4. **Knobs.** The token budget trades TTFT against TPOT. Smaller chunks give smoother ITL but more KV re-reads per chunk. This is now vLLM's default V1 scheduler behaviour [R].

### Prefill/decode disaggregation
DistServe (OSDI '24) https://arxiv.org/abs/2401.09670 ; Splitwise (ISCA '24) https://arxiv.org/abs/2311.18677 ; Mooncake (Kimi) https://arxiv.org/abs/2407.00079 [R]
1. **Why split.** Prefill is compute-bound and drives TTFT; decode is memory-bound and drives TPOT. Colocating them couples the two SLOs, and each prefers different parallelism and batching.
2. **DistServe** optimises **goodput**: requests/s that meet *both* the TTFT and TPOT SLOs. It reports up to 7.4× more requests, or 12.6× tighter SLOs, vs colocated vLLM.
3. **Splitwise** runs the phases on different hardware generations (e.g. older GPUs for decode). It reports ~1.4× throughput at 20% lower cost, or ~2.35× throughput at the same cost and power.
4. **The cost is KV transfer.** It is `KV_bytes_per_token × prompt_len` per request. It needs a fast interconnect, or layer-wise streaming that overlaps with prefill.
5. **Mooncake** makes the KV cache a first-class distributed store (CPU DRAM/SSD pooling).

---

## 6. Speculative decoding

Leviathan et al. 2023 https://arxiv.org/abs/2211.17192 (also Chen et al., DeepMind https://arxiv.org/abs/2302.01318); Medusa https://arxiv.org/abs/2401.10774 ; EAGLE https://arxiv.org/abs/2401.15077 ; EAGLE-3 https://arxiv.org/abs/2503.01840

1. **Why it works.** A memory-bound target model can score γ+1 positions in about the time it takes to score one, because the weights are read once.
2. **The loop.** A cheap drafter proposes γ tokens. The target model verifies them all in a single forward pass. You keep the longest accepted prefix, plus one token from the target.
3. **Lossless acceptance rule.** For draft token x with drafter prob q(x) and target prob p(x), accept with probability `min(1, p(x)/q(x))`. On rejection, sample from `norm(max(0, p − q))`. The output distribution is then *exactly* p.
   - Greedy special case: accept iff `argmax p == x`.
   - Implement this precisely and test it statistically (a histogram of 1e5 samples vs p). It makes a lovely blog figure.
4. **Expected tokens per target call**, with per-token acceptance rate α (assumed i.i.d.):

   `E[tokens] = (1 − α^{γ+1}) / (1 − α)`

   **Expected speedup**, where c = drafter cost / target cost per token:

   `(1 − α^{γ+1}) / ((1 − α)(γc + 1))`

   Example: α=0.8, γ=4, c=0.05 gives ≈ 3.36 / 1.2 ≈ **2.8×**. The optimal γ grows with α and shrinks with c.
5. **Drafter options.**
   - A small model from the same family.
   - n-gram / prompt-lookup: free, and great for code or editing.
   - **Medusa:** extra decoding heads on the target's last hidden state, plus tree attention that verifies many candidate branches in one pass. It reports ~2.2× (Medusa-1, lossless) and 2.3 to 3.6× (Medusa-2, fine-tuned backbone) [R].
   - **EAGLE:** autoregression at the *feature* level (it predicts the next hidden state using the token shifted by one) and gets much better acceptance. EAGLE-1 reported ~2.7 to 3.5× on LLaMA2-Chat 70B [R].
   - **EAGLE-3:** drops feature prediction for direct token prediction over fused multi-layer features, with "training-time test". Reported speedups reach ~6.5× in the paper; independent measurements are more like 1.5 to 2× on real 8B serving [V: search results, workload-dependent].
6. **Gotcha: speculation's value falls as batch size grows.** At high batch, decode becomes compute-bound and the "free" verification tokens are no longer free. Serving systems throttle or disable speculation under load.
   - This makes nanoinfer on an M4 at batch 1 the *ideal* setting for spec decoding. Expect a big win: GPT-2 small drafting for GPT-2 medium/large/XL shares the tokenizer.
7. **Tree verification** needs a custom attention mask (each node attends only to its ancestors) and KV rollback for rejected branches. Paged KV makes rollback just "truncate the block table".

---

## 7. Training references: llm.c, nanoGPT, nanochat

- **llm.c** https://github.com/karpathy/llm.c [V for the README; R for the rest]
  - GPT-2 training in raw C/CUDA.
  - **Multi-GPU uses MPI + NCCL.** `mpirun -np N ./train_gpt2cu`. Multi-node can bootstrap NCCL via MPI, a shared filesystem or **TCP sockets** [V].
  - It implements **ZeRO-1** (optimizer state sharded with ReduceScatter/AllGather) [R].
  - Headline runs [R]:
    - GPT-2 124M in ~90 min on 8×A100 for ~$20 (Discussion #481: https://github.com/karpathy/llm.c/discussions/481).
    - GPT-2 1.6B in ~24 h on 8×H100 for ~$672 (Discussion #677).
  - **Read** `train_gpt2.cu` for the gradient-reduction and ZeRO-1 code paths and the multi_gpu_config struct. That is the cleanest raw-NCCL training reference in existence.
- **nanoGPT** https://github.com/karpathy/nanoGPT. Its DDP via `torchrun` (~10 lines) is the canonical "what DDP does" example: gradient AllReduce bucketed and overlapped with backward, plus `no_sync()` during gradient accumulation.
- **nanochat** https://github.com/karpathy/nanochat (2025): an end-to-end ChatGPT-clone pipeline (tokenizer, pretrain, SFT, RL, a KV-cached inference engine, web UI) on one 8×H100 node, with a ~$100 "speedrun" (~4 h) [R].
  - Its `engine.py` KV-cache inference loop is a short, readable serving baseline to compare against.
  - It is a good template for designing a rented-node experiment as a single script.

---

## 8. Renting GPUs cheaply and getting the most learning per dollar

Prices move weekly. Check them on the day [?].

| Provider | H100 80GB (per GPU-hr) | Notes |
|---|---|---|
| **Modal** | ~$3.95 (SXM, billed per second) [V] | **$30/month free on Starter** [V] ≈ 7.5 H100-hours, or ~55 min of 8×H100. Serverless Python-first; Rust binaries ship fine in a custom image. Multi-GPU per container via `gpu="H100:8"` [R]. A100-80GB ~$2.50, L4 ~$0.80 [V]. |
| **Vast.ai** | ~$1.5–2.2 [V: secondary sources] | Marketplace; reliability and interconnect vary. Lots of cheap **RTX 4090/3090** boxes at ~$0.2–0.5/GPU-hr [?]. **Check whether the box is NVLink or PCIe-only.** |
| **RunPod** | ~$2.0 community, ~$2.4–3.4 secure [V: secondary] | Per-minute pods, easy SSH, Docker images. |
| **Lambda** | ~$2.9–4.0 [V: sources disagree] | 8× SXM nodes with proper NVLink; on-demand availability varies. |

**Experiment design for maximum learning per dollar:**
1. **Do 100% of development on the Mac.** Same `Communicator` trait, emulated links, CPU or Metal backend. Rent only to *measure*, never to debug. Ship one container or script that builds, runs every experiment, writes JSON results and exits. Use per-second billing (Modal) or a hard `shutdown` timer on pods.
2. **Cheap rung (~$2–5):** 2–4× RTX 4090 on Vast for 1–2 h.
   - Bring up cudarc + NCCL.
   - Run `nccl-tests all_reduce_perf -b 8 -e 1G -f 2` to get a busbw vs message-size curve on PCIe.
   - Run your own ring all-reduce over NCCL send/recv vs NCCL's native one.
   - Run TP=2 GPT-2 decode.
   - Consumer GPUs often have P2P disabled, which *nicely illustrates* communication-bound TP.
3. **Real rung (~$25–60):** 8×H100 SXM (NVLink/NVSwitch) for 1–2 h, ideally on Modal's free credit plus a little spend.
   - Re-run nccl-tests to see the NVLink curve and the ring vs tree vs NVLS crossover.
   - Run TP=1/2/4/8 latency per token for GPT-2 XL (1.5B) and a ~7B model.
   - Run batch sweeps: throughput vs TPOT curves with continuous batching.
   - Run DP vs TP replicas at a fixed GPU count.
   - Optionally run a 15-minute llm.c 124M DDP/ZeRO-1 training run as a "training also works" reference point.
4. **Total budget: ~$50–100** covers everything above. Pre-download weights into the image or a volume, so you do not pay H100 rates to download from HF.

---

## 9. Metrics vocabulary (use these consistently in the logs and the blog)

- **TTFT** (time to first token): queueing + prefill + first sample. It is dominated by prompt length and queueing.
- **TPOT** (time per output token) = `(E2E − TTFT) / (n_out − 1)`, the mean per request.
- **ITL** (inter-token latency): the *distribution* of gaps between tokens. Report p50/p90/p99, because chunked prefill and preemption show up in the tail and not the mean.
- **E2E latency:** TTFT + (n−1)·TPOT.
- **Throughput:** output tokens/s, or total tokens/s, at a given load. Always plot it **against latency** (TPOT p50 or p99), sweeping request rate or batch size. A single throughput number is meaningless.
- **Goodput** (DistServe): requests/s that meet the SLOs, e.g. TTFT < 500 ms *and* TPOT < 50 ms. This is the honest serving metric.
- **MFU** (model FLOPs utilisation) = achieved model FLOPs / peak.
  - Training costs ≈ 6·N FLOPs per token (+ attention).
  - Inference forward costs ≈ 2·N per token (+ attention).
  - Relevant mainly for prefill and training.
- **MBU** (model bandwidth utilisation; Databricks' term, https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices [R]) = `(param_bytes + KV_bytes_read) / t_step / peak_BW`. **This is the decode metric** and nanoinfer's headline number on the M4: "we hit X% of the memory roofline".
- **Load generation:** use a Poisson arrival process with sampled prompt and output lengths (ShareGPT-like) for open-loop tests. Closed-loop fixed concurrency is easier, but it hides queueing.

---

## (a) Proposed ladder: serving + distributed steps for nanoinfer

Assumes the single-device engine ladder (naive forward, then KV cache, kernels, quantisation) runs in parallel per research doc 01. Each step ends with a measurement and a log entry.

**On the Mac (free):**
1. **Metrics harness first.** Per-request TTFT/TPOT/ITL, MBU vs a measured STREAM bandwidth, JSON output, and a Poisson load generator. Everything after this step is judged by it.
2. **Static batching** (padded `[B,S]`). Sweep the throughput-vs-latency curve across B and show the padding waste.
3. **Continuous batching** (Orca-style) with a ragged token batch and per-sequence attention, behind a scheduler loop and an SSE streaming API (tokio).
4. **Paged KV block manager**: free list, block tables, ref counts, preemption by recompute. Then a paged attention kernel (Metal). Show measured KV utilisation vs contiguous allocation.
5. **Prefix caching**, first as hashed blocks, then as a radix tree with LRU. Use a system-prompt-heavy workload and show the TTFT drop.
6. **Chunked prefill with a token budget.** Show the ITL p99 improvement under mixed load.
7. **Speculative decoding**: GPT-2 small drafting for GPT-2 large/XL, then prompt-lookup n-gram drafting. Verify losslessness statistically and plot speedup vs γ against the formula.
8. **Collective library**: processes plus a shared-memory/TCP `Communicator`, with naive, ring, tree and all_to_all. Add the link emulator, then reproduce the `2(N−1)/N` curve and the ring/tree crossover.
9. **Tensor-parallel GPT-2 across N local processes** (Megatron column/row split, KV sharded by head). Bit-compare against TP=1 within tolerance, and count all-reduces per token.
10. **Pipeline-parallel GPT-2** (layers split across processes, micro-batched decode) to compare TP vs PP latency and throughput under emulated slow links.
11. **Toy disaggregation**: a prefill process and a decode process with KV shipped over TCP. Measure transfer cost per prompt token.
12. **(Optional) DP training of a tiny GPT-2** with ring all-reduce of gradients, then ZeRO-1, all on CPU, to understand the training side (llm.c as reference).

**On rented GPUs (est. ~$50–100 total):**
- **R1 (~$5, 2–4× 4090, Vast):** NCCL backend for the `Communicator`; nccl-tests vs your own ring; TP=2 decode of GPT-2 XL on PCIe.
- **R2 (~$30–60, 8×H100 SXM, Modal/Lambda/RunPod, 1–2 h, fully scripted):** nccl-tests on NVLink; TP 1/2/4/8 latency per token on a ~7B model; continuous-batching throughput-vs-TPOT curves; TP=8 vs 8 replicas vs TP=2×4 replicas at a fixed GPU count; optional 15-min llm.c 124M DDP run.
- **R3 (optional, ~$20):** 2-GPU disaggregation (prefill on one GPU, decode on the other, KV over NVLink via NCCL send/recv) vs colocated, measured as goodput at fixed SLOs.

---

## (b) Learner-copy intuitions and analogies

- **Decode is a librarian, not a mathematician.** For every token the engine must carry the whole library (all the weights) from the shelves to the desk, and only does one sum per book. Batching means answering 64 people's questions per trip to the shelves.
- **The KV cache is the model's notes.** Without it, the model re-reads the whole conversation for every word. With it, it only reads its notes. The notes grow with every token, and they, not the model, are what fills memory.
- **PagedAttention is virtual memory.** Instead of reserving a whole notebook per guest "in case they talk for 2,048 tokens", hand out 16-token pages as needed and keep an index card (the block table) of where each guest's pages are.
- **Continuous batching is a bus that never waits at the terminal.** Passengers get off at their stop and new ones board at every stop. Static batching is a coach that leaves only when full and returns only when the last passenger is home.
- **Prefix caching:** do not re-read the same 2,000-word system prompt for every customer. Keep it on the desk, bookmarked.
- **Chunked prefill:** the newcomer with a 10-page question gets served a page at a time, so everyone already in the conversation keeps getting replies.
- **Speculative decoding is an intern drafting and a senior signing off.** The senior can check five words as fast as writing one, so as long as the intern is usually right, the work goes faster. The acceptance rule guarantees the result is exactly what the senior would have written.
- **Ring all-reduce is passing dishes around a table.** Everyone cuts their dish into N pieces. Each round you pass one piece left and add what you receive. After N−1 rounds each person holds one fully summed piece, and N−1 more rounds share them. Nobody's link is idle and nobody is a bottleneck, which is why the time barely depends on how many people are at the table.
- **Tensor vs pipeline parallelism.** TP is four chefs on one dish, each chopping a quarter, who must sync after every step (needs a fast kitchen intercom: NVLink). PP is an assembly line: each station does its layers and passes the tray on, with little talking, but the line takes time to fill (the bubble).
- **Disaggregation is separate kitchens for prep and service.** Prep (prefill) is a burst of heavy chopping. Service (decode) is a long trickle of plates. Sharing a kitchen means a big prep order stalls the plates.
- **Roofline:** every kernel is either "waiting on the truck" (memory) or "waiting on the factory" (compute). Optimising the wrong one does nothing.

---

## (c) Concepts that make great live visualisations (driven by the real engine)

1. **Roofline plot with live dots.** Each kernel or step from the running engine is plotted at its measured intensity and throughput against the M4's measured roofline. A batch-size slider moves the decode-matmul dot from the bandwidth slope toward the compute ceiling.
2. **Batching timeline (Gantt).** Rows are request slots and the x-axis is decode iterations. A toggle switches static vs continuous batching on the same recorded trace, so idle cells (static) visibly turn into new requests (continuous). Chunked prefill shows up as prefill chunks interleaved into each iteration, with an ITL sparkline below.
3. **Paged KV block table.** On the left, a grid of physical KV blocks coloured by owning sequence. On the right, each sequence's block table, with arrows. Live while generating: blocks get allocated, freed, shared (ref-count badges) on prefix hits, and copy-on-write on fork. A side counter compares wasted memory against contiguous allocation.
4. **Radix tree of cached prefixes.** It grows as requests arrive, with node size proportional to tokens, hits flashing, and LRU leaves fading on eviction.
5. **Speculative decoding playback.** Drafted tokens appear in grey, then turn green (accepted) or red (rejected, resampled), showing p and q bars per token. A γ slider sits next to a theoretical-vs-measured speedup chart.
6. **Animated ring all-reduce.** N nodes in a circle with chunked vectors. Step through ReduceScatter then AllGather, with a byte counter per link. Toggle ring vs tree vs naive, and a message-size slider shows the latency/bandwidth crossover using numbers measured from the engine's own `Communicator` (and the H100 nccl-tests run as overlay).
7. **Megatron TP dataflow.** Matrices split by column/row across N coloured devices, with tensors flowing through one layer and a flash on each all-reduce. A per-token time breakdown shows compute vs communication, so you can watch TP *hurt* GPT-2 small at batch 1 and help a 7B model.
8. **Pipeline bubble diagram.** A stages × time grid for GPipe vs 1F1B vs interleaved, with sliders for p and m, a bubble fraction (p−1)/m readout, and memory-per-stage bars.
9. **Throughput vs latency frontier.** A live scatter from the load generator as the request rate ramps, with the SLO box drawn and goodput shaded. It compares engine versions along the hill-climb.
10. **Memory budget bar.** Weights | KV | activations | free, updating as batch and context change, with a "max concurrent sequences" readout. It shows where GQA and int8 KV would move the line.

---

### Key sources
- Scaling book: https://jax-ml.github.io/scaling-book/ (roofline, sharding, inference chapters)
- Ultra-Scale Playbook: https://huggingface.co/spaces/nanotron/ultrascale-playbook
- Megatron-LM: https://arxiv.org/abs/1909.08053 , https://arxiv.org/abs/2104.04473 , https://arxiv.org/abs/2205.05198 ; GPipe https://arxiv.org/abs/1811.06965 ; ZeRO https://arxiv.org/abs/1910.02054
- NCCL perf notes: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md ; cudarc https://github.com/coreylowman/cudarc ; candle TP example (above) ; MLX distributed (above)
- Orca https://www.usenix.org/conference/osdi22/presentation/yu ; vLLM https://arxiv.org/abs/2309.06180 ; SGLang https://arxiv.org/abs/2312.07104 ; Sarathi-Serve https://arxiv.org/abs/2403.02310 ; DistServe https://arxiv.org/abs/2401.09670 ; Splitwise https://arxiv.org/abs/2311.18677 ; Mooncake https://arxiv.org/abs/2407.00079
- Speculative: https://arxiv.org/abs/2211.17192 , https://arxiv.org/abs/2302.01318 , Medusa https://arxiv.org/abs/2401.10774 , EAGLE https://arxiv.org/abs/2401.15077 , EAGLE-3 https://arxiv.org/abs/2503.01840
- llm.c https://github.com/karpathy/llm.c ; nanoGPT https://github.com/karpathy/nanoGPT ; nanochat https://github.com/karpathy/nanochat
- Pricing: https://modal.com/pricing ; secondary price roundups (vendor-affiliated, verify live): https://tech-insider.org/runpod-vs-lambda-vs-vast-ai-2026/ , https://www.spheron.network/blog/lambda-cloud-h100-pricing-2026/
- MBU / inference metrics: https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices
