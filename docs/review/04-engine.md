# Review 04: the engine, end to end

Scope: `crates/core`, `crates/cli`, `crates/metal`, `crates/webgpu`, `crates/wasm`,
`scripts/`, `bench/`, the tests, plus `docs/LOG.md`, `docs/ARCHITECTURE.md`,
`bench/headtohead.json` (and the LOG's "Head-to-head" section) and
`research/DISTILLED.md`. Read-only review at commit `12ed363` (19 Oct 2026).

**What I ran** (one job at a time, `nice -n 10`, load 1.8–3.3 throughout):

| command | result |
|---|---|
| `cargo clippy --workspace --all-targets` | clean apart from 6 style warnings (webgpu `too_many_arguments`, `manual_is_multiple_of`, 4× `chunks_exact_to_as_chunks` in examples and tests) |
| `cargo test -p nanoinfer-core --release -- --test-threads=2` | 18 unit + 6 reference tests pass (14 s). **Without `--features threads`**: see §2.4 |
| `cargo test -p nanoinfer-core --release --features threads` | 22 unit + 6 reference tests pass |
| `cargo test -p nanoinfer-metal --release` | 3 batch + 1 kernel + 6 reference tests pass (20 s) |
| `cargo test -p nanoinfer-webgpu --release` | 2 reference tests pass (27 s). No stage skipped on this M4 |
| `cargo check` of `crates/wasm` (wasm32, `--features webgpu`) and core for wasm32 | builds |
| core for `x86_64-apple-darwin` | not checked: the target isn't installed. Nothing has ever built the CPU engine for a non-aarch64 native target |

No benchmarks were run. All performance numbers below come from `docs/LOG.md` and
`bench/*.json`. Numbers marked **est.** are my own back-of-envelope figures.

---

## 0. Summary

The engine does what it set out to do. It is a readable, heavily tested GPT-2
engine with three backends and ~60 stages you can still run. Its Metal decode is
the fastest of the four engines at f16, Q8 and Q4 on this machine, at matched
bytes and quality. The LOG is unusually honest: dead ends, machine noise and a
corrected probe are all written down. Clippy is close to clean. Every test
passes.

Five problems matter more than the rest:

1. **Stage metadata is spread across ~8 `match` statements** in
   `stages/mod.rs`, and they have already drifted apart. That drift is the
   `head_format` bug: q8-prefill reports 173 MB/token when it actually reads
   139.5 MB. Metal's `Recipe` struct is the better pattern, and core should copy it.
2. **Packed weights belong to the `Session`, not the model.** Every CPU session
   re-quantises the whole model and holds its own copy, on top of the f32 `Model`
   that always stays resident. That is fine for a benchmark and blocks a server.
3. **The GPT-2 graph is written out by hand at least 7 times** (core
   `naive`, `kv_cache`, `packed`, `kv16`; Metal `encode_forward` and the fused
   decode path; Metal `batch`; WebGPU `plan`). Supporting Llama or Qwen means
   touching every copy unless the GPT-2 stages are frozen as a "museum" and
   modern models go onto one or two current paths.
4. **The modern-model blockers are concrete and fixable.** The loader is F32 only
   with hard-coded `n_head = n_embd / 64`. `Config` has 5 fields and no
   `n_kv_head`, `d_ff`, norm or activation type, or RoPE. The tokenizer only
   reads `vocab.json` + `merges.txt` with the GPT-2 regex. The Metal attention
   kernels hard-code `HD = 64` (9 places).
5. **Open-source hygiene is missing.** There is no `LICENSE` file (Cargo says
   MIT), no CI, and no `rust-version`. The README's "where we are" table stops at
   stage 1 and the ARCHITECTURE doc predates WebGPU. 536 of 541
   `bench/results.jsonl` lines carry a `-dirty` sha. No remote is configured yet.

---

## 1. Architecture

### 1.1 Crate boundaries

```
core   (no fs / threads / clocks; wasm32-clean)  ← cli (mmap, clock, QoS, every bench)
  ↑                                               ← metal  (objc2-metal, macOS only; empty elsewhere)
  └──────────────────────────────────────────────← webgpu (wgpu 30, native Metal + browser)
wasm (separate workspace) → core (+ webgpu behind a feature)
```

What works:

- **Core is honestly portable.** It has no clock (the `Clock` trait is injected),
  takes bytes rather than paths, and keeps the `threads` feature off by default.
  The spin pools are `cfg`-gated away on wasm, and wasm threads go through rayon
  instead (`kernels/spin.rs::par_for`). Core does build for wasm32 (verified).
- **The GPU crates reuse core** for the loader, tokenizer, config and sampler, and
  keep only GPU maths themselves. That is the right boundary.
- **`crates/wasm` is a separate workspace**, so native builds never pull
  wasm-bindgen. Its `stages()` reads `Stage::ALL`, so new CPU stages appear in the
  browser automatically.

What doesn't:

- **The CLI is a grab-bag.** It holds `metal_serve.rs` (582 lines, the
  continuous-batching scheduler), `metal_spec.rs` (speculative decoding),
  `metal_batch.rs`, `accel.rs` (a `Kernels` impl), `ppl.rs` and the benches. The
  scheduler and the speculative decoder are engine features, not CLI features. A
  server would want them in a library crate.
- **Core still contains parallel runtimes.** `pool.rs` and `fastpool.rs` exist
  under `threads`. That is fine for the hillclimb, but it means core ships two
  global, process-wide spin pools plus a rayon pool. See §2.3.
- `crates/wasm` is outside the workspace, so `cargo clippy --workspace` and
  `cargo test` never touch its 1,100-line `lib.rs`, and it has no tests of its own.

### 1.2 The Stage / Kernels / Trace design

**Kernels.** This is a ZST trait with default methods that fall back to
`reference`, and it is monomorphised per stage. It is a good teaching device:
stage *n* is visibly "stage *n−1* plus one override". But it only covers f32
weights (`matmul(&[f32] w)`). From stage 8 on, the real interface is
`PackedMatrix` + `Par` + `Head`, so the forward passes go through two
abstraction layers, and `Kernels::matmul` has become dead weight in the
`LowLat` and `AttnNeon` sets. Those sets override it with `spin::matmul`, but
the packed forward never calls it.

**Trace.** `const ENABLED` with `if T::ENABLED` guards is genuinely zero-cost
(measured at ~1% with a full recorder). The FLOP and byte counts are analytic,
which suits a learner. One bug was found: `Stage::decode_bytes`/`head_format` are
part of what the trace and bench report, and they are wrong for q8-prefill
(§2.6). Note that `Recorder` grows without bound over a long generation.

**Stage.** This is a flat enum with 20 CPU variants, and its properties live in
separate `match`es:

| property | where |
|---|---|
| name | `Stage::name` |
| kernel label | `Stage::kernels` |
| threaded? | `Stage::multithreaded` |
| block format | `Stage::weight_format` |
| head format | `Stage::head_format` (**missing `Q8Prefill`**) |
| packed variant | `PackedWeights::for_stage` + the 7-variant `PackedWeights` enum |
| uses f16 KV | `uses_kv16` in `Session::with_max_seq` |
| forward | a ~160-line `match` in `Session::step` (lines 471–630), 12 near-identical packed arms of `match &self.packed { X(p) => packed::forward::<K, M, _, P, T>(…7 args…), _ => unreachable!() }` |

Adding a stage means editing 6–8 places, and forgetting one compiles fine. The
`head_format` bug is exactly that failure. The Metal crate already solved this
problem: `MetalStage::recipe()` returns a `Recipe` struct of orthogonal knobs,
and one `encode_forward` reads the knobs. That design scales (20 Metal stages
cost one line each) and reads well.

**Recommendation.** Keep the enum, since the browser and CLI depend on its
names, but have each variant map to one `const StageSpec { name, kernels,
threads, blocks: WeightFormat, head: HeadKind, kv: KvKind, forward: ForwardKind }`
in a single table. Derive `weight_format`, `head_format`, `decode_bytes` and
`uses_kv16` from it. The `Session::step` arms can collapse into a macro, or into
a `dyn`-free dispatch on `(ForwardKind, PackedWeights)`. This is a mechanical
change: no old stage's code path changes, so the never-overwrite rule still
holds.

### 1.3 Is "never overwrite a stage" scaling?

At ~60 stages it is still working, but it is starting to bend:

- **Kernel files: yes.** One file per idea (`q8.rs`, `q8hilo.rs`, `q8gemm.rs`,
  `attn_neon.rs`, …) is easy to read, easy to diff and easy to teach from.
- **Forward passes: not really.** `packed.rs::forward` re-inlines
  `blocks::qkv`/`proj_and_mlp`, because `blocks` is written against
  `&Layer` f32 weights and `K::matmul`. `kv16.rs::forward` (319 lines) then
  re-inlines it again with a different cache. So three CPU files carry the same
  11-op block sequence and the same trace descriptors. A bug in the
  trace-descriptor arithmetic, or a Llama port, means three edits.
  Fix: make `blocks` generic over a `Linear` trait (implemented by `&[f32]` +
  `K` and by every `PackedMatrix` + `P`), and over a `KvStore` trait
  (f32 row-major, f16 head-major). One block function then serves stages 1–15.
  Old stages keep their exact kernels, so their measured behaviour is unchanged.
- **Runtimes.** "Never touch stage 6's pool" produced `fastpool.rs`, a 90%
  copy of `pool.rs`. That was defensible for measurement, but it means two global
  pools, two sets of spin threads, and two places to fix the panic-safety issue
  in §2.3.
- **Dead code.** `stages/packed.rs::FoldedHead` (the LN-gain folding head) is
  defined and documented but used by no stage. `metal::Matmul::Matvec` is matched
  but no recipe selects it. The `metal-fused-ln` stage is a documented dead end
  that is kept runnable on purpose, which is fine. Mark the unused items
  `#[doc(hidden)]` with a LOG pointer, or delete them.

**Suggested policy going forward.** Stages 0–15, metal-* and webgpu-* stay
frozen as the "GPT-2 museum", byte for byte. New work (modern models, serving,
distributed) targets one current path per backend and is allowed to evolve. The
museum is what the blog runs. The current path is what users run.

### 1.4 Core's wasm-compatibility constraints

These are all sound and well documented:

- no `Instant`: the `Clock` is injected;
- no fs: `from_safetensors(&[u8])`;
- no threads by default: the `threads` feature, plus `cfg(not(target_family = "wasm"))` on the pools.

Costs to be aware of:

- **Everything is copied into `Vec<f32>`.** On wasm that is unavoidable for an
  unaligned fetch buffer. On native it means the CLI mmaps 500 MB and then
  allocates and transposes 500 MB more. A 1B model at f32 needs 4–5 GB of RAM
  before packing. For modern models, add a "pack while loading" path that
  converts bf16 straight into the target format, one tensor at a time.
- Inline `asm!` (`fcvtn`/`fcvtl` in `kv16.rs`) is used because f16 NEON
  intrinsics aren't stable. That is fine, but readers need the comment that's
  already there.
- **No runtime CPU feature detection.** `#[target_feature(enable = "dotprod")]`
  functions in `q8.rs`, `q4.rs`, `q8hilo.rs` and `q8gemm.rs` are called without
  `is_aarch64_feature_detected!`. Every Apple chip has dotprod. A Cortex-A72
  (Raspberry Pi 4) or Graviton 1 would hit an illegal instruction. x86 gets the
  portable fallbacks only (no AVX2), and nobody has ever compiled them natively
  (§2.4).

### 1.5 Metal and WebGPU duplication of forward logic

Each GPU backend writes its own forward: Metal has `forward.rs` (828 lines,
covering the prefill/naive path and the fused/pipelined decode path) and
`batch.rs` (617 lines, ragged and paged). WebGPU has `forward.rs::plan`
(766 lines). This is unavoidable to a degree: the GPU sides build command
streams, not function calls. But the *graph* (which ops, in what order, with
which shapes) is the same GPT-2 block every time.

Two facts make it worse than necessary:

- Metal has two forwards that do the same job. `MetalSession` (single stream)
  and `Batch` (ragged, paged) both cover decode and prefill. `Batch` already
  covers batch-1 decode, spec verification and chunked prefill, and its kernels
  are batch-invariant (tested). For new work, `Batch` should become *the* Metal
  forward and `MetalSession` the museum.
- `Recipe` constraints are only written in doc comments: "`lanes` needs
  `Fusion::Epilogue`", "`kv_f16` needs `Attn::Flash`". A bad custom recipe through
  the public `MetalEngine::with_recipe` reaches `unimplemented!()` in `linear()`
  at encode time. Add `Recipe::validate() -> Result` and call it in `with_recipe`.

A shared, backend-agnostic `ModelGraph`, meaning a list of ops with shapes that
every backend lowers, would remove most of this. It is also a big refactor, and
it would make the museum code less direct for learners. My recommendation: don't
do it for GPT-2. Do it once, for the modern-model path (§3.4), and only for
`core::packed` + `metal::batch` (+ webgpu later).

### 1.6 Config and model generality: what blocks Llama / Qwen

| assumption | where | blocks |
|---|---|---|
| F32 tensors only | `safetensors.rs:80` | every modern checkpoint (bf16) |
| one file | `weights.rs`, CLI `load_model` | sharded `model-0000x-of-0000y.safetensors` + `index.json` (Llama 3.2 1B is one file, 3B is sharded) |
| `n_head = n_embd / 64` inferred from shapes; no `config.json` | `weights.rs:70–73` | everything |
| `Config` = {vocab, ctx, embd, head, layer} | `config.rs` | GQA (`n_kv_head`), `head_dim ≠ C/H` (Qwen3: 128), `d_ff ≠ 4C` (Qwen2.5-0.5B: 4864 vs 896), eps, rope θ/scaling, tie flag, bias flags |
| `n_params`, `decode_bytes` hard-code `12·C²` per layer | `config.rs`, `WeightFormat::decode_bytes`, `MetalStage::decode_weight_bytes` | every ceiling for non-GPT-2 models |
| LayerNorm with bias, tanh-GELU, learned `wpe` | `blocks.rs`, every backend's shaders | RMSNorm, SiLU-gated MLP (SwiGLU), RoPE |
| fused `[3C, C]` QKV, Conv1D transpose | `weights.rs::Layer` | separate q/k/v (`nn.Linear`, already `[out, in]`), with K/V narrower than Q under GQA |
| tied head = `wte` | `Model`, `PlainHead`, `RescoredHead`, Metal `Weights` | untied `lm_head.weight` (SmolLM2 and Qwen2.5-0.5B are tied, Llama 3.2 1B is tied, Llama 3 8B and Qwen2.5-7B untied) |
| KV row width = C | `KvCache`, `KvCache16`, Metal/WebGPU KV buffers | GQA, where KV is `n_kv_head·head_dim` (4–7× smaller: a feature) |
| `HD = 64` | 9 `const HD = 64` in `attn.metal` and friends; `attend64*` fast paths (with portable fallbacks) on CPU | Qwen3 (128). SmolLM2, Qwen2.5-0.5B and Llama 3.2 1B are all 64, so this is not the first blocker |
| GPT-2 BPE: `vocab.json` + `merges.txt`, GPT-2 regex, only `<\|endoftext\|>` special | `tokenizer.rs` | Llama 3 (`tokenizer.json` only, different regex with `\p{N}{1,3}` and case-insensitive contractions, many special tokens), chat templates |
| `n_ctx ≤ 1024`, stack scratch `[0f32; 1024]` (heap fallback exists) | `lowlat.rs`, Metal buffers sized `n_ctx` | 32k contexts need paged or growable buffers everywhere (Metal `Batch` already pages) |
| WebGPU 128 MiB binding limit | webgpu weights | Qwen2.5's 151,936 × 896 head at Q8 is ~145 MB, so it must be split across two bindings |

The good news: once the loader and config are generalised, the *kernels* mostly
carry over. RMSNorm is LayerNorm without mean and bias. RoPE is a small
elementwise kernel applied to q and k before the KV write (or fused into the QKV
epilogue on Metal, next to `EPI_QKV`). GQA is an index change in attention
(`kv_h = h / (n_head / n_kv_head)`). SwiGLU is a `[2·d_ff, C]` fused gate|up
matmul plus a `silu(a)·b` epilogue. The plan is in §3.4.

---

## 2. Correctness and robustness

### 2.1 Numerics

- f32 stages match HF to < 1e-3 max |Δlogit| on CPU, Metal and WebGPU, and
  reproduce HF's 24-token greedy continuations. That is solid.
- **The tolerances are sensible but loose for f16.** The CPU f16 bound is 0.25
  max |Δlogit|, and quantised formats are judged by KL (Q8 < 0.01, Q4 < 0.1) on
  the next-token distribution of 4 prompts. The checklist in DISTILLED asked for
  f16 ~1e-2. The quality gate that really matters is `nanoinfer ppl` on
  WikiText-2 (Δppl, KL, top-1), and it is excellent, but it isn't in `cargo test`.
- **The rescored LM head mixes two logit sources.** Candidates get f16-exact
  logits and everything else keeps Q8-approximate logits. That is fine for greedy
  and for sampling (the tail mass is below e^-8). It does mean `logits` aren't a
  single consistent function, which matters for logprob output in a future API.
  Document it, or rescore the top-k you return.
- `sampler::argmax` ignores NaN unless index 0 is NaN, in which case it returns 0
  forever. That is harmless today, but a NaN check in debug builds would catch a
  bad kernel early.
- The reductions are deterministic per stage. Metal `Batch` is
  batch-invariant up to `MAX_NB` rows, by design and by test. That is a strong
  property and a good thing to advertise.

### 2.2 Tests: coverage gaps

| gap | consequence |
|---|---|
| **Core runs only *prefill* for stages 2–15** (`last_logits_match_hf_for_every_stage` does one `step` with the whole prompt). Token-by-token decode is only tested for `KvCache`. | The decode matvecs (`dot_q8`, `dot_q4`, two-level `q8hilo`, `attend64`, `attend64_f16`, the rescored head) are never compared against HF. They are only covered by kernel unit tests and by ppl runs. Metal *does* run token-by-token for every stage, so copy that loop. |
| Fixture prompts are 5, 10, 6 and 27 tokens. | CPU attention above the `lowlat` parallel thresholds, the `kv16` head-major cache at long context, and the 4×4 prefill tiles with ragged tails are not tested against a longer sequence. Metal has a context-300 test (`split_and_flash_attention_match_simd_at_long_context`); core has none. |
| No multi-session or multi-thread test of a `Session`. | The concurrent-dispatcher fix (§2.3) has unit tests on the pool, but nothing checks two sessions decoding at once. |
| `cargo test -p nanoinfer-core` **without `--features threads` doesn't compile `pool.rs`/`fastpool.rs` at all**, and runs stages 5–15 single-threaded. | Workspace `cargo test` unifies features (the CLI turns `threads` on), so the README command is fine. Per-crate test runs, and any future CI matrix, silently test the serial paths. |
| **Silent skips.** Every reference test returns early and passes when `weights/gpt2` is missing. Core and Metal honour `NANOINFER_REQUIRE_WEIGHTS=1`; **WebGPU doesn't**. WebGPU also `eprintln!`s and skips stages that `!supported(&gpu)` (subgroups, f16). | On a CI runner without weights or subgroups, the GPU test suites pass while testing nothing. Make the skip count visible (print a summary, fail when `REQUIRE_WEIGHTS` is set) in all three suites. |
| No test that `Stage::decode_bytes` equals the packed model's real `bytes()`. | That test would have caught the `head_format` bug. |
| `crates/wasm` has no tests and isn't in the workspace. | Browser-only regressions are found by hand. |

### 2.3 Unsafe audit

There are 55 `unsafe` sites in core, plus Metal/WebGPU FFI and the wasm SIMD.

**Pointer arithmetic in the kernels** (`tile_4x4`, `prefill_cols`, the
`attend*` functions, `narrow`/`widen`). Each has a `# Safety` contract, and the
callers check lengths up front (for example the `assert!` in `spin::matmul`).
The contracts rely on `d_in` being a multiple of the tile or block size in some
paths. Examples: `Q8Matrix::pack` uses `as_chunks::<32>()` and silently drops a
tail, and Metal `matmul_tiled` reads `x[gr*d_in + k0 + k]` with no `k < d_in`
guard. GPT-2, Qwen2.5 and Llama dims are all multiples of 32, so this is latent.
Add `assert!(d_in % QK == 0)` in `pack` and in the GPU weight upload.

**`OutPtr` (`unsafe impl Send + Sync`).** Disjoint-column writes are correct by
construction in every caller I read (`par_cols` chunk bounds, attention units,
rescoring candidates being distinct indices). It is idiomatic and fine.

**Spin pools** (`pool.rs`, `fastpool.rs`). I walked through the protocol:

- *Job lifetime:* the dispatcher publishes `job` (Release) before `state`
  (SeqCst). A worker that observes epoch E reads E's job, or a later one only if
  E has finished, in which case its CAS (pool) fails or its claim lands in E+1
  and reads E+1's job (fastpool). A chunk can't be claimed for a finished job,
  because the dispatcher doesn't return until every chunk is counted. Correct.
- *fastpool's over-claiming:* `fetch_add` past `n` is bounded by one extra claim
  per thread per job, so with `n < 2^15` the 16-bit index can't carry into `n`.
  Correct, but non-obvious. The `assert!(n_chunks < 1 << 15)` is load-bearing;
  say so in the comment.
- *Lost wakeup:* sleeping++ (SeqCst) under the lock, a re-check of the epoch, then
  wait; the dispatcher stores the epoch (SeqCst) and then reads `sleeping`.
  Correct.
- *The earlier deadlock* (commit `8c08b53`): two concurrent dispatchers could
  deadlock. The fix is a `try_lock`, and the loser runs its job serially. That is
  safe, but **silently degrades a second session to one thread**. A server
  running two CPU sessions on two threads would see one of them run ~5× slower,
  with no signal. Log it once, or make the pool per-session.
- **Not panic-safe (real issue).** If `f` panics on a worker, that worker thread
  dies, its chunk is never counted, and the dispatcher spins forever, which looks
  like a deadlock. If `f` panics on the *dispatching* thread inside `work()`,
  `run` unwinds and frees the `Job` stack frame while workers may still be
  executing it: a use-after-free. Fix: wrap the chunk call in
  `catch_unwind`, record the payload, count the chunk anyway, and re-raise after
  the join. Alternatively abort on panic inside pool workers. Kernel panics are
  realistic here: an index out of bounds on a bad shape.
- Three runtimes can coexist in one process: rayon, `pool` and `fastpool`, each
  with N threads. `bench` only uses one at a time. A process that runs stage 6
  and stage 11 alternately (the web lab natively, or a sweep) has 2N spinning
  threads for up to `SPIN_LIMIT + YIELD_LIMIT` (≈20k iterations, i.e. a few ms of
  yields) after each switch. That is worth knowing when reading interleaved A/B
  numbers.

**Metal** (`gpu.rs`):

- Buffer lifetimes are fine. `commandBuffer()` (not
  `…WithUnretainedReferences`) retains every bound buffer, so a dropped `Buf` can't
  be freed under in-flight work. `buffer_no_copy` is `unsafe` with a correct
  contract, and it is currently unused (weights are copied, see §1.4).
- **`Buf::slice` / `Buf::write` are safe functions that can data-race the GPU.**
  Their correctness rests on "we always wait before reading". That holds for
  every caller today, but it is an unsound public API: `nanoinfer_metal::gpu` is
  `pub`. Make them `unsafe`, or only hand out slices from a `Pending::wait()`
  result.
- A failed command buffer `panic!`s (`Pending::wait`, `Runner::flush`), and
  allocation failures `expect("alloc")`. For a server these should be `Result`s.
  A GPU watchdog timeout or OOM shouldn't kill the process.
- `Runner` has no `Drop`. If encoding panics mid-forward, the encoder is released
  without `endEncoding`, which Metal's validation layer treats as fatal. That
  only matters together with the point above.
- Metal shaders compile at runtime with `MTLMathMode::Safe` on every
  `Gpu::new()`, and every `MetalEngine` creates its own `Gpu`. That is fine, but
  the draft and target `BatchEngine`s in `metal-spec` compile the 3,000-line source
  twice. Caching or sharing one `Gpu` would help.

**WebGPU** uses `Rc` and is `!Send` by design (browser). OK.

### 2.4 Error handling and bad input

- Core's `Error` enum is small and good. `Session::step` validates empty input,
  out-of-vocabulary tokens and context overflow. Metal's `check()` mirrors it.
- **Panics on malformed checkpoints.** `wte_shape[0]` and `[1]` are indexed
  without a length check (an empty `shape` array panics).
  `u64 as usize` and `8 + n` aren't checked (fine on 64-bit, since `get` catches
  it, but a debug build panics on overflow). Shape products aren't overflow
  checked. None of this is exploitable, but a fuzz target over
  `SafeTensors::parse` + `Model::from_safetensors` would take ten minutes to write.
- The tokenizer is robust: `token_bytes` returns `&[]` for unknown ids, and
  decoding is lossy UTF-8, like HF.
- GPU errors are `String`s. A typed error enum would let a server tell user
  errors from device errors.

### 2.5 Benchmark methodology

The LOG learned most of these lessons the hard way (QoS on E-cores, a busy
machine, the probe timing bug, the "80 GB/s" GPU bandwidth that turned out to be
contention). What's still open:

1. **The sha doesn't identify the code**: 536 of 541 lines in
   `bench/results.jsonl` are `-dirty`. Record `git diff | sha256` with the line,
   or refuse to record from a dirty tree unless `--allow-dirty`. Also record
   rustc version, `RUSTFLAGS`/profile, load average (the headtohead harness does
   this; `bench` itself doesn't) and power state.
2. **Prefill is one sample per run**: one ~25 ms (CPU) or ~10 ms (GPU) forward,
   ×5 runs, which explains the "469 ± 119" style spreads. Time *k* prefills per
   run on fresh sessions and take the median.
3. **Allocation and page faults inside the timed prefill.** `Activations::ensure`
   allocates inside the first `step`, and the KV `vec![0.0; n]` is calloc'd
   (lazily mapped), so the first touch of every KV page happens during timed
   decode. That costs roughly 1% at P = 64 (est.) and more at long context. Run
   one throwaway `step` + `reset` before timing, or pre-touch the buffers.
4. **The repack happens per run.** `Session::new` re-quantises the whole model
   each run. It is untimed, but it churns ~300 MB of allocations between timed
   runs, which can move the allocator and cache state.
5. **Head-to-head ordering is fixed** (llama.cpp GPU → nanoinfer GPU → MLX →
   torch → CPU), the same in all three rounds. GPU clock state measurably depends
   on what ran before (that is llama.cpp's bimodality), and nanoinfer always runs
   straight after llama.cpp has warmed the GPU. Randomise the order per round, or
   interleave A/B pairs, as the LOG does internally.
6. **The load gate is ≤ 8 on a 10-core machine.** That is high for 10-thread CPU
   runs (rows were recorded up to 8.7). "Best of 3 rounds" for every engine
   mitigates this but doesn't remove it. For CPU rows, gate at ≤ 2 or report the
   per-round spread.
7. **QoS parity.** nanoinfer sets `USER_INTERACTIVE` on every thread. llama-bench
   doesn't by default, so its threads inherit the default QoS. That may help us at
   10 threads, where E-core placement decides the result. Rerun llama.cpp under
   `taskpolicy`, with `--prio`, or with the same QoS before claiming the 10-thread
   win. (Unverified: I didn't check llama.cpp's macOS thread QoS code path.)
8. **The pipelined decode isn't a streaming decode.** The metal-* decode
   numbers (`decode_greedy`) commit all 64 command buffers up front, with no
   host-side sampling, detokenisation or stop check between tokens. llama-bench
   synchronises per token, and MLX pipelines one token ahead. The comparison is
   fair as "what the GPU can do", and the LOG says so. But a chat server needs a
   per-token readback. At ~130 µs commit + wait per token (LOG probe) on a
   1.27 ms token, a per-token-sync mode would land around ~715 tok/s for
   metal-lanes (est.), still ahead of MLX's 694. Publish that number next to the
   787, and build the server on a depth-2 lookahead (commit t+1 before reading
   t) to recover most of the gap.
9. CPU `decode_tok_s` is the median of whole-run throughput, which matches
   llama-bench's statistic. The bench also prints a per-token median, which is
   more flattering. Make sure the web figures use the former when comparing
   engines (the headtohead JSON does).

### 2.6 Known bugs

- **`Stage::head_format` misses `Q8Prefill`** (`stages/mod.rs:161`). Q8Prefill
  uses `RescoredHead<Q8HiLoMatrix>` (`PackedWeights::Q8G`), but `head_format`
  falls through to `F16`. So `decode_bytes` reports 173 MB instead of 139.5 MB,
  and every bench line and ceiling percentage for q8-prefill is understated by
  about 20%. The headtohead JSON was hand-corrected. Fix it via the `StageSpec`
  table (§1.2) plus a test asserting `decode_bytes ≈ PackedModel::bytes()`.
- `Session::step` extends `tokens` before running the forward, so a forward that
  panics leaves `tokens` and the KV length inconsistent. That is cosmetic today.
- The README stage table (stage 1 only) and parts of ARCHITECTURE are stale
  (§4.2).

---

## 3. Performance

### 3.1 Where we stand (from the LOG and `bench/headtohead.json`)

| | nanoinfer | best other | gap |
|---|---|---|---|
| GPU decode Q4 + Q8 head | **787** (metal-lanes) | MLX 694, llama.cpp 622 | +13% / +27% |
| GPU decode Q8 | **612** | MLX 534 | +15% |
| GPU decode f16 | **364** | MLX 342 | +6% |
| GPU decode f32 | 176 (metal-mma) | llama.cpp **192** | −8% |
| GPU prefill P=64 | 4.5–6.7k | llama.cpp **10.3–11.1k** | **1.6–1.9×** |
| CPU decode Q4 / Q8 / f16 / f32 | 742 / 585 / 328 / 165 | llama.cpp 753 / 593 / 358 / 197 | −1.5% / −1% / −8% / −16% |
| CPU prefill Q8 / Q4 | 3.5k / 2.3k | llama.cpp **5.4k / 5.7k** | **1.5× / 2.5×** |

### 3.2 Concrete plans for the known losses

**(a) GPU prefill: f16 MMA staging, bigger tiles, double buffering.** Expected
gain: 1.4–1.7× (est.).

The current state is `matmul_mma*` (prefill.metal, quant.metal): 32 × 64 output
tiles, 128 threads, `simdgroup_float8x8`. X and W are staged as **f32** in
threadgroup memory, with one barrier pair per K step of 32 and no double
buffering. At P = 64 the qkv GEMM is only 2 × 36 = 72 threadgroups on a 10-core
GPU, so a single wave is under-occupied.

llama.cpp's `kernel_mul_mm` dequantises to **half** in threadgroup memory and
multiplies `simdgroup_half8x8` into `simdgroup_float8x8` accumulators. That
halves threadgroup bytes and load instructions per MAC, lets the tile grow at the
same occupancy, and is "f16 math" even for f32 weights (which is why its "f32"
prefill is 8.3k).

Plan, as new kernels and a new stage:
1. `matmul_mma_h_{f16,q8_0,q4_0}`: dequantise W to half and convert X to half on
   the threadgroup store. Keep f32 accumulators.
2. Double-buffer the K loop (two threadgroup buffers, one barrier per step).
3. For P ≤ 128, use 32 × 32 tiles or split-K on `fc_proj` (K = 3072, only 768
   outputs) to fill all 10 cores. For P ≥ 128, reuse `matmul_mma2` (64 × 64) from
   `batch.metal`, which `examples/gemm_sweep.rs` already tuned.
4. Flash-attention prefill is already simdgroup-matrix. Give it the same half
   staging for Q and K.

Quality: f16 rounding of activations before the matmul. Gate it with
`quant_quality` and `ppl`. The f16 weight stage already showed +0.01% ppl.

Files: `crates/metal/src/shaders/quant.metal`, `prefill.metal` (new kernels in a
new file, `prefill_h.metal`, to keep the rule), `forward.rs::linear` (a new
`Matmul::SimdHalf`), a `MetalStage::PrefillHalf` recipe, and `batch.rs::gemm`.

**(b) CPU prefill: i8mm `smmla` on repacked weights.** Expected gain: Q8 prefill
3.5k → 5–6k, Q4 2.3k → ~5k (est.).

`q8gemm.rs` computes a 4 × 4 tile with `sdot` (16 MACs per instruction).
`smmla` (FEAT_I8MM, present on M2 and later) multiplies a 2 × 8 by an 8 × 2
int8 block into a 2 × 2 int32 tile: 32 MACs per instruction, the same register
footprint, and fewer horizontal adds.

It needs weights interleaved so that 8 consecutive k from 2 (or 4) output rows
are adjacent. ggml's `q8_0_4x8` stores 4 rows × 8 bytes, repeating, with the 4
row scales together. The activations get the same treatment, as rows quantised
to Q8 and paired.

Plan:
1. `kernels/q8mm.rs`: `Q8x4Matrix` (pack = Q8_0 then interleave 4 rows), with a
   `smmla` 4 × 4 or 8 × 4 micro-kernel through `asm!` (`smmla` isn't in stable
   `std::arch` intrinsics; check `vmmlaq_s32` stability on 1.99, it may be
   available). Keep decode on the existing `dot_q8` reading the same interleaved
   layout, or keep a second copy for decode. Repacking costs nothing at load,
   but two copies double memory.
2. Q4: unpack nibbles to int8 into a stack panel per 4-row strip (as decode
   already does into registers) and feed the same `smmla` tile. That removes the
   f32-dequantise prefill path that makes q4-kv16 prefill 2.3k.
3. Add `is_aarch64_feature_detected!("i8mm")` and fall back to `sdot`.
   This is also the moment to add the missing dotprod detection (§1.4).

Files: new `kernels/q8mm.rs`, new `Stage::Q8Mmla` / `Q4Mmla`, `stages/mod.rs`
spec table, and `quant.rs` (interleave helper).

**(c) CPU quant blocks: f16 scales instead of f32.** Expected gain: −6–9% bytes,
+3–6% decode (est.).

`BlockQ8 { d: f32, qs: [i8; 32] }` is 36 B, against ggml's 34 B. Q4 is 20 B
against 18 B. A packed 34-byte struct misaligns the int8 loads, so do what Metal
did: **struct-of-arrays**, with `qs: Vec<[i8; 32]>` and `d: Vec<f16 bits>` per
row. Loads stay 16-byte aligned, and one `fcvtl` widens 4 or 8 scales at once.
Decode is ~50–60 GB/s against ~86 GB/s achievable, so it isn't purely bandwidth
bound. Expect about half of the byte saving to show up in tok/s. Quality is
unchanged (scale rounding to f16 is ≪ the int8 step).

Files: new `quant.rs` types (`BlockQ8Soa`), new `kernels/q8s.rs` / `q4s.rs`,
and new stages.

**(d) CPU f16 and f32 onto the low-latency stack.** Expected gain: +10–15% decode
(est.).

f16 (328) and fast-ops f32 (165) still use stage 6's `SpinPar`, plus stage 7
attention with the f32 row-major KV. The Q8 move to `FastPar` was worth +13%
(315 → 356). An `f16-fast` stage =
`kv16::forward::<AttnNeon, F16Matrix, PlainHead<F16Matrix>, FastPar>`, and its
f32 equivalent, is a few lines in the stage table because every piece exists.
Expected f16 ~360–375, level with or ahead of llama.cpp's 358 (est.).

**(e) Metal f32 decode: a `metal-pipelined-f32` recipe.** Expected gain: 176 →
~190–200 tok/s (est.).

metal-mma predates GPU argmax, epilogue fusion and pipelining. The `mve_*_f32`
epilogue kernels already exist (`fused.metal: EPI_SET("f32", WF32, 4)`), so this
is
`Recipe { dtype: F32, head_dtype: F32, ..pipelined }` plus `attn: Flash`, one
line in `MetalStage::recipe`. f16 pipelined reaches 75% of ceiling. At 75–80%,
f32 would land at 182–194, closing the gap to llama.cpp (192).

### 3.3 Remaining ideas, ranked

| idea | expected | notes |
|---|---|---|
| **Prompt-lookup (n-gram) speculative decoding** on Metal `Batch` | 1.3–2.5× on repetitive or structured text, ~1.0× on prose (est. from the measured α table) | The verifier exists (`metal_spec.rs`, k+1-row verify ≈ 1.27× a step up to 8 rows). The draft becomes a suffix match in the context, at zero GPU cost, so c ≈ 0 and speedup ≈ E/v(k+1). No second model, which suits a 124M target where a draft model doesn't exist. Greedy-only acceptance is fine for now. |
| **CPU fused epilogues** (bias+GELU into `fc`, residual into `proj`/`fc_proj`, KV write into `qkv`, LN into `quantize_activations`) | +5–10% CPU decode (est.) | The LOG puts ~2.4 ms per token of the CPU budget in fixed cost across ~49 parallel sections. Fusion cuts that to ~25 and removes 4 activation passes. Metal got +5–10% from the same move. |
| **f16 KV everywhere** | +7% short / +60% at 900 ctx (measured on q8) | CPU f16, f32 and fast-ops still use the f32 cache. Free with (d). WebGPU has `kv16`. |
| **Batch > 1 on CPU** | ~B× aggregate throughput up to the compute ridge (~B = 4–8 for Q8 `sdot`, est.) | Decode reads every weight per token. B rows share the read. `Q8GemmMatrix`'s 4 × 4 tile already *is* a batch-4 decode kernel. What's missing is a `Batch`-like ragged session on CPU, which is needed anyway for a CPU server. |
| **Q4_K-style quants** (super-blocks of 256, 6-bit scales and mins) | Δppl +8% → ~+3–4% at ~4.5 bpw (llama.cpp lore, unverified on GPT-2) | Q4_0 on GPT-2 costs +7–8% ppl. K-quants buy back most of it at the same bytes. They are more complex (two-level scales), so put them on the current path only. |
| Q6_K/Q8 head on CPU Q4 stages | already done via the rescored head | |
| Fused ln_f + head + argmax on Metal | < 1% (LOG analysed it) | Skip. |
| SME on M4 for CPU prefill (FMOPA, streaming mode) | f32 prefill 2.2k → Accelerate-level 2.7k+ (est.) | Research-grade (inline asm, streaming-mode ABI). Only for the f32 museum story. |
| Pad vocab to 50304 | small | DISTILLED law 4. Helps tile edges on GPU. |

### 3.4 What a modern 0.5–1B model needs

Target order: **SmolLM2-360M** (Llama arch, 960 wide, 32 layers, 15 heads / 5 KV
heads, head_dim 64, tied, GPT-2-like size) → **Qwen2.5-0.5B** (896 wide, 24
layers, 14 / 2 heads, head_dim 64, `d_ff` 4864, QKV bias, tied, 152k vocab, RoPE θ
1e6) → **Llama-3.2-1B** (2048 wide, 16 layers, 32 / 8 heads, tied, llama3 RoPE
scaling).

Steps:

1. **Loader.** Parse `config.json` (CLI side; core takes the struct). Add
   safetensors `BF16`/`F16` → f32 conversion per tensor, `model.safetensors.index.json`
   shards, and an `Arch` name map (HF Llama/Qwen tensors are `nn.Linear`
   `[out, in]`, so no transpose). Concatenate q|k|v into one fused
   `[C + 2·kv_dim, C]` matrix and gate|up into `[2·d_ff, C]`. **Pack while
   loading** so a 1B model never exists in f32. Files: `config.rs`,
   `safetensors.rs`, `weights.rs`, new `arch.rs`.
2. **Config v2.** Add `n_kv_head`, `head_dim`, `d_ff`, `norm: {LayerNorm, RmsNorm}`,
   `eps`, `act: {GeluTanh, SwiGlu}`, `pos: {Learned, Rope{θ, scaling}}`,
   `tie_embeddings`, `qkv_bias`, `bos/eos`. Base `n_params`/`decode_bytes` on it.
3. **Kernels.** `rmsnorm`, `rope` (an f32 cos/sin table computed in f64 at load,
   since precision matters past 8k positions), `silu_mul` epilogue, and GQA in
   `attend*` (`kv_h = h / group`). The KV cache becomes `n_kv_head·head_dim`
   wide: Qwen2.5's KV is 7× smaller than an MHA cache of the same width.
4. **Forward.** One generic block over `Linear`/`KvStore` traits (§1.3), on
   the packed CPU path and on Metal `Batch`. Not on the museum stages.
5. **Tokenizer.** A `tokenizer.json` reader (BPE model, merges, added tokens),
   the Llama 3 / Qwen pre-tokenizer regex (written by hand as now, with tests
   against HF on a few thousand lines), special tokens, and a minimal chat
   template (ChatML for Qwen, the Llama 3 header format). Full Jinja is out of
   scope.
6. **Quality gates.** Fixtures from `gen_reference.py` for the new arch, ppl
   against our own f32, plus KL/top-1. Qwen2.5 has large activation outliers, so
   check Q8 *activation* quantisation (the CPU decode quantises x to Q8) before
   trusting Q4 numbers.
7. **Ceilings (est., 105 GB/s practical GPU).** Qwen2.5-0.5B Q8 ≈ 525 MB/token →
   ~200 tok/s; Q4 blocks + Q8 head ≈ 345 MB → ~300 tok/s. Llama-3.2-1B Q4 + Q8
   head ≈ 0.85 GB → ~125 tok/s. At these sizes per-token fixed cost matters less
   and the matvecs become bandwidth-bound for everyone. Expect the head-to-head
   gaps to narrow, as the LOG predicts.
8. **WebGPU.** Split the 145 MB Q8 head across two bindings (128 MiB limit), and
   shard the downloads (DISTILLED §3).

---

## 4. Code quality and open-source readiness

### 4.1 Readability as teaching material

These are strengths, and they are real:

- Module docs explain *why* before *how*. Examples: the KV-cache rationale in
  `kv_cache.rs`, the layout note in `weights.rs`, the head-major argument in
  `kv16.rs` with numbers, and the pool's protocol in `pool.rs`/`fastpool.rs`.
- Constants cite the LOG experiment that chose them
  (`RescoredHead::MARGIN`, "swept 2–15"). The ZST kernel-set pattern makes each
  stage's delta obvious.
- Names are consistent and short. Comments use units (µs, GB/s, bytes).

Weaknesses:

- **`stages/mod.rs` (636 lines) is the first file a learner opens**, and half of
  it is the 12-arm `Session::step` match. Collapse it (§1.2), and add a top-of-file
  table "stage → forward → matrix → head → KV → pool" that mirrors
  `StageSpec`.
- Stage numbering diverges from file names (`lowlat` = stage 11, `attn_neon` =
  13, `kv16` = 14, `q8gemm` = 15) and from the Metal and WebGPU numbering. A
  `docs/STAGES.md` (or a generated table) mapping every stage to its file, its
  LOG section and its headline number would help readers a lot. The web lab
  probably wants that data anyway.
- Large files: `wasm/lib.rs` 1,100 lines, `metal/forward.rs` 828,
  `webgpu/forward.rs` 766, `batch.metal` 29 KB. They read fine but would benefit
  from splitting along the Recipe axes (decode fused vs prefill vs attention).
- Several hot functions carry `#[allow(clippy::too_many_arguments)]` with 8–10
  positional `usize` arguments (`n, d_in, d_out, b, e`). That suits a kernel, but
  a `MatShape { n, d_in, d_out }` would make call sites self-describing for
  learners.

### 4.2 Docs accuracy

- **README** is stale. Its "Where we are" table lists stages 0–1 only. There is
  no mention of Metal, WebGPU, the browser, metal-serve, metal-spec or the
  head-to-head. The quick start is correct. Rewrite it around the LOG's summary
  tables and the head-to-head verdict.
- **ARCHITECTURE.md**: `wasm/` is described as "(later)" but exists. There is no
  `webgpu/` entry. The core module table lacks `quant`, `pool`, `fastpool`,
  `kernels/packed` and the `PackedMatrix`/`Par`/`Head` traits, which carry
  stages 8–15. "How to add a stage" lists 5 steps but the real count is ~8
  places. The Metal paragraph doesn't mention `Batch`, paged KV or serving. Bench
  docs don't mention `metal-bench`, `metal-batch`, `metal-serve`, `metal-spec`,
  the `webgpu` bench or `scripts/headtohead`.
- The core crate description ("GPT-2 … runs on wasm32") is accurate.
- LOG: excellent. One nit: the top-of-file "1 thread, f32" machine note
  describes round 1 only.

### 4.3 Licence, CI, toolchain

- **No `LICENSE` file**, even though `license = "MIT"` is in Cargo. Add
  `LICENSE` (MIT, matching Cargo) before the repo goes public. Check that the
  vendored ideas (ggml block layouts) need no notice. They're formats, not code,
  so a credit line in the README is courtesy, not obligation.
- **No CI.** Proposed `.github/workflows/ci.yml`:
  - `macos-14` (arm64): `cargo fmt --check`, `cargo clippy --workspace
    --all-targets -D warnings`, `cargo test --workspace --release` with
    `NANOINFER_REQUIRE_WEIGHTS=1`. Cache `weights/gpt2` (~525 MB) with
    `actions/cache`, keyed on `fetch_weights.sh`. GitHub's arm64 macOS runners
    expose a paravirtualised Metal device. Treat Metal and WebGPU tests as
    best-effort (feature support may differ, so print the skip counts) and
    **never benchmark on CI**.
  - `ubuntu-latest` (x86_64) plus `ubuntu-24.04-arm`: core + CLI build and
    tests (the portable fallbacks and the cfg paths have never been compiled
    natively off Apple).
  - wasm: `cargo build -p nanoinfer-core --target wasm32-unknown-unknown` and
    `crates/wasm` with `--features webgpu`.
- **Toolchain.** Core uses `slice::as_chunks` (stable since 1.88). Add
  `rust-version = "1.88"` to the workspace, or a `rust-toolchain.toml`. The
  wasm threads build needs nightly; pin it in `build.sh`.

### 4.4 Reproducibility of benchmarks

`scripts/headtohead/` is a real, scripted harness: setup, run, warm check,
ppl, aggregate. Few hobby projects have one. Gaps: dirty shas (§2.5), a
llama.cpp commit that is pinned for the source build but not for Homebrew's
(build 11429 is recorded, which helps), fixed run order, and no `machine.json`
(macOS version, power source, `pmset` thermal state). The `bench/*.json` files
are generated data, committed on purpose for the web page. That is fine, but
mark them as generated (a header field or a README in `bench/`).

### 4.5 Repo hygiene

- `.git` is 8.7 MB, 259 tracked files, and the largest tracked file is 348 KB
  (`bench/results.jsonl`, `web/public/recorded/native.jsonl`). The logit
  fixtures are 4 × 200 KB. Healthy.
- `.gitignore` covers `target/`, `weights/`, `*.bin`, `*.safetensors` and
  `.venv/`. `web/` has its own. `weights/` and `target/` carry
  `.metadata_never_index` (good, after the Spotlight incident).
- Untracked: `web/public/lab/`, `web/screenshots/`, `docs/review/`. Decide
  whether `lab/` is committed.
- **No git remote is configured**, although `Cargo.toml` names
  `github.com/joshuaramkissoon/nanoinfer`.

### 4.6 Commit history

73 commits, all on 18 October, one author, **no AI co-author trailers**
(consistent with the OSS rule). Messages are descriptive and stage-scoped
("Stage 9: Q8_0 blocks with int8 sdot decode, f16 LM head"). Some later commits
are large mixed "learn:" and "web:" commits, which is fine. Consider tagging each
stage's commit (`cpu-stage-09`, `metal-stage-08a`) so readers can check out
history. DISTILLED asked for that ("structural steps get git tags").

---

## 5. Missing pieces for the original goals

DISTILLED's ladder, against what exists:

| ladder step | status |
|---|---|
| 0–17 (CPU, Metal, WebGPU, browser) | **done**, beyond plan in places (rescored head, lanes, kv8) |
| 18 serving harness (TTFT/TPOT, Poisson) | **done** as `metal-serve` (simulated arrivals, no network) |
| 19 continuous batching | **done** in simulation (Orca-style, chunked prefill, token budget) |
| 20 paged KV | **done** (16-position pages, block tables, `truncate` rollback). **Prefix caching / refcount / CoW: not done** |
| 21 chunked prefill | done |
| 22 speculative decoding | **done** with draft models (gpt2 → medium/large, up to 4.25×). **n-gram / prompt-lookup: not done**; acceptance is greedy-only (no rejection sampling for temperature > 0) |
| 23 collectives (`Communicator`, ring/tree, link emulator) | **not started** |
| 24 TP / PP | **not started** |
| 25 disaggregated prefill/decode | not started |
| R1/R2 rented GPUs (CUDA/NCCL) | not started; there is no CUDA backend |
| a real server (SSE, OpenAI API) | **not started** |
| modern model support | **not started** (§3.4) |
| training | not in scope so far (no backward pass, no autograd) |

### 5.1 Distributed inference

Proposed order (all on one Mac first, then the rented step):

1. **The `Communicator` trait** in a new `crates/dist`: `rank()`, `size()`,
   `send/recv(&[f32])`, `all_reduce_sum`, `all_gather`, `barrier`. Backends:
   **shared memory** (N processes, a `memmap2` ring buffer + futex/spin flags:
   fast, shows the algorithm), **TCP** (loopback first, then two Macs over
   Thunderbolt or 10 GbE), and an **emulated link** (added α latency and B
   bandwidth) so the 2(N−1)/N law can be swept on one machine.
2. **Ring and tree all-reduce**, plus an `nccl-tests`-style bench (bus
   bandwidth vs message size) feeding V14. Expected: for GPT-2 TP decode's
   24 × 3 KB all-reduces per token, latency dominates. At ~20–50 µs per TCP hop
   that is +0.5–1.2 ms per token against a ~2.4 ms CPU token (est.), i.e. **TP
   makes GPT-2 small slower**. That is the teaching result (law 10). Shared memory
   should get it to ~2–5 µs per hop (est.), roughly break-even.
3. **Tensor parallel (Megatron).** Split qkv and fc by columns (heads per rank,
   with the KV cache split by head) and attn_proj and fc_proj by rows: 2
   all-reduces per layer. This needs the "one generic block over a `Linear`
   trait" refactor (§1.3), with `ShardedLinear` as an implementation. Don't fork
   another forward.
4. **Pipeline parallel.** Give each process a layer range, send the activation
   row, and use micro-batches for throughput. This is easy on top of `Batch`
   (rows already ragged). It shows the pipeline bubble.
5. **Later:** expert parallelism (the web course already covers MoE), toy
   disaggregation (prefill process → KV pages over TCP → decode process; the
   paged KV makes the transfer unit natural), then R1: a CUDA backend is a
   whole new crate (cudarc) and is the most expensive item in this list.

### 5.2 A real server

There is a good base: `Batch` (ragged, paged, batch-invariant) and the
continuous scheduler in `cli/src/metal_serve.rs`. Plan:

1. Move the scheduler into a library crate, `crates/serve` (it depends on metal
   for now, behind a backend trait so CPU can join after "batch > 1 on CPU").
2. **Per-row sampling.** `Batch::step` returns greedy tokens only. Add logits
   readback for sampled rows, or GPU top-k + sampling, with temperature, top-k,
   top-p, min-p, seed, stop sequences, `max_tokens` and logprobs. This is the
   biggest engine change the server needs.
3. **The HTTP layer** (axum + tokio): `POST /v1/completions` and
   `/v1/chat/completions` (OpenAI schema, `stream: true` SSE), `GET /v1/models`,
   and `/metrics` (TTFT, TPOT, queue depth, KV pages in use). Run one engine thread
   owning the `Batch`, and talk to request handlers over channels. Metal objects
   stay on one thread, which also sidesteps the global spin-pool contention.
   Streaming detokenisation must buffer partial UTF-8 (`token_bytes` is already
   byte-level).
4. **Lifecycle.** Cancellation (free the sequence's pages), admission control
   on KV pages (it exists in simulation), preemption by recompute when pages run
   out, and prefix caching (hash full pages, refcount, CoW). The paged KV makes
   this incremental.
5. **Pipelining.** Use a depth-2 lookahead (commit forward t+1 before reading t's
   tokens) to keep most of metal-pipelined's advantage while streaming (§2.5,
   item 8).

---

## 6. Top 15 engine improvements (impact × effort)

S = hours, M = a day or two, L = a week or more. Ordered by value per effort,
with the prerequisites for modern-model and serving work pulled forward.

| # | improvement | impact | effort | files |
|---|---|---|---|---|
| 1 | **Single `StageSpec` table** for CPU stages, deriving `weight_format`, `head_format`, `decode_bytes`, `uses_kv16`, packed kind and kernels label; collapse the `Session::step` arms. Fixes the q8-prefill byte bug. Add a test that `decode_bytes` matches the packed `bytes()`. | med (correctness of every reported ceiling; learner clarity) | S | `crates/core/src/stages/mod.rs`, `crates/core/tests/reference.rs` |
| 2 | **OSS basics:** `LICENSE` (MIT), `rust-version`, README rewrite (current tables, head-to-head verdict), ARCHITECTURE refresh (webgpu, wasm, packed traits, batch, serve, spec), git remote, stage tags | high for the launch | S | `LICENSE`, `Cargo.toml`, `README.md`, `docs/ARCHITECTURE.md` |
| 3 | **CI**: macOS arm64 (fmt, clippy `-D warnings`, workspace tests with cached weights and `NANOINFER_REQUIRE_WEIGHTS=1`), Linux x86_64 and arm64 core builds, wasm builds. WebGPU tests honour `REQUIRE_WEIGHTS`; every suite prints its skip count | high (prevents silent regressions and vacuous test passes) | S–M | `.github/workflows/ci.yml`, `crates/webgpu/tests/reference.rs`, `crates/metal/tests/*.rs` |
| 4 | **Close the test gaps:** token-by-token decode against HF for *every* CPU stage, a context-300 CPU test (attention thresholds, kv16), a two-session concurrency test, a `Model::from_safetensors` fuzz/robustness test | high (stages 2–15 decode kernels are untested against HF today) | S | `crates/core/tests/reference.rs`, new `crates/core/tests/robustness.rs` |
| 5 | **Pool panic safety** (`catch_unwind` per chunk, count it, re-raise after the join), a one-time warning when a concurrent dispatcher falls back to serial; make `Buf::slice`/`write` `unsafe`; return `Result` instead of `panic!` on Metal command buffer errors; `Recipe::validate()` | med (no hangs or UB on a kernel bug; needed before a server) | S | `crates/core/src/pool.rs`, `fastpool.rs`, `crates/metal/src/gpu.rs`, `crates/metal/src/forward.rs`, `lib.rs` |
| 6 | **`metal-pipelined-f32`** recipe (epilogue f32 kernels already exist) | closes the f32 GPU decode loss (176 → ~190–200 est.) | S | `crates/metal/src/lib.rs` (+ a test row is automatic via `MetalStage::ALL`) |
| 7 | **CPU `f16-fast` / `f32-fast` stages** on `FastPar` + `AttnNeon` + kv16 | f16 decode 328 → ~360–375 (est.), f32 similar; level with llama.cpp | S | `crates/core/src/stages/mod.rs` (spec rows), possibly `stages/kv16.rs` for an f32-head variant |
| 8 | **Move packed weights out of `Session`** into a shared `Arc<Packed>` keyed by (model, format); sessions borrow it. Bench stops repacking per run | high for serving and memory (N sessions = 1 copy) | M | `crates/core/src/stages/mod.rs`, `stages/packed.rs`, `crates/cli/src/bench.rs`, `ppl.rs`, `crates/wasm/src/lib.rs` |
| 9 | **Metal half-staged prefill GEMM** (`simdgroup_half8x8`, double-buffered K, small-P tiles or split-K) | GPU prefill 1.4–1.7× (est.), closing most of the 1.6–1.9× gap | M | new `crates/metal/src/shaders/prefill_h.metal`, `forward.rs::linear`, `lib.rs` (`Matmul::SimdHalf`, new stage), `batch.rs::gemm` |
| 10 | **CPU i8mm `smmla` prefill** on 4-row interleaved Q8, Q4 unpacked to int8 on the same tile; runtime detection of `dotprod`/`i8mm` | CPU prefill Q8 3.5k → 5–6k, Q4 2.3k → ~5k (est.) | M | new `crates/core/src/kernels/q8mm.rs`, `quant.rs`, `stages/mod.rs`, existing `q8.rs`/`q4.rs`/`q8hilo.rs`/`q8gemm.rs` (feature checks) |
| 11 | **Benchmark hygiene:** refuse or flag dirty-tree records (store a diff hash), record rustc, flags, load and power; k prefills per run; pre-touch buffers before timing; randomise the head-to-head order; QoS parity for llama.cpp; publish a per-token-readback ("streaming") Metal decode number | med (credibility of every published number) | S–M | `crates/cli/src/bench.rs`, `metal_bench.rs`, `scripts/headtohead/run.sh`, `aggregate.py` |
| 12 | **Model generality, part 1:** `Config` v2, bf16/f16 safetensors, shards, `config.json`, arch name maps, pack-while-loading, `tokenizer.json` + Llama 3/Qwen pre-tokenizer + special tokens | unlocks every modern model | M–L | `crates/core/src/config.rs`, `safetensors.rs`, `weights.rs`, `tokenizer.rs`, new `arch.rs`, `crates/cli/src/main.rs` (loading), `scripts/gen_reference.py` |
| 13 | **Model generality, part 2:** one generic block over `Linear` + `KvStore` traits; RMSNorm, RoPE, GQA, SwiGLU, untied head on the CPU packed path, then Metal `Batch`; SmolLM2-360M → Qwen2.5-0.5B → Llama-3.2-1B with HF fixtures and ppl gates | the headline feature for "useful engine"; ~200–300 tok/s Qwen2.5-0.5B on GPU (est. ceiling) | L | `crates/core/src/stages/blocks.rs` (generic), new `stages/llama.rs`, new `kernels/{rmsnorm,rope}.rs`, `kernels/attn_neon.rs` + `stages/kv16.rs` (GQA), `crates/metal/src/batch.rs`, new `shaders/llama.metal` (`HD` as a function constant) |
| 14 | **Prompt-lookup speculative decoding** on Metal `Batch` (an n-gram draft from the context, the existing k+1 verify) | 1.3–2.5× on structured text at no model cost (est.); works for single models where no draft exists (all of GPT-2 small, most 0.5B models) | S–M | `crates/cli/src/metal_spec.rs` → move the core loop into `crates/metal/src/spec.rs`, a new `--draft ngram` option |
| 15 | **Server crate:** move the continuous scheduler out of the CLI, add per-row sampling in `Batch`, an OpenAI-compatible `/v1/(chat/)completions` with SSE, cancellation, KV admission control, depth-2 pipelining; prefix caching after that | the original serving goal, turned into something people can point a client at | L | new `crates/serve/`, `crates/metal/src/batch.rs` (sampling, logits readback), `crates/cli/src/metal_serve.rs` (becomes a thin driver) |

Just outside the top 15: f16 block scales on CPU (−6–9% bytes), CPU fused
epilogues (+5–10%), CPU batch > 1 (a prerequisite for a CPU server), the
`Communicator` trait and ring all-reduce (§5.1, start of the distributed track),
Q4_K quants, and deleting or hiding `FoldedHead` and `Matmul::Matvec`.

**If only three things happen before launch:** #1, #2 and #3 (all S). **If
the next month is engine work:** #8 → #12 → #13 → #15. That turns the
GPT-2 museum into a small, fast engine for current 0.5–1B models with a
real API, while the museum stays intact for the blog.
