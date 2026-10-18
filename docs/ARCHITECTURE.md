# Architecture

## Crates

```
crates/
  core/    nanoinfer-core: loader, tokenizer, sampler, CPU stages 0-15, trace.
           Pure Rust: no fs, no clock, threads only behind the `threads` feature.
           Builds for wasm32-unknown-unknown.
  cli/     nanoinfer-cli, binary `nanoinfer`: mmaps weights, owns the clock and
           thread QoS, runs generate / bench / ppl / membw / accel / tokenize,
           and the Metal-only benches metal-batch, metal-serve and metal-spec.
  metal/   nanoinfer-metal (macOS only; empty elsewhere): GPT-2 on the Apple GPU
           through objc2-metal. MSL shaders in src/shaders/, compiled at startup.
  webgpu/  nanoinfer-webgpu: GPT-2 on WebGPU through wgpu 30 and WGSL. The same
           host code runs natively (wgpu's Metal backend) and in a browser.
  wasm/    nanoinfer-wasm: wasm-bindgen facade over core (+ webgpu behind the
           `webgpu` feature) for the web page. Its own workspace, so native
           builds never pull wasm-bindgen; built by crates/wasm/build.sh.
scripts/   fetch_weights.sh, fetch_wikitext.sh, gen_reference.py (HF fixtures),
           export_webgpu.py, headtohead/ (the llama.cpp / MLX / PyTorch harness)
tests/fixtures/  committed reference logits + token ids (~800 KB)
bench/     results.jsonl (one JSON line per bench run) and the summaries the
           web page reads: headtohead, webgpu, batching, serving, specdec .json
```

The GPU crates reuse core for everything that isn't GPU maths: the loader
(`Model`, already transposed to `[out, in]`), the tokenizer, the config and the
sampler. Each backend writes its own forward, because a GPU forward builds a
command stream rather than calling functions.

Core takes everything as bytes. `Model::from_safetensors(&[u8])` and
`Tokenizer::from_vocab_and_merges(&[u8], &[u8])` don't care whether the bytes
came from an mmap (CLI) or a `fetch()` (browser).

### Core modules

| module | what |
|---|---|
| `safetensors` | ~80-line parser: u64 header length, JSON header, raw data (F32 only) |
| `weights` | `Model` = `Config` + `Vec<f32>` weights. HF `Conv1D` `[in,out]` matrices are transposed to `[out,in]` at load |
| `config` | `Config` (vocab, ctx, embd, head, layer), parameter count, f32 decode bytes |
| `tokenizer` | byte-level BPE + a hand-written GPT-2 pre-tokenizer (the regex has a look-ahead) |
| `kernels` | the `Kernels` trait and one zero-sized kernel set per idea (`Reference`, `Unrolled`, `Neon`, `Tiled`, `Threaded`, `Spin`, `FastOps`, `LowLat`, `AttnNeon`); the packed-weight interfaces and formats (below) |
| `quant` | f16 conversion and the Q8_0 / Q4_0 block types |
| `pool`, `fastpool` | the spinning thread pools of stages 6 and 11 (`threads` feature, not on wasm) |
| `stages` | `Stage`, its table `Stage::spec`, `Session`, `blocks` (shared pieces of a GPT-2 block), one file per forward strategy (`naive`, `kv_cache`, `packed`, `kv16`) |
| `trace` | `Trace` sink trait, `Op` descriptors with FLOP/byte estimates, `Clock`, ready-made `Recorder` |
| `sampler` | argmax, top-k probabilities, temperature sampling with a seeded xorshift RNG |

## Sessions and stages

```rust
let model = Model::from_safetensors(&bytes)?;
let mut s = Session::new(&model, Stage::Q8Kv16);
let logits = s.step(&model, &prompt_ids, &mut NoTrace)?;     // prefill
let logits = s.step(&model, &[next], &mut NoTrace)?;         // decode, repeat
```

Every stage has the same API and must produce the same logits (within its
format's tolerance). What differs is how much work a `step` does and how. `Session`
doesn't borrow the model (you pass `&Model` per call), so the wasm wrapper can
own both without self-referential structs.

### The stage table

Each CPU stage is one row of `Stage::spec()`:

| field | what |
|---|---|
| `name` | the `--stage` name, also used by the browser |
| `kernels` | label of the kernel set |
| `threaded` | splits work across cores (with the `threads` feature) |
| `packing` | which packed copy of the weights the session builds: none (f32), f16, Q8_0 or Q4_0 blocks, with a plain f16 head or the rescored head |
| `kv` | none (stage 0), f32 row-major, or f16 head-major (stage 14+) |

`weight_format`, `head_format` and `decode_bytes` (the ceiling's denominator)
are derived from `packing`, the same value that picks the packed weights, so
the reported bytes per token can't drift from what decode really streams. A unit
test checks `decode_bytes` against the packed matrices' `bytes()` for every
stage.

A stage is then a forward strategy plus its type parameters. Up to stage 7 that
is `kv_cache::forward::<K: Kernels, T: Trace>`. From stage 8 it is
`packed::forward` or `kv16::forward` over `<K, M: PackedMatrix, H: Head, P: Par, T>`.
Each `Stage` variant monomorphises into its own code. Runtime selection costs
one `match` per forward call (in `Session::step`; still separate from the
table, see the improvement plan) and nothing in the inner loops.

### Quantised-weight interfaces (core)

- `PackedMatrix` (`kernels/packed.rs`): a `[d_out, d_in]` matrix in some
  format, with `pack(&[f32])`, `bytes()` and `matmul::<P>()`. Decode (n < 4 rows)
  runs the format's own matvec on the packed bytes. Prefill dequantises each
  4-row panel once and runs a 4×4 f32 tile, except `Q8GemmMatrix` (stage 15),
  which quantises the activations and runs an int8 `sdot` tile. Implementations:
  `F16Matrix`, `Q8Matrix`, `Q4Matrix`, `Q8HiLoMatrix` (two-level int8
  activations, for the head), `Q8GemmMatrix`.
- Blocks are ggml's Q8_0 / Q4_0 layout with an **f32** scale per 32 weights
  (36 / 20 bytes, so 9 / 5 bits per weight). The GPU backends use f16 scales.
- `Head` (`stages/packed.rs`): how `ln_f` + the tied LM head run.
  `PlainHead<M>` is one matrix. `RescoredHead<A>` scores every token with a Q8
  approximate head, then rescores the candidates within 8 nats of the top
  exactly in f16. `FoldedHead` (LayerNorm gain folded in) exists but no stage
  uses it.
- `Par`: which runtime a packed kernel splits columns on: `SpinPar` (stage 6's
  pool) or `FastPar` (stage 11's low-latency pool). A type parameter, so each
  stage keeps the runtime it was measured with.
- `PackedModel<M, H>`: every block matrix as `M` plus the head as `H`, built
  once per `Session` from the f32 `Model`. (Sessions don't share it yet, and the
  f32 `Model` stays resident too.)

### Rule: never overwrite an old kernel or stage

The course and the lab run the *historical* code side by side. An optimisation
adds a new kernel set, matrix type or forward file plus a new `Stage` variant.
It never edits the old ones. `Kernels` methods default to the reference
implementation, so a new set overrides only what it changes.

### How to add a CPU stage

1. Add a variant to `Stage` and to `Stage::ALL` (`stages/mod.rs`).
2. Add its row in `Stage::spec`. A new packing also needs a `Packing` variant,
   a `PackedWeights` variant and its arm in `PackedWeights::for_stage` / `bytes`.
3. New kernel? Add `kernels/<name>.rs`: a ZST `impl Kernels` overriding only
   the ops you speed up, or a new `PackedMatrix`. New strategy? Add
   `stages/<name>.rs` with a `forward` built from `stages::blocks`.
4. Add the arm in `Session::step`.
5. `cargo test --release -- --test-threads=2`. The reference tests and the
   byte-accounting test iterate `Stage::ALL`, so the new stage is checked
   against HF automatically.
6. `nanoinfer bench --stage <name>`, `nanoinfer ppl` if it's lossy, then the
   LOG.md entry.

## GPU backends

Both GPU crates follow the same pattern as core: a stage enum whose every
variant stays runnable, and a `Recipe` struct of orthogonal knobs that one
forward reads. Adding a GPU stage is usually one line: a new variant whose
`recipe()` turns on one more knob.

### Metal (`crates/metal`)

- `MetalStage::recipe()` → `Recipe { sync, dtype, head_dtype, matvec, matmul,
  norm, attn, fused, gpu_argmax, kv_f16, kv_q8, pipelined, lanes }`. Some
  knobs depend on others (`lanes`, `kv_f16` and `kv_q8` need epilogue fusion and
  flash attention); that is documented on the fields, not checked.
- `MetalEngine` / `MetalSession` (`forward.rs`): single-stream prefill and
  decode. `Sync::PerOp` commits and waits after every dispatch (stage 1);
  `PerForward` encodes a whole token into one command buffer; `pipelined`
  commits every greedy token's command buffer up front, each reading the token
  the previous one's GPU argmax wrote.
- `Batch` / `BatchEngine` (`batch.rs`): one ragged forward over many rows (one
  row = one token of one sequence), with a paged KV cache (16-position pages
  and per-sequence block tables). It serves batched decode, chunked prefill,
  continuous batching and speculative verification. Its kernels are
  batch-invariant up to `MAX_NB` rows (a sequence's tokens are bit-identical
  alone or batched; tested).
- Weights (`weights.rs`): each matrix is uploaded once in the stage's dtype.
  Q8_0 and Q4_0 are struct-of-arrays (all quants, then all f16 scales: 34 / 18
  bytes per 32 weights), so loads stay aligned.
- The schedulers live in the CLI for now: `metal_batch.rs` (throughput vs batch
  size), `metal_serve.rs` (static vs continuous batching on simulated
  arrivals) and `metal_spec.rs` (speculative decoding with a GPT-2 draft model,
  greedy acceptance).
- `examples/`: the dispatch / bandwidth probe, matvec, lanes and GEMM sweeps,
  and the quantisation-quality table.

### WebGPU (`crates/webgpu`)

- `WgpuStage::recipe()` → `Recipe { sync, dtype, head_dtype, matvec, matmul,
  par, ln64, fused, ln_fold, attn4, flashdec, kv16, wide_q8, gpu_argmax,
  resident }`. `forward.rs` plans a forward as a list of dispatches.
- Everything is `async` and `Rc`-based, because a browser can't block on the
  GPU. Natively, wrap calls in `gpu::block_on`.
- `WgpuStage::supported(&gpu)` reports stages that need subgroups or f16 the
  adapter lacks. The reference tests skip those, or fail when
  `NANOINFER_REQUIRE_WEIGHTS` is set.
- Weights: f32, f16 (two per u32) or Q8_0 (quants + a separate f16 scale
  buffer). The tied head is split into parts that each fit WebGPU's default
  128 MiB binding limit.
- `resident` greedy decode encodes many tokens per submit and reads them back
  once, which matters more in a tab, where every round trip crosses the GPU
  process.

## Trace design

The CPU forward is generic over `T: Trace` and reports:

- `forward_start/end(ForwardInfo)`: stage, phase (prefill/decode), new tokens,
  rows actually computed (`n_computed`: the whole context for naive), context
  length
- `op_start/end(Op)`: kind (`embed|ln|matmul|attn|gelu|residual|kv_write|lm_head`),
  name (`qkv`, `fc_proj`, ...), layer, dims, and **analytic** FLOPs and bytes
  from the shape (packed matmuls report the packed matrix's real bytes)
- `kv_cache(len, capacity)`
- `logits(&[f32])` if `wants_logits()` (for top-k bars)
- `attention(layer, head, query_pos, probs)` if `wants_attention()` returns
  `Some((layer, head))`. Recomputed for that one head only when requested, so
  the fast path never materialises attention maps.

**Zero overhead:** `Trace` has an associated `const ENABLED: bool`. `NoTrace`
sets it to false and every call site is `if T::ENABLED { ... }`, so untraced
builds contain no tracing code. Measured: attaching a full timing `Recorder`
costs ~1% at stage 1. (`Recorder` keeps every event, so it grows with the
length of a generation.)

**No clock in core:** wasm32-unknown-unknown has no `Instant`. Events carry no
timestamps. Sinks that want timing take a `Clock` (any `Fn() -> f64` works):
`Instant` in the CLI, `performance.now()` in the browser. `trace::Recorder<C:
Clock>` records a flat list of `OpEvent { op, forward, start, duration }` and
`ForwardEvent`, plus the latest top-k and attention map.

**On disk:** `nanoinfer generate --trace-out t.jsonl [--attention L,H]` writes
JSONL events (`run_start`, `op`, `kv`, `capture`, `token`, `run_end`). The web
page's recorded fallback (`web/public/recorded/`) packs the same events more
tightly: interned op definitions plus per-forward durations.

The GPU backends don't implement `Trace`. They report per-forward `RunStats`
(dispatches, command buffers or submits per token, timings) instead.

## Benchmarking

`nanoinfer bench --stage S --prompt-len P --tokens N --runs R` times one
prefill forward and N single-token decode forwards per run, separately. After
warmup it reports median ± stddev and the decode ceiling (`--bandwidth-gbs`,
default 120, the M4's spec, ÷ `Stage::decode_bytes`), and appends a JSON line to
`bench/results.jsonl`. It also prints achieved decode GB/s and prefill/decode
GFLOP/s (matmul FLOPs only). `--with-trace` attaches a timing `Recorder`, to
measure tracing overhead. `--stage metal-*` and `--stage webgpu-*` are routed to
the GPU benches with the same output; `nanoinfer webgpu-probe` measures wgpu's
dispatch and bandwidth costs.

`nanoinfer metal-batch`, `metal-serve` and `metal-spec` write
`bench/batching.json`, `serving.json` and `specdec.json` for the web page.

`nanoinfer membw --sweep 1,4,10` measures real CPU read bandwidth (each thread
sums its own slice of a 512 MB buffer), which gives the realistic ceiling.

`nanoinfer ppl --stage S --reference unrolled` runs WikiText-2
(`scripts/fetch_wikitext.sh`) through the decode path in windows and reports
perplexity, plus KL / top-1 agreement / max |Δlogit| against an f32 reference
stage in lockstep. This is the quality gate for lossy (f16, quantised) stages.

`nanoinfer accel` runs the KV-cache forward with Apple Accelerate's
`cblas_sgemv`/`cblas_sgemm` as the matmul (a CLI-only `Kernels` impl, since core
can't link a macOS framework): a reference point for what Apple's own BLAS
(AMX/SME) gets. `--threads N --qos interactive|utility|background` (global
flags) size the rayon pool and the core spin pools and set every thread's macOS
QoS class (background ⇒ E-cores).

`scripts/headtohead/` runs the same workload through llama.cpp, MLX and PyTorch
MPS: `setup.sh` (downloads, converts, builds llama.cpp), `run.sh`,
`warm_check.sh`, `ppl.sh`, then `aggregate.py` into `bench/headtohead.json`.
Run one heavy job at a time: parallel benches on one machine corrupt each
other's numbers.
