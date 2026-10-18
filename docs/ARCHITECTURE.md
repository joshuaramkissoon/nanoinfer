# Architecture

## Crates

```
crates/
  core/   nanoinfer-core: the engine. Pure Rust, no fs / threads / clocks.
          Builds for wasm32-unknown-unknown.
  cli/    nanoinfer-cli: binary `nanoinfer`. mmaps weights, owns the clock,
          runs generate / bench / tokenize.
  wasm/   (later) wasm-bindgen wrapper around core for the in-browser figures.
scripts/  fetch_weights.sh (HF download), gen_reference.py (HF reference fixtures)
tests/fixtures/  committed reference logits + token ids (~800 KB)
bench/results.jsonl  one JSON line per bench run (the hillclimb's history)
```

The core takes everything as bytes. `Model::from_safetensors(&[u8])` and
`Tokenizer::from_vocab_and_merges(&[u8], &[u8])` don't care whether the bytes came from
an mmap (CLI) or a `fetch()` (browser).

### Core modules

| module | what |
|---|---|
| `safetensors` | ~80-line parser: u64 header length, JSON header, raw data |
| `weights` | `Model` = `Config` + `Vec<f32>` weights. HF `Conv1D` `[in,out]` matrices are transposed to `[out,in]` at load |
| `tokenizer` | byte-level BPE + a hand-written GPT-2 pre-tokenizer (the regex has a look-ahead) |
| `kernels` | `Kernels` trait + `Reference` (scalar) implementation of matmul, layernorm, gelu, softmax, attention, embed, residual |
| `stages` | `Stage` enum, `Session` (per-conversation state), `blocks` (shared pieces of a GPT-2 block), one file per forward strategy |
| `trace` | `Trace` sink trait, `Op` descriptors with FLOP/byte estimates, `Clock`, ready-made `Recorder` |
| `sampler` | argmax, top-k probabilities, temperature sampling with a seeded xorshift RNG |

## Sessions and stages

```rust
let model = Model::from_safetensors(&bytes)?;
let mut s = Session::new(&model, Stage::KvCache);
let logits = s.step(&model, &prompt_ids, &mut NoTrace)?;     // prefill
let logits = s.step(&model, &[next], &mut NoTrace)?;         // decode, repeat
```

Every stage has the same API and must produce the same logits. What differs is how
much work a `step` does: `Naive` recomputes the whole context, `KvCache` only the
new tokens. `Session` doesn't borrow the model (you pass `&Model` per call), so a
wasm wrapper can own both without self-referential structs.

A stage is a pair: a forward strategy (`stages/naive.rs`, `stages/kv_cache.rs`)
and a kernel set (`kernels::Reference`, later `Simd`, ...). Both are generic
parameters (`forward::<K: Kernels, T: Trace>`), so each `Stage` variant
monomorphises into its own specialised code. Runtime selection costs one `match`
per forward call, and nothing in the inner loops.

### Rule: never overwrite an old kernel or stage

The blog runs the *historical* code side by side. An optimisation adds a new kernel
set or a new forward file plus a new `Stage` variant. It never edits the old ones.
`Kernels` methods default to the reference implementation, so a new set overrides
only what it changes.

### How to add a stage

1. Add a variant to `Stage` (`stages/mod.rs`), with its name in `Stage::name`,
   `Stage::ALL` and `Stage::kernels`.
2. New kernel? Add `kernels/<name>.rs` and a ZST `impl Kernels for <Name>`, overriding
   just the ops you speed up. New strategy? Add `stages/<name>.rs` with a
   `forward<K, T>` built from `stages::blocks`.
3. Add the match arm in `Session::step` (and in `Session::with_max_seq` if it needs a
   KV cache).
4. `cargo test --release`. The reference tests iterate `Stage::ALL`, so the new stage is
   checked against HF automatically.
5. `nanoinfer bench --stage <name>`, then write the LOG.md and WRITEUP.md entries.

## Trace design

The forward pass is generic over `T: Trace` and reports:

- `forward_start/end(ForwardInfo)`: stage, phase (prefill/decode), new tokens, rows
  actually computed (`n_computed`: the whole context for naive), context length
- `op_start/end(Op)`: kind (`embed|ln|matmul|attn|gelu|residual|kv_write|lm_head`),
  name (`qkv`, `fc_proj`, ...), layer, dims, and **analytic** FLOPs and bytes from
  the shape
- `kv_cache(len, capacity)`
- `logits(&[f32])` if `wants_logits()` (for top-k bars)
- `attention(layer, head, query_pos, probs)` if `wants_attention()` returns
  `Some((layer, head))`. Recomputed for that one head only when requested, so the
  fast path never materialises attention maps.

**Zero overhead:** `Trace` has an associated `const ENABLED: bool`. `NoTrace` sets it to
false and every call site is `if T::ENABLED { ... }`, so untraced builds contain no
tracing code. Measured: attaching a full timing `Recorder` costs ~1% at stage 1.

**No clock in core:** wasm32-unknown-unknown has no `Instant`. Events carry no
timestamps. Sinks that want timing take a `Clock` (any `Fn() -> f64` works):
`Instant` in the CLI, `performance.now()` in the browser. `trace::Recorder<C: Clock>`
records a flat list of `OpEvent { op, forward, start, duration }` and
`ForwardEvent`, plus the latest top-k and attention map.

**On disk:** `nanoinfer generate --trace-out t.jsonl [--attention L,H]` writes JSONL
events (`run_start`, `op`, `kv`, `capture`, `token`, `run_end`) following the schema in
`research/03-explorable-writeup-and-wasm.md` §3.4. Recorded traces from this machine
can back the blog figures when live mode isn't available. Differences from that
schema today: attention is a single `attn` op (not split into scores / softmax / mix),
ops are named by string rather than an interned index, and the format is JSON rather
than binary records.

## Benchmarking

`nanoinfer bench --stage S --prompt-len P --tokens N --runs R` times one prefill
forward and N single-token decode forwards per run, separately. After warmup it
reports median ± stddev and the decode ceiling (`--bandwidth-gbs`, default 120 for the
M4, ÷ `Config::decode_weight_bytes`), and appends a JSON line to `bench/results.jsonl`.
`--with-trace` attaches a timing `Recorder`, to measure tracing overhead.
The ceiling uses `Stage::weight_format().decode_bytes(cfg)` (matrices at the stage's
bits per weight, biases/LayerNorm at f32). Bench also prints achieved decode GB/s
and prefill/decode GFLOP/s (matmul FLOPs only).

`nanoinfer membw --sweep 1,4,10` measures real CPU read bandwidth (each thread
sums its own slice of a 512 MB buffer), which gives the realistic ceiling.

`nanoinfer ppl --stage S --reference unrolled` runs WikiText-2
(`scripts/fetch_wikitext.sh`) through the decode path in windows and reports
perplexity, plus KL / top-1 agreement / max |Δlogit| against an f32 reference stage
in lockstep. This is the quality gate for lossy (f16, quantised) stages.
