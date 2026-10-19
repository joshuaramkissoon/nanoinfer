# nanoinfer

A GPT-2 inference engine written from scratch in Rust, for CPU, Metal and WebGPU, made fast one measured step at a time.

It starts at a naive forward pass (0.37 tok/s) and climbs, one stage per idea, to 792 tok/s on an M4's GPU. Every
stage is still in the code and still runs, so you can compare any two. The journal of how it got there, including the
dead ends, is [`docs/LOG.md`](docs/LOG.md).

## What's in it

- **One engine, four backends.** A portable core (loader, byte-level BPE tokenizer, kernels) with NEON + threaded CPU
  stages, a Metal backend, a WebGPU backend (native and in the browser) and a WebAssembly build that runs in a tab.
- **About 60 stages, each a measured hypothesis.** KV cache, accumulator unrolling, NEON, tiling, a low-latency
  thread pool, f16 / Q8_0 / Q4_0 weights, a rescored LM head, an f16 KV cache, GPU kernels shaped for SIMD-groups,
  flash-decoding, batched and paged decode, continuous batching and speculative decoding.
- **Correctness first.** Every stage on every backend is tested against Hugging Face reference logits; lossy stages
  are gated on WikiText-2 perplexity, KL and top-1 agreement against f32.
- **Measured against the hardware.** Each result is reported as a share of its speed limit (memory bandwidth ÷ bytes
  read per token), and against llama.cpp and MLX in a randomised head-to-head on the same machine.
- **A course built on it.** [`web/`](web/) is an interactive course and lab that run the real engine in the browser,
  so every figure shows live numbers rather than mock-ups.

## The climb

Decode, GPT-2 124M, batch 1, one Apple M4. "% of limit" divides by memory bandwidth ÷ bytes read per token for that
stage's number format.

| Stage | What changed | Decode tok/s | % of limit |
|---|---|---|---|
| naive | f32, recompute the whole sequence per token | 0.37 | 0.15% |
| kv-cache | cache keys and values | 19 | 8% |
| unrolled | 16 independent accumulators (now memory-bound) | 101 | 41% |
| fast-ops | threads + low-latency pool, vectorised ops | 146 | 60% |
| q4-kv16 | 4-bit weights, f16 LM head and KV cache (CPU) | 497 | 40% |
| metal-lanes | 4-bit on the GPU, one command buffer per token | 792 | 59% |

## Results

GPT-2 124M on one Apple M4 (base: 4P + 6E cores, 10-core GPU, 24 GB). Same job for every engine: read a 64-token
prompt, then write 64 tokens greedily, batch 1. Each engine gets the same bytes per token where the formats allow it,
and its quality is checked against its own f32 model. Three rounds in a random order, strictly one engine at a time,
best round for everyone; re-run 19 Oct on a quiet machine (1-min load ≤ 2 before CPU jobs, ≤ 2.5 before GPU jobs).
Method, conditions, every row and the quality table: [`docs/LOG.md`](docs/LOG.md#head-to-head-re-run-19-oct-night)
and [`bench/headtohead.json`](bench/headtohead.json).

**GPU (Metal) decode, tok/s.** nanoinfer has three decode loops, and which one you compare matters:

- **per-token sync**: commit a token, wait, read it back, then encode the next. llama-bench's `tg` does exactly this
  (`llama_synchronize` after every token), so **this column is the apples-to-apples one against llama.cpp**.
- **one in flight**: token t+1 is committed before t is read back. A real server can run this (a stop is noticed one
  token late and rolled back); MLX's async `stream_generate` does the same, so it's the fair one against MLX.
- **pipelined**: all 64 tokens queued up front, the CPU never waits. What the GPU can do; no streaming server sees it.

| format | nanoinfer per-token sync | llama.cpp (syncs per token) | nanoinfer one in flight | MLX (one ahead) | nanoinfer pipelined |
|---|---|---|---|---|---|
| f32 | 179.5 | **192.0** | 191.1 | 177.5 | 194.6 |
| f16 | 315.2 | **334.0** | **360.0** | 339.8 | 360.9 |
| Q8 | 504.5 | **510.8** | **607.8** | 546.3 | 610.8 (WebGPU: 571) |
| Q4 + Q8 head | 611.2 | **636.3** | **789.1** | 700.2 | 792.1 |

**GPU prefill (P = 64), tok/s:** llama.cpp **8,040 / 9,967 / 10,336 / 10,604** (f32 / f16 / Q8 / Q4); nanoinfer
4,857 / 5,251 / 6,054 / 6,870; MLX 3,557 / 5,457 / 6,278 / 7,448.

**CPU, tok/s at each engine's best thread count**

| format | nanoinfer decode | llama.cpp decode | nanoinfer prefill | llama.cpp prefill |
|---|---|---|---|---|
| f32 | 168 | **199** | 2,274 | 1,231 (pure NEON), 2,729 (Accelerate) |
| f16 | 328 | **361** | 2,232 | **2,281** |
| Q8 | 578 | **596** | 3,475 | **5,579** |
| Q4 + Q8 head | 736 | **764** | 2,290 | **5,787** |

Where we stand:

- **Like for like with llama.cpp, nanoinfer is slightly behind at every format.** With per-token sync it trails by
  1% (Q8, a tie), 4% (Q4), 6% (f16) and 7% (f32). Reading each token back costs it ~0.36 ms, more than llama.cpp
  pays for the same round trip.
- **With one token in flight, it's the fastest of the four at f16, Q8 and Q4, and level at f32**: 789 tok/s at Q4 vs
  MLX 700 (+13%) and llama.cpp's per-token-sync 636; 608 at Q8 vs 546 / 511. On a model this small most of a token
  is fixed cost between kernels; the Metal climb removed most of it (one command buffer per token, argmax on the GPU),
  and keeping one token in flight hides the rest. That's a fair comparison with MLX (same idea); against llama.cpp
  it compares our server loop with its benchmark loop.
- **Prefill is the loss: 1.5–1.9× slower than llama.cpp on the GPU** (it stages its matrix tiles in half precision;
  ours are f32) and 1.6–2.5× on the CPU for quantised formats (it uses the M4's int8 matrix instructions; we use the
  older dot-product ones).
- **CPU decode is close but behind:** 3–4% at Q8 and Q4, 9% at f16, 16% at f32 (older stages that never moved onto
  the low-latency thread pool). At 10 threads the order flips (llama.cpp drops with the E-cores).
- WebGPU (the same WGSL through `wgpu`) reaches 571 tok/s at Q8 natively and 558 in Chrome on the same machine,
  about 93% of the matching hand-written Metal stage. In the browser's CPU fallback (wasm, one thread) the quantised
  stages now decode at ~195 tok/s with simd128 int8 dots, 1.5× the best f32 stage
  ([`docs/LOG.md`](docs/LOG.md#browser-stage-4-simd128-int8-dots-and-f16-widening-in-core-19-oct)).

All of this is GPT-2 small, where per-token fixed costs dominate, and that favours this design. On a 7B model every
engine would be bandwidth-bound and the gaps would shrink.

## Run it

Needs Rust 1.98 or later. Measured on Apple Silicon only (see the scope section below).

```sh
scripts/fetch_weights.sh        # GPT-2 124M from Hugging Face -> weights/gpt2 (~550 MB, gitignored)

# generate text on any stage (`--stage` takes any name in the LOG)
cargo run --release -p nanoinfer-cli -- generate --stage q8-kv16 --prompt "The meaning of life is" --tokens 40
cargo run --release -p nanoinfer-cli -- generate --stage naive --tokens 10 --profile    # per-op time breakdown

# benchmark a stage: prefill and decode tok/s, plus its share of the memory-bandwidth ceiling
cargo run --release -p nanoinfer-cli -- bench --stage q8-kv16 --tokens 64
cargo run --release -p nanoinfer-cli -- bench --stage metal-lanes      # Apple GPU (macOS): pipelined, streaming, lookahead
cargo run --release -p nanoinfer-cli -- bench --stage webgpu-vec       # wgpu, native

# quality gate for lossy stages: WikiText-2 perplexity, KL and top-1 against f32
scripts/fetch_wikitext.sh
cargo run --release -p nanoinfer-cli -- ppl --stage q8-kv16 --reference unrolled

# every stage, on every backend, against Hugging Face reference logits
cargo test --release -- --test-threads=2
```

Each bench appends a line to `bench/results.jsonl` with the commit, rustc, build flags, load average, QoS and power
source. It refuses to record when `crates/` or the Cargo files have uncommitted changes (`--allow-dirty` records anyway,
marked `git_dirty`; `--no-record` skips the file).

Without weights the reference tests print SKIP and pass. Set `NANOINFER_REQUIRE_WEIGHTS=1` to make that a failure
(the WebGPU suite also fails then if any stage is unsupported on the GPU).

The web course and lab (needs Node and `wasm-bindgen-cli`; details in [`web/README.md`](web/README.md)):

```sh
node web/scripts/to-f16.mjs     # once: the f16 weights the page downloads
cd web && npm install && npm run wasm && npm run dev
```

Then open the course at <http://localhost:5199/learn> or the lab at <http://localhost:5199/lab>.

Regenerate the reference fixtures (needs `uv`; nothing is installed globally):

```sh
uv run --with torch --with transformers --with numpy python scripts/gen_reference.py
```

## Repo layout

```
crates/core     the engine: loader, tokenizer, CPU stages 0-15, trace. No fs, threads or clock; builds for wasm32
crates/cli      `nanoinfer`: generate, bench, ppl, membw, plus the Metal batch / serve / speculative benches
crates/metal    Apple GPU backend (objc2-metal, MSL): single-stream stages and a batched, paged forward
crates/webgpu   WebGPU backend (wgpu 30, WGSL): runs natively and in the browser
crates/wasm     wasm-bindgen facade for the web page (its own workspace)
web/            the course (/learn) and the lab (/lab): React + Vite, driven by the engine in the tab
bench/          measured results the docs and the web page read (results.jsonl, headtohead.json, ...)
scripts/        weights and fixtures, plus the head-to-head harness (scripts/headtohead/)
research/       distilled notes from the engines, papers and explainers this draws on
docs/           LOG (the journal), WRITEUP (notes for the write-up), ARCHITECTURE
```

How the code fits together: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Scope

- **GPT-2 family only.** The loader, config, tokenizer and kernels assume GPT-2: f32 safetensors, LayerNorm, GELU,
  learned positions, a tied head, 64-wide heads. GPT-2 medium and large load too
  (`scripts/fetch_weights.sh openai-community/gpt2-medium`; the speculative-decoding bench uses them). Llama- and
  Qwen-style models are not supported yet.
- **Measured on one machine**, an M4. The CPU fast paths are aarch64 NEON with the dot-product extension, which every
  Apple chip has. Other targets get portable fallbacks that are not benchmarked or tested off Apple.
- **Not a server.** Continuous batching, paged KV and speculative decoding exist as benches on Metal (`metal-serve`,
  `metal-spec`), with simulated arrivals and greedy sampling only. There is no HTTP API.
- Why Rust: one language for the engine, the native benches and the browser build (core compiles to wasm unchanged),
  with SIMD intrinsics and no runtime between the code and the hardware.

## Docs

- [`docs/LOG.md`](docs/LOG.md): every stage's hypothesis, change and measured result, failures included.
- [`docs/WRITEUP.md`](docs/WRITEUP.md): working notes for the write-up and the course.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): crates, stages, traces and how to add a stage.
- The course: <http://localhost:5199/learn> when the web dev server is running.

## Stage tags

Every stage is still runnable at `HEAD` (`--stage <name>`), but each also has an annotated git tag on the commit
that introduced it, so you can read the code exactly as it was when that stage landed:

- CPU: `stage-00-naive`, `stage-01-kv-cache`, … `stage-15-q8-prefill` (numbered as in the LOG)
- Metal: `metal-01-naive` … `metal-11-spec` (numbered as the GPU stages in `docs/WRITEUP.md`), plus
  `metal-12-pipelined-f32`
- WebGPU: `webgpu-01-port`, `webgpu-02-round2`, `webgpu-03-vec`; the browser rungs: `browser-rungs`

```sh
git tag -n1 'stage-*'              # list them with a one-line description
git checkout stage-09-q8           # read the code as it was (git checkout - to come back)
git diff stage-08-f16 stage-09-q8  # exactly what one idea changed
```

## Licence

- Code: MIT, see [`LICENSE`](LICENSE) (matches `license = "MIT"` in `Cargo.toml`).
- Prose and figures (the course text in `web/src/learn/`, the writing in `docs/*.md`, and the figures as visual works):
  CC BY 4.0, see [`LICENSE-CONTENT`](LICENSE-CONTENT). Where a file mixes the two, the code is MIT and the prose it
  renders is CC BY 4.0.
