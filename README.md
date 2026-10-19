# nanoinfer

A GPT-2 inference engine written from scratch in Rust, for CPU, Metal and WebGPU, made fast one measured step at a time.

It starts at a naive forward pass (0.37 tok/s) and climbs, one stage per idea, to 787 tok/s on an M4's GPU. Every
stage is still in the code and still runs, so you can compare any two. The journal of how it got there, including the
dead ends, is [`docs/LOG.md`](docs/LOG.md).

## Results

GPT-2 124M on one Apple M4 (base: 4P + 6E cores, 10-core GPU, 24 GB). Same job for every engine: read a 64-token
prompt, then write 64 tokens greedily, batch 1. Each engine gets the same bytes per token where the formats allow it,
and its quality is checked against its own f32 model. Three rounds, strictly one engine at a time, best round for
everyone. Method, every row and the quality table: [`docs/LOG.md`](docs/LOG.md#head-to-head-nanoinfer-vs-llamacpp-mlx-pytorch-mps-18-oct-evening)
and [`bench/headtohead.json`](bench/headtohead.json).

**GPU (Metal), tok/s**

| format | nanoinfer decode | MLX | llama.cpp | nanoinfer prefill | llama.cpp prefill |
|---|---|---|---|---|---|
| f32 | 176 | 178 | **192** | 4,509 | **8,271** |
| f16 | **364** | 342 | 333 | 5,208 | **10,283** |
| Q8 | **612** (WebGPU: 569) | 534 | 502 | 5,856 | **10,706** |
| Q4 + Q8 head | **787** | 694 | 622 | 6,725 | **10,935** |

**CPU, tok/s at each engine's best thread count**

| format | nanoinfer decode | llama.cpp decode | nanoinfer prefill | llama.cpp prefill |
|---|---|---|---|---|
| f32 | 165 | **197** | 2,242 | 1,249 (pure NEON), 2,724 (Accelerate) |
| f16 | 328 | **358** | 2,237 | **2,357** |
| Q8 | 585 | **593** | 3,548 | **5,423** |
| Q4 + Q8 head | 742 | **753** | 2,274 | **5,683** |

Where we stand:

- **GPU decode is the win.** nanoinfer is the fastest of the four engines at f16, Q8 and Q4, at the same bytes and
  quality. On a model this small most of a token is fixed cost between kernels, and the Metal climb removed most of
  it (one command buffer per token, argmax on the GPU, no CPU wait between tokens).
- **Prefill is the loss: about 2× slower than llama.cpp.** 1.6–1.9× on the GPU (llama.cpp stages its matrix tiles in
  half precision; ours are f32) and 1.5–2.5× on the CPU for quantised formats (it uses the M4's int8 matrix
  instructions; we use the older dot-product ones).
- **CPU decode is roughly level.** Q8 and Q4 are within 1.5% of llama.cpp. f16 and f32 are 8% and 16% behind: those
  older stages never moved onto the low-latency thread pool the quantised ones use.
- f32 decode on the GPU was 8% behind llama.cpp with `metal-mma`, which predates the fused, pipelined decode path.
  `metal-pipelined-f32` (19 Oct) puts f32 on that path: +10% in an interleaved A/B (168 → 186 tok/s in a slower
  session), about level with llama.cpp's 192 once scaled to head-to-head conditions (est.; the re-run will confirm).
- **The GPU decode numbers are pipelined**: every token's work is queued up front and the CPU never waits, which is
  what the GPU can do but not what a chat server sees. Reading each token back before encoding the next costs ~0.37 ms
  per token, and then nanoinfer is *behind* llama.cpp and MLX at every format. Keeping one token in flight (commit
  t+1 before reading t) gets it all back:

  | format | pipelined | streaming, per-token sync | streaming, one token ahead |
  |---|---|---|---|
  | f32 | 186 | 174 | 185 |
  | f16 | 347 | 305 | 339 |
  | Q8 | 589 | 486 | 577 |
  | Q4 + Q8 head | 760 | 593 | 760 |

  Same session, 19 Oct, ~3–4% slower than the head-to-head session (metal-lanes 760 vs 787). Details:
  [`docs/LOG.md`](docs/LOG.md#metal-f32-on-the-pipelined-path-and-what-streaming-costs-19-oct).
- WebGPU (the same WGSL through `wgpu`) reaches 556–569 tok/s at Q8 natively and 558 in Chrome on the same machine,
  about 90% of the matching hand-written Metal stage.

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

## Licence

- Code: MIT, see [`LICENSE`](LICENSE) (matches `license = "MIT"` in `Cargo.toml`).
- Prose and figures (the course text in `web/src/learn/`, the writing in `docs/*.md`, and the figures as visual works):
  CC BY 4.0, see [`LICENSE-CONTENT`](LICENSE-CONTENT). Where a file mixes the two, the code is MIT and the prose it
  renders is CC BY 4.0.
