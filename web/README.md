# nanoinfer · live

The demo page for nanoinfer: GPT-2 124M running in the browser through the real engine (`crates/core`, compiled to
WebAssembly by `crates/wasm`). Every figure is drawn from what the engine reports while it runs: per-op timings, FLOPs,
bytes, KV length, next-token probabilities and attention maps. None of it is a mock-up.

## Run it

```sh
scripts/fetch_weights.sh          # once: HF gpt2 -> weights/gpt2 (gitignored)
cd web
npm install
npm run wasm                      # builds crates/wasm -> web/src/engine/pkg (needs wasm-bindgen-cli 0.2.129)
npm run dev                       # http://localhost:5199
```

Click **Load GPT-2**. In local dev the 548 MB f32 checkpoint is served straight from `../weights/gpt2` by a small Vite
middleware, which also serves `../bench/results.jsonl` (the native hillclimb results) at `/bench/results.jsonl`.

Toolchain: Rust with the `wasm32-unknown-unknown` target, and `cargo install wasm-bindgen-cli --version 0.2.129` (the
version must match the `wasm-bindgen` pin in `crates/wasm/Cargo.toml`). `wasm-opt` is used if it's on PATH.

Env overrides: `VITE_WEIGHTS_BASE` (where the page fetches `model.safetensors`, `vocab.json` and `merges.txt`),
`VITE_BENCH_URL` (results.jsonl), and `NANOINFER_WEIGHTS` (the directory the dev server serves).

## Architecture

```
crates/wasm  (standalone crate, own [workspace]; path dep on ../core)
  Engine      weights + tokenizer + N sessions ("slots"), each on any Stage
  forward()   one real forward pass -> packed Float64Array trace
  stages()    Stage::ALL from core (new stages appear automatically)
  attention_maps()  analysis pass: all 144 heads' T×T matrices, via core's blocks
  bench_bandwidth / bench_peak_gflops   the two roofs, measured in-tab

web/src/engine
  worker.ts   owns the wasm. Streams the checkpoint into wasm memory chunk by
              chunk (never held twice in JS), caches files in OPFS, decodes traces
  client.ts   typed RPC to the worker + one tiny store (useSyncExternalStore)
              that every panel reads, plus the vocab for decoding tokens
  generate.ts the generation loop (main thread: one RPC per token, so it's
              trivially cancellable) and the fixed-shape Ladder benchmark
web/src/panels   one file per figure; registry.tsx is the page order
web/src/ui       Figure frame, theme (reads jkmr's --m-* CSS variables)
```

- **One engine, one run.** The hero (fig. 1) writes the shared `run` (tokens + every forward's trace). The time strip,
  byte budget, roofline and attention panels all redraw from that same run. Panels that need the engine to themselves
  (the KV race, the Ladder benchmark) open their own session slots and `claim()` the engine so runs don't interleave.
- **Packed trace.** Each `forward()` returns a header, then top-k, then an optional attention row, then 10 f64s per op
  (`kind, name, layer, d0, d1, d2, flops, bytes, start_ms, dur_ms`). Names and kinds are interned in grow-only tables, so
  new op kinds from future stages decode without code changes. Layout constants come from `trace_layout()`.
- **Timing.** `performance.now()` in the worker. The dev server sends COOP/COEP, so the page is cross-origin isolated
  and the timer resolution is ~5 µs (~100 µs otherwise). The page shows which one it got.
- **Isolation headers.** COOP `same-origin` and COEP `require-corp`, set on every response. They're needed for
  threaded wasm later (SharedArrayBuffer). jkmr would need them site-wide (see research/03 §3.6).
- **Theme.** `src/styles.css` copies jkmr's `.minimal` tokens (`--m-bg`, `--m-accent`, `--m-heat-0..4`, ...). Canvas and
  SVG figures read them through `usePalette()`, which re-reads when `data-theme` changes on the `.minimal` host. The theme
  toggle uses jkmr's localStorage key.
- **Reduced motion.** All animation and transitions are disabled under `prefers-reduced-motion`, and the token stream
  doesn't use the blur-in effect.

## Adding a panel

1. Create `src/panels/MyPanel.tsx` exporting a default component. Read shared state with
   `useEngine((s) => s.run)` (or `s.config`, `s.bandwidthGBs`, `s.stages`, ...). Render `<NeedsEngine>` until the data you
   need exists.
2. If it has to drive the engine itself: `if (!claim("my-panel")) return;`, then `const slot = await engine.open(stage, maxSeq)`
   and `engine.forward(slot, tokens)` / `engine.sample(...)`. Finish with `engine.close(slot)` and `release("my-panel")`
   in a `finally`.
3. Add an entry to `PANELS` in `src/panels/registry.tsx` with `id`, `title`, a 1–3 sentence learner `caption`, and
   optional `intro` prose. Numbering and the frame are automatic.
4. Colours: use `usePalette()` and `kindColor(pal, op.kind)`, so the figure follows the theme and matches the others.
   Measure width with `useWidth()` (a callback ref, so it works for elements that mount late).

## Embedding in jkmr

The panels only depend on React, `src/engine/*`, `src/ui/*` and `src/lib/*`, plus the CSS in `src/styles.css` (scoped to
`.minimal`, with an `nv-` prefix). Plan: vendor the built `pkg/` and these folders into jkmr, register panels behind
````live <id>```` fences in `toSegments` (research/03 §6), and lazy-load the engine so the home page bundle is unchanged.

## Weights: now and later

- **Now (local dev):** the raw HF f32 `model.safetensors` (548 MB). It's streamed into wasm memory, parsed and
  transposed by core (`Model::from_safetensors`). The wasm heap peaks at about 1.05 GB during load. Files are cached in
  OPFS and written while they download. A ".ok" marker means a file was written completely. On the next visit the page
  loads from the cache automatically (it never auto-downloads).
- **Later (production):** pre-transposed f16 or Q8_0 shards of 16–32 MB with a `manifest.json` (sha256, offsets), hosted
  on the HF Hub. Use whole-shard GETs, because Range requests fail CORS on the Xet hop. That needs a core loader for
  packed shards (see `NEEDS_FROM_CORE.md`). Never ship f32 over the wire.

## Known issues

- Single-threaded, simd128 only. No relaxed-SIMD (FMA) build, threads or WebGPU yet.
- Some browsers' storage quota is too small to cache 548 MB (incognito-like profiles cap OPFS at about 64 MB). The page
  still works there; it just downloads again next time.
- The first bandwidth probe right after loading reads about 40% low, so the page takes the best of two probes. It's still
  noisy (48–62 GB/s on an M4) and can move when other processes compete for memory bandwidth.
- No recorded-trace fallback yet: figures show a placeholder until the engine loads. Native `trace_out` JSONL from the
  CLI could be replayed here.
- `neon` doesn't exist on wasm: core quietly falls back to the `unrolled` kernels, and the page marks that stage with *.
