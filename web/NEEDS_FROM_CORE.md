# What the web demo would like from `crates/core`

Written by the web/wasm side instead of editing core. Nothing here blocks the page today; each item lists the
workaround in use.

1. **Stage metadata: description and platform fallback.** `neon` silently runs the `unrolled` kernels on wasm32, and
   the threaded stages run on one thread. `crates/wasm` now derives a `wasmNote` per stage (`wasm_note()`, from
   `Stage::multithreaded()` / `weight_format()` plus two hard-coded names). A core `Stage::description()` and
   `Stage::kernels_on_this_target()` would make that automatic for future stages.

1b. **Threads on wasm.** Stage 6+ start OS threads via `crate::pool` (`std::thread::spawn` panics on
   wasm32-unknown-unknown), so wasm never enables core's `threads` feature. If the pool had a rayon (or injectable
   spawner) backend, `crates/wasm` could run core's own threaded stages on its wasm-bindgen-rayon pool instead of the
   separate `wasm-threads` rung.

2. **An attention capture for every head.** `Trace::wants_attention` returns a single `(layer, head)`. The attention
   small multiples need all 144 heads' full causal matrices, so `crates/wasm::Engine::attention_maps` re-implements the
   forward loop from `stages::blocks::{embed, qkv, attention, proj_and_mlp}` and `kernels::reference::attention_probs`.
   If those signatures change, the wasm crate breaks. Better: a core `analysis::attention_maps(model, tokens) -> Vec<f32>`,
   or `Trace::wants_all_attention() -> bool` with a hook that receives `(layer, qkv_rows, n)`.

3. **A loader for packed web weights.** `Model::from_safetensors` needs the whole 548 MB f32 file plus a full transposed
   copy, which peaks the wasm heap at about 1.05 GB. For production the page wants pre-transposed f16 or Q8_0 shards:
   `Model::from_packed_shards(manifest, shards)` (or an incremental `ModelBuilder::push_shard`). Then the heap only holds
   the final weights and nothing goes over the wire as f32.

4. **Per-stage bytes per token.** `Stage::weight_format().decode_bytes(cfg)` already exists and the page uses
   `bits_per_weight` for its ceilings. Once quantised stages arrive, please keep `Op::bytes` for matmuls honest for the
   packed format (it currently assumes 4 bytes per weight in `Op::matmul`). The byte treemap and roofline read it directly.

4b. **A logit-lens / residual hook.** `Engine::logit_lens` and `residual_stream` re-run the forward from `blocks`
   (same as attention maps). A `Trace::residual(layer, rows)` callback would let any stage report it.

5. **(Nice to have) a reusable `Sampler::sample_with(logits, temperature, top_k)`** that leaves the RNG alone, so the hero
   can re-sample cached logits for different temperatures without reseeding. The workaround mutates the public
   `temperature` and `top_k` fields.
