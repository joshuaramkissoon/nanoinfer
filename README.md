# nanoinfer

A small LLM inference engine in Rust, built from a naive baseline and optimised one measured step at a time.

- `research/` — distilled notes from the best resources (engines, kernels, serving, distributed, explorable explanations)
- `docs/LOG.md` — the hillclimb journal: every step, its hypothesis, and the measured result

## Quick start

```sh
source ~/.cargo/env
scripts/fetch_weights.sh                      # GPT-2 124M from Hugging Face -> weights/gpt2 (~525 MB)
cargo run --release -p nanoinfer-cli -- generate --prompt "The meaning of life is" --tokens 40
cargo run --release -p nanoinfer-cli -- generate --stage naive --tokens 10 --profile
cargo run --release -p nanoinfer-cli -- bench --stage kv-cache --tokens 64
cargo test --release                          # checks every stage against HF reference logits
cargo build -p nanoinfer-core --target wasm32-unknown-unknown
```

Regenerate the reference fixtures (needs `uv`, nothing installed globally):

```sh
uv run --with torch --with transformers --with numpy python scripts/gen_reference.py
```

## Where we are

| Stage | Decode tok/s (M4, 1 thread, f32) | % of 243 tok/s ceiling |
|---|---|---|
| 0 naive | 0.37 | 0.15% |
| 1 kv-cache | 18.95 | 7.8% |

See `docs/LOG.md` for the journal and `docs/ARCHITECTURE.md` for how the code is laid out.
