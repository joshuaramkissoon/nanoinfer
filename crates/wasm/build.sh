#!/usr/bin/env bash
# Build the engine for the browser: wasm32 + SIMD128, single-threaded.
# Output goes to web/src/engine/pkg (committed? no: gitignored, rebuilt by `npm run wasm`).
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"
OUT=../../web/src/engine/pkg
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir "$OUT" --out-name nanoinfer \
  target/wasm32-unknown-unknown/release/nanoinfer_wasm.wasm
if command -v wasm-opt >/dev/null; then
  wasm-opt -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int \
    "$OUT/nanoinfer_bg.wasm" -o "$OUT/nanoinfer_bg.wasm" || true
fi
ls -la "$OUT"
