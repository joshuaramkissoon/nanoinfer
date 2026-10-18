#!/usr/bin/env bash
# Build the engine for the browser. Three builds, picked at runtime by the page
# (web/src/engine/worker.ts) from what the browser supports:
#
#   pkg/          simd128, single thread          (every browser with wasm SIMD)
#   pkg-relaxed/  + relaxed-SIMD (fused mul-add)  (Chrome 114+, Firefox 145+)
#   pkg-threads/  + relaxed-SIMD + atomics: core's `threads` stage on a
#                 wasm-bindgen-rayon pool         (needs cross-origin isolation;
#                 built with nightly, skipped if nightly + rust-src are missing)
#
# Output is gitignored and rebuilt by `npm run wasm`. `ONLY=simd` etc. builds one.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"
WEB=../../web/src/engine
ONLY=${ONLY:-all}

bindgen() { # <target-dir> <out-dir>
  wasm-bindgen --target web --out-dir "$2" --out-name nanoinfer "$1/wasm32-unknown-unknown/release/nanoinfer_wasm.wasm"
  if command -v wasm-opt >/dev/null; then
    wasm-opt -O3 --enable-simd --enable-relaxed-simd --enable-bulk-memory --enable-threads \
      --enable-nontrapping-float-to-int "$2/nanoinfer_bg.wasm" -o "$2/nanoinfer_bg.wasm" || true
  fi
  echo "{\"name\":\"nanoinfer\",\"type\":\"module\",\"main\":\"nanoinfer.js\"}" > "$2/package.json"
  ls -la "$2"/*.wasm
}

if [[ $ONLY == all || $ONLY == simd ]]; then
  RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown --target-dir target/simd --features webgpu
  bindgen target/simd $WEB/pkg
fi

if [[ $ONLY == all || $ONLY == relaxed ]]; then
  RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" cargo build --release --target wasm32-unknown-unknown --target-dir target/relaxed --features webgpu
  bindgen target/relaxed $WEB/pkg-relaxed
fi

if [[ $ONLY == all || $ONLY == threads ]]; then
  if rustup run nightly rustc --version >/dev/null 2>&1 && rustup component list --toolchain nightly --installed | grep -q rust-src; then
    # Shared memory needs an explicit maximum; the f32 load peaks at ~1.05 GB.
    RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd,+atomics,+bulk-memory,+mutable-globals -C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=4294967296 -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base" \
      rustup run nightly cargo build --release --target wasm32-unknown-unknown --target-dir target/threads \
      --features threads,webgpu -Z build-std=panic_abort,std
    bindgen target/threads $WEB/pkg-threads
  else
    echo "skipping threads build: needs 'rustup toolchain install nightly --component rust-src'" >&2
  fi
fi
