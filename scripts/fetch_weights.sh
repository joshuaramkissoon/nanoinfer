#!/usr/bin/env bash
# Download GPT-2 124M (openai-community/gpt2) weights + tokenizer files from
# Hugging Face into weights/gpt2/. These are large (~550MB) and gitignored.
set -euo pipefail

MODEL="${1:-openai-community/gpt2}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/weights/$(basename "$MODEL")"
BASE="https://huggingface.co/$MODEL/resolve/main"

mkdir -p "$OUT"
for f in config.json model.safetensors vocab.json merges.txt tokenizer.json; do
  if [ -s "$OUT/$f" ]; then
    echo "have $f"
  else
    echo "fetching $f"
    curl -fL --retry 3 -o "$OUT/$f.part" "$BASE/$f"
    mv "$OUT/$f.part" "$OUT/$f"
  fi
done
echo "done -> $OUT"
ls -lh "$OUT"
