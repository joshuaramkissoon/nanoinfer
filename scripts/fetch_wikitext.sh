#!/usr/bin/env bash
# Download the WikiText-2 (raw) test split and extract it to plain text at
# weights/wikitext-2-raw-test.txt, for `nanoinfer ppl`. Gitignored, like weights.
# Needs `uv` (pyarrow is pulled in on the fly to read the parquet file).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/weights/wikitext-2-raw-test.txt"
TMP="$(mktemp -d)"
curl -fL --retry 3 -o "$TMP/test.parquet" \
  "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"
uv run -q --with pyarrow python -I -c '
import sys, pyarrow.parquet as pq
open(sys.argv[2], "w").write("".join(pq.read_table(sys.argv[1]).column("text").to_pylist()))
' "$TMP/test.parquet" "$OUT"
rm -rf "$TMP"
echo "wrote $OUT ($(wc -c < "$OUT") bytes)"
