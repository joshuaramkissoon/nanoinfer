#!/usr/bin/env bash
# Quality side of the head-to-head (WikiText-2 test set), one job at a time.
# Each engine is compared with ITS OWN f32 model; absolute ppl differs across
# tools because the scoring windows differ:
#   nanoinfer ppl / ppl_mlx.py: 4 x 512 windows, context reset, all 2044 positions
#   llama-perplexity: -c 512 --chunks 8, scores the 2nd half of each chunk (2048 positions)
#   Metal formats (examples/quant_quality.rs): 2 x 512 windows, token-by-token on the GPU kernels
# Writes $H2H/ppl.json.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
H2H=${H2H:-$HOME/.cache/nanoinfer-h2h}
T="$REPO/weights/wikitext-2-raw-test.txt"
cd "$H2H"
wait_load() { while [ "$(sysctl -n vm.loadavg | awk '{print ($2>8)}')" = 1 ]; do sleep 20; done; }

: > ppl_raw.txt
for s in f16 q8-kv16 q4-kv16; do
  wait_load; echo "== nanoinfer $s" >> ppl_raw.txt
  (cd "$REPO" && "$H2H/bin/nanoinfer" --weights weights/gpt2 --threads 6 ppl --stage $s --windows 4 2>/dev/null) >> ppl_raw.txt
done
wait_load; echo "== nanoinfer metal quant_quality" >> ppl_raw.txt
(cd "$REPO" && cargo run -q --release -p nanoinfer-metal --example quant_quality 2>/dev/null) >> ppl_raw.txt

wait_load
llama-perplexity -m gguf/gpt2-f32.gguf -f "$T" -c 512 --chunks 8 -ngl 99 --kl-divergence-base kld-f32.bin >/dev/null 2>&1
for m in f16 q8_0 q4_0-q8head q4_0-default; do
  wait_load; echo "== llama.cpp $m" >> ppl_raw.txt
  llama-perplexity -m gguf/gpt2-$m.gguf -f "$T" -c 512 --chunks 8 -ngl 99 --kl-divergence-base kld-f32.bin --kl-divergence 2>&1 \
    | grep -E "^Mean PPL\(Q\)  |^Mean PPL\(base\)|^Mean    KLD|^Same top p" >> ppl_raw.txt
done

wait_load
uv run -q --python 3.12 --with mlx-lm python "$REPO/scripts/headtohead/ppl_mlx.py" "$T" mlx/gpt2-f32 \
  f16=mlx/gpt2-f16 q8g32=mlx/gpt2-q8g32 q4g32=mlx/gpt2-q4g32 q4g32-q8head=mlx/gpt2-f16:q4q8head 2>/dev/null | tail -1 > ppl_mlx.json

python3 "$REPO/scripts/headtohead/parse_ppl.py" ppl_raw.txt ppl_mlx.json > ppl.json
cat ppl.json
