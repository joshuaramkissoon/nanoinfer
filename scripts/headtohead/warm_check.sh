#!/usr/bin/env bash
# llama.cpp's Metal decode is bimodal on this machine (~450 vs ~625 tok/s for
# Q4_0): it syncs every token, and the GPU clock it gets depends on what ran
# before. Give it its best case: heat the GPU with a nanoinfer metal-lanes run,
# then llama-bench tg64@d64 with 15 reps. aggregate.py takes the max of this
# and the round results. Sequential, one job at a time. -> $H2H/raw/warm.jsonl
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
H2H=${H2H:-$HOME/.cache/nanoinfer-h2h}
OUT="$H2H/raw/warm.jsonl"; : > "$OUT"
for m in gpt2-f32 gpt2-f16 gpt2-q8_0 gpt2-q4_0-q8head gpt2-q4_0-default; do
  while [ "$(sysctl -n vm.loadavg | awk '{print ($2>8)}')" = 1 ]; do sleep 20; done
  (cd "$REPO" && "$H2H/bin/nanoinfer" bench --stage metal-lanes --runs 5 --no-record >/dev/null 2>&1)
  llama-bench -m "$H2H/gguf/$m.gguf" -p 0 -n 64 -d 64 -r 15 -ngl 99 -o jsonl 2>/dev/null \
    | python3 -c 'import json,sys; d=json.loads(sys.stdin.readline()); d.update(engine="llama.cpp", model_file=sys.argv[1], tag="gpu-warm", round=0, load1=float(sys.argv[2])); print(json.dumps(d))' "$m" "$(sysctl -n vm.loadavg | awk '{print $2}')" >> "$OUT"
done
