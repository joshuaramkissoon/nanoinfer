#!/usr/bin/env bash
# Head-to-head throughput: nanoinfer vs llama.cpp vs MLX vs PyTorch MPS on GPT-2
# 124M. Prompt 64, then 64 greedy decode tokens at context 64..128, batch 1.
# STRICTLY SEQUENTIAL: one engine at a time, waits for 1-min load <= MAXLOAD
# before every bench (parallel heavy jobs panicked this machine once).
# Raw results: $H2H/raw/round<R>.jsonl. Then: aggregate.py -> bench/headtohead.json
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
H2H=${H2H:-$HOME/.cache/nanoinfer-h2h}
ROUNDS=${ROUNDS:-2}
RUNS=${RUNS:-7}
MAXLOAD=${MAXLOAD:-8}
NI="$H2H/bin/nanoinfer"
W="$REPO/weights/gpt2"
G="$H2H/gguf"
LB_BREW=llama-bench
LB_CPU="$H2H/llama.cpp/build-cpu/bin/llama-bench"
SCRIPTS="$REPO/scripts/headtohead"

guard() {
  while :; do
    l=$(sysctl -n vm.loadavg | awk '{print $2}')
    if awk -v l="$l" -v m="$MAXLOAD" 'BEGIN{exit !(l<=m)}'; then break; fi
    echo "  load $l > $MAXLOAD, waiting" >&2; sleep 20
  done
  if pgrep -x llama-bench >/dev/null || pgrep -f "target/release/nanoinfer bench" >/dev/null; then
    echo "another bench is running; aborting" >&2; exit 1
  fi
}
emit() { # wrap a JSON payload with round + load
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update(round=int(sys.argv[2]), load1=float(sys.argv[3]), tag=sys.argv[4]); print(json.dumps(d))' "$1" "$2" "$(sysctl -n vm.loadavg | awk '{print $2}')" "$3"
}
llama() { # tag binary model extra-args...
  local tag=$1 bin=$2 model=$3; shift 3
  for test in "-p 64 -n 0" "-p 0 -n 64 -d 64"; do
    guard
    $bin -m "$G/$model.gguf" $test -r "$RUNS" -o jsonl "$@" 2>/dev/null | while read -r line; do
      emit "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d["engine"]="llama.cpp"; d["model_file"]=sys.argv[2]; print(json.dumps(d))' "$line" "$model")" "$R" "$tag" >> "$OUT"
    done
  done
}
nano() { # tag stage threads
  guard
  local tmp; tmp=$(mktemp)
  local targs=(--threads "$3"); case $2 in metal*|webgpu*) targs=();; esac
  (cd "$REPO" && "$NI" --weights "$W" ${targs[@]+"${targs[@]}"} bench --stage "$2" --prompt-len 64 --tokens 64 \
      --runs "$RUNS" --warmup 1 --out "$tmp" --note headtohead >/dev/null 2>&1)
  emit "$(python3 -c 'import json,sys; d=json.loads(open(sys.argv[1]).read().strip().splitlines()[-1]); d["engine"]="nanoinfer"; d["threads"]=int(sys.argv[2]) if not d["stage"].startswith(("metal","webgpu")) else 0; print(json.dumps(d))' "$tmp" "$3")" "$R" "$1" >> "$OUT"
  rm -f "$tmp"
}
mlx() { guard; emit "$(cd "$H2H" && uv run -q --python 3.12 --with mlx-lm python "$SCRIPTS/bench_mlx.py" "$@" --runs "$RUNS" 2>/dev/null | tail -1)" "$R" "mlx-$3" >> "$OUT"; }
tmps() { guard; emit "$(cd "$H2H" && uv run -q --python 3.12 --with torch --with transformers python "$SCRIPTS/bench_torch.py" hf/gpt2 prompt64.json "$1" --runs "$RUNS" 2>/dev/null | tail -1)" "$R" "torch-$1" >> "$OUT"; }

for R in $(seq "${ROUND_START:-1}" "$ROUNDS"); do
  OUT="$H2H/raw/round$R.jsonl"; : > "$OUT"
  echo "== round $R -> $OUT" >&2
  # ---------- GPU ----------
  for m in gpt2-f32 gpt2-f16 gpt2-q8_0 gpt2-q4_0-q8head gpt2-q4_0-default; do
    echo "llama.cpp metal $m" >&2; llama gpu "$LB_BREW" "$m" -ngl 99
  done
  for s in metal-mma metal-pipelined-f16 metal-pipelined-q8 metal-lanes webgpu-kv16 webgpu-vec; do
    echo "nanoinfer $s" >&2; nano gpu "$s" 1
  done
  echo "mlx" >&2
  mlx mlx/gpt2-f32 prompt64.json f32
  mlx mlx/gpt2-f16 prompt64.json f16
  mlx mlx/gpt2-q8g32 prompt64.json q8g32
  mlx mlx/gpt2-q4g32 prompt64.json q4g32
  mlx mlx/gpt2-f16 prompt64.json q4g32-q8head --q4-q8head
  echo "torch mps" >&2
  tmps float32; tmps float16
  # ---------- CPU ----------
  for t in 4 6 10; do
    for m in gpt2-f32 gpt2-f16 gpt2-q8_0 gpt2-q4_0-q8head gpt2-q4_0-default; do
      echo "llama.cpp cpu (brew, BLAS) $m t=$t" >&2; llama cpu-blas "$LB_BREW" "$m" -ngl 0 -nopo 1 -t "$t"
      echo "llama.cpp cpu (no BLAS) $m t=$t" >&2;   llama cpu-noblas "$LB_CPU" "$m" -ngl 0 -t "$t"
    done
    for s in fast-ops f16 q8-kv16 q4-kv16 q8-prefill; do
      echo "nanoinfer $s t=$t" >&2; nano cpu "$s" "$t"
    done
  done
done
echo "done" >&2
