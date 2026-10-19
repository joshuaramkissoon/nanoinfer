#!/usr/bin/env bash
# Head-to-head throughput: nanoinfer vs llama.cpp vs MLX vs PyTorch MPS on GPT-2
# 124M. Prompt 64, then 64 greedy decode tokens at context 64..128, batch 1.
# STRICTLY SEQUENTIAL: one engine at a time, waits for 1-min load <= MAXLOAD
# (GPU jobs) or MAXLOAD_CPU (CPU jobs, which compete with the load for cores)
# before every bench (parallel heavy jobs panicked this machine once).
# ORDER: every round runs the GPU jobs, then the CPU jobs, each block in a
# fresh random order (seed = SEED + round, recorded with every row), so no
# engine always runs straight after another has warmed the GPU or the cores.
# nanoinfer refuses to record from a dirty engine tree; ALLOW_DIRTY=1 passes
# --allow-dirty (rows are then marked git_dirty).
# Raw results: $H2H/raw/round<R>.jsonl. Then: aggregate.py -> bench/headtohead.json
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
H2H=${H2H:-$HOME/.cache/nanoinfer-h2h}
ROUNDS=${ROUNDS:-2}
RUNS=${RUNS:-7}
MAXLOAD=${MAXLOAD:-8}
MAXLOAD_CPU=${MAXLOAD_CPU:-2}
SEED=${SEED:-$(date +%s)}
NI="$H2H/bin/nanoinfer"
W="$REPO/weights/gpt2"
G="$H2H/gguf"
LB_BREW=llama-bench
LB_CPU="$H2H/llama.cpp/build-cpu/bin/llama-bench"
SCRIPTS="$REPO/scripts/headtohead"

guard() { # [max load] (default MAXLOAD; CPU jobs pass MAXLOAD_CPU)
  local max=${1:-$MAXLOAD}
  while :; do
    l=$(sysctl -n vm.loadavg | awk '{print $2}')
    if awk -v l="$l" -v m="$max" 'BEGIN{exit !(l<=m)}'; then break; fi
    echo "  load $l > $max, waiting" >&2; sleep 20
  done
  if pgrep -x llama-bench >/dev/null || pgrep -f "target/release/nanoinfer bench" >/dev/null; then
    echo "another bench is running; aborting" >&2; exit 1
  fi
}
emit() { # wrap a JSON payload with round, load, order position and seed
  python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d.update(round=int(sys.argv[2]), load1=float(sys.argv[3]), tag=sys.argv[4], order=int(sys.argv[5]), seed=int(sys.argv[6])); print(json.dumps(d))' "$1" "$2" "$(sysctl -n vm.loadavg | awk '{print $2}')" "$3" "$POS" "$((SEED + R))"
}
maxload() { case $1 in cpu*) echo "$MAXLOAD_CPU";; *) echo "$MAXLOAD";; esac; }
llama() { # tag binary model extra-args...
  local tag=$1 bin=$2 model=$3; shift 3
  for test in "-p 64 -n 0" "-p 0 -n 64 -d 64"; do
    guard "$(maxload "$tag")"
    $bin -m "$G/$model.gguf" $test -r "$RUNS" -o jsonl "$@" 2>/dev/null | while read -r line; do
      emit "$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d["engine"]="llama.cpp"; d["model_file"]=sys.argv[2]; print(json.dumps(d))' "$line" "$model")" "$R" "$tag" >> "$OUT"
    done
  done
}
nano() { # tag stage threads
  guard "$(maxload "$1")"
  local tmp; tmp=$(mktemp)
  local targs=(--threads "$3"); case $2 in metal*|webgpu*) targs=();; esac
  local dirty=(); [ "${ALLOW_DIRTY:-0}" = 1 ] && dirty=(--allow-dirty)
  (cd "$REPO" && "$NI" --weights "$W" ${targs[@]+"${targs[@]}"} bench --stage "$2" --prompt-len 64 --tokens 64 \
      --runs "$RUNS" --warmup 1 --out "$tmp" --note headtohead ${dirty[@]+"${dirty[@]}"} >/dev/null 2>&1)
  emit "$(python3 -c 'import json,sys; d=json.loads(open(sys.argv[1]).read().strip().splitlines()[-1]); d["engine"]="nanoinfer"; d["threads"]=int(sys.argv[2]) if not d["stage"].startswith(("metal","webgpu")) else 0; print(json.dumps(d))' "$tmp" "$3")" "$R" "$1" >> "$OUT"
  rm -f "$tmp"
}
mlx() { guard; emit "$(cd "$H2H" && uv run -q --python 3.12 --with mlx-lm python "$SCRIPTS/bench_mlx.py" "$@" --runs "$RUNS" 2>/dev/null | tail -1)" "$R" "mlx-$3" >> "$OUT"; }
tmps() { guard; emit "$(cd "$H2H" && uv run -q --python 3.12 --with torch --with transformers python "$SCRIPTS/bench_torch.py" hf/gpt2 prompt64.json "$1" --runs "$RUNS" 2>/dev/null | tail -1)" "$R" "torch-$1" >> "$OUT"; }

# One job per line: "label|command". Each block is shuffled per round.
gpu_jobs() {
  for m in gpt2-f32 gpt2-f16 gpt2-q8_0 gpt2-q4_0-q8head gpt2-q4_0-default; do
    echo "llama.cpp metal $m|llama gpu $LB_BREW $m -ngl 99"
  done
  for s in metal-mma metal-pipelined-f32 metal-pipelined-f16 metal-pipelined-q8 metal-lanes webgpu-kv16 webgpu-vec; do
    echo "nanoinfer $s|nano gpu $s 1"
  done
  echo "mlx f32|mlx mlx/gpt2-f32 prompt64.json f32"
  echo "mlx f16|mlx mlx/gpt2-f16 prompt64.json f16"
  echo "mlx q8g32|mlx mlx/gpt2-q8g32 prompt64.json q8g32"
  echo "mlx q4g32|mlx mlx/gpt2-q4g32 prompt64.json q4g32"
  echo "mlx q4g32-q8head|mlx mlx/gpt2-f16 prompt64.json q4g32-q8head --q4-q8head"
  echo "torch mps float32|tmps float32"
  echo "torch mps float16|tmps float16"
}
cpu_jobs() {
  for t in 4 6 10; do
    for m in gpt2-f32 gpt2-f16 gpt2-q8_0 gpt2-q4_0-q8head gpt2-q4_0-default; do
      echo "llama.cpp cpu (brew, BLAS) $m t=$t|llama cpu-blas $LB_BREW $m -ngl 0 -nopo 1 -t $t"
      echo "llama.cpp cpu (no BLAS) $m t=$t|llama cpu-noblas $LB_CPU $m -ngl 0 -t $t"
    done
    for s in fast-ops f16 q8-kv16 q4-kv16 q8-prefill; do
      echo "nanoinfer $s t=$t|nano cpu $s $t"
    done
  done
}
shuffle() { python3 -c 'import random,sys; l=sys.stdin.read().splitlines(); random.Random(int(sys.argv[1])).shuffle(l); print("\n".join(l))' "$1"; }

for R in $(seq "${ROUND_START:-1}" "$ROUNDS"); do
  OUT="$H2H/raw/round$R.jsonl"; : > "$OUT"
  echo "== round $R -> $OUT (order seed $((SEED + R)))" >&2
  POS=0
  # GPU block, then CPU block, each in this round's random order. The job
  # strings are our own (no spaces inside arguments), so word splitting is safe.
  while IFS='|' read -r label cmd; do
    POS=$((POS + 1))
    echo "[$POS] $label" >&2
    # shellcheck disable=SC2086
    $cmd </dev/null
  done < <({ gpu_jobs | shuffle "$((SEED + R))"; cpu_jobs | shuffle "$((SEED + R + 1000))"; })
done
echo "done" >&2
