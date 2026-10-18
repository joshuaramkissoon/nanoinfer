#!/usr/bin/env bash
# One-time setup for the head-to-head: fetch llama.cpp (converter + a CPU-only,
# no-BLAS build), convert our HF GPT-2 to GGUF (F32/F16/Q8_0/Q4_0 variants) and
# to MLX (F32/F16 + 8/4-bit, group 32). Everything lands in $H2H (default
# ~/.cache/nanoinfer-h2h), nothing in the repo. Needs brew llama.cpp, cmake, uv.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
H2H=${H2H:-$HOME/.cache/nanoinfer-h2h}
mkdir -p "$H2H"/{gguf,mlx,raw,bin}
cd "$H2H"
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama.cpp

# HF dir without the h.*.attn.bias mask buffers the converter can't map.
uv run --with safetensors --with numpy python "$REPO/scripts/headtohead/strip_hf.py" "$REPO/weights/gpt2" hf/gpt2
uv run --python 3.12 --with transformers python "$REPO/scripts/headtohead/prompt.py" hf/gpt2 prompt64.json

CONV="uv run --python 3.12 --with torch --with numpy<2.3 --with sentencepiece --with transformers==4.57.6 --with ./llama.cpp/gguf-py --with protobuf<5"
for t in f32 f16; do
  [ -f gguf/gpt2-$t.gguf ] || $CONV python llama.cpp/convert_hf_to_gguf.py hf/gpt2 --outtype $t --outfile gguf/gpt2-$t.gguf
done
Q=llama-quantize
[ -f gguf/gpt2-q8_0.gguf ]          || $Q gguf/gpt2-f32.gguf gguf/gpt2-q8_0.gguf Q8_0
# llama.cpp's default Q4_0 puts the tied token_embd/LM head in Q6_K:
[ -f gguf/gpt2-q4_0-default.gguf ]  || $Q gguf/gpt2-f32.gguf gguf/gpt2-q4_0-default.gguf Q4_0
# ours (metal-lanes / q4-kv16) keeps a Q8_0 head: match it.
[ -f gguf/gpt2-q4_0-q8head.gguf ]   || $Q --output-tensor-type q8_0 --token-embedding-type q8_0 gguf/gpt2-f32.gguf gguf/gpt2-q4_0-q8head.gguf Q4_0

uv run -q --python 3.12 --with ./llama.cpp/gguf-py python "$REPO/scripts/headtohead/gguf_bytes.py" gguf/*.gguf > gguf_bytes.json

# llama.cpp CPU kernels without Accelerate/AMX (brew's build routes pp matmuls to BLAS).
if [ ! -x llama.cpp/build-cpu/bin/llama-bench ]; then
  cmake -S llama.cpp -B llama.cpp/build-cpu -DGGML_METAL=OFF -DGGML_BLAS=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build llama.cpp/build-cpu -j 4 --target llama-bench llama-perplexity
fi

MLX="uv run --python 3.12 --with mlx-lm"
[ -d mlx/gpt2-f32 ]   || $MLX mlx_lm.convert --hf-path hf/gpt2 --mlx-path mlx/gpt2-f32 --dtype float32
[ -d mlx/gpt2-f16 ]   || $MLX mlx_lm.convert --hf-path hf/gpt2 --mlx-path mlx/gpt2-f16 --dtype float16
[ -d mlx/gpt2-q8g32 ] || $MLX mlx_lm.convert --hf-path hf/gpt2 --mlx-path mlx/gpt2-q8g32 -q --q-bits 8 --q-group-size 32 --dtype float16
[ -d mlx/gpt2-q4g32 ] || $MLX mlx_lm.convert --hf-path hf/gpt2 --mlx-path mlx/gpt2-q4g32 -q --q-bits 4 --q-group-size 32 --dtype float16

(cd "$REPO" && cargo build --release -p nanoinfer-cli && cp target/release/nanoinfer "$H2H/bin/nanoinfer")
echo "setup done in $H2H"
