#!/usr/bin/env bash
# Start llama-server with a GGUF model on the GPU, OpenAI-compatible API on http://127.0.0.1:${PORT:-8080}/v1.
#   tools/serve.sh models/gguf/Qwen3.5-9B-Q4_K_M.gguf            # teacher for data generation and labelling
#   NP=8 CTX=65536 tools/serve.sh models/gguf/Qwen3.5-9B-Q4_K_M.gguf
# llama.cpp binaries: tools/llama.cpp/cuda (official ggml-org release b11381, CUDA 13.4) with the CUDA runtime DLLs
# copied from the venv's torch/lib.
set -euo pipefail
cd "$(dirname "$0")/.."
exec tools/llama.cpp/cuda/llama-server.exe -m "$1" --jinja -ngl 99 -fa on \
  -np "${NP:-4}" -c "${CTX:-32768}" --port "${PORT:-8080}" --host 127.0.0.1 --no-webui "${@:2}"
