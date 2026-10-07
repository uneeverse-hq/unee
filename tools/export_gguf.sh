#!/usr/bin/env bash
# Merged Hugging Face model -> GGUF (f16) -> quantized GGUF, with llama.cpp's converter at the matching build tag.
#   tools/export_gguf.sh models/unee-v1/merged models/gguf/unee-v1 [Q4_K_M]
# Writes <prefix>-f16.gguf and <prefix>-<QUANT>.gguf. Needs tools/llama-src (sparse clone of llama.cpp b11381:
# convert_hf_to_gguf.py, conversion/, gguf-py/) and the llama.cpp binaries in tools/llama.cpp.
set -euo pipefail
cd "$(dirname "$0")/.."
src="$1"; prefix="$2"; quant="${3:-Q4_K_M}"
PYTHONUTF8=1 .venv/Scripts/python tools/llama-src/convert_hf_to_gguf.py "$src" --outfile "$prefix-f16.gguf" --outtype f16 --no-mtp  # LoRA-merged models have no MTP weights
tools/llama.cpp/cpu/llama-quantize.exe "$prefix-f16.gguf" "$prefix-$quant.gguf" "$quant"
ls -la "$prefix"-*.gguf
