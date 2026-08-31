#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 HF_MODEL_DIR OUTPUT.gguf [f16|bf16|q8_0|q4_k_m]" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"
outtype="${3:-f16}"

case "$outtype" in
  f16|bf16|q8_0) ;;
  q4_k_m)
    # Q4_K_M must be produced from a 16-bit GGUF source.  The temporary is
    # intentionally adjacent to the requested output so the filesystem does
    # not need to copy the model before block-wise quantization.
    temp_gguf="${2}.source-f16.gguf"
    trap 'rm -f "$temp_gguf"' EXIT
    "$python_bin" "$root/third_party/llama.cpp/convert_hf_to_gguf.py" \
      "$1" --outfile "$temp_gguf" --outtype f16
    quantizer="${LLMQUANTIZE:-$root/build/bin/llmquantize}"
    if [[ ! -x "$quantizer" ]]; then
      echo "Q4_K_M requires $quantizer; run scripts/build_arm.sh first or set LLMQUANTIZE." >&2
      exit 1
    fi
    "$quantizer" "$temp_gguf" "$2" --threads "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
    exit 0
    ;;
  *)
    echo "Unsupported GGUF output type: $outtype" >&2
    exit 2
    ;;
esac

"$python_bin" "$root/third_party/llama.cpp/convert_hf_to_gguf.py" \
  "$1" --outfile "$2" --outtype "$outtype"
