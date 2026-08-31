#!/usr/bin/env bash
# Install the target-native external-KV decode wheel included by package_project.
set -euo pipefail

deploy_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
wheel=("${deploy_dir}"/wheels/tfdl_llmdecode-*.whl)
if [[ ! -f "${wheel[0]}" ]]; then
  echo "missing deploy/wheels/tfdl_llmdecode-*.whl; rebuild the portable package" >&2
  exit 1
fi
# The wheel is intentionally rebuilt in-place while the native ABI evolves.
# Reinstall even when its Python version string is unchanged, otherwise pip
# keeps an older ELF bundle and silently preserves a broken RUNPATH.
"${PYTHON:-python3}" -m pip install --force-reinstall --no-deps "${wheel[0]}"
