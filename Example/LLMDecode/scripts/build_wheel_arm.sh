#!/usr/bin/env bash
# Build a target-native llama.cpp binary and embed it in an installable wheel.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
build_dir="${1:-${project_dir}/build}"
dist_dir="${2:-${project_dir}/dist}"
python_bin="${PYTHON:-python3}"

bash "${project_dir}/scripts/build_arm.sh" "${build_dir}"
install -D -m 0755 "${build_dir}/bin/llmdecode" \
  "${project_dir}/python_package/tfdl_llmdecode/bin/llmdecode"
install -D -m 0755 "${build_dir}/bin/llmquantize" \
  "${project_dir}/python_package/tfdl_llmdecode/bin/llmquantize"
shopt -s nullglob
for library in "${build_dir}/bin"/libllama.so* "${build_dir}/bin"/libggml*.so*; do
  install -D -m 0755 "${library}" \
    "${project_dir}/python_package/tfdl_llmdecode/bin/$(basename "${library}")"
done
shopt -u nullglob
for artifact in "${project_dir}/python_package/tfdl_llmdecode/bin"/llmdecode \
  "${project_dir}/python_package/tfdl_llmdecode/bin"/llmquantize \
  "${project_dir}/python_package/tfdl_llmdecode/bin"/libllama.so.* \
  "${project_dir}/python_package/tfdl_llmdecode/bin"/libggml*.so.*; do
  [[ -f "${artifact}" ]] || continue
  if ! readelf -d "${artifact}" 2>/dev/null | grep -Fq 'Library runpath: [$ORIGIN]'; then
    echo 'non-relocatable wheel ELF (expected RUNPATH $ORIGIN): '"${artifact}" >&2
    exit 1
  fi
done
"${python_bin}" -m pip wheel --no-deps --no-build-isolation --wheel-dir "${dist_dir}" "${project_dir}/python_package"
