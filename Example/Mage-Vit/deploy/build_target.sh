#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_root="$(cd "${project_root}/../.." && pwd)"
build_dir="${MEGAVIT_BUILD_DIR:-${project_root}/build}"

if [[ -n "${MEGAVIT_PYTHON:-}" ]]; then
  python_bin="${MEGAVIT_PYTHON}"
elif [[ -x "${sdk_root}/.venv-tfdl-linux/bin/python" ]]; then
  python_bin="${sdk_root}/.venv-tfdl-linux/bin/python"
else
  python_bin="$(command -v python3)"
fi

# TFDL2's Python extension is a native ABI consumer of the SDK headers and
# libraries.  Always rebuild it after an SDK/lib update unless explicitly
# disabled; an old Python/build extension can otherwise shadow site-packages.
if [[ "${MEGAVIT_REBUILD_TFDL_PYTHON:-1}" == "1" ]]; then
  (
    cd "${sdk_root}/Python"
    "${python_bin}" setup.py clean --all
  )
  "${python_bin}" -m pip install \
    --no-build-isolation --no-cache-dir --force-reinstall --no-deps \
    "${sdk_root}/Python"
fi

addon_path="${project_root}/deploy/runtime/libTFDLAddOn.so"
if [[ ! -f "${addon_path}" ]]; then
  echo "missing deployment addon: ${addon_path}" >&2
  exit 1
fi
LD_LIBRARY_PATH="${sdk_root}/lib:${project_root}/deploy/runtime:${LD_LIBRARY_PATH:-}" \
  "${python_bin}" -c \
  'import sys; from TFDL2.utils import LoadCustomOp; import TFDL2.TFDL2 as native; LoadCustomOp(sys.argv[1]); print("TFDL2 Python/addon ABI: OK", native.__file__)' \
  "${addon_path}"

cmake -S "${project_root}" -B "${build_dir}" \
  -DMEGAVIT_WITH_FFMPEG=ON \
  -DMEGAVIT_WITH_TFDL=ON \
  -DMEGAVIT_BUILD_TESTS=ON \
  -DTFDL_SDK_ROOT="${sdk_root}"
cmake --build "${build_dir}" -j"${MEGAVIT_BUILD_JOBS:-2}"
ctest --test-dir "${build_dir}" --output-on-failure

capabilities="$(${build_dir}/megavit_frontend --capabilities)"
if [[ "${capabilities}" != *'"ffmpeg":true'* ]]; then
  echo "megavit_frontend was built without FFmpeg development libraries" >&2
  exit 1
fi
if [[ "${capabilities}" != *'"tfdl":true'* ]]; then
  echo "megavit_frontend was built without TFDL support" >&2
  exit 1
fi
echo "frontend capabilities: ${capabilities}"
