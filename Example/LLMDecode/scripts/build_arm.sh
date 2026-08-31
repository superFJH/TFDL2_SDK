#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-$root/build}"
lto="${LLMDECODE_LTO:-ON}"

# Ubuntu development images occasionally point cc at a newer GCC than c++.
# GCC LTO requires matching plugin versions, so when the caller did not pin a
# toolchain, prefer the GCC pair matching the default C++ compiler.
if [[ "$lto" == "ON" && -z "${CC:-}" && -z "${CXX:-}" ]]; then
  cxx_major="$(c++ -dumpfullversion -dumpversion 2>/dev/null | cut -d. -f1 || true)"
  gcc_candidate="gcc-${cxx_major}"
  gxx_candidate="g++-${cxx_major}"
  if [[ "$cxx_major" =~ ^[0-9]+$ ]] && command -v "$gcc_candidate" >/dev/null && command -v "$gxx_candidate" >/dev/null; then
    export CC="$(command -v "$gcc_candidate")"
    export CXX="$(command -v "$gxx_candidate")"
  fi
fi

cmake_args=(
  -S "$root"
  -B "$build_dir"
  -DCMAKE_BUILD_TYPE=Release
  -DLLMDECODE_KLEIDIAI=ON
  -DLLMDECODE_NATIVE=ON
  "-DLLMDECODE_LTO=$lto"
)
if [[ -n "${LLMDECODE_KLEIDIAI_SOURCE_DIR:-}" ]]; then
  cmake_args+=(
    "-DFETCHCONTENT_SOURCE_DIR_KLEIDIAI=${LLMDECODE_KLEIDIAI_SOURCE_DIR}"
  )
fi
cmake "${cmake_args[@]}"
cmake --build "$build_dir" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
