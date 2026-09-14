#!/usr/bin/env bash
#
# build.sh - cross-build this project's vision binary in the SDK container for the
# Modalix DevKit (aarch64). Offline build step; no inference.
#
#   vision-local/build.sh
#
# Output: vision-local/build-devkit/drone-seminar-parallel-vision
# Uses only files inside this project (sources in src/, toolchain in cmake/) plus the
# installed SDK sysroot and cross compilers.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${HERE}/build-devkit}"
TOOLCHAIN="${TOOLCHAIN:-${HERE}/cmake/aarch64-modalix.cmake}"
export CC="${CC:-aarch64-linux-gnu-gcc}"
export CXX="${CXX:-aarch64-linux-gnu-g++}"
[ -f "${TOOLCHAIN}" ] || { echo "ERROR: toolchain not found: ${TOOLCHAIN}" >&2; exit 1; }
cmake -S "${HERE}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}"
cmake --build "${BUILD_DIR}" -j"$(nproc 2>/dev/null || echo 8)"
echo "Built: ${BUILD_DIR}/drone-seminar-usb-vision"
