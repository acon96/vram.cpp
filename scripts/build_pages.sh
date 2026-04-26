#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
wasm_build_dir="${VRAM_WASM_BUILD_DIR:-build-wasm}"
public_wasm_dir="${repo_root}/ui/public/wasm"
llama_version="$(git -C "${repo_root}/vendor/llama-cpp" describe --tags --match 'b[0-9]*' --abbrev=0 2>/dev/null || true)"
llama_sha="$(git -C "${repo_root}/vendor/llama-cpp" rev-parse --short HEAD 2>/dev/null || true)"

if [[ -z "${llama_version}" ]]; then
    llama_version="unknown"
fi

if [[ -n "${llama_sha}" ]]; then
    llama_version="${llama_version} (${llama_sha})"
fi

if ! command -v emcmake >/dev/null 2>&1; then
    echo "error: emcmake not found in PATH. Activate emsdk first." >&2
    exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "error: npm not found in PATH." >&2
    exit 1
fi

echo "[build-pages] configuring wasm build (${wasm_build_dir})"
emcmake cmake \
    -S "${repo_root}" \
    -B "${repo_root}/${wasm_build_dir}" \
    -DVRAM_BUILD_TESTS=OFF

echo "[build-pages] building wasm predictor"
cmake --build "${repo_root}/${wasm_build_dir}" --target vram_predictor -j4

mkdir -p "${public_wasm_dir}"
cp "${repo_root}/${wasm_build_dir}/vram_predictor.js" "${public_wasm_dir}/vram_predictor.js"
cp "${repo_root}/${wasm_build_dir}/vram_predictor.wasm" "${public_wasm_dir}/vram_predictor.wasm"

echo "[build-pages] building ui bundle"
pushd "${repo_root}/ui" >/dev/null
npm ci
VITE_WASM_BASE_URL="${VITE_WASM_BASE_URL:-./wasm/}" VITE_BASE_URL="${VITE_BASE_URL:-/vram.cpp/}" VITE_LLAMA_VERSION="${VITE_LLAMA_VERSION:-${llama_version}}" npm run build
touch dist/.nojekyll
popd >/dev/null

echo "[build-pages] done: ui/dist"