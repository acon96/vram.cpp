# vram-cpp

WASM-first VRAM predictor scaffold for llama.cpp fit-style memory estimation.

## Current status

This repository currently contains Phase 0 + early Phase 2 scaffolding:

- JSON schemas for predictor request/response
- C ABI surface for JS/WASM interop
- Emscripten target skeleton exporting predictor entrypoints
- Submodule vendoring for llama.cpp in `vendor/llama-cpp`
- GGUF prefix parser prototype for partial metadata extraction
- HF progressive prefix range planning helper
- HF URL resolver for direct model-file range requests
- Local predictor API path that progressively parses GGUF prefixes from disk
- Unit tests for parser/range behavior and predictor API integration

## Build (native dev smoke check)

```bash
cmake -S . -B build -DVRAM_ENABLE_VENDOR_LLAMA=OFF
cmake --build build
./build/vram_predictor_dev
ctest --test-dir build --output-on-failure
```

## Build (Emscripten)

In each shell session, load emsdk first:

```bash
source ~/emsdk/emsdk_env.sh
```

```bash
emcmake cmake -S . -B build-wasm -DVRAM_ENABLE_VENDOR_LLAMA=OFF
cmake --build build-wasm
```

Detailed setup and troubleshooting: `docs/EMSCRIPTEN_SETUP.md`.

Expected artifacts:

- `vram_predictor_wasm.js`
- `vram_predictor_wasm.wasm`

## Next implementation steps

1. Add native `llama-fit-params` parity golden tests for 2-3 full model fixtures
2. Execute remote HF range-fetch loop and feed fetched prefixes into parser
3. Begin wiring wasm wrapper to llama fit internals
