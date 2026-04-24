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
- Unit tests for parser/range behavior

## Build (native dev smoke check)

```bash
cmake -S . -B build -DVRAM_ENABLE_VENDOR_LLAMA=OFF
cmake --build build
./build/vram_predictor_dev
ctest --test-dir build --output-on-failure
```

## Build (Emscripten)

```bash
emcmake cmake -S . -B build-wasm -DVRAM_ENABLE_VENDOR_LLAMA=OFF
cmake --build build-wasm
```

Expected artifacts:

- `vram_predictor_wasm.js`
- `vram_predictor_wasm.wasm`

## Next implementation steps

1. Integrate llama.cpp as `vendor/llama-cpp`
2. Build byte-range HTTP fetch helper over resolved HF URLs
3. Connect GGUF prefix parser to predictor request flow
4. Add first golden validation tests
