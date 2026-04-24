# Emscripten Setup (macOS)

This project now builds successfully with Emscripten using `~/emsdk`.

## One-time activation

```bash
~/emsdk/emsdk activate 5.0.6
```

## Per-shell setup

```bash
source ~/emsdk/emsdk_env.sh
emcc -v
```

If `emcc -v` prints compiler info, the toolchain is active.

## Optional auto-setup for new shells

```bash
echo 'source "$HOME/emsdk/emsdk_env.sh"' >> ~/.zprofile
```

Then open a new terminal (or run `source ~/.zprofile`).

## Build wasm target

```bash
source ~/emsdk/emsdk_env.sh
emcmake cmake -S . -B build-wasm -DVRAM_ENABLE_VENDOR_LLAMA=OFF -DVRAM_BUILD_TESTS=OFF
cmake --build build-wasm
```

Expected artifacts:

- `build-wasm/vram_predictor_wasm.js`
- `build-wasm/vram_predictor_wasm.wasm`

## Troubleshooting

- `emcc: command not found`:
  - You likely did not source `~/emsdk/emsdk_env.sh` in the current shell.
- `emsdk --version` fails:
  - `emsdk` does not support `--version`; use `~/emsdk/emsdk list` and `emcc -v`.
