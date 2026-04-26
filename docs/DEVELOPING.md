# Developing

## Build (native dev smoke check)

```bash
cmake -S . -B build
cmake --build build
./build/vram_predictor
ctest --test-dir build --output-on-failure
```

## Build (Emscripten)

In each shell session, load emsdk first:

```bash
source ~/emsdk/emsdk_env.sh
```

```bash
emcmake cmake -S . -B build-wasm -DVRAM_BUILD_TESTS=OFF
cmake --build build-wasm --target vram_predictor -j4
```

Expected artifacts:

- `vram_predictor.js`
- `vram_predictor.wasm`

## Emscripten Setup (macOS)

This project now builds successfully with Emscripten using `~/emsdk`.

### One-time activation

```bash
~/emsdk/emsdk activate 5.0.6
```

### Per-shell setup

```bash
source ~/emsdk/emsdk_env.sh
emcc -v
```

If `emcc -v` prints compiler info, the toolchain is active.

### Optional auto-setup for new shells

```bash
echo 'source "$HOME/emsdk/emsdk_env.sh"' >> ~/.zprofile
```

Then open a new terminal (or run `source ~/.zprofile`).

### Build wasm target

```bash
source ~/emsdk/emsdk_env.sh
emcmake cmake -S . -B build-wasm -DVRAM_BUILD_TESTS=OFF
cmake --build build-wasm
```

Expected artifacts:

- `build-wasm/vram_predictor.js`
- `build-wasm/vram_predictor.wasm`

## Troubleshooting

- `emcc: command not found`:
  - You likely did not source `~/emsdk/emsdk_env.sh` in the current shell.
- `emsdk --version` fails:
  - `emsdk` does not support `--version`; use `~/emsdk/emsdk list` and `emcc -v`.
