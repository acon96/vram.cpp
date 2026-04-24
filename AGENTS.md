# AGENTS.md

## Scope
These instructions apply to the vram-cpp repository.

## Project Goal
Build a browser-first VRAM predictor that reuses llama.cpp fit behavior through a WASM-friendly API.

## Canonical References
Use links instead of duplicating details:
- Project overview and baseline commands: [README.md](README.md)
- Roadmap and progress tracking: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- Emscripten setup and troubleshooting: [docs/EMSCRIPTEN_SETUP.md](docs/EMSCRIPTEN_SETUP.md)
- API contracts (request + response shapes): [docs/API.md](docs/API.md)

## Repository Map

### C++ (`cpp/`)
- API bridge: `cpp/src/predictor_api.cpp`, `cpp/include/vram/predictor_api.h`
- Prefix parser: `cpp/src/gguf_prefix_parser.cpp`, `cpp/include/vram/gguf_prefix_parser.h`
- Hugging Face range planning/fetch: `cpp/src/hf_range_fetch_helper.cpp`, `cpp/include/vram/hf_range_fetch_helper.h`
- In-process fit executor: `cpp/src/fit_executor.cpp`, `cpp/include/vram/fit_executor.h`
- WASM entrypoint: `cpp/src/predictor_wasm_main.cpp`
- Native fit harness tool: `cpp/tools/vram_fit_harness.cpp`
- Integration tests: `cpp/tests/`

### Web / Browser interop (`web/`)
- WASM JS helper: `web/vram_predictor_browser.js`
- Smoke harness HTML: `web/browser_helper_smoke.html`

### Frontend (`ui/`)
- SvelteJS app: `ui/src/`
- Main app component: `ui/src/App.svelte`
- Parameter panel: `ui/src/components/ParamPanel.svelte`
- File upload: `ui/src/components/FileUpload.svelte`
- Results table: `ui/src/components/ResultsTable.svelte`
- WASM predictor wrapper: `ui/src/lib/predictor.js`

## Build and Test Commands
Use these first unless a task explicitly needs alternatives.

Native (no vendor llama):
```bash
cmake -S . -B build -DVRAM_ENABLE_VENDOR_LLAMA=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

Vendor-enabled native fit execution/parity:
```bash
cmake -S . -B build-vendor -DVRAM_ENABLE_VENDOR_LLAMA=ON
cmake --build build-vendor
ctest --test-dir build-vendor --output-on-failure
```

WASM build:
```bash
source ~/emsdk/emsdk_env.sh
emcmake cmake -S . -B build-wasm -DVRAM_ENABLE_VENDOR_LLAMA=OFF
cmake --build build-wasm --target vram_predictor_wasm -j4
```

Vendor-enabled WASM build:
```bash
source ~/emsdk/emsdk_env.sh
emcmake cmake -S . -B build-wasm-vendor -DVRAM_ENABLE_VENDOR_LLAMA=ON -DVRAM_BUILD_TESTS=OFF
cmake --build build-wasm-vendor --target vram_predictor_wasm -j4
```

## Test Inputs and Environment
- Some fit tests are optional and skip unless env vars are set:
  - `VRAM_LLAMA_FIT_MODELS` (comma/semicolon list)
  - `VRAM_LLAMA_FIT_MODEL` (single fallback)
  - `VRAM_LLAMA_FIT_BINARY`
  - `VRAM_FIT_HARNESS_BINARY`
- `.fixtures/` is gitignored; do not assume fixtures exist in fresh clones.
    - If you need to use a real GGUF file, search in this folder

## API Contract Guardrails
- Keep responses compact and stable (item 14 simplification).
- System info shape should stay minimal (`ok`, `version`, `target`, `features`).
- Fit shape should stay grouped (`targets`, `overrides`, `recommended`, `memoryBytes`, `breakdown`, optional `command`).
- Metadata/HF planning should use lean keys (`source`, `resolvedUrl`, `requests`, `metadata`) without wrapper noise.
- If API shape changes, update all of:
  - `cpp/src/predictor_api.cpp`
  - `docs/API.md`
  - API tests under `cpp/tests/`
  - Browser helper assumptions in `web/vram_predictor_browser.js`

## First Session Notes (2026-04-23)
Record of repo-specific lessons learned during the first coding session:
- WASM fit execution can fail if threaded logger paths are used under Emscripten; keep fit logging conservative in browser runs.
- Metadata-only GGUF fixtures may require larger prefix caps; 4 MiB avoided false `insufficient_prefix_bytes` failures for some vocab models.
- When terminal buffers look stale, trust fresh build/test/browser reruns over old terminal scrollback.
- `hf_range_plan` was merged into `hf_range_fetch_helper`; keep a single helper surface (no duplicate planning modules).
- Compact response contract is now validated in native, vendor, and browser wasm paths; preserve this as the baseline for item 15 UI work.

## Editing and Commit Practices
- Keep changes scoped to the current implementation item; do not batch unrelated work.
- Update plan status/changelog in `IMPLEMENTATION_PLAN.md` when a milestone is completed.
- Commit regularly with small, reviewable checkpoints.
- Avoid direct edits inside `vendor/llama-cpp` unless the task is explicitly about vendor patch maintenance; prefer patch-based updates under `patches/`.
