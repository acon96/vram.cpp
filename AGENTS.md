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

### Root / project control
- Build configuration and target wiring: `CMakeLists.txt`
- Repository-specific agent guidance: `AGENTS.md`
- High-level user/developer docs: `README.md`
- Implementation progress and decision log: `IMPLEMENTATION_PLAN.md`
- Vendor sync-log patch maintenance: `patches/llama-emscripten-sync-fit-logs.patch`
- Third-party source subtree: `vendor/llama-cpp/`
- Local/generated build outputs: `build*/`, `a.out.js`, `a.out.wasm`

### C++ (`cpp/`)
- API bridge: `cpp/src/predictor_api.cpp`, `cpp/include/vram/predictor_api.h`
- Prefix parser: `cpp/src/gguf_prefix_parser.cpp`, `cpp/include/vram/gguf_prefix_parser.h`
- Hugging Face range planning/fetch: `cpp/src/hf_range_fetch_helper.cpp`, `cpp/include/vram/hf_range_fetch_helper.h`
- In-process fit executor: `cpp/src/fit_executor.cpp`, `cpp/include/vram/fit_executor.h`
- Simulated ggml backend: `cpp/src/sim_backend.cpp`, `cpp/include/vram/sim_backend.h`
- WASM entrypoint: `cpp/src/predictor_wasm_main.cpp`
- Native fit harness tool: `cpp/tools/vram_fit_harness.cpp`
- Public headers live under `cpp/include/vram/`
- Unit/integration coverage lives under `cpp/tests/`

### C++ test surfaces (`cpp/tests/`)
- Parser coverage: `parser_tests.cpp`
- Predictor API contract coverage: `predictor_api_tests.cpp`
- HF helper coverage: `hf_range_fetch_helper_tests.cpp`
- Optional native llama fit parity coverage: `llama_fit_parity_tests.cpp`
- Vendor-enabled in-process fit execution coverage: `predictor_api_fit_execution_tests.cpp`

### Web / Browser interop (`web/`)
- WASM JS helper: `web/vram_predictor_browser.js`
- Smoke harness HTML: `web/browser_helper_smoke.html`

### Frontend (`ui/`)
- Vite + Svelte app root: `ui/src/`
- Main app shell and view switch: `ui/src/App.svelte`
- Global frontend styles: `ui/src/app.css`
- Parameter panel: `ui/src/components/ParamPanel.svelte`
- File upload: `ui/src/components/FileUpload.svelte`
- Results table: `ui/src/components/ResultsTable.svelte`
- Raw JSON browser harness: `ui/src/components/JsonHarness.svelte`
- Worker-side fit execution bridge: `ui/src/lib/predictor_fit_worker.js`
- Main-thread worker client: `ui/src/lib/predictor_worker_client.js`
- Shared formatting helpers: `ui/src/lib/format.js`
- Direct browser predictor wrapper: `ui/src/lib/predictor.js`
- UI entrypoint: `ui/src/main.js`
- Local wasm asset server: `ui/scripts/serve-wasm-assets.mjs`

### Semantic runtime flow
- JSON request entrypoint: `cpp/src/predictor_api.cpp`
- Shared fit execution path: `cpp/src/fit_executor.cpp`
- Simulated device injection for deterministic fits: `cpp/src/sim_backend.cpp`
- Standalone native harness uses the same shared executor path as the API/browser fit path: `cpp/tools/vram_fit_harness.cpp`
- Browser UI fit requests run through the dedicated worker, not directly on the main thread: `ui/src/lib/predictor_fit_worker.js` + `ui/src/lib/predictor_worker_client.js`
- Browser raw-request debugging path is exposed at `/?view=harness` via `ui/src/components/JsonHarness.svelte`

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

UI dev/build:
```bash
cd ui
npm install
npm run assets:serve
npm run dev
npm run build
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

## Recent Repo Notes
- The old vendor memory-override fit patch has been removed; deterministic fit simulation now goes through `sim_backend` and stock `common_fit_params(...)`.
- The Emscripten synchronous fit logging patch must be preserved. With browser debug fit logs enabled, the default llama/common logging path can attempt to spawn a thread and crash the non-pthreads wasm build.
- The native fit harness now uses the same shared `execute_fit_request(...)` path as the in-process predictor API/browser fit flow.
- The browser raw JSON harness at `/?view=harness` is the fastest way to validate exact request/response behavior against a local GGUF file.
- WASM fit execution should keep fit logging conservative by default even though the sync-log patch exists.
- Metadata-only GGUF fixtures may require larger prefix caps; 4 MiB avoided false `insufficient_prefix_bytes` failures for some vocab models.
- When terminal buffers look stale, trust fresh build/test/browser reruns over old terminal scrollback.
- `hf_range_plan` was merged into `hf_range_fetch_helper`; keep a single helper surface (no duplicate planning modules).
- Compact response contract is now validated in native, vendor, and browser wasm paths; preserve this as the baseline for UI and API follow-up work.

## Editing and Commit Practices
- Keep changes scoped to the current implementation item; do not batch unrelated work.
- Update plan status/changelog in `IMPLEMENTATION_PLAN.md` when a milestone is completed.
- Commit regularly with small, reviewable checkpoints.
- Avoid direct edits inside `vendor/llama-cpp` unless the task is explicitly about vendor patch maintenance; prefer patch-based updates under `patches/`.
- If a task touches wasm fit logging behavior, update both the live vendor delta and `patches/llama-emscripten-sync-fit-logs.patch`.
