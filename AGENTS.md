# AGENTS.md

## Scope
These instructions apply to the vram-cpp repository.

## Project Goal
Build a browser-first VRAM predictor that reuses llama.cpp fit behavior through a WASM-friendly API.

## Canonical References
Use links instead of duplicating details:
- Project overview and baseline commands: [README.md](README.md)
- Roadmap and progress tracking: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- setup and troubleshooting: [docs/DEVELOPING.md](docs/DEVELOPING.md)
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
- In-process fit executor: `cpp/src/fit_executor.cpp`, `cpp/include/vram/fit_executor.h`
- Simulated ggml backend: `cpp/src/sim_backend.cpp`, `cpp/include/vram/sim_backend.h`
- WASM entrypoint: `cpp/src/predictor_main.cpp`
- Native fit harness executable (`vram_predictor`) now uses: `cpp/src/predictor_main.cpp`
- Public headers live under `cpp/include/vram/`
- Unit/integration coverage lives under `cpp/tests/`

### C++ test surfaces (`cpp/tests/`)
- Predictor API contract coverage: `predictor_api_tests.cpp`
- Optional native llama fit parity coverage: `llama_fit_parity_tests.cpp`
- Vendor-enabled in-process fit execution coverage: `predictor_api_fit_execution_tests.cpp`

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
- Browser WASM helper: `ui/src/lib/vram_predictor_browser.js`
- Shared formatting helpers: `ui/src/lib/format.js`
- Direct browser predictor wrapper: `ui/src/lib/predictor.js`
- UI entrypoint: `ui/src/main.js`
- Local wasm asset server: `ui/scripts/serve-wasm-assets.mjs`

### Semantic runtime flow
- JSON request entrypoint: `cpp/src/predictor_api.cpp`
- Shared fit execution path: `cpp/src/fit_executor.cpp`
- Simulated device injection for deterministic fits: `cpp/src/sim_backend.cpp`
- Standalone native harness executable uses the same shared JSON entrypoint and fit path as the API/browser flow: `cpp/src/predictor_main.cpp` + `cpp/src/fit_executor.cpp`
- Browser UI fit requests run through the dedicated worker, not directly on the main thread: `ui/src/lib/predictor_fit_worker.js` + `ui/src/lib/predictor_worker_client.js`
- Browser raw-request debugging path is exposed at `/?view=harness` via `ui/src/components/JsonHarness.svelte`

## Build and Test Commands
Use these first unless a task explicitly needs alternatives.

Native validation build:
```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Deployable WASM build:
```bash
source ~/emsdk/emsdk_env.sh && emcmake cmake -S . -B build-wasm -DVRAM_BUILD_TESTS=OFF
cmake --build build-wasm --target vram_predictor -j4
```
> Note: This build takes 60+ seconds to link the final WASM binary, so only run this when you need to validate browser behavior or update the WASM build after C++ changes. For faster iterative development, use the native build and tests, which share all core logic and are much quicker to build and run.

UI development server:
```bash
cd ui
npm ci
npm run assets:serve # (in a separate terminal; serves wasm assets from build-wasm)
npm run dev
```

UI production build:
```bash
cd ui
npm ci
npm run build
```

## Test Inputs and Environment
- Some fit tests are optional and skip unless env vars are set:
  - `VRAM_LLAMA_FIT_MODELS` (comma/semicolon list)
  - `VRAM_LLAMA_FIT_MODEL` (single fallback)
  - `VRAM_LLAMA_FIT_BINARY`
  - `VRAM_PREDICTOR_BINARY`
- `.fixtures/` is gitignored; do not assume fixtures exist in fresh clones.
    - If you need to use a real GGUF file, search in this folder; the developer has likely downloaded fixtures here for testing. Just make sure you search the folder directly before trying to download large files again.

## API Contract Guardrails
- Keep responses compact and stable (item 14 simplification).
- System info shape should stay minimal; do not include unnecessary fields such as `ok`, `version`, `status`, or `message` that do not add value for the UI or client logic.
- Fit shape should stay grouped (`targets`, `overrides`, `recommended`, `memoryBytes`, `breakdown`, optional `command`).
- Metadata/HF planning should use lean keys (`source`, `resolvedUrl`, `requests`, `metadata`) without wrapper noise.
- If API shape changes, update all of:
  - `cpp/src/predictor_api.cpp`
  - `docs/API.md`
  - API tests under `cpp/tests/`
  - Browser helper assumptions in `ui/src/lib/vram_predictor_browser.js`

## Recent Repo Notes
- The old vendor memory-override fit patch has been removed; deterministic fit simulation now goes through `sim_backend` and stock `common_fit_params(...)`.
- The Emscripten synchronous fit logging patch must be preserved. With browser debug fit logs enabled, the default llama/common logging path can attempt to spawn a thread and crash the non-pthreads wasm build.
- The native fit harness now uses the same shared `execute_fit_request(...)` path as the in-process predictor API/browser fit flow.
- The browser raw JSON harness at `/?view=harness` is the fastest way to validate exact request/response behavior against a local GGUF file.
- WASM fit execution should keep fit logging conservative by default even though the sync-log patch exists.
- Metadata-only GGUF fixtures may require larger prefix caps; 4 MiB avoided false `insufficient_prefix_bytes` failures for some vocab models.
- When terminal buffers look stale, trust fresh build/test/browser reruns over old terminal scrollback.
- HF remote fetch execution is intentionally JS-side in browser flows; C++ keeps local parsing and HF request planning only.
- Compact response contract is now validated in native, vendor, and browser wasm paths; preserve this as the baseline for UI and API follow-up work.
- Cleaned up a bunch of duplicate or unused code paths and parameters in the API and fit execution logic; Going forward, if you alter a large feature or flow, do not leave the old code in place unless there is a clear consumer that is still utilizing the code path.

## Editing and Commit Practices
- Keep changes scoped to the current implementation item; do not batch unrelated work.
- Update plan status/changelog in `IMPLEMENTATION_PLAN.md` when a milestone is completed.
- Commit regularly with small, reviewable checkpoints.
- Avoid direct edits inside `vendor/llama-cpp` unless the task is explicitly about vendor patch maintenance; prefer patch-based updates under `patches/`.
- If a task touches wasm fit logging behavior, update both the live vendor delta and `patches/llama-emscripten-sync-fit-logs.patch`.
- If a file becomes corrupted while editing it, the fastest way to clean it up is to delete the file and then create it again with the create file tool.
  - Do NOT try to run a python script or write to other arbitrary files to try and replace the file.
  - If the file you are re-creating is long, then you must do it in multiple smaller chunks by first creating the file and then appending to it with the edit file tool. - This is because the edit file tool has a token limit and if you exceed it, the file will become corrupted again.

## Common Deliverables
- When implementing a new feature or fixing a bug, the expected deliverables are:
  - The updated code checked into the repository with a clear commit message.
  - Updated or new tests that validate the change.
  - If the change affects the API contract, an update to `docs/API.md` reflecting the new request/response shapes.
  - Building and testing the change locally to ensure it works as expected before committing.
- Do not give a large summary when finishing your change. You can just respond saying you are done. 
  - The commit message and code comments should be enough to explain the change.
  - If you need to provide additional context, consider adding comments in the code or updating the relevant documentation files instead.