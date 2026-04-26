# VRAM.cpp: a llama.cpp WASM Memory Predictor - Implementation Plan

## 1. Goal

Build a browser app that compiles from llama.cpp source to WebAssembly and predicts model memory usage using llama-fit style logic plus model metadata inputs:

- architecture details from GGUF metadata
- tensor dtypes and dimensions
- context size and KV cache settings
- quantization level (weights + KV)
- target device memory layout assumptions

The app should support both local GGUF files and remote Hugging Face models with header-only metadata fetch.
This app will enable users to quickly estimate system requirements for running a given model with specific context and quantization settings, without needing to download the full model or run a native fit tool. Because it utilizes the same underlying logic as llama-fit, it will provide accurate estimates and recommendations for fitting models into available memory, while being accessible directly from the browser.

## 2. Key Findings From Current Repo

### Existing memory-fit logic

- Core fitting algorithm exists in common/fit.cpp and public helper declarations in common/fit.h.
- common_fit_params already computes projected model/context/compute memory and adjusts n_ctx, n_gpu_layers, tensor split, and tensor buffer overrides.
- Deterministic memory simulation now routes through `sim_backend` and stock `common_fit_params(...)`; only the Emscripten-safe fit logging hook remains patched in vendor llama.cpp.
- Memory breakdown relies on llama_get_memory_breakdown from src/llama-ext.h (currently C++ extension API, not stable C API in include/llama.h).

### Existing parameter surface needed by UI

- Context size, KV cache dtype K/V, mmap, n_gpu_layers, fit targets, etc. already map through common/common.cpp conversion helpers.
- CLI argument definitions in common/arg.cpp already document supported knobs and defaults.

### Existing remote/HF plumbing (important for header-only mode)

- common/download.cpp already supports:
  - HEAD + ETag + Accept-Ranges checks
  - resumable Range downloads
  - custom request headers via common_remote_get_content
- hf-cache API already resolves repo tree and direct file URLs without downloading full files.

### WASM build constraints

- Root CMake enables Emscripten and MEMORY64 support (LLAMA_WASM_MEM64) and allows memory growth.
- Normal tools/examples are excluded under EMSCRIPTEN in tools/CMakeLists.txt and examples/CMakeLists.txt.
- Therefore, this needs a new Emscripten-compatible target, not reuse of existing native tool targets.

## 3. Product Architecture

## 3.1 Frontend

- Single-page web UI (TS + Worker) for:
  - model source: local upload or Hugging Face repo/file
  - runtime knobs: n_ctx, n_batch, n_ubatch, cache_type_k/v, offload flags
  - device assumptions: host RAM and optional virtual device list with free-memory targets
  - output: total + breakdown + sensitivity graph (memory vs context)

## 3.2 WASM engine

Create a small C ABI wrapper around llama/common fit internals:

- Input JSON (or struct) describing model and runtime params
- Output JSON breakdown and fit recommendation
- Expose stable C entrypoints for JS interop

Planned wrapper responsibilities:

1. Build model/context params from request
2. Run one projection pass (no auto-fit)
3. Run fit pass (llama-fit behavior)
4. Return:
   - projected model/context/compute memory by device + host
   - fitted n_ctx / n_gpu_layers / tensor_split / overrides
   - warnings/failure reasons

## 3.3 Two prediction modes

### Mode A: Metadata-first estimator (default for web)

- Uses GGUF metadata + tensor info only
- Fast and works with partial remote fetch
- Produces:
  - weight memory by dtype
  - estimated KV cache memory from n_ctx and K/V cache types
  - optional architecture-derived graph statistics

### Mode B: Full llama-fit projection (higher fidelity)

- Uses existing llama/common fit flow in wasm
- Best accuracy for model/context/compute split
- Requires model bytes accessible in wasm virtual FS
- For very large models, may be limited by browser memory and bandwidth

Recommendation: ship Mode A first, then add Mode B as optional advanced mode.

## 4. Hugging Face Header-Only Strategy (New Requirement)

Goal: get tensor dtypes and dimensions without downloading full GGUF.

### Proposed flow

1. Resolve repo and model file via existing HF cache tree API (already in common/hf-cache.cpp).
2. Perform byte-range GET on the GGUF file URL:
   - start with a small prefix (for example 1-4 MiB)
   - parse GGUF header + KV + tensor-info table
   - if parser indicates incomplete data, grow range progressively
3. Stop once tensor metadata is complete; do not fetch tensor payload.

### Implementation options

Option 1 (preferred):

- Add a dedicated header parser in wasm module using gguf structures/parsing logic.
- Feed parser an in-memory byte buffer from range fetches.
- Return compact JSON summary of tensor names, shapes, and ggml types.

Option 2 (fallback):

- Write fetched prefix to wasm FS and use existing GGUF file parser path.
- If parser insists on full file size checks, switch to Option 1 parser.

### Why this is feasible in this repo

- common_remote_get_content already supports custom headers, so Range: bytes=... is available.
- Existing download layer already handles auth headers and HF token flow patterns.

## 5. Proposed Repo Implementation Plan

## Phase 0: Design lock + app skeleton

- Define request/response schema for VRAM predictor API. Set up as a single C function with JSON string input and output for easy JS interop and well defined contract.
- Set up llama.cpp as a submodule in a new vendor/llama-cpp folder with CMake integration.

## Phase 1: Refine Emscripten predictor target

- Add new target under tools or a new wasm-specific folder.
- Build with EMSCRIPTEN guard and explicit exported C functions.
- Link against llama + llama-common as needed.

Deliverable:

- predictor.js + predictor.wasm with callable API from JS.

## Phase 2: Metadata parser path

- Implement GGUF metadata/tensor-info extraction from partial bytes.
- Add iterative range fetch helper for HF URLs.
- Normalize output into model profile object.

Deliverable:

- robust dtype/dims extraction without full file download.

## Phase 3: Memory math engine

User comment: SKIP THIS PHASE. This is what everyone else does. the entire point of this project is to utilize the actual llama-fit code in wasm. any validation or benchmarking can be done against the native llama-fit code.

- ~~Compute:~~
  - ~~weight memory by tensor type~~
  - ~~KV cache memory by n_ctx, n_layer, n_head_kv, n_embd_head_k/v, and cache dtypes~~
  - ~~optional per-layer approximation graph~~
- ~~Validate against known models and native llama-fit output.~~

~~Deliverable:~~
- ~~estimator output with confidence flags and assumptions.~~

## Phase 4: llama-fit integration mode

- Use existing llama-fit internals to run a full projection pass in wasm with the same input parameters.
- Patch out any file I/O to work with in-memory buffers or virtual FS as needed, and ensure that enough of the file has been fetched to satisfy llama-fit’s needs.
- Return per-device model/context/compute and fitted allocation suggestions.

Deliverable:

- advanced accuracy mode for users with sufficient browser resources.

## Phase 5: Web app

- Worker-based compute pipeline
- Interactive controls and comparison charts:
  - memory vs n_ctx
  - memory vs kv dtype
  - memory vs quant choice
- Export/share JSON scenarios.

## 6. Validation Plan

- Golden tests against native outputs from tools/fit-params for several models and context sizes.
- Unit tests for HF header-range parser with synthetic truncated buffers.
- Regression suite for:
  - split GGUF models
  - MoE models
  - recurrent/hybrid architecture metadata

## 7. Risks and Mitigations

Risk: llama-ext memory breakdown API is C++ staging API.

- Mitigation: keep wasm wrapper internal and avoid exposing llama-ext types directly to JS.

Risk: browser memory limits for full-fit mode.

- Mitigation: ship metadata-only mode first; make full-fit opt-in and clearly labeled.

Risk: HF response variations (redirects, range support, auth).

- Mitigation: fallback from range to bounded full-prefix fetch with max cap and clear UI errors.

Risk: mismatch between estimator and runtime backend behavior.

- Mitigation: calibration against native fit tool outputs and add model-specific correction factors only when necessary.

## 8. Recommended MVP Scope

1. Header-only GGUF metadata extraction (local + HF)
2. Deterministic memory estimator for weights and KV cache
3. UI controls for context, KV dtype, and quant profile
4. Optional suggested fit settings (n_ctx and offload guidance)

This gives immediate value with low bandwidth cost and creates a clean base for deeper llama-fit wasm integration next.

## 9. Immediate Next Steps

### "Mainline" implementation tasks:
1. [x] Create predictor API schema file (request/response JSON).
2. [x] Add Emscripten target skeleton with one exported function returning version/system info.
3. [x] Implement GGUF metadata parse prototype and iterative HF prefix range planning helper.
4. [x] Add first golden fixture tests for GGUF metadata parsing on 3 vendored GGUF files.
5. [x] Connect local GGUF prefix parser flow to `vram_predictor_predict_json` with progressive prefix attempts.
6. [ ] Add native `llama-fit-params` parity golden tests for 2-3 full model GGUF fixtures. Use ggml-org/gemma-3-270m-GGUF because it has a Q8_0 quant that is only ~200mb
  - [x] Added custom in-process harness (`vram_fit_harness`) that links llama/common code directly.
  - [x] Added optional parity test comparing harness + llama-fit-params outputs when fixture/binaries are provided via env vars.
  - [x] Executed first real fixture parity run on `gemma-3-270m-Q8_0.gguf`.
  - [x] Added reusable llama/common patch for overriding detected device/host free memory during fit.
  - [x] Added and validated two additional full-model fixtures: `Qwen2.5-0.5B-Instruct.Q4_0.gguf` and `gemma-3-1b-it-Q2_K.gguf`.
7. [x] Build HF URL resolution + progressive byte-range request planning helper and wire it to `model.source = huggingface` request handling.
8. [x] Execute remote HF range fetch loop and feed downloaded prefixes into parser.
9. [x] Begin wiring in-process predictor API execution for llama-fit override mode so the exported API can return actual fitted results instead of planning-only output.
  - [x] Add a reusable internal fit execution helper linked from predictor core when vendor llama/common is enabled.
  - [x] Execute fit requests in-process from `vram_predictor_predict_json` with explicit device/host memory overrides.
  - [x] Validate the execution path in a vendor-enabled native build and carry the same surface into a vendor-enabled wasm build.
10. [x] Return detailed model/context/compute memory breakdowns from the in-process API path instead of fitted parameters only.
11. [x] Add a JS-side wasm integration flow that mounts model bytes and calls `fit.execute_in_process`.
12. [x] Put together a simple html UI page that can exercise the wasm predictor API with local file and HF URL inputs and display the output breakdowns and fit recommendations.
13. [x] Add more fixtures and edge cases to the test suite, including split GGUF models, metadata-only files, and heterogeneous systems.
  - [x] Added predictor API edge-case coverage for metadata-only GGUF fixtures using multiple vocab-only model files.
  - [x] Added split-shard Hugging Face planning coverage for `-00001-of-00002.gguf` request flows.
  - [x] Added heterogeneous multi-GPU fit planning coverage with mixed free/total device memory overrides.
  - [x] Validated browser wasm metadata API against 3 additional vendored GGUF files (`ggml-vocab-gpt-neox`, `ggml-vocab-llama-spm`, `ggml-vocab-starcoder`) after increasing fetch cap.
  - [x] Extended vendor fit execution integration test to accept `VRAM_LLAMA_FIT_MODELS` (comma/semicolon list) for running the same assertions across 2-3 full model fixtures.
  - [x] Validated full-model native+wasm parity/execution paths for three fixtures: `gemma-3-270m-Q8_0.gguf`, `Qwen2.5-0.5B-Instruct.Q4_0.gguf`, and `gemma-3-1b-it-Q2_K.gguf`.
14. [x] Clean up the API surface to be simple and stable. remove unnecessary fields not required for the app to function or hard coded values. This will be a private, internal API interface so it doesn't need all the random metadata/wrapper stuff that it has now.
  - [x] Removed wrapper-oriented response fields (`phase`, `engine`, `apiVersion`, duplicated mode metadata, and descriptive limitations) from predictor responses.
  - [x] Introduced a compact fit response contract with grouped keys (`targets`, `overrides`, `recommended`, `memoryBytes`, `breakdown`, `command`).
  - [x] Updated predictor response schema and API tests to validate the simplified contract.
  - [x] Consolidated duplicate HF prefix range-planning logic into a single internal implementation and removed redundant files.
  - [x] Revalidated native tests, vendor fit/parity tests, and wasm browser fit execution with the new contract.
15. [x] Build a proper UI with an actual UI framework that is interactive and visually compares different fit scenarios across different models, quantization levels, or context sizes.
  - [x] Selected Svelte 5 + Vite as the UI framework (`ui/` directory). Chosen for minimal bundle size, reactive primitives without heavy runtime, and straightforward GitHub Pages deployment.
  - [x] Scaffolded `ui/` with Vite + Svelte template; cleaned up boilerplate, replaced `app.css` with project design tokens (warm cream / dark green palette, dark mode support).
  - [x] Implemented `FileUpload.svelte` — drag-and-drop / file-picker for local `.gguf` files with file name + size display.
  - [x] Implemented `ParamPanel.svelte` — runtime config (n_ctx slider, KV cache K/V type selectors, n_gpu_layers), host RAM, per-GPU VRAM (add/remove up to 4 GPUs), fit target MiB, and target free MiB fields.
  - [x] Implemented `ResultsTable.svelte` — recommended n_ctx/GPU-layers chips, per-device memory breakdown table (model / KV cache / compute / total / capacity) with usage progress bars and colour-coded thresholds.
  - [x] Wired all components in `App.svelte` with a two-column layout (config sidebar + results panel), async WASM loading via `initPredictor`, loading/error state display, and `predictMountedFit` integration.
  - [x] Added `lib/predictor.js` — lazy WASM initialisation helper (loads script tag, dynamic-imports browser helper, caches client promise).
  - [x] Added `lib/predictor_fit_worker.js` and `lib/predictor_worker_client.js` to execute fit prediction in a dedicated web worker, preventing long-running synchronous wasm calls from blocking the main UI thread.
  - [x] Added `lib/format.js` — MiB/GiB/byte conversion and display helpers.
  - [x] Verified production build (`npm run build` in `ui/`) produces clean output with no errors.
  - [x] Wired `VITE_WASM_BASE_URL` to accept full asset URLs (cross-port local dev) or relative paths (static hosting), and added a dedicated local assets server script that serves wasm/helper files in place without copy steps.
  - [x] Hardened `n_gpu_layers` UI handling so `-1` is preserved as "all layers on GPU" instead of drifting through invalid intermediate numeric states.
  - [x] Added a raw JSON harness view at `/?view=harness` so wasm requests can be pasted, submitted, and inspected directly through the Svelte app.
16. [x] Finish wiring up ability to select models from HuggingFace repos directly in the UI, using the existing HF cache API to resolve model files and feeding them into the same header-range-fetch + parse flow used by the API tests.
  - [x] Added a `HuggingFaceSearch` UI component that searches HF repos and loads candidate `.gguf` files from the selected repo/revision tree.
  - [x] Wired a metadata validation preflight that runs browser-side range fetch + parser calls through the wasm worker and surfaces errors/metadata in the UI.
  - [x] Updated the model input layout to an explicit top-level "Upload" OR "Search on HF" switch next to existing upload flow.
17. [x] Properly implement fit execution in the wasm module using the existing llama/common fit code, with simulated backend devices for deterministic memory reporting and API-level backend profile selection.
  - [x] Added a new simulated backend module (`cpp/src/sim_backend.cpp`, `cpp/include/vram/sim_backend.h`) that exposes profile-aware fake ggml GPU devices (CUDA/Metal/Vulkan/Generic) with configurable free/total memory and null-terminated `ggml_backend_dev_t *` wiring for `llama_model_params.devices`.
  - [x] Routed in-process fit execution through `sim_backend` + stock `common_fit_params(...)` so predictor API fit execution no longer calls patched `common_fit_params_with_memory_override(...)`.
  - [x] Threaded optional per-device backend profile selection through API request parsing (`device.gpus[].backend`) and validated it in predictor API tests.
  - [x] Stabilized wasm in-process fit breakdown collection by using no-allocation model/context setup and falling back to non-fatal summary rows when detailed breakdown collection fails.
  - [x] Refined simulated device memory accounting so `ggml_backend_dev_memory` reports post-allocation free bytes (reducing negative/unbounded unaccounted memory artifacts in debug tables).
  - [x] Corrected `target_free_mib` to fit-margin conversion so large simulated free-memory values do not inflate margin requirements and force unexpected host placement.
  - [x] Removed the vendor `common_fit_params_with_memory_override(...)` surface and rewired `vram_fit_harness` onto the shared `execute_fit_request(...)` path.
18. [x] Implement remaining features, knobs, and UI polish, such as:
  - [x] Add explicit UI controls for choosing per-device backend profiles in the Svelte parameter panel (cuda, metal, vulkan, etc.)
  - [x] Add hard coded profiles for common GPUs with total memory and backend profile presets
  - [x] Add support for split modes (layer, row, tensor) and other llama.cpp knobs that affect memory allocation/usage
  - [x] Figure out how to visually indicate the progress of the fit execution (i.e. number of attempts/iterations?)
  - [x] probably need to detect the number of iterations the fit algorithm has gone through and show "Fitting... (attempt ##)" below the spinner
    - to detect the iterations as well as the current ngl and nctx being tested, we should parse the logs coming out of the fit execution. to do that we need to stop sending out the debug logs and only send the info level logs. this also means we should re-check if us enabling the virtual pthreads thingy allows the built in logging system for llama.cpp to work with emscripten.
  - [x] allow cancelling the fit attempt if it's taking too long or the user wants to adjust parameters and try again. this would involve adding a cancellation mechanism to the API and the wasm worker, and then adding a "Cancel" button in the UI that triggers it.
  - [x] finish wiring up the min ctx checkbox and add a tooltip explaining that enabling it tells llama.cpp to treat it as a minimum context size and might go higher if it fits. this will also require passing it to the backend correctly.
  - [x] When I click on the "Retrieve Tensor Info" button. there should be a loading info text block that updates continuously through the gguf fetch process, while it is parsed/validated, as well as any follow up requests that happen to pull down further shard chunks.
  - [x] remove the "parroting" back of values in `fit_execution_result` `fit_execution_request` and just generally make that API interface simpler, there's a ton of unnecessary info being passed around there
  - [x] don't surface the true debug logs in the web viewer. they come across on stderr instead of stdout. there's a ton of noise from each model load and fit attempt that isn't really useful for the user to see
  - [x] make sure you catch the attempts and iterations of the fit loop that happen at the beginning when trying a few different layouts and prior to iterating over the n_gpu_layers and n_ctx sizes. the text you need to look for might be slightly different for those first few loops
19. [ ] Set up github actions to build and deploy the app to a GitHub Pages site; once that works we want to grab any new llama.cpp model architectures (run nightly)
  - [ ] Create an initial "unified" build process that builds the wasm module and packages the bundle with the UI for static hosting; likely some sort of vite build plugin
  - [ ] create a GitHub actions pipeline that runs the build process whenever there is a push to the main branch; should publish the built app to a GitHub Pages site using the pre-built action
  - [ ] Add a nightly workflow that checks for new commits to the llama.cpp repo, and if it detects any, updates the submodule commit reference and pushes a commit to the main branch to trigger a rebuild and redeploy of the app with the latest llama.cpp changes.

### Fixes needed:
- [x] de-duplicate the "Target Free MiB", "Fit target (MiB)" and the "Free VRAM" parameters in the gpu device section. 
  - Removed global fitTargetMiB/targetFreeMiB; replaced with per-GPU `bufferMiB` (keep-free margin) fed to both fit_target_mib and target_free_mib in the fit engine.
- [x] KV cache quantization types are only FP16, Q8_0 and Q4_0. remove all other options
- [x] split gguf files: use original HF filename when mounting prefix bytes in WASM FS so llama.cpp sees the correct shard naming convention
- [x] There is something wrong with the re-implementation of the tensor attribution code. We should just use the stock code in fit.cpp instead of re-writing it ourself and breaking something in the process
  - `fill_breakdown_fallback` shouldn't exist at all. if it fails, we have a problem
  - replace `collect_memory_breakdown` with `common_get_device_memory_data` from `fit.cpp`; they effectively do the same thing and our version is reporting weird things with multi-gpus

### Tweaks:
- [x] Auto assign the device index based on the order they are in the UI. the user doesn't need to select them directly
- [x] Metadata preview should be collapsed and not cause the UI to jump when it loads/appears. basically put the drawer behind a button that is enabled once the metadata is verified
- [x] Move the 'Runtime' parameters section to be on the same row as the 'Model' section and expand the remaining parameters to fill that row and make it into the 'Hardware Config' section.
- [ ] Disable as much of the remaining llama.cpp build as possible to reduce the wasm binary size

### Cleanup:
- [x] Remove the `web/` folder and move the wasm helper and worker files into `ui/lib/` since they are only used by the UI and not shared with any other potential consumers of the wasm module. the smoke test can be deleted
- [x] replace all the custom gguf parsing code with @huggingface/gguf because it can already handle remote ggufs stored on HuggingFace with built in support for range requests, and it will be more robust and better maintained than a custom implementation. just need to make sure it supports retrieving the raw bytes of the metadata without trying to fetch the whole file, which it should be able to do since it already supports range requests.
    - browser-side HF metadata now uses `@huggingface/gguf`, and native/local metadata now goes through vendored GGUF/ggml readers instead of the removed custom parser.
    - HuggingFace gguf "validate" button should show all the individual sub-steps in the hint text box so it doesn't look like it "froze"
- [x] reduce the number of "build targets" for the cpp part of the project. 
  - there shouldn't be the ability to build **without** llama.cpp vendored in. unit testing against the core logic without llama.cpp doesn't help a ton since the main point of the project is to run the actual llama-fit code in wasm or native.
  - removed the old optional-vendor CMake path and the related `VRAM_ENABLE_VENDOR_LLAMA` / `VRAM_HAS_LLAMA_FIT_EXECUTION` compatibility branches.
- [ ] Work through the entire codebase and find any unnecessary complications in the arguments, responses, and API surfaces. 
  - The goal would be to remove extra logic and handling for scenarios that don't exist in the codebase.
  - Basically a reverse YAGNI pass to simplify the code and make it easier to maintain. For example, if there are any parameters that are accepted but not actually used anywhere in the code, those should be removed.
  - If there are any response fields that are calculated but not actually returned or used by the UI, those should be removed as well.
  - The idea is to have a clean and minimal codebase that only includes what is actually needed for the app to function, without extra noise or complexity from unused features or hypothetical scenarios.
- [ ] replace the readme with a simple, concise description of the project and links to the github pages url.

## 10. Change Log

- 2026-04-23: Bootstrapped repository skeleton with CMake, include/src layout, API schemas, and WASM-exported C ABI stubs.
- 2026-04-23: Added `.gitignore` and `README.md` for baseline housekeeping and reproducible build entrypoints.
- 2026-04-23: Updated immediate next steps with completion markers to support phased implementation tracking.
- 2026-04-23: Added llama.cpp as a git submodule at `vendor/llama-cpp` with optional CMake integration.
- 2026-04-23: Switched API JSON serialization to nlohmann JSON from the vendored llama.cpp dependency.
- 2026-04-23: Implemented GGUF prefix metadata parser and HF prefix range planning helper with parser unit tests.
- 2026-04-23: Added golden regression tests using 3 vendored GGUF fixtures and split native fit parity testing into an explicit follow-up step requiring full model fixtures.
- 2026-04-23: Integrated local GGUF prefix parsing into predictor API requests and added API-level integration tests.
- 2026-04-23: Added HF file URL resolver and progressive range request planner with helper tests, plus API support for Hugging Face request planning output.
- 2026-04-23: Validated first Emscripten build (`vram_predictor_wasm.js/.wasm`) and added a dedicated setup guide for emsdk activation and wasm build commands.
- 2026-04-23: Implemented platform-aware HF range execution backend (Emscripten fetch in browser/WASM, curl fallback in native) and fed fetched prefixes directly into the GGUF parser.
- 2026-04-23: Replaced executable-driven fit orchestration in predictor API with harness planning semantics and added a native in-process fit harness (`vram_fit_harness`) linked to llama/common code.
- 2026-04-23: Built and ran first native parity comparison using `gemma-3-270m-Q8_0.gguf` against `llama-fit-params`.
- 2026-04-23: Added a maintainable vendor patch to llama/common fit APIs so predictor requests and the native harness can override detected host/device memory for deterministic hardware targeting.
- 2026-04-23: Fixed browser in-process wasm fit execution path by suppressing threaded common-fit logger usage under Emscripten and verified successful browser-side fit execution against the local Gemma fixture.
- 2026-04-23: Started implementation item 13 by adding API tests for metadata-only GGUF fixtures, split-shard HF request planning, and heterogeneous multi-GPU planning scenarios.
- 2026-04-23: Verified browser wasm metadata requests across three additional vendored GGUF fixtures and adjusted fetch byte-cap assumptions to avoid false `insufficient_prefix_bytes` failures.
- 2026-04-23: Updated vendor fit execution integration tests to support multi-model matrix runs via `VRAM_LLAMA_FIT_MODELS` while keeping `VRAM_LLAMA_FIT_MODEL` as a fallback.
- 2026-04-23: Added requested full-model fixtures from QuantFactory (Qwen2.5-0.5B-Instruct Q4_0) and unsloth (gemma-3-1b-it Q2_K), then validated native fit execution + parity and browser wasm in-process fit across all three full-model fixtures.
- 2026-04-23: Completed item 14 API-surface cleanup by simplifying predictor response shapes, updating schema/tests, and validating native+vendor+wasm behavior on full-model fixtures.
- 2026-04-23: Consolidated duplicate HF prefix range-planning modules and removed the redundant standalone implementation.
- 2026-04-23: Began replacing fit-mode planning-only API responses with in-process execution wiring so the exported predictor API can directly consume the override-capable llama/common path.
- 2026-04-23: Added a browser-side wasm helper that mounts local model bytes into the Emscripten FS and issues `fit.execute_in_process` requests against the vendor-enabled predictor module.
- 2026-04-23: Added post-fit model/context instantiation inside the in-process API executor so fit responses now include detailed device and host model/context/compute breakdowns in both vendor-native and vendor-wasm builds.
- 2026-04-25: Removed the legacy custom C++ GGUF prefix parser, switched local metadata loading to vendored GGUF/ggml APIs, and made vendored llama.cpp mandatory across native and wasm builds.
- 2026-04-23: Added explicit `fit.execute_in_process` API execution, validated it in a vendor-enabled native test path, and produced a successful vendor-enabled Emscripten build after aligning the predictor target with llama.cpp's wasm64 configuration.
- 2026-04-24: Scaffolded Svelte 5 + Vite UI in `ui/` with FileUpload, ParamPanel, and ResultsTable components wired to the WASM predictor bridge; verified clean production build.
- 2026-04-24: Switched UI wasm asset wiring to a URL-driven model (`VITE_WASM_BASE_URL`) with a dedicated CORS-enabled local assets server so Vite dev can run against in-place build artifacts on a separate port.
- 2026-04-24: Added `sim_backend` scaffolding and CMake wiring so the project can construct simulated profile-based ggml GPU devices and pass them via `llama_model_params.devices` in vendor-enabled builds.
- 2026-04-24: Migrated in-process fit execution to stock `common_fit_params(...)` + simulated backend devices and removed in-process dependency on patched memory-override fit symbols.
- 2026-04-24: Added backend-profile request parsing (`device.gpus[].backend`), updated API docs/AGENTS references, and expanded predictor API tests for valid/invalid backend profile handling.
- 2026-04-24: Fixed wasm fit post-pass breakdown failures by switching breakdown collection to no-allocation model/context setup and adding graceful fallback warnings instead of hard API failure when breakdown collection cannot initialize.
- 2026-04-24: Updated simulated backend memory reporting to subtract live simulated allocations from free bytes, preventing large unsigned underflow artifacts in debug unaccounted-memory output.
- 2026-04-24: Moved browser fit execution onto a dedicated worker path (`predictor_fit_worker`) so long fit runs no longer execute on the main UI thread.
- 2026-04-24: Fixed fit margin derivation for `target_free_mib` by basing margins on desired free memory plus base headroom, preventing large simulated GPUs from being effectively treated as "must stay almost entirely free".
- 2026-04-24: Updated `n_gpu_layers` input parsing in the Svelte parameter panel to reliably preserve `-1` and clamp invalid values.
- 2026-04-24: Added Hugging Face repo search + GGUF file selection UI, integrated metadata validation/submission flow through the WASM worker predictor API, and surfaced metadata responses in the results panel.
- 2026-04-24: Addressed UI tweaks and bug fixes: deduplicated fit-target params into per-GPU bufferMiB, trimmed KV cache type options to f16/q8_0/q4_0, fixed split GGUF filename mounting, auto-assigned device index from position, collapsed metadata preview behind toggle button, restructured layout into Model+Runtime row and Hardware Config row (split ParamPanel into RuntimePanel + HardwarePanel).
- 2026-04-25: Moved the browser helper into `ui/src/lib`, removed the old `web/` smoke harness, and switched the browser Hugging Face metadata parse flow to `@huggingface/gguf` while fetching only the exact stub-prefix bytes needed for wasm fit mounting.
- 2026-04-25: Completed item 18 fit UX/runtime slice: added split-mode runtime controls (`layer`/`row`/`tensor`) wired through predictor API + fit executor, added worker-side fit progress parsing (attempt/n_ctx/n_gpu_layers) from info-level fit logs, and added cancellable runs via worker cancellation + UI cancel button.
- 2026-04-25: Completed item 18 fit UX/runtime slice: added split-mode runtime controls (`layer`/`row`/`tensor`) wired through predictor API + fit executor, added worker-side fit progress parsing (attempt/n_ctx/n_gpu_layers) from info-level fit logs, and added cancellable runs via worker cancellation + UI cancel button.
- 2026-04-25: Completed remaining item 18 polish: filtered noisy stderr logs from UI log view (only stdout shown); added early fit-phase progress patterns (initial layout, layer fill, MoE, extra-layer); wired min ctx checkbox to backend min_ctx with tooltip; added live step-by-step status during HF tensor fetch; removed parroted-back targets/overrides fields from fit response and fit_execution_result struct.
