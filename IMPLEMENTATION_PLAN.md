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
- Added local vendor patch surface for deterministic memory simulation via `common_fit_params_with_memory_override(...)`.
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
  - [ ] Add 1-2 additional full-model fixtures to complete this item.
7. [x] Build HF URL resolution + progressive byte-range request planning helper and wire it to `model.source = huggingface` request handling.
8. [x] Execute remote HF range fetch loop and feed downloaded prefixes into parser.
9. [x] Begin wiring in-process predictor API execution for llama-fit override mode so the exported API can return actual fitted results instead of planning-only output.
  - [x] Add a reusable internal fit execution helper linked from predictor core when vendor llama/common is enabled.
  - [x] Execute fit requests in-process from `vram_predictor_predict_json` with explicit device/host memory overrides.
  - [x] Validate the execution path in a vendor-enabled native build and carry the same surface into a vendor-enabled wasm build.

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
- 2026-04-23: Began replacing fit-mode planning-only API responses with in-process execution wiring so the exported predictor API can directly consume the override-capable llama/common path.
- 2026-04-23: Added explicit `fit.execute_in_process` API execution, validated it in a vendor-enabled native test path, and produced a successful vendor-enabled Emscripten build after aligning the predictor target with llama.cpp's wasm64 configuration.
