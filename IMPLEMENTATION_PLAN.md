# Implementation Plan

## Overview
The initial MVP implementation has been completed. As further enhancements, fixes, and tweaks are identified, they will be added to the "Tasks to Pick Up" section below. Each task should be clearly described and prioritized as they are identified.

## Tasks to Pick Up

### "Mainline" implementation tasks:
1. [ ] TODO

### Fixes:
- [ ] TODO

### Tweaks:
- [ ] TODO

### Cleanup:
- [ ] TODO

## Change Log
As you work, make sure to update this change log with a brief description of each significant change or milestone, along with the date and a reference to the relevant commit(s). This will help maintain a clear history of the project's evolution and provide context for future contributors.

- 2026-04-23: Bootstrapped repository skeleton with CMake, include/src layout, API schemas, and WASM-exported C ABI stubs.
- 2026-04-23: Added `.gitignore` and `README.md` for baseline housekeeping and reproducible build entrypoints.
- 2026-04-23: Updated immediate next steps with completion markers to support phased implementation tracking.
- 2026-04-23: Added llama.cpp as a git submodule at `vendor/llama-cpp` with optional CMake integration.
- 2026-04-23: Switched API JSON serialization to nlohmann JSON from the vendored llama.cpp dependency.
- 2026-04-23: Implemented GGUF prefix metadata parser and HF prefix range planning helper with parser unit tests.
- 2026-04-23: Added golden regression tests using 3 vendored GGUF fixtures and split native fit parity testing into an explicit follow-up step requiring full model fixtures.
- 2026-04-23: Integrated local GGUF prefix parsing into predictor API requests and added API-level integration tests.
- 2026-04-23: Added HF file URL resolver and progressive range request planner with helper tests, plus API support for Hugging Face request planning output.
- 2026-04-23: Validated first Emscripten build (`vram_predictor.js/.wasm`) and added a dedicated setup guide for emsdk activation and wasm build commands.
- 2026-04-23: Implemented platform-aware HF range execution backend (Emscripten fetch in browser/WASM, curl fallback in native) and fed fetched prefixes directly into the GGUF parser.
- 2026-04-23: Replaced executable-driven fit orchestration in predictor API with harness planning semantics and added a native in-process fit harness (`vram_predictor`) linked to llama/common code.
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
- 2026-04-26: Reduced wasm llama.cpp surface by forcing a minimal Emscripten option set in root CMake (disabled OpenSSL/web UI/tooling extras, OpenMP/llamafile/repack paths, and explicitly pinned non-CPU ggml backends off).
- 2026-04-26: Unified native harness entrypoint by wiring `vram_predictor` to `cpp/src/predictor_main.cpp`, removing the separate harness-only CLI source, and enabling direct JSON request input (argv/stdin) for local fit testing.
- 2026-04-26: Added GitHub Pages automation: unified wasm+UI static build script (`scripts/build_pages.sh`), push-to-main Pages deploy workflow using `actions/deploy-pages`, and nightly llama.cpp submodule sync workflow that auto-commits upstream updates.

## Document Milestones
This document (the task list in particular) is regularly pruned and updated as the project evolves. Sometimes you need to see the history to understand why certain decisions were made or how the plan evolved, but the current state of the document should always reflect the latest thinking and direction for the project.  The following git commits represent major milestones in the evolution of this implementation plan and can be used to dig further into the history of the project:
- 2026-04-26 - `fcd69b1133945f0f71744208d335c0276fd66005`: Initial Release complete with full implementation and GitHub Pages deployment. 
