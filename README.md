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
- HF range execution backend (browser fetch in wasm, curl fallback in native)
- Local predictor API path that progressively parses GGUF prefixes from disk
- Native in-process llama-fit harness (`vram_fit_harness`) linked against llama/common libraries
- Vendored llama/common fit patch for explicit device/host memory overrides during fitting
- Predictor API fit execution path behind explicit `fit.execute_in_process` opt-in when built with vendor llama support
- In-process fit responses now include detailed device/host model, context, and compute breakdowns
- Browser-side wasm helper for mounting local files and calling `fit.execute_in_process`
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

Vendor-enabled fit build:

```bash
emcmake cmake -S . -B build-wasm-vendor -DVRAM_ENABLE_VENDOR_LLAMA=ON -DVRAM_BUILD_TESTS=OFF
cmake --build build-wasm-vendor
```

Detailed setup and troubleshooting: `docs/EMSCRIPTEN_SETUP.md`.

Expected artifacts:

- `vram_predictor_wasm.js`
- `vram_predictor_wasm.wasm`

## Next implementation steps

1. Expand native llama-fit parity golden tests to 2-3 fixtures (Gemma Q8_0 baseline is implemented)
2. Return detailed model/context/compute breakdowns from the in-process fit API path
3. Put together a simple html UI page that exercises the mounted-file wasm flow end to end

## Native Fit Harness

Build harness and llama-common stack:

```bash
cmake -S . -B build-fit-harness -DVRAM_ENABLE_VENDOR_LLAMA=ON -DVRAM_BUILD_WASM=OFF -DVRAM_BUILD_TESTS=OFF
cmake --build build-fit-harness --target vram_fit_harness -j 8
```

Example run:

```bash
build-fit-harness/vram_fit_harness \
	--model .fixtures/gemma-3-270m-Q8_0.gguf \
	--fit-target-mib 512 \
	--fit-ctx 1024 \
	-c 4096
```

Deterministic hardware override example:

```bash
build-fit-harness/vram_fit_harness \
	--model .fixtures/gemma-3-270m-Q8_0.gguf \
	--fit-target-mib 512 \
	--target-free-mib 2048 \
	--override-device-free-mib 4096 \
	--override-device-total-mib 8192 \
	--override-host-free-mib 32768 \
	--override-host-total-mib 32768 \
	--fit-ctx 1024 \
	-c 4096
```

The native harness currently uses a vendor patch in `vendor/llama-cpp/common/fit.h` and `vendor/llama-cpp/common/fit.cpp` via `common_fit_params_with_memory_override(...)`, with the corresponding rebaseable patch stored at `patches/llama-fit-memory-override.patch`.

The in-process predictor API now uses stock `common_fit_params(...)` with an explicit simulated backend device list (`llama_model_params.devices`) so fit execution no longer depends on the memory-override patch surface.

Predictor API fit execution example:

```json
{
	"mode": "fit",
	"model": {"source": "local", "path": "/models/gemma-3-270m-Q8_0.gguf"},
	"runtime": {"n_ctx": 4096, "n_gpu_layers": -1, "cache_type_k": "f16", "cache_type_v": "f16"},
	"device": {
		"host_ram_bytes": 34359738368,
		"fit_target_mib": [512],
		"target_free_mib": [2048],
		"gpus": [{"id": "gpu0", "backend": "cuda", "free_bytes": 4294967296, "total_bytes": 8589934592}]
	},
	"fit": {"min_ctx": 1024, "execute_in_process": true}
}
```

In-process fit responses include a `fit.memoryBreakdown` object with per-device and host `modelMiB`, `contextMiB`, and `computeMiB` values, plus a top-level `memory` summary for byte-oriented consumers.

Browser-side mounted-file helper:

```js
import createVRAMPredictorModule from '../build-wasm-vendor/vram_predictor_wasm.js';
import { createBrowserPredictorClient } from '../web/vram_predictor_browser.js';

const client = await createBrowserPredictorClient({
	moduleFactory: createVRAMPredictorModule,
	moduleOptions: {
		locateFile: (path) => `../build-wasm-vendor/${path}`,
	},
});

const modelPath = await client.mountBrowserFile(fileInput.files[0]);
const result = client.predictMountedFit({
	modelPath,
	hostRamBytes: 32 * 1024 * 1024 * 1024,
	fitTargetMiB: [512],
	targetFreeMiB: [2048],
	gpus: [{ id: 'gpu0', free_bytes: 4 * 1024 * 1024 * 1024, total_bytes: 8 * 1024 * 1024 * 1024 }],
	nCtx: 4096,
	minCtx: 1024,
});
```

The helper lives at `web/vram_predictor_browser.js` and assumes a vendor-enabled wasm build so the mounted file can be passed directly into the in-process fit path.
