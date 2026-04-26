# vram.cpp UI (Svelte + Vite)

This UI loads the predictor wasm artifacts from an external assets origin set by `VITE_WASM_BASE_URL`.

## Local Development (Two Servers)

1. Build vendor-enabled wasm artifacts from repo root:

```bash
source ~/emsdk/emsdk_env.sh
cmake -S . -B build-wasm -DVRAM_ENABLE_VENDOR_LLAMA=ON -DVRAM_BUILD_TESTS=OFF
cmake --build build-wasm --target vram_predictor -j4
```

2. In `ui/`, the Vite dev server and start the assets server (defaults to `http://127.0.0.1:8123/assets/`) simultaneously:

```bash
npm run dev
```

> Alternatively you can run these in separate terminals if you prefer with `npm run assets:serve` and `npm run web`.

The checked-in [ui/.env.development](.env.development) points Vite dev to `http://127.0.0.1:8123/assets/`.

## Environment Variables

- `VITE_WASM_BASE_URL`: Full URL or relative path containing:
	- `vram_predictor.js`
	- `vram_predictor.wasm`
	- `vram_predictor_browser.js` (served from `ui/src/lib/vram_predictor_browser.js` by the local asset server)

- `VITE_DEBUG_WASM`: Set to `1` to force verbose frontend/wasm debug logs in the browser console.
  - In dev mode, debug logs are enabled by default.
  - At runtime, you can override with `window.__VRAM_DEBUG__ = true` or `window.__VRAM_DEBUG__ = false`.
  - The helper also supports local storage toggles: `localStorage.setItem('vram.debug', '1')` or `localStorage.setItem('vram.debug', '0')`.

- `VITE_DEBUG_WASM_FIT_LOGS`: Set to `1` to forward `show_fit_logs=true` to the backend fit call.
	- Default is off, even in dev mode.
	- In browser wasm builds this can trigger `thread constructor failed: Not supported` in some llama/common logging paths.

- `VITE_BASE_URL`: Vite base path for deployed app routing/asset resolution.

Assets server options:

- `WASM_ASSETS_HOST` (default `127.0.0.1`)
- `WASM_ASSETS_PORT` (default `8123`)
- `WASM_ASSETS_PATH` (default `/assets`)
- `WASM_BUILD_DIR` (default `build-wasm`)

## GitHub Pages / Production

For production builds, set `VITE_WASM_BASE_URL` to the deployed wasm asset URL (full URL or site-relative path), then run:

```bash
npm run build
```

This keeps local multi-port development and final static hosting on the same configuration surface.
