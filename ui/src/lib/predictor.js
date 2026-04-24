/**
 * Browser WASM predictor bridge.
 *
 * Dynamically loads the vram_predictor_wasm.js script (which sets
 * globalThis.createVRAMPredictorModule) and returns a client wrapping
 * the mounted-file fit path from vram_predictor_browser.js.
 */

let clientPromise = null;

async function loadScript(src) {
    return new Promise((resolve, reject) => {
        const existing = document.querySelector(`script[src="${src}"]`);
        if (existing) {
            resolve();
            return;
        }
        const s = document.createElement('script');
        s.src = src;
        s.onload = () => resolve();
        s.onerror = () => reject(new Error(`Failed to load WASM script: ${src}`));
        document.head.appendChild(s);
    });
}

/**
 * Initialize the predictor.
 *
 * @param {object} opts
 * @param {string} opts.wasmJsUrl   - URL to vram_predictor_wasm.js
 * @param {string} opts.browserHelperUrl - URL to vram_predictor_browser.js
 * @returns {Promise<object>} client created by createBrowserPredictorClient
 */
export async function initPredictor({ wasmJsUrl, browserHelperUrl }) {
    if (clientPromise) return clientPromise;

    clientPromise = (async () => {
        await loadScript(wasmJsUrl);

        const factory = globalThis.createVRAMPredictorModule;
        if (typeof factory !== 'function') {
            throw new Error(
                'createVRAMPredictorModule not found on globalThis after loading WASM script. ' +
                'Ensure the vendor-enabled WASM build is used.'
            );
        }

        const { createBrowserPredictorClient } = await import(/* @vite-ignore */ browserHelperUrl);

        const client = await createBrowserPredictorClient({
            moduleFactory: factory,
            moduleOptions: {
                locateFile: (path) => {
                    const base = wasmJsUrl.substring(0, wasmJsUrl.lastIndexOf('/') + 1);
                    return base + path;
                },
            },
        });

        return client;
    })();

    return clientPromise;
}

/**
 * Reset the cached client promise (useful when changing WASM paths).
 */
export function resetPredictor() {
    clientPromise = null;
}
