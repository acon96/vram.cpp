/**
 * Browser WASM predictor bridge.
 *
 * Dynamically loads the vram_predictor_wasm.js script (which sets
 * globalThis.createVRAMPredictorModule) and returns a client wrapping
 * the mounted-file fit path from vram_predictor_browser.js.
 */

let clientPromise = null;

function isDebugEnabled() {
    if (typeof globalThis.__VRAM_DEBUG__ === 'boolean') {
        return globalThis.__VRAM_DEBUG__;
    }
    return false;
}

function debugLog(event, payload) {
    if (!isDebugEnabled()) return;
    console.log(`[vram-ui] ${event}`, payload);
}

function debugError(event, payload) {
    if (!isDebugEnabled()) return;
    console.error(`[vram-ui] ${event}`, payload);
}

function toAbsoluteUrl(urlLike) {
    return new URL(urlLike, window.location.href).toString();
}

async function loadScript(src) {
    const absoluteSrc = toAbsoluteUrl(src);

    debugLog('loadScript.start', { src: absoluteSrc });

    return new Promise((resolve, reject) => {
        const existing = Array.from(document.querySelectorAll('script')).find(
            (script) => script.src === absoluteSrc
        );
        if (existing) {
            debugLog('loadScript.reuseExistingTag', { src: absoluteSrc });
            resolve();
            return;
        }
        const s = document.createElement('script');
        s.src = absoluteSrc;
        s.onload = () => {
            debugLog('loadScript.loaded', { src: absoluteSrc });
            resolve();
        };
        s.onerror = () => {
            debugError('loadScript.error', { src: absoluteSrc });
            reject(new Error(`Failed to load WASM script: ${absoluteSrc}`));
        };
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
    if (clientPromise) {
        debugLog('initPredictor.cacheHit', {
            wasmJsUrl,
            browserHelperUrl,
        });
        return clientPromise;
    }

    clientPromise = (async () => {
        const startedAt = globalThis.performance?.now?.() ?? Date.now();
        const wasmScriptUrl = toAbsoluteUrl(wasmJsUrl);
        const helperModuleUrl = toAbsoluteUrl(browserHelperUrl);

        debugLog('initPredictor.start', {
            wasmScriptUrl,
            helperModuleUrl,
        });

        await loadScript(wasmScriptUrl);

        const factory = globalThis.createVRAMPredictorModule;
        if (typeof factory !== 'function') {
            throw new Error(
                'createVRAMPredictorModule not found on globalThis after loading WASM script. ' +
                'Ensure the vendor-enabled WASM build is used.'
            );
        }

        debugLog('initPredictor.factoryReady', {
            factoryType: typeof factory,
        });

        const { createBrowserPredictorClient } = await import(/* @vite-ignore */ helperModuleUrl);

        debugLog('initPredictor.helperImported', {
            hasCreateBrowserPredictorClient: typeof createBrowserPredictorClient === 'function',
        });

        const client = await createBrowserPredictorClient({
            moduleFactory: factory,
            moduleOptions: {
                locateFile: (path) => new URL(path, wasmScriptUrl).toString(),
            },
        });

        const finishedAt = globalThis.performance?.now?.() ?? Date.now();
        debugLog('initPredictor.clientReady', {
            elapsedMs: Math.round((finishedAt - startedAt) * 100) / 100,
            hasPredictMountedFit: typeof client?.predictMountedFit === 'function',
        });

        return client;
    })().catch((error) => {
        debugError('initPredictor.error', {
            wasmJsUrl,
            browserHelperUrl,
            error,
        });
        clientPromise = null;
        throw error;
    });

    return clientPromise;
}

/**
 * Reset the cached client promise (useful when changing WASM paths).
 */
export function resetPredictor() {
    debugLog('resetPredictor', {});
    clientPromise = null;
}
