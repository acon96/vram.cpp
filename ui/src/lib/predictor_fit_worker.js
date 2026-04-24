/* eslint-disable no-restricted-globals */

let modulePromise = null;
let moduleInstance = null;
let moduleConfig = null;
let debugEnabled = false;

function debugLog(event, payload) {
    if (!debugEnabled) {
        return;
    }
    console.log(`[vram-fit-worker] ${event}`, payload);
}

function debugError(event, payload) {
    if (!debugEnabled) {
        return;
    }
    console.error(`[vram-fit-worker] ${event}`, payload);
}

function sanitizePathSegment(name) {
    return String(name || 'model.gguf').replace(/[^A-Za-z0-9._-]/g, '_');
}

function ensureDirectory(fs, dirPath) {
    const segments = dirPath.split('/').filter(Boolean);
    let current = '';
    for (const segment of segments) {
        current += '/' + segment;
        try {
            fs.mkdir(current);
        } catch (error) {
            if (error == null || error.code !== 'EEXIST') {
                try {
                    const stat = fs.stat(current);
                    if (!fs.isDir(stat.mode)) {
                        throw error;
                    }
                } catch (_) {
                    throw error;
                }
            }
        }
    }
}

function isThreadCtorError(response) {
    return response?.ok === false
        && typeof response?.error === 'string'
        && response.error.includes('thread constructor failed: Not supported');
}

function buildPredictRequest(options, modelPath) {
    const {
        hostRamBytes,
        fitTargetMiB = [1024],
        targetFreeMiB = [],
        gpus = [],
        nCtx = 4096,
        nBatch,
        nUbatch,
        nGpuLayers = -1,
        minCtx = 0,
        cacheTypeK = 'f16',
        cacheTypeV = 'f16',
        showFitLogs = false,
    } = options;

    const runtime = {
        n_ctx: nCtx,
        n_gpu_layers: nGpuLayers,
        cache_type_k: cacheTypeK,
        cache_type_v: cacheTypeV,
    };

    const parsedNBatch = Number(nBatch);
    const parsedNUbatch = Number(nUbatch);

    if (Number.isFinite(parsedNBatch) && parsedNBatch > 0) {
        runtime.n_batch = Math.trunc(parsedNBatch);
    }

    if (Number.isFinite(parsedNUbatch) && parsedNUbatch > 0) {
        runtime.n_ubatch = Math.trunc(parsedNUbatch);
    }

    return {
        mode: 'fit',
        model: {
            source: 'local',
            path: modelPath,
        },
        runtime,
        device: {
            host_ram_bytes: hostRamBytes,
            fit_target_mib: fitTargetMiB,
            target_free_mib: targetFreeMiB,
            gpus,
        },
        fit: {
            min_ctx: minCtx,
            execute_in_process: true,
            show_fit_logs: showFitLogs,
        },
    };
}

function callPredict(module, request) {
    const requestJson = JSON.stringify(request);
    const raw = module.ccall(
        'vram_predictor_predict_json',
        'string',
        ['string'],
        [requestJson]
    );

    return JSON.parse(raw);
}

function cloneRequest(request) {
    return JSON.parse(JSON.stringify(request));
}

async function ensureModule(config) {
    const normalizedConfig = {
        wasmJsUrl: new URL(config.wasmJsUrl, self.location.href).toString(),
        mountRoot: config.mountRoot || '/models',
    };

    if (moduleInstance != null && moduleConfig?.wasmJsUrl === normalizedConfig.wasmJsUrl) {
        return moduleInstance;
    }

    if (modulePromise != null) {
        return modulePromise;
    }

    modulePromise = (async () => {
        if (typeof self.createVRAMPredictorModule !== 'function') {
            debugLog('ensureModule.importScripts', { wasmJsUrl: normalizedConfig.wasmJsUrl });
            self.importScripts(normalizedConfig.wasmJsUrl);
        }

        if (typeof self.createVRAMPredictorModule !== 'function') {
            throw new Error('createVRAMPredictorModule is unavailable in worker scope');
        }

        const module = await self.createVRAMPredictorModule({
            locateFile: (path) => new URL(path, normalizedConfig.wasmJsUrl).toString(),
        });

        if (module == null || module.FS == null || typeof module.ccall !== 'function') {
            throw new Error('worker predictor module is missing FS or ccall methods');
        }

        ensureDirectory(module.FS, normalizedConfig.mountRoot);

        moduleInstance = module;
        moduleConfig = normalizedConfig;

        debugLog('ensureModule.ready', {
            wasmJsUrl: normalizedConfig.wasmJsUrl,
            mountRoot: normalizedConfig.mountRoot,
        });

        return module;
    })().catch((error) => {
        modulePromise = null;
        debugError('ensureModule.error', { error });
        throw error;
    });

    return modulePromise;
}

async function handleGetSystemInfo(message) {
    const module = await ensureModule(message.config);
    const raw = module.ccall('vram_predictor_get_system_info_json', 'string', [], []);
    return JSON.parse(raw);
}

async function handlePredict(message) {
    const module = await ensureModule(message.config);
    const mountRoot = moduleConfig?.mountRoot || '/models';
    ensureDirectory(module.FS, mountRoot);

    const safeName = sanitizePathSegment(message.fileName);
    const virtualPath = `${mountRoot}/${message.jobId}_${safeName}`;

    module.FS.writeFile(virtualPath, new Uint8Array(message.fileBuffer));

    try {
        const request = buildPredictRequest(message.options, virtualPath);
        let response = callPredict(module, request);

        if (isThreadCtorError(response) && request.fit.show_fit_logs) {
            debugError('handlePredict.retryWithoutFitLogs', { firstResponse: response });
            request.fit.show_fit_logs = false;
            response = callPredict(module, request);
        }

        return response;
    } finally {
        try {
            module.FS.unlink(virtualPath);
        } catch (error) {
            debugError('handlePredict.unlinkError', {
                virtualPath,
                error,
            });
        }
    }
}

async function handlePredictJson(message) {
    const module = await ensureModule(message.config);
    const mountRoot = moduleConfig?.mountRoot || '/models';
    ensureDirectory(module.FS, mountRoot);

    const request = cloneRequest(message.request || {});
    let virtualPath = null;

    if (message.fileBuffer != null) {
        const safeName = sanitizePathSegment(message.fileName);
        virtualPath = `${mountRoot}/${message.jobId}_${safeName}`;
        module.FS.writeFile(virtualPath, new Uint8Array(message.fileBuffer));

        if (request?.model?.source === 'local') {
            request.model.path = virtualPath;
        }
    }

    try {
        return callPredict(module, request);
    } finally {
        if (virtualPath != null) {
            try {
                module.FS.unlink(virtualPath);
            } catch (error) {
                debugError('handlePredictJson.unlinkError', {
                    virtualPath,
                    error,
                });
            }
        }
    }
}

self.onmessage = async (event) => {
    const message = event.data || {};

    if (message.type === 'set-debug') {
        debugEnabled = message.enabled === true;
        return;
    }

    if (message.type !== 'predict' && message.type !== 'predict-json' && message.type !== 'get-system-info') {
        return;
    }

    const jobId = message.jobId;

    try {
        if (message.type === 'get-system-info') {
            const result = await handleGetSystemInfo(message);
            self.postMessage({
                type: 'result',
                jobId,
                result,
            });
            return;
        }

        const result = message.type === 'predict-json'
            ? await handlePredictJson(message)
            : await handlePredict(message);
        self.postMessage({
            type: 'result',
            jobId,
            result,
        });
    } catch (error) {
        self.postMessage({
            type: 'error',
            jobId,
            error: error?.message || String(error),
        });
    }
};
