function sanitizePathSegment(name) {
    return name.replace(/[^A-Za-z0-9._-]/g, '_');
}

function isDebugEnabled() {
    if (typeof globalThis.__VRAM_DEBUG__ === 'boolean') {
        return globalThis.__VRAM_DEBUG__;
    }

    try {
        const value = globalThis.localStorage?.getItem('vram.debug');
        if (value === '1' || value === 'true') {
            return true;
        }
        if (value === '0' || value === 'false') {
            return false;
        }
    } catch (_) {
    }

    return false;
}

function debugLog(event, payload) {
    if (!isDebugEnabled()) return;
    console.log(`[vram-wasm] ${event}`, payload);
}

function debugError(event, payload) {
    if (!isDebugEnabled()) return;
    console.error(`[vram-wasm] ${event}`, payload);
}

function fitStatusName(statusCode) {
    if (statusCode === 0) return 'fit_success';
    if (statusCode === 1) return 'fit_failure_no_viable_allocation';
    if (statusCode === 2) return 'fit_error_hard_failure';
    if (statusCode == null) return 'fit_status_unset';
    return `fit_status_unknown_${statusCode}`;
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

export async function createBrowserPredictorClient(options = {}) {
    const {
        moduleFactory,
        moduleOptions = {},
        mountRoot = '/models',
    } = options;

    if (typeof moduleFactory !== 'function') {
        throw new Error('moduleFactory is required');
    }

    debugLog('createBrowserPredictorClient.start', {
        mountRoot,
        moduleOptions,
        hasFactory: typeof moduleFactory === 'function',
    });

    const initStartedAt = globalThis.performance?.now?.() ?? Date.now();
    const module = await moduleFactory(moduleOptions);
    const initFinishedAt = globalThis.performance?.now?.() ?? Date.now();

    debugLog('createBrowserPredictorClient.moduleReady', {
        elapsedMs: Math.round((initFinishedAt - initStartedAt) * 100) / 100,
        hasFS: module?.FS != null,
        hasCcall: typeof module?.ccall === 'function',
    });

    if (module == null || module.FS == null || typeof module.ccall !== 'function') {
        throw new Error('predictor wasm module is missing FS or ccall runtime methods');
    }

    ensureDirectory(module.FS, mountRoot);

    debugLog('filesystem.mountRootReady', { mountRoot });

    return {
        module,

        getSystemInfo() {
            const startedAt = globalThis.performance?.now?.() ?? Date.now();
            const raw = module.ccall('vram_predictor_get_system_info_json', 'string', [], []);
            const finishedAt = globalThis.performance?.now?.() ?? Date.now();

            debugLog('getSystemInfo.raw', {
                raw,
                elapsedMs: Math.round((finishedAt - startedAt) * 100) / 100,
            });

            const parsed = JSON.parse(raw);
            debugLog('getSystemInfo.parsed', parsed);
            return parsed;
        },

        async mountBrowserFile(file, targetName = file?.name ?? 'model.gguf') {
            if (file == null || typeof file.arrayBuffer !== 'function') {
                throw new Error('mountBrowserFile requires a browser File or Blob-like object');
            }

            const mountStartedAt = globalThis.performance?.now?.() ?? Date.now();
            const safeName = sanitizePathSegment(targetName || 'model.gguf');
            const virtualPath = `${mountRoot}/${safeName}`;
            const bytes = new Uint8Array(await file.arrayBuffer());
            module.FS.writeFile(virtualPath, bytes);

            const mountFinishedAt = globalThis.performance?.now?.() ?? Date.now();
            debugLog('mountBrowserFile.done', {
                sourceName: file?.name,
                sourceType: file?.type,
                sourceSizeBytes: file?.size,
                virtualPath,
                bytesWritten: bytes.length,
                elapsedMs: Math.round((mountFinishedAt - mountStartedAt) * 100) / 100,
            });

            return virtualPath;
        },

        unmountFile(virtualPath) {
            module.FS.unlink(virtualPath);
            debugLog('unmountFile.done', { virtualPath });
        },

        predict(request) {
            const requestJson = JSON.stringify(request);
            const startedAt = globalThis.performance?.now?.() ?? Date.now();

            debugLog('predict.request', {
                request,
                requestJson,
            });

            const raw = module.ccall(
                'vram_predictor_predict_json',
                'string',
                ['string'],
                [requestJson]
            );

            const finishedAt = globalThis.performance?.now?.() ?? Date.now();
            debugLog('predict.rawResponse', {
                raw,
                elapsedMs: Math.round((finishedAt - startedAt) * 100) / 100,
            });

            try {
                const parsed = JSON.parse(raw);
                debugLog('predict.parsedResponse', parsed);

                const statusCode = parsed?.fit?.status;
                debugLog('predict.status', {
                    ok: parsed?.ok,
                    statusCode,
                    statusText: fitStatusName(statusCode),
                    warnings: parsed?.fit?.warnings,
                });

                globalThis.__VRAM_LAST_REQUEST__ = request;
                globalThis.__VRAM_LAST_RESPONSE_RAW__ = raw;
                globalThis.__VRAM_LAST_RESPONSE__ = parsed;

                return parsed;
            } catch (error) {
                debugError('predict.parseError', {
                    error,
                    raw,
                });
                throw error;
            }
        },

        predictMountedFit(options) {
            const {
                modelPath,
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

            if (typeof modelPath !== 'string' || modelPath.length === 0) {
                throw new Error('predictMountedFit requires modelPath');
            }

            if (!Number.isFinite(hostRamBytes)) {
                throw new Error('predictMountedFit requires hostRamBytes');
            }

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

            return this.predict({
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
            });
        },
    };
}