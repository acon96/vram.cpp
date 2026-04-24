function sanitizePathSegment(name) {
    return name.replace(/[^A-Za-z0-9._-]/g, '_');
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

    const module = await moduleFactory(moduleOptions);
    if (module == null || module.FS == null || typeof module.ccall !== 'function') {
        throw new Error('predictor wasm module is missing FS or ccall runtime methods');
    }

    ensureDirectory(module.FS, mountRoot);

    return {
        module,

        getSystemInfo() {
            const raw = module.ccall('vram_predictor_get_system_info_json', 'string', [], []);
            return JSON.parse(raw);
        },

        async mountBrowserFile(file, targetName = file?.name ?? 'model.gguf') {
            if (file == null || typeof file.arrayBuffer !== 'function') {
                throw new Error('mountBrowserFile requires a browser File or Blob-like object');
            }

            const safeName = sanitizePathSegment(targetName || 'model.gguf');
            const virtualPath = `${mountRoot}/${safeName}`;
            const bytes = new Uint8Array(await file.arrayBuffer());
            module.FS.writeFile(virtualPath, bytes);
            return virtualPath;
        },

        unmountFile(virtualPath) {
            module.FS.unlink(virtualPath);
        },

        predict(request) {
            const raw = module.ccall(
                'vram_predictor_predict_json',
                'string',
                ['string'],
                [JSON.stringify(request)]
            );
            return JSON.parse(raw);
        },

        predictMountedFit(options) {
            const {
                modelPath,
                hostRamBytes,
                fitTargetMiB = [1024],
                targetFreeMiB = [],
                gpus = [],
                nCtx = 4096,
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

            return this.predict({
                mode: 'fit',
                model: {
                    source: 'local',
                    path: modelPath,
                },
                runtime: {
                    n_ctx: nCtx,
                    n_gpu_layers: nGpuLayers,
                    cache_type_k: cacheTypeK,
                    cache_type_v: cacheTypeV,
                },
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