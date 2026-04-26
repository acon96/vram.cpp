/* eslint-disable no-restricted-globals */

let modulePromise = null;
let moduleInstance = null;
let moduleConfig = null;
let debugEnabled = false;
let activeJobId = null;
const lineBuffers = new Map();
const progressStateByJob = new Map();

function getBufferKey(jobId, isStdout) {
    return `${jobId}:${isStdout ? 'out' : 'err'}`;
}

function resetJobTracking(jobId) {
    progressStateByJob.delete(jobId);
    lineBuffers.delete(getBufferKey(jobId, true));
    lineBuffers.delete(getBufferKey(jobId, false));
}

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

function parseLogicalSizeBytes(rawValue) {
    const parsed = Number(rawValue);
    if (!Number.isFinite(parsed)) {
        return 0;
    }

    const truncated = Math.trunc(parsed);
    if (truncated <= 0) {
        return 0;
    }

    return truncated;
}

function installSparseLogicalSize(module, virtualPath, logicalSizeBytes, downloadedBytes) {
    if (!module?.FS || typeof module.FS.lookupPath !== 'function') {
        return false;
    }

    if (!Number.isFinite(logicalSizeBytes) || !Number.isFinite(downloadedBytes)) {
        return false;
    }

    const logicalSize = Math.trunc(logicalSizeBytes);
    const prefixSize = Math.trunc(downloadedBytes);

    if (logicalSize <= prefixSize || prefixSize <= 0) {
        return false;
    }

    const lookup = module.FS.lookupPath(virtualPath, { follow: true });
    const node = lookup?.node;
    if (!node || !node.stream_ops || typeof node.stream_ops.read !== 'function') {
        return false;
    }

    const baseRead = node.stream_ops.read;
    node.stream_ops.read = (stream, buffer, offset, length, position) => {
        const currentNode = stream.node;
        if (position >= currentNode.usedBytes) {
            return 0;
        }

        const size = Math.min(currentNode.usedBytes - position, length);
        const availablePrefix = position < prefixSize
            ? Math.min(size, prefixSize - position)
            : 0;

        if (availablePrefix > 0) {
            const source = currentNode.contents.subarray(position, position + availablePrefix);
            buffer.set(source, offset);
        }

        if (availablePrefix < size) {
            buffer.fill(0, offset + availablePrefix, offset + size);
        }

        return size;
    };

    node.usedBytes = logicalSize;
    node.mtime = node.ctime = Date.now();

    debugLog('installSparseLogicalSize.applied', {
        virtualPath,
        logicalSize,
        prefixSize,
    });

    return true;
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
        splitMode = 'layer',
        minCtx = 0,
        showFitLogs = false,
    } = options;

    const runtime = {
        n_ctx: nCtx,
        min_ctx: minCtx,
        n_batch: nBatch,
        n_ubatch: nUbatch,
        n_gpu_layers: nGpuLayers,
        split_mode: splitMode,
    };

    return {
        model: modelPath,
        runtime,
        device: {
            host_ram_bytes: hostRamBytes,
            fit_target_mib: fitTargetMiB,
            target_free_mib: targetFreeMiB,
            gpus,
        },
        show_fit_logs: showFitLogs,
    };
}

function postFitProgress(jobId, patch) {
    if (jobId == null || patch == null || typeof patch !== 'object') {
        return;
    }

    const prev = progressStateByJob.get(jobId) || {
        attempt: 0,
        nCtx: null,
        nGpuLayers: null,
        nglByDevice: {},
        lastLine: '',
    };

    const next = {
        ...prev,
        ...patch,
        nglByDevice: {
            ...(prev.nglByDevice || {}),
            ...(patch.nglByDevice || {}),
        },
    };

    const nglValues = Object.values(next.nglByDevice || {})
        .map((v) => Number(v))
        .filter((v) => Number.isFinite(v) && v >= 0);
    if (nglValues.length > 0) {
        next.nGpuLayers = nglValues.reduce((a, b) => a + b, 0);
    }

    progressStateByJob.set(jobId, next);
    self.postMessage({
        type: 'progress',
        jobId,
        progress: {
            attempt: next.attempt,
            nCtx: next.nCtx,
            nGpuLayers: next.nGpuLayers,
            lastLine: next.lastLine,
        },
    });
}

function parseFitProgressLine(jobId, rawLine, isStdout) {
    const line = String(rawLine || '').trim();
    if (!line) return;

    const patch = /** @type {any} */ ({});
    if (isStdout) {
        patch.lastLine = line;
    }

    if (line.includes('memory for test allocation by device')) {
        const prevAttempt = progressStateByJob.get(jobId)?.attempt || 0;
        patch.attempt = prevAttempt + 1;
    }

    if (line.includes('getting device memory data for initial parameters')) {
        patch.lastLine = 'Analyzing initial layout…';
    }

    if (line.includes('cannot meet free memory target')) {
        patch.lastLine = 'Adjusting layer distribution…';
    }

    if (line.includes('filling dense layers') || line.includes('filling dense-only layers')) {
        patch.lastLine = 'Filling layer layout…';
    }

    if (line.includes('converting dense-only layers')) {
        patch.lastLine = 'Finalizing layer layout…';
    }

    if (line.includes('trying to fit one extra layer')) {
        patch.lastLine = 'Trying to fit one extra layer…';
    }

    const ctxMatch = /context size reduced from\s+\d+\s+to\s+(\d+)/i.exec(line);
    if (ctxMatch) {
        patch.nCtx = Number.parseInt(ctxMatch[1], 10);
    }

    const nglSetMatch = /set ngl_per_device\[(\d+)\]\.n_layer=(\d+)/i.exec(line);
    if (nglSetMatch) {
        const idx = Number.parseInt(nglSetMatch[1], 10);
        const nLayer = Number.parseInt(nglSetMatch[2], 10);
        patch.nglByDevice = { [idx]: nLayer };
    }

    const nglPairMatch = /set ngl_per_device\[(\d+)\]\.\(n_layer,\s*n_part\)=\((\d+),\s*(\d+)\)/i.exec(line);
    if (nglPairMatch) {
        const idx = Number.parseInt(nglPairMatch[1], 10);
        const nLayer  = Number.parseInt(nglPairMatch[2], 10);
        patch.nglByDevice = { [idx]: nLayer };
    }

    postFitProgress(jobId, patch);
}

function onModuleLogChunk(jobId, chunk, isStdout) {
    if (jobId == null) {
        return;
    }

    const chunkText = String(chunk ?? '');
    const bufferKey = getBufferKey(jobId, isStdout);
    const prev = lineBuffers.get(bufferKey) || '';

    // Emscripten print callbacks may deliver one complete log line without a trailing newline.
    // Parse these immediately so UI progress/log output updates live.
    if (prev.length === 0 && chunkText.length > 0 && !/[\r\n]/.test(chunkText)) {
        parseFitProgressLine(jobId, chunkText, isStdout);
        return;
    }

    const merged = prev + chunkText;
    const lines = merged.split(/\r?\n/);
    const tail = lines.pop() || '';

    for (const line of lines) {
        parseFitProgressLine(jobId, line, isStdout);
    }

    lineBuffers.set(bufferKey, tail);
}

function flushModuleLogBuffer(jobId) {
    if (jobId == null) {
        return;
    }

    for (const isStdout of [true, false]) {
        const bufferKey = getBufferKey(jobId, isStdout);
        const tail = lineBuffers.get(bufferKey);
        if (tail && tail.trim().length > 0) {
            parseFitProgressLine(jobId, tail, isStdout);
        }
        lineBuffers.delete(bufferKey);
    }
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
            print: (text) => onModuleLogChunk(activeJobId, text, true),
            printErr: (text) => onModuleLogChunk(activeJobId, text, false),
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

async function handlePredict(message) {
    const module = await ensureModule(message.config);
    const mountRoot = moduleConfig?.mountRoot || '/models';
    ensureDirectory(module.FS, mountRoot);

    const safeName = sanitizePathSegment(message.fileName);
    const virtualPath = `${mountRoot}/${message.jobId}_${safeName}`;
    const fileBytes = new Uint8Array(message.fileBuffer);
    const logicalSizeBytes = parseLogicalSizeBytes(message?.options?.virtualFileSizeBytes);

    module.FS.writeFile(virtualPath, fileBytes);
    if (logicalSizeBytes > fileBytes.byteLength) {
        installSparseLogicalSize(module, virtualPath, logicalSizeBytes, fileBytes.byteLength);
    }

    // Mount stub shard files if the primary file is a sharded GGUF.
    // llama.cpp opens sibling shards by name (e.g. model-00002-of-00007.gguf)
    // but only reads their GGUF headers (no_alloc=true), so the stub bytes are
    // sufficient — no tensor data is needed.
    const shardPaths = [];
    const rawShards = Array.isArray(message?.options?.shardFiles) ? message.options.shardFiles : [];
    for (const shard of rawShards) {
        if (!shard?.bytes || !shard?.shardNo) continue;
        // Replace only the shard index in the primary filename, keeping the
        // total shard count from the original name (e.g. 00001 -> 00002).
        const shardName = safeName.replace(
            /(-)\d{5}(-of-\d{5}\.gguf)$/i,
            `$1${String(shard.shardNo).padStart(5, '0')}$2`,
        );
        const shardPath = `${mountRoot}/${message.jobId}_${shardName}`;
        const shardBytes = new Uint8Array(shard.bytes);
        module.FS.writeFile(shardPath, shardBytes);
        // Apply the sparse-read trick so that reads past our stub prefix return
        // zeros instead of EOF errors.  With no_alloc=true only the GGUF header
        // + tensor descriptors are read, never the tensor data section.
        const shardLogicalSize = parseLogicalSizeBytes(shard.logicalSizeBytes);
        if (shardLogicalSize > shardBytes.byteLength) {
            installSparseLogicalSize(module, shardPath, shardLogicalSize, shardBytes.byteLength);
        }
        shardPaths.push(shardPath);
        debugLog('handlePredict.mountedShard', { shardPath, bytes: shardBytes.byteLength, shardLogicalSize });
    }

    try {
        const request = buildPredictRequest(message.options, virtualPath);
        activeJobId = message.jobId;
        resetJobTracking(message.jobId);

        postFitProgress(message.jobId, {
            attempt: 0,
            nCtx: Number.isFinite(Number(request?.runtime?.n_ctx)) ? Number(request.runtime.n_ctx) : null,
            nGpuLayers: Number.isFinite(Number(request?.runtime?.n_gpu_layers)) ? Number(request.runtime.n_gpu_layers) : null,
            lastLine: 'fit started',
        });

        let response = callPredict(module, request);

        if (isThreadCtorError(response) && request.show_fit_logs) {
            debugError('handlePredict.retryWithoutFitLogs', { firstResponse: response });
            request.show_fit_logs = false;
            response = callPredict(module, request);
        }

        return response;
    } finally {
        flushModuleLogBuffer(message.jobId);
        resetJobTracking(message.jobId);
        if (activeJobId === message.jobId) {
            activeJobId = null;
        }
        for (const sp of shardPaths) {
            try { module.FS.unlink(sp); } catch (_) { /* best-effort */ }
        }
        try {
            module.FS.unlink(virtualPath);
        } catch (error) {
            debugError('handlePredict.unlinkError', { virtualPath, error });
        }
    }
}

async function handlePredictJson(message) {
    const module = await ensureModule(message.config);
    const mountRoot = moduleConfig?.mountRoot || '/models';
    ensureDirectory(module.FS, mountRoot);

    const request = typeof structuredClone === 'function'
        ? structuredClone(message.request || {})
        : JSON.parse(JSON.stringify(message.request || {}));
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

    const jobId = message.jobId;

    try {
        if (message.type === 'predict') {
            const result = await handlePredict(message);
            self.postMessage({
                type: 'result',
                jobId,
                result,
            });
        } else if (message.type === 'predict-json') {
             const result = await handlePredictJson(message);
             self.postMessage({
                 type: 'result',
                 jobId,
                 result,
             });
        } else {
            throw new Error(`Unknown message type: ${message.type}`);
        }
    } catch (error) {
        self.postMessage({
            type: 'error',
            jobId,
            error: error?.message || String(error),
        });
    }
};
