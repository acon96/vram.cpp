<script>
    import FileUpload from './components/FileUpload.svelte';
    import HardwarePanel from './components/HardwarePanel.svelte';
    import HuggingFaceSearch from './components/HuggingFaceSearch.svelte';
    import JsonHarness from './components/JsonHarness.svelte';
    import ResultsTable from './components/ResultsTable.svelte';
    import RuntimePanel from './components/RuntimePanel.svelte';
    import { initPredictorWorker } from './lib/predictor_worker_client.js';
    import { giBToBytes } from './lib/format.js';

    // ── WASM paths ────────────────────────────────────────────────────────────
    // VITE_WASM_BASE_URL accepts a full URL (for cross-port local assets)
    // or a relative path (for static GitHub Pages deployments).
    const configuredWasmBase = import.meta.env.VITE_WASM_BASE_URL ?? './wasm/';
    const wasmBaseUrl = new URL(configuredWasmBase, window.location.href);
    const wasmJsUrl = new URL('vram_predictor_wasm.js', wasmBaseUrl).toString();
    const wasmDebugEnabled = import.meta.env.VITE_DEBUG_WASM === '1' || import.meta.env.DEV;
    const wasmFitLogsEnabled = import.meta.env.VITE_DEBUG_WASM_FIT_LOGS === '1';
    const appView = new URLSearchParams(window.location.search).get('view') ?? 'app';
    const isJsonHarnessView = appView === 'harness';
    const hfInitialPrefixBytes = 2 * 1024 * 1024;
    const hfMaxPrefixBytes = 16 * 1024 * 1024;
    const hfPrefixChunkBytes = 2 * 1024 * 1024;
    const hardwareConfigStorageKey = 'vram_cpp_hardware_config_v1';
    const ggufTypeUint8 = 0;
    const ggufTypeInt8 = 1;
    const ggufTypeUint16 = 2;
    const ggufTypeInt16 = 3;
    const ggufTypeUint32 = 4;
    const ggufTypeInt32 = 5;
    const ggufTypeFloat32 = 6;
    const ggufTypeBool = 7;
    const ggufTypeString = 8;
    const ggufTypeArray = 9;
    const ggufTypeUint64 = 10;
    const ggufTypeInt64 = 11;
    const ggufTypeFloat64 = 12;
    globalThis.__VRAM_DEBUG__ = wasmDebugEnabled;

    function debugLog(event, payload) {
        if (!wasmDebugEnabled) return;
        console.log(`[vram-ui] ${event}`, payload);
    }

    function debugError(event, payload) {
        if (!wasmDebugEnabled) return;
        console.error(`[vram-ui] ${event}`, payload);
    }

    function describeFitStatus(statusCode) {
        if (statusCode === 0) {
            return 'fit_success';
        }
        if (statusCode === 1) {
            return 'fit_failure_no_viable_allocation';
        }
        if (statusCode === 2) {
            return 'fit_error_hard_failure';
        }
        return `fit_status_unknown_${statusCode}`;
    }

    debugLog('config', {
        wasmDebugEnabled,
        wasmFitLogsEnabled,
        configuredWasmBase,
        wasmJsUrl,
    });

    function normalizeHardwareGpu(input, fallbackIndex) {
        const totalGiB = Number.isFinite(Number(input?.totalGiB))
            ? Math.max(0.5, Number(input.totalGiB))
            : 8;
        const freeGiB = Number.isFinite(Number(input?.freeGiB))
            ? Math.max(0, Math.min(totalGiB, Number(input.freeGiB)))
            : totalGiB;
        const bufferMiB = Number.isFinite(Number(input?.bufferMiB))
            ? Math.max(0, Math.trunc(Number(input.bufferMiB)))
            : 512;

        return {
            name: typeof input?.name === 'string' && input.name.trim().length > 0
                ? input.name
                : `GPU ${fallbackIndex}`,
            totalGiB,
            freeGiB,
            bufferMiB,
        };
    }

    function readStoredHardwareConfig() {
        if (typeof window === 'undefined' || window.localStorage == null) {
            return {};
        }

        try {
            const raw = window.localStorage.getItem(hardwareConfigStorageKey);
            if (!raw) {
                return {};
            }

            const parsed = JSON.parse(raw);
            if (parsed == null || typeof parsed !== 'object') {
                return {};
            }

            const gpus = Array.isArray(parsed.gpus)
                ? parsed.gpus.slice(0, 4).map((gpu, index) => normalizeHardwareGpu(gpu, index))
                : undefined;

            return {
                hostRamGiB: Number.isFinite(Number(parsed.hostRamGiB))
                    ? Math.max(1, Number(parsed.hostRamGiB))
                    : undefined,
                gpus,
            };
        } catch (error) {
            debugError('hardwareConfig.read.failed', { error });
            return {};
        }
    }

    const defaultParams = {
        nCtx: 4096,
        nBatch: 2048,
        nUbatch: 512,
        cacheTypeK: 'f16',
        cacheTypeV: 'f16',
        nGpuLayers: -1,
        hostRamGiB: 32,
        gpus: [],
    };
    const storedHardwareConfig = readStoredHardwareConfig();

    // ── State ─────────────────────────────────────────────────────────────────
    let modelSource = $state('huggingface');
    let selectedFile = $state(null);
    let hfSelection = $state({
        repo: '',
        file: '',
        fileSizeBytes: 0,
        revision: 'main',
        token: '',
        resolvedUrl: '',
        validated: false,
        response: null,
        metadata: null,
        error: '',
    });
    let hfPreparedFit = $state(null);

    let params = $state({
        ...defaultParams,
        hostRamGiB: storedHardwareConfig.hostRamGiB ?? defaultParams.hostRamGiB,
        gpus: Array.isArray(storedHardwareConfig.gpus)
            ? storedHardwareConfig.gpus
            : defaultParams.gpus,
    });

    let status = $state('idle'); // 'idle' | 'loading-wasm' | 'running' | 'done' | 'error'
    let errorMsg = $state('');
    let result = $state(null);

    function buildHfSelectionCacheKey(selection) {
        const repo = String(selection?.repo || '').trim();
        const file = String(selection?.file || '').trim();
        const revision = String(selection?.revision || 'main').trim() || 'main';
        const token = String(selection?.token || '').trim();
        return `${repo}||${file}||${revision}||${token}`;
    }

    function clearPreparedHfFit(reason) {
        if (hfPreparedFit != null) {
            debugLog('hf.prepared.clear', { reason });
        }
        hfPreparedFit = null;
    }

    // ── Handlers ──────────────────────────────────────────────────────────────
    function handleModelSourceChange(nextSource) {
        modelSource = nextSource;
        if (nextSource !== 'huggingface') {
            clearPreparedHfFit('model_source_changed');
        }
        result = null;
        status = 'idle';
        errorMsg = '';
    }

    function handleFile(file) {
        debugLog('handleFile', {
            name: file?.name,
            sizeBytes: file?.size,
            type: file?.type,
        });
        clearPreparedHfFit('local_file_selected');
        modelSource = 'local';
        selectedFile = file;
        result = null;
        status = 'idle';
        errorMsg = '';
    }

    function handleHuggingFaceSelectionChange(selection) {
        const nextCacheKey = buildHfSelectionCacheKey(selection);
        if (selection?.validated !== true || hfPreparedFit?.cacheKey !== nextCacheKey) {
            clearPreparedHfFit('hf_selection_changed');
        }

        hfSelection = {
            repo: selection?.repo ?? '',
            file: selection?.file ?? '',
            fileSizeBytes: Number.isFinite(Number(selection?.fileSizeBytes))
                ? Math.max(0, Math.trunc(Number(selection.fileSizeBytes)))
                : 0,
            revision: selection?.revision ?? 'main',
            token: selection?.token ?? '',
            resolvedUrl: selection?.resolvedUrl ?? '',
            validated: selection?.validated === true,
            response: selection?.response ?? null,
            metadata: selection?.metadata ?? null,
            error: selection?.error ?? '',
        };

        result = null;
        status = 'idle';
        errorMsg = '';
    }

    function handleParamsChange(p) {
        debugLog('handleParamsChange', p);
        params = p;
    }

    $effect(() => {
        if (typeof window === 'undefined' || window.localStorage == null) {
            return;
        }

        const hardwareConfig = {
            hostRamGiB: params.hostRamGiB,
            gpus: Array.isArray(params.gpus)
                ? params.gpus.slice(0, 4).map((gpu, index) => normalizeHardwareGpu(gpu, index))
                : [],
        };

        try {
            window.localStorage.setItem(hardwareConfigStorageKey, JSON.stringify(hardwareConfig));
        } catch (error) {
            debugError('hardwareConfig.write.failed', { error });
        }
    });

    function encodePathPreservingSlashes(path) {
        return String(path || '')
            .split('/')
            .map((segment) => encodeURIComponent(segment))
            .join('/');
    }

    function encodeRepoPath(repo) {
        return String(repo || '')
            .split('/')
            .filter((segment) => segment.length > 0)
            .map((segment) => encodeURIComponent(segment))
            .join('/');
    }

    function buildCanonicalHfFileUrl(selection) {
        const encodedRepo = encodeRepoPath(selection.repo || '');
        const encodedFile = encodePathPreservingSlashes(selection.file || '');
        const encodedRevision = encodeURIComponent((selection.revision || 'main').trim() || 'main');

        if (encodedRepo.length === 0 || encodedFile.length === 0) {
            return '';
        }

        return `https://huggingface.co/${encodedRepo}/resolve/${encodedRevision}/${encodedFile}`;
    }

    function buildHfRangePlan(initialBytes, maxBytes, stepBytes) {
        const plan = [];
        if (!Number.isFinite(maxBytes) || maxBytes <= 0) {
            return plan;
        }

        const firstChunk = Number.isFinite(initialBytes) && initialBytes > 0
            ? Math.trunc(initialBytes)
            : Math.min(2 * 1024 * 1024, Math.trunc(maxBytes));
        const chunkStep = Number.isFinite(stepBytes) && stepBytes > 0
            ? Math.trunc(stepBytes)
            : firstChunk;

        if (firstChunk <= 0 || chunkStep <= 0) {
            return plan;
        }

        for (let current = firstChunk; current <= maxBytes; current += chunkStep) {
            plan.push({ start: 0, end: current - 1 });
        }

        const lastEnd = plan.length > 0 ? plan[plan.length - 1].end : -1;
        if (lastEnd + 1 < maxBytes) {
            plan.push({ start: 0, end: Math.trunc(maxBytes) - 1 });
        }

        return plan;
    }

    function buildLocalMetadataRequestForPrefix(prefixBytes) {
        return {
            mode: 'metadata',
            model: {
                source: 'local',
                path: '__MOUNTED_MODEL__',
            },
            runtime: {
                n_ctx: params.nCtx,
                cache_type_k: params.cacheTypeK,
                cache_type_v: params.cacheTypeV,
            },
            device: {
                host_ram_bytes: giBToBytes(params.hostRamGiB),
            },
            fetch: {
                initial_bytes: prefixBytes,
                max_bytes: prefixBytes,
                growth_factor: 2.0,
            },
        };
    }

    function buildPredictFitInput() {
        const gpus = params.gpus.map((g, i) => {
            const parsedName = typeof g.name === 'string' ? g.name.trim() : '';
            const id = parsedName.length > 0 ? parsedName : `gpu${i}`;

            return {
                id,
                name: parsedName,
                index: i,
                backend: 'cuda',
                free_bytes: giBToBytes(g.freeGiB),
                total_bytes: giBToBytes(g.totalGiB),
            };
        });

        // Per-GPU keep-free buffer feeds into both fit_target_mib and target_free_mib.
        const bufferTargets = params.gpus.map((g) => g.bufferMiB ?? 512);
        const fitTargets = bufferTargets.length > 0 ? bufferTargets : [512];
        const freeTargets = bufferTargets.length > 0 ? bufferTargets : [];

        return {
            hostRamBytes: giBToBytes(params.hostRamGiB),
            fitTargetMiB: fitTargets,
            targetFreeMiB: freeTargets,
            gpus,
            nCtx: params.nCtx,
            nBatch: params.nBatch,
            nUbatch: params.nUbatch,
            nGpuLayers: params.nGpuLayers,
            cacheTypeK: params.cacheTypeK,
            cacheTypeV: params.cacheTypeV,
            minCtx: 512,
            // Keep llama/common fit logs opt-in only in wasm. Some logging
            // paths can try to spawn threads that are unavailable.
            showFitLogs: wasmFitLogsEnabled,
        };
    }

    async function fetchHfPrefixBytes(url, start, end, token, includeAuthHeader) {
        const headers = {
            Accept: 'application/octet-stream',
            Range: `bytes=${start}-${end}`,
        };

        const trimmedToken = (token || '').trim();
        if (includeAuthHeader && trimmedToken.length > 0) {
            headers.Authorization = `Bearer ${trimmedToken}`;
        }

        const response = await fetch(url, {
            method: 'GET',
            headers,
            redirect: 'follow',
        });

        const contentRange = response.headers.get('content-range') || '';
        const contentLengthHeader = response.headers.get('content-length') || '';
        const acceptRanges = response.headers.get('accept-ranges') || '';

        if (!response.ok) {
            throw {
                message: `hf_range_fetch_failed_http_${response.status}`,
                httpStatus: response.status,
                finalUrl: response.url || url,
                contentRange,
                contentLength: contentLengthHeader,
            };
        }

        const buffer = await response.arrayBuffer();
        if (buffer.byteLength === 0) {
            throw {
                message: 'empty_response',
                httpStatus: response.status,
                finalUrl: response.url || url,
                contentRange,
                contentLength: contentLengthHeader,
            };
        }

        return {
            bytes: new Uint8Array(buffer),
            httpStatus: response.status,
            redirected: response.redirected === true,
            finalUrl: response.url || url,
            contentRange,
            contentLength: contentLengthHeader,
            acceptRanges,
        };
    }

    function parseTotalBytesFromFetchHeaders(fetchResult) {
        const rangeHeader = String(fetchResult?.contentRange || '').trim();
        if (rangeHeader.length > 0) {
            const slashIndex = rangeHeader.lastIndexOf('/');
            if (slashIndex >= 0 && slashIndex + 1 < rangeHeader.length) {
                const totalPart = rangeHeader.slice(slashIndex + 1).trim();
                if (totalPart !== '*') {
                    const parsed = Number(totalPart);
                    if (Number.isFinite(parsed) && parsed > 0) {
                        return Math.trunc(parsed);
                    }
                }
            }
        }

        const lengthHeader = String(fetchResult?.contentLength || '').trim();
        if (lengthHeader.length > 0) {
            const parsed = Number(lengthHeader);
            if (Number.isFinite(parsed) && parsed > 0) {
                return Math.trunc(parsed);
            }
        }

        return 0;
    }

    function ggufScalarTypeByteSize(valueType) {
        if (valueType === ggufTypeUint8 || valueType === ggufTypeInt8 || valueType === ggufTypeBool) {
            return 1;
        }
        if (valueType === ggufTypeUint16 || valueType === ggufTypeInt16) {
            return 2;
        }
        if (valueType === ggufTypeUint32 || valueType === ggufTypeInt32 || valueType === ggufTypeFloat32) {
            return 4;
        }
        if (valueType === ggufTypeUint64 || valueType === ggufTypeInt64 || valueType === ggufTypeFloat64) {
            return 8;
        }
        return 0;
    }

    function readU32LE(bytes, offset) {
        if (offset + 4 > bytes.length) {
            return null;
        }
        const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        return dv.getUint32(offset, true);
    }

    function readI32LE(bytes, offset) {
        if (offset + 4 > bytes.length) {
            return null;
        }
        const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        return dv.getInt32(offset, true);
    }

    function readU64LEAsNumber(bytes, offset) {
        if (offset + 8 > bytes.length) {
            return null;
        }
        const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const low = dv.getUint32(offset, true);
        const high = dv.getUint32(offset + 4, true);
        const value = high * 4294967296 + low;
        return Number.isSafeInteger(value) ? value : null;
    }

    function readI64LEAsNumber(bytes, offset) {
        if (offset + 8 > bytes.length) {
            return null;
        }
        const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const low = dv.getUint32(offset, true);
        const highSigned = dv.getInt32(offset + 4, true);
        const value = highSigned * 4294967296 + low;
        return Number.isSafeInteger(value) ? value : null;
    }

    function readFloat32LE(bytes, offset) {
        if (offset + 4 > bytes.length) {
            return null;
        }
        const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        return dv.getFloat32(offset, true);
    }

    function readFloat64LE(bytes, offset) {
        if (offset + 8 > bytes.length) {
            return null;
        }
        const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        return dv.getFloat64(offset, true);
    }

    function readGgufString(bytes, offset) {
        const len = readU64LEAsNumber(bytes, offset);
        if (len == null) {
            return null;
        }

        const start = offset + 8;
        const end = start + len;
        if (end > bytes.length) {
            return null;
        }

        const decoder = new TextDecoder('utf-8');
        return {
            value: decoder.decode(bytes.subarray(start, end)),
            nextOffset: end,
        };
    }

    function skipGgufValue(bytes, offset, valueType) {
        if (valueType === ggufTypeString) {
            const str = readGgufString(bytes, offset);
            return str == null ? null : str.nextOffset;
        }

        if (valueType === ggufTypeArray) {
            const elemType = readU32LE(bytes, offset);
            const count = readU64LEAsNumber(bytes, offset + 4);
            if (elemType == null || count == null || elemType === ggufTypeArray) {
                return null;
            }

            let next = offset + 12;
            if (elemType === ggufTypeString) {
                for (let i = 0; i < count; i += 1) {
                    const str = readGgufString(bytes, next);
                    if (str == null) {
                        return null;
                    }
                    next = str.nextOffset;
                }
                return next;
            }

            const elemSize = ggufScalarTypeByteSize(elemType);
            if (elemSize <= 0) {
                return null;
            }
            const byteCount = elemSize * count;
            if (!Number.isSafeInteger(byteCount)) {
                return null;
            }
            const end = next + byteCount;
            return end <= bytes.length ? end : null;
        }

        const scalarSize = ggufScalarTypeByteSize(valueType);
        if (scalarSize <= 0) {
            return null;
        }
        const next = offset + scalarSize;
        return next <= bytes.length ? next : null;
    }

    function readGgufNumericValue(bytes, offset, valueType) {
        if (valueType === ggufTypeUint8) {
            return bytes[offset];
        }
        if (valueType === ggufTypeInt8) {
            return (bytes[offset] << 24) >> 24;
        }
        if (valueType === ggufTypeUint16) {
            const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
            return dv.getUint16(offset, true);
        }
        if (valueType === ggufTypeInt16) {
            const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
            return dv.getInt16(offset, true);
        }
        if (valueType === ggufTypeUint32) {
            return readU32LE(bytes, offset);
        }
        if (valueType === ggufTypeInt32) {
            return readI32LE(bytes, offset);
        }
        if (valueType === ggufTypeFloat32) {
            return readFloat32LE(bytes, offset);
        }
        if (valueType === ggufTypeBool) {
            return bytes[offset] !== 0 ? 1 : 0;
        }
        if (valueType === ggufTypeUint64) {
            return readU64LEAsNumber(bytes, offset);
        }
        if (valueType === ggufTypeInt64) {
            return readI64LEAsNumber(bytes, offset);
        }
        if (valueType === ggufTypeFloat64) {
            return readFloat64LE(bytes, offset);
        }
        return null;
    }

    function extractContextLengthFromPrefix(prefixBytes) {
        const bytes = prefixBytes instanceof Uint8Array
            ? prefixBytes
            : new Uint8Array(prefixBytes || []);

        if (bytes.length < 24) {
            return 0;
        }

        if (bytes[0] !== 0x47 || bytes[1] !== 0x47 || bytes[2] !== 0x55 || bytes[3] !== 0x46) {
            return 0;
        }

        let pos = 4;
        const version = readU32LE(bytes, pos);
        if (version == null || version < 2 || version > 3) {
            return 0;
        }
        pos += 4;

        const tensorCount = readU64LEAsNumber(bytes, pos);
        const kvCount = readU64LEAsNumber(bytes, pos + 8);
        if (tensorCount == null || kvCount == null) {
            return 0;
        }
        pos += 16;

        let architecture = '';
        const candidates = [];

        for (let i = 0; i < kvCount; i += 1) {
            const keyParsed = readGgufString(bytes, pos);
            if (keyParsed == null) {
                return 0;
            }
            const key = keyParsed.value;
            pos = keyParsed.nextOffset;

            const valueType = readU32LE(bytes, pos);
            if (valueType == null) {
                return 0;
            }
            pos += 4;

            if (key === 'general.architecture' && valueType === ggufTypeString) {
                const parsedString = readGgufString(bytes, pos);
                if (parsedString == null) {
                    return 0;
                }
                architecture = parsedString.value;
                pos = parsedString.nextOffset;
                continue;
            }

            const isContextKey = key === 'context_length' || key.endsWith('.context_length');
            if (isContextKey) {
                const numericValue = readGgufNumericValue(bytes, pos, valueType);
                const next = skipGgufValue(bytes, pos, valueType);
                if (next == null) {
                    return 0;
                }
                pos = next;

                if (numericValue != null && Number.isFinite(Number(numericValue))) {
                    const parsedCtx = Math.trunc(Number(numericValue));
                    if (parsedCtx > 0) {
                        candidates.push({ key, value: parsedCtx });
                    }
                }
                continue;
            }

            const next = skipGgufValue(bytes, pos, valueType);
            if (next == null) {
                return 0;
            }
            pos = next;
        }

        if (candidates.length === 0) {
            return 0;
        }

        if (architecture.length > 0) {
            const exactKey = `${architecture}.context_length`;
            const exact = candidates.find((entry) => entry.key === exactKey);
            if (exact != null) {
                return exact.value;
            }
        }

        return candidates[0].value;
    }

    function buildPreparedHfFitResult(selection, requestUrl, requests, attempts, prefixBytes, fetchResult, metadataResponse) {
        const sizeFromHeaders = parseTotalBytesFromFetchHeaders(fetchResult);
        const sizeFromSelection = Number.isFinite(Number(selection?.fileSizeBytes))
            ? Math.max(0, Math.trunc(Number(selection.fileSizeBytes)))
            : 0;
        const logicalFileSizeBytes = sizeFromHeaders > 0
            ? sizeFromHeaders
            : sizeFromSelection;
        const contextLength = extractContextLengthFromPrefix(prefixBytes);

        // Preserve the original HF filename so llama.cpp sees a valid name when
        // the file is mounted in the WASM virtual FS. This is especially important
        // for split-shard models whose filenames must match the shard naming convention.
        const originalFileName = String(selection?.file || '')
            .split('/')
            .filter(Boolean)
            .pop() || 'model.gguf';

        return {
            ...metadataResponse,
            source: 'huggingface',
            resolvedUrl: requestUrl,
            requests,
            attempts,
            prefixBytes,
            logicalFileSizeBytes,
            contextLength,
            originalFileName,
            cacheKey: buildHfSelectionCacheKey({ ...selection, resolvedUrl: requestUrl }),
        };
    }

    async function runFitFromPreparedPrefix(client, preparedFit, predictInput) {
        // Use the original HF filename so split-shard naming is preserved.
        const fileName = preparedFit.originalFileName || 'hf_cached_prefix.gguf';
        const fitFile = new File([preparedFit.prefixBytes], fileName, {
            type: 'application/octet-stream',
        });

        const fitInput = {
            ...predictInput,
            virtualFileSizeBytes: preparedFit.logicalFileSizeBytes > preparedFit.prefixBytes.byteLength
                ? preparedFit.logicalFileSizeBytes
                : undefined,
        };

        debugLog('hf.cachedFit.start', {
            resolvedUrl: preparedFit.resolvedUrl,
            fitFileBytes: preparedFit.prefixBytes.byteLength,
            logicalFileSizeBytes: preparedFit.logicalFileSizeBytes,
            attempts: preparedFit.attempts,
        });

        const fitResponse = await client.predictMountedFit(fitFile, fitInput);
        return {
            ...fitResponse,
            source: 'huggingface',
            resolvedUrl: preparedFit.resolvedUrl,
            requests: preparedFit.requests,
            attempts: preparedFit.attempts,
            hfMetadata: preparedFit.metadata ?? null,
        };
    }

    async function runHfMetadataFromBrowser(client, selection) {
        const resolvedUrl = (selection?.resolvedUrl || '').trim();
        const canonicalUrl = buildCanonicalHfFileUrl(selection || {});
        if (resolvedUrl.length === 0 && canonicalUrl.length === 0) {
            return {
                ok: false,
                source: 'huggingface',
                error: 'invalid_huggingface_model_descriptor',
            };
        }

        const plan = buildHfRangePlan(hfInitialPrefixBytes, hfMaxPrefixBytes, hfPrefixChunkBytes);

        const candidateUrls = [];
        if (resolvedUrl.length > 0) {
            candidateUrls.push(resolvedUrl);
        }
        if (canonicalUrl.length > 0 && canonicalUrl !== resolvedUrl) {
            candidateUrls.push(canonicalUrl);
        }

        debugLog('hf.browserMetadata.start', {
            repo: selection?.repo || '',
            file: selection?.file || '',
            revision: selection?.revision || 'main',
            hasResolvedUrl: resolvedUrl.length > 0,
            candidateUrls,
            initialBytes: hfInitialPrefixBytes,
            maxBytes: hfMaxPrefixBytes,
            chunkStepBytes: hfPrefixChunkBytes,
            plan,
            hasToken: (selection?.token || '').trim().length > 0,
        });

        let lastFailure = {
            ok: false,
            source: 'huggingface',
            error: 'hf_range_fetch_failed',
            detail: 'no_candidate_urls',
        };

        for (let candidateIndex = 0; candidateIndex < candidateUrls.length; candidateIndex += 1) {
            const requestUrl = candidateUrls[candidateIndex];
            const includeAuthHeader = requestUrl.startsWith('https://huggingface.co/');
            const requests = [];
            const attempts = [];
            let lastErrorResponse = null;
            let fetchFailure = null;

            debugLog('hf.browserMetadata.candidate.start', {
                candidateIndex,
                requestUrl,
                includeAuthHeader,
                hasToken: (selection?.token || '').trim().length > 0,
            });

            for (let attemptIndex = 0; attemptIndex < plan.length; attemptIndex += 1) {
                const range = plan[attemptIndex];
                const requestedBytes = range.end - range.start + 1;
                const requestHeaders = [
                    { name: 'Range', value: `bytes=${range.start}-${range.end}` },
                    { name: 'Accept', value: 'application/octet-stream' },
                ];

                if (includeAuthHeader && (selection?.token || '').trim().length > 0) {
                    requestHeaders.push({ name: 'Authorization', value: 'Bearer ***' });
                }

                requests.push({
                    url: requestUrl,
                    start: range.start,
                    end: range.end,
                    headers: requestHeaders,
                });

                debugLog('hf.browserMetadata.attempt.fetch.start', {
                    candidateIndex,
                    attemptIndex,
                    requestUrl,
                    range,
                    requestedBytes,
                    includeAuthHeader,
                });

                let fetchResult;
                try {
                    fetchResult = await fetchHfPrefixBytes(
                        requestUrl,
                        range.start,
                        range.end,
                        selection?.token || '',
                        includeAuthHeader,
                    );
                } catch (error) {
                    debugError('hf.browserMetadata.attempt.fetch.error', {
                        candidateIndex,
                        attemptIndex,
                        requestUrl,
                        range,
                        requestedBytes,
                        error: error?.message ?? String(error),
                        httpStatus: error?.httpStatus,
                        finalUrl: error?.finalUrl,
                        contentRange: error?.contentRange,
                        contentLength: error?.contentLength,
                    });

                    fetchFailure = {
                        ok: false,
                        source: 'huggingface',
                        error: 'hf_range_fetch_failed',
                        detail: error?.message ?? String(error),
                        resolvedUrl: requestUrl,
                        requests,
                        attempts,
                    };
                    break;
                }

                const prefixBytes = fetchResult.bytes;
                debugLog('hf.browserMetadata.attempt.fetch.done', {
                    candidateIndex,
                    attemptIndex,
                    requestUrl,
                    range,
                    requestedBytes,
                    fetchedBytes: prefixBytes.byteLength,
                    httpStatus: fetchResult.httpStatus,
                    redirected: fetchResult.redirected,
                    finalUrl: fetchResult.finalUrl,
                    contentRange: fetchResult.contentRange,
                    contentLength: fetchResult.contentLength,
                    acceptRanges: fetchResult.acceptRanges,
                });

                const localRequest = buildLocalMetadataRequestForPrefix(prefixBytes.byteLength);
                const localFile = new File([prefixBytes], `hf_prefix_${range.end + 1}.gguf`, {
                    type: 'application/octet-stream',
                });

                const response = await client.predictMountedJson(localFile, localRequest);
                attempts.push({
                    attemptIndex,
                    start: range.start,
                    end: range.end,
                    requestedBytes,
                    fetchedBytes: prefixBytes.byteLength,
                    httpStatus: fetchResult.httpStatus,
                    finalUrl: fetchResult.finalUrl,
                    contentRange: fetchResult.contentRange,
                    parserOk: response?.ok === true,
                    parserError: response?.error || '',
                    minimumRequiredBytes: response?.minimumRequiredBytes || 0,
                    bytesConsumed: response?.metadata?.bytesConsumed || 0,
                });

                debugLog('hf.browserMetadata.attempt.parse.result', {
                    candidateIndex,
                    attemptIndex,
                    requestUrl,
                    fetchedBytes: prefixBytes.byteLength,
                    response,
                });

                if (response?.ok === true) {
                    debugLog('hf.browserMetadata.success', {
                        candidateIndex,
                        attemptIndex,
                        requestUrl,
                        fetchedBytes: prefixBytes.byteLength,
                        metadata: {
                            version: response?.metadata?.version,
                            kvCount: response?.metadata?.kvCount,
                            tensorCount: response?.metadata?.tensorCount,
                            bytesConsumed: response?.metadata?.bytesConsumed,
                        },
                    });

                    return buildPreparedHfFitResult(
                        selection,
                        requestUrl,
                        requests,
                        attempts,
                        prefixBytes,
                        fetchResult,
                        response,
                    );
                }

                lastErrorResponse = response;
                if (response?.error !== 'insufficient_prefix_bytes') {
                    debugError('hf.browserMetadata.nonPrefixError', {
                        candidateIndex,
                        attemptIndex,
                        requestUrl,
                        response,
                    });

                    return {
                        ...response,
                        source: 'huggingface',
                        resolvedUrl: requestUrl,
                        requests,
                        attempts,
                    };
                }
            }

            if (fetchFailure != null) {
                lastFailure = fetchFailure;
                continue;
            }

            debugError('hf.browserMetadata.candidate.exhausted', {
                candidateIndex,
                requestUrl,
                attempts,
                lastErrorResponse,
            });

            lastFailure = {
                ok: false,
                source: 'huggingface',
                error: lastErrorResponse?.error || 'insufficient_prefix_bytes',
                detail: '',
                minimumRequiredBytes: lastErrorResponse?.minimumRequiredBytes || 0,
                resolvedUrl: requestUrl,
                requests,
                attempts,
            };
        }

        debugError('hf.browserMetadata.failed', lastFailure);

        return lastFailure;
    }

    async function runHfFitFromBrowser(client, selection, predictInput) {
        const resolvedUrl = (selection?.resolvedUrl || '').trim();
        const canonicalUrl = buildCanonicalHfFileUrl(selection || {});
        if (resolvedUrl.length === 0 && canonicalUrl.length === 0) {
            return {
                ok: false,
                source: 'huggingface',
                error: 'invalid_huggingface_model_descriptor',
            };
        }

        const plan = buildHfRangePlan(hfInitialPrefixBytes, hfMaxPrefixBytes, hfPrefixChunkBytes);

        const candidateUrls = [];
        if (resolvedUrl.length > 0) {
            candidateUrls.push(resolvedUrl);
        }
        if (canonicalUrl.length > 0 && canonicalUrl !== resolvedUrl) {
            candidateUrls.push(canonicalUrl);
        }

        debugLog('hf.browserFit.start', {
            repo: selection?.repo || '',
            file: selection?.file || '',
            revision: selection?.revision || 'main',
            hasResolvedUrl: resolvedUrl.length > 0,
            candidateUrls,
            initialBytes: hfInitialPrefixBytes,
            maxBytes: hfMaxPrefixBytes,
            chunkStepBytes: hfPrefixChunkBytes,
            plan,
            hasToken: (selection?.token || '').trim().length > 0,
            predictInput,
        });

        let lastFailure = {
            ok: false,
            source: 'huggingface',
            error: 'hf_range_fetch_failed',
            detail: 'no_candidate_urls',
        };

        for (let candidateIndex = 0; candidateIndex < candidateUrls.length; candidateIndex += 1) {
            const requestUrl = candidateUrls[candidateIndex];
            const includeAuthHeader = requestUrl.startsWith('https://huggingface.co/');
            const requests = [];
            const attempts = [];
            let lastErrorResponse = null;
            let fetchFailure = null;

            debugLog('hf.browserFit.candidate.start', {
                candidateIndex,
                requestUrl,
                includeAuthHeader,
                hasToken: (selection?.token || '').trim().length > 0,
            });

            for (let attemptIndex = 0; attemptIndex < plan.length; attemptIndex += 1) {
                const range = plan[attemptIndex];
                const requestedBytes = range.end - range.start + 1;
                const requestHeaders = [
                    { name: 'Range', value: `bytes=${range.start}-${range.end}` },
                    { name: 'Accept', value: 'application/octet-stream' },
                ];

                if (includeAuthHeader && (selection?.token || '').trim().length > 0) {
                    requestHeaders.push({ name: 'Authorization', value: 'Bearer ***' });
                }

                requests.push({
                    url: requestUrl,
                    start: range.start,
                    end: range.end,
                    headers: requestHeaders,
                });

                debugLog('hf.browserFit.attempt.fetch.start', {
                    candidateIndex,
                    attemptIndex,
                    requestUrl,
                    range,
                    requestedBytes,
                    includeAuthHeader,
                });

                let fetchResult;
                try {
                    fetchResult = await fetchHfPrefixBytes(
                        requestUrl,
                        range.start,
                        range.end,
                        selection?.token || '',
                        includeAuthHeader,
                    );
                } catch (error) {
                    debugError('hf.browserFit.attempt.fetch.error', {
                        candidateIndex,
                        attemptIndex,
                        requestUrl,
                        range,
                        requestedBytes,
                        error: error?.message ?? String(error),
                        httpStatus: error?.httpStatus,
                        finalUrl: error?.finalUrl,
                        contentRange: error?.contentRange,
                        contentLength: error?.contentLength,
                    });

                    fetchFailure = {
                        ok: false,
                        source: 'huggingface',
                        error: 'hf_range_fetch_failed',
                        detail: error?.message ?? String(error),
                        resolvedUrl: requestUrl,
                        requests,
                        attempts,
                    };
                    break;
                }

                const prefixBytes = fetchResult.bytes;
                debugLog('hf.browserFit.attempt.fetch.done', {
                    candidateIndex,
                    attemptIndex,
                    requestUrl,
                    range,
                    requestedBytes,
                    fetchedBytes: prefixBytes.byteLength,
                    httpStatus: fetchResult.httpStatus,
                    redirected: fetchResult.redirected,
                    finalUrl: fetchResult.finalUrl,
                    contentRange: fetchResult.contentRange,
                    contentLength: fetchResult.contentLength,
                    acceptRanges: fetchResult.acceptRanges,
                });

                const localRequest = buildLocalMetadataRequestForPrefix(prefixBytes.byteLength);
                const metadataFile = new File([prefixBytes], `hf_prefix_${range.end + 1}.gguf`, {
                    type: 'application/octet-stream',
                });

                const metadataResponse = await client.predictMountedJson(metadataFile, localRequest);
                attempts.push({
                    attemptIndex,
                    start: range.start,
                    end: range.end,
                    requestedBytes,
                    fetchedBytes: prefixBytes.byteLength,
                    httpStatus: fetchResult.httpStatus,
                    finalUrl: fetchResult.finalUrl,
                    contentRange: fetchResult.contentRange,
                    parserOk: metadataResponse?.ok === true,
                    parserError: metadataResponse?.error || '',
                    minimumRequiredBytes: metadataResponse?.minimumRequiredBytes || 0,
                    bytesConsumed: metadataResponse?.metadata?.bytesConsumed || 0,
                });

                debugLog('hf.browserFit.attempt.parse.result', {
                    candidateIndex,
                    attemptIndex,
                    requestUrl,
                    fetchedBytes: prefixBytes.byteLength,
                    response: metadataResponse,
                });

                if (metadataResponse?.ok === true) {
                    const preparedFit = buildPreparedHfFitResult(
                        selection,
                        requestUrl,
                        requests,
                        attempts,
                        prefixBytes,
                        fetchResult,
                        metadataResponse,
                    );
                    hfPreparedFit = preparedFit;

                    debugLog('hf.browserFit.attempt.fit.start', {
                        candidateIndex,
                        attemptIndex,
                        requestUrl,
                        fitFileBytes: preparedFit.prefixBytes.byteLength,
                        logicalFileSizeBytes: preparedFit.logicalFileSizeBytes,
                        logicalSizeSource: preparedFit.logicalFileSizeBytes > 0 ? 'headers_or_tree' : 'none',
                    });

                    const fitResponse = await runFitFromPreparedPrefix(client, preparedFit, predictInput);

                    debugLog('hf.browserFit.attempt.fit.result', {
                        candidateIndex,
                        attemptIndex,
                        requestUrl,
                        response: fitResponse,
                        fitStatusText: describeFitStatus(fitResponse?.fit?.status),
                    });

                    return fitResponse;
                }

                lastErrorResponse = metadataResponse;
                if (metadataResponse?.error !== 'insufficient_prefix_bytes') {
                    debugError('hf.browserFit.nonPrefixError', {
                        candidateIndex,
                        attemptIndex,
                        requestUrl,
                        response: metadataResponse,
                    });

                    return {
                        ...metadataResponse,
                        source: 'huggingface',
                        resolvedUrl: requestUrl,
                        requests,
                        attempts,
                    };
                }
            }

            if (fetchFailure != null) {
                lastFailure = fetchFailure;
                continue;
            }

            debugError('hf.browserFit.candidate.exhausted', {
                candidateIndex,
                requestUrl,
                attempts,
                lastErrorResponse,
            });

            lastFailure = {
                ok: false,
                source: 'huggingface',
                error: lastErrorResponse?.error || 'insufficient_prefix_bytes',
                detail: '',
                minimumRequiredBytes: lastErrorResponse?.minimumRequiredBytes || 0,
                resolvedUrl: requestUrl,
                requests,
                attempts,
            };
        }

        debugError('hf.browserFit.failed', lastFailure);
        return lastFailure;
    }

    async function validateHfSelection(selection) {
        const client = await initPredictorWorker({
            wasmJsUrl,
            debugEnabled: wasmDebugEnabled,
        });

        const response = await runHfMetadataFromBrowser(client, selection);
        if (response?.ok === true && response?.prefixBytes instanceof Uint8Array) {
            hfPreparedFit = response;
            if (Number.isFinite(Number(response.contextLength)) && Number(response.contextLength) > 0) {
                const detectedNctx = Math.max(1, Math.trunc(Number(response.contextLength)));
                params = {
                    ...params,
                    nCtx: detectedNctx,
                };
                debugLog('hf.browserMetadata.detectedContextLength', {
                    detectedNctx,
                });
            }

            const { prefixBytes, logicalFileSizeBytes, contextLength, originalFileName, cacheKey, ...responseForUi } = response;
            return responseForUi;
        }

        return response;
    }

    function buildFitFailureDetails(response, predictInput) {
        const fitStatusText = describeFitStatus(response?.fit?.status);
        const fitWarnings = Array.isArray(response?.fit?.warnings) && response.fit.warnings.length > 0
            ? ` warnings=${response.fit.warnings.join(',')}`
            : '';
        const backendError = typeof response?.error === 'string' ? ` backendError=${response.error}` : '';
        const deviceCountInResponse = Array.isArray(response?.fit?.breakdown?.devices)
            ? response.fit.breakdown.devices.length
            : 0;
        const sentGpuOverrides = Array.isArray(predictInput?.gpus)
            ? predictInput.gpus.length
            : 0;

        return {
            message: `WASM fit failed (${fitStatusText}).${fitWarnings}${backendError}`,
            diagnostics: {
                fitStatus: response?.fit?.status,
                fitStatusText,
                warnings: response?.fit?.warnings,
                sentGpuOverrides,
                deviceCountInResponse,
                response,
            },
        };
    }

    async function runPrediction() {
        if (modelSource === 'local') {
            if (!selectedFile) {
                errorMsg = 'Please select a GGUF file first.';
                status = 'error';
                return;
            }
        } else if (!hfSelection.validated) {
            errorMsg = 'Please select a valid Hugging Face model first.';
            status = 'error';
            return;
        }

        status = 'loading-wasm';
        errorMsg = '';
        result = null;

        let client;
        try {
            debugLog('runPrediction.initPredictorWorker.start', {
                wasmJsUrl,
            });

            client = await initPredictorWorker({
                wasmJsUrl,
                debugEnabled: wasmDebugEnabled,
            });

            try {
                const systemInfo = await client.getSystemInfo();
                debugLog('runPrediction.systemInfo', systemInfo);
            } catch (systemInfoError) {
                debugError('runPrediction.systemInfo.error', { systemInfoError });
            }
        } catch (err) {
            status = 'error';
            errorMsg = `Failed to load WASM module: ${err.message}`;
            debugError('runPrediction.initPredictorWorker.error', { err });
            return;
        }

        status = 'running';
        try {
            const predictInput = buildPredictFitInput();

            if (modelSource === 'huggingface') {
                const selectionCacheKey = buildHfSelectionCacheKey(hfSelection);
                const hasPreparedFit = hfPreparedFit?.cacheKey === selectionCacheKey
                    && hfPreparedFit?.prefixBytes instanceof Uint8Array;

                debugLog('runPrediction.hfFitFromBrowser.input', {
                    selection: hfSelection,
                    predictInput,
                    hasPreparedFit,
                });

                const predictStartedAt = performance.now();
                const res = hasPreparedFit
                    ? await runFitFromPreparedPrefix(client, hfPreparedFit, predictInput)
                    : await runHfFitFromBrowser(client, hfSelection, predictInput);
                const predictElapsedMs = Math.round((performance.now() - predictStartedAt) * 100) / 100;

                debugLog('runPrediction.hfFitFromBrowser.output', {
                    predictElapsedMs,
                    response: res,
                    fitStatusText: describeFitStatus(res?.fit?.status),
                });

                result = res;
                if (res?.ok === false) {
                    status = 'error';
                    if (res?.fit != null) {
                        const failure = buildFitFailureDetails(res, predictInput);
                        errorMsg = failure.message;
                        debugError('runPrediction.hfFitFromBrowser.failed', failure.diagnostics);
                    } else {
                        const detail = res?.detail ? ` (${res.detail})` : '';
                        errorMsg = `HF fit request failed: ${res?.error ?? 'unknown_error'}${detail}`;
                    }
                } else {
                    status = 'done';
                }
                return;
            }

            debugLog('runPrediction.predictMountedFit.input', predictInput);

            const predictStartedAt = performance.now();
            const res = await client.predictMountedFit(selectedFile, predictInput);
            const predictElapsedMs = Math.round((performance.now() - predictStartedAt) * 100) / 100;

            debugLog('runPrediction.predictMountedFit.output', {
                predictElapsedMs,
                response: res,
                fitStatusText: describeFitStatus(res?.fit?.status),
            });

            result = res;
            if (res?.ok === false) {
                status = 'error';
                const failure = buildFitFailureDetails(res, predictInput);
                errorMsg = failure.message;
                debugError('runPrediction.predictMountedFit.failed', failure.diagnostics);
            } else {
                status = 'done';
            }
        } catch (err) {
            status = 'error';
            errorMsg = err.message ?? String(err);
            debugError('runPrediction.error', { err });
        }
    }

    const isRunning = $derived(status === 'loading-wasm' || status === 'running');
    const canRunPrediction = $derived(
        modelSource === 'local'
            ? selectedFile != null
            : hfSelection.validated === true
    );
    const runButtonText = $derived('Predict VRAM');
    const resultsTitle = $derived('Memory Breakdown');
    const statusLabel = $derived({
        idle: '',
        'loading-wasm': 'Preparing WASM worker…',
        running: 'Running fit prediction…',
        done: '',
        error: '',
    }[status] ?? '');
</script>

<div class="app-shell">
    <header class="app-header">
        <div class="header-inner">
            <div class="logo-row">
                <span class="logo-text">vram.cpp</span>
            </div>
            <p class="tagline">Predict LLM VRAM usage using llama.cpp fit logic</p>
        </div>
    </header>

    <main class="app-main">
        {#if isJsonHarnessView}
            <JsonHarness wasmJsUrl={wasmJsUrl} debugEnabled={wasmDebugEnabled} />
        {:else}
        <!-- Row 1: Model input + Runtime params side-by-side -->
        <div class="model-runtime-row">
            <div class="model-col">
                <section class="panel-section">
                    <h2 class="panel-title">Model</h2>
                    <div class="source-switch" role="tablist" aria-label="Model source">
                        <button
                            class="source-btn"
                            class:active={modelSource === 'huggingface'}
                            type="button"
                            onclick={() => handleModelSourceChange('huggingface')}
                        >
                            Search on HuggingFace
                        </button>
                        <span class="source-or">OR</span>
                        <button
                            class="source-btn"
                            class:active={modelSource === 'local'}
                            type="button"
                            onclick={() => handleModelSourceChange('local')}
                        >
                            Upload
                        </button>
                    </div>

                    {#if modelSource === 'local'}
                        <FileUpload onfile={handleFile} />
                    {:else}
                        <HuggingFaceSearch
                            onselectionchange={handleHuggingFaceSelectionChange}
                            onvalidate={validateHfSelection}
                        />
                    {/if}
                </section>
            </div>

            <section class="panel-section runtime-col">
                <h2 class="panel-title">Runtime</h2>
                <RuntimePanel {params} onchange={handleParamsChange} />
            </section>
        </div>

        <!-- Row 2: Hardware config spans full width -->
        <section class="panel-section hardware-section">
            <h2 class="panel-title">Hardware Config</h2>
            <HardwarePanel {params} onchange={handleParamsChange} />
        </section>

        <section class="results-section">
            <div class="action-row">
                <button
                    class="run-btn"
                    type="button"
                    onclick={runPrediction}
                    disabled={isRunning || !canRunPrediction}
                >
                    {#if isRunning}
                        <span class="spinner" aria-hidden="true"></span>
                        {statusLabel}
                    {:else}
                        ▶ {runButtonText}
                    {/if}
                </button>

                {#if modelSource === 'huggingface' && !hfSelection.validated}
                    <p class="hint-msg">Select a valid Hugging Face model to enable fitting.</p>
                {:else if modelSource === 'local' && selectedFile == null}
                    <p class="hint-msg">Please upload a GGUF file to enable fitting.</p>
                {/if}

                {#if status === 'error'}
                    <p class="error-msg">{errorMsg}</p>
                {/if}
            </div>
            <section class="results-panel">
                <h2 class="panel-title">
                    {resultsTitle}
                    {#if status === 'done'}
                        <span class="ok-badge">✓</span>
                    {/if}
                </h2>

                {#if isRunning}
                    <div class="loading-state">
                        <span class="spinner lg" aria-hidden="true"></span>
                        <p>{statusLabel}</p>
                    </div>
                {:else}
                    <ResultsTable {result} />
                {/if}
            </section>
        </section>
        {/if}
    </main>

    <footer class="app-footer">
        <span>
            Powered by <a href="https://github.com/ggml-org/llama.cpp" target="_blank" rel="noreferrer">llama.cpp</a>
            fit logic compiled to WebAssembly.
        </span>
    </footer>
</div>

<style>
    .app-shell {
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        background: var(--bg);
    }

    /* ── Header ── */
    .app-header {
        background: var(--header-bg);
        border-bottom: 1px solid var(--border);
        padding: 0 24px;
    }

    .header-inner {
        max-width: 1280px;
        margin: 0 auto;
        padding: 16px 0;
        display: flex;
        align-items: baseline;
        gap: 20px;
        flex-wrap: wrap;
    }

    .logo-row {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .logo-text {
        font-size: 1.3rem;
        font-weight: 700;
        font-family: var(--mono);
        color: var(--text-primary);
    }

    .tagline {
        margin: 0;
        font-size: 0.88rem;
        color: var(--text-muted);
    }

    /* ── Main layout ── */
    .app-main {
        flex: 1;
        max-width: 1280px;
        width: 100%;
        margin: 0 auto;
        padding: 24px;
        display: flex;
        flex-direction: column;
        gap: 20px;
        box-sizing: border-box;
    }

    /* Row 1: Model + Runtime side by side */
    .model-runtime-row {
        display: grid;
        grid-template-columns: minmax(340px, 1fr) minmax(340px, 1fr);
        gap: 20px;
        align-items: start;
    }

    @media (max-width: 800px) {
        .model-runtime-row {
            grid-template-columns: 1fr;
        }
    }

    /* Row 2: Hardware config full width */
    .hardware-section {
        width: 100%;
        box-sizing: border-box;
    }

    .source-switch {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        gap: 8px;
    }

    .source-btn {
        border: 1px solid var(--border);
        background: var(--surface-raised);
        color: var(--text-primary);
        border-radius: 10px;
        padding: 9px 12px;
        font-size: 0.9rem;
        font-weight: 600;
        cursor: pointer;
    }

    .source-btn.active {
        background: var(--accent);
        border-color: transparent;
        color: var(--on-accent);
    }

    .source-or {
        font-size: 0.74rem;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        font-family: var(--mono);
    }

    /* ── Config panel (removed — layout replaced by model-runtime-row + hardware-section) ── */

    .panel-section {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 18px;
        display: flex;
        flex-direction: column;
        gap: 14px;
    }

    .panel-title {
        margin: 0;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Results panel ── */
    .results-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 18px;
        display: flex;
        flex-direction: column;
        gap: 16px;
        min-height: 320px;
    }

    /* ── Action row ── */
    .action-row {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding-bottom: 20px;
    }

    .run-btn {
        width: 100%;
        padding: 13px;
        background: var(--accent);
        color: var(--on-accent);
        border: none;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        transition: opacity 0.2s, transform 0.1s;
    }

    .run-btn:hover:not(:disabled) {
        opacity: 0.88;
    }

    .run-btn:active:not(:disabled) {
        transform: scale(0.98);
    }

    .run-btn:disabled {
        opacity: 0.45;
        cursor: not-allowed;
    }

    .error-msg {
        margin: 0;
        font-size: 0.85rem;
        color: var(--error);
        padding: 8px 12px;
        background: var(--error-bg);
        border-radius: 6px;
    }

    .hint-msg {
        margin: 0;
        font-size: 0.82rem;
        color: var(--text-muted);
        padding: 8px 12px;
        border: 1px dashed var(--border);
        border-radius: 6px;
    }

    /* ── Loading state ── */
    .loading-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        flex: 1;
        gap: 14px;
        color: var(--text-muted);
        padding: 48px 0;
    }

    .loading-state p {
        margin: 0;
        font-size: 0.9rem;
    }

    /* ── Spinner ── */
    .spinner {
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid transparent;
        border-top-color: currentColor;
        border-right-color: currentColor;
        border-radius: 50%;
        animation: spin 0.7s linear infinite;
    }

    .spinner.lg {
        width: 28px;
        height: 28px;
        border-width: 3px;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .ok-badge {
        display: inline-flex;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: var(--accent);
        color: var(--on-accent);
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
    }

    /* ── Footer ── */
    .app-footer {
        border-top: 1px solid var(--border);
        padding: 14px 24px;
        text-align: center;
        font-size: 0.8rem;
        color: var(--text-muted);
    }

    .app-footer a {
        color: var(--accent);
        text-decoration: none;
    }

    .app-footer a:hover {
        text-decoration: underline;
    }
</style>
