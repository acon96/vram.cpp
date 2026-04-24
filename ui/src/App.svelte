<script>
    import FileUpload from './components/FileUpload.svelte';
    import HuggingFaceSearch from './components/HuggingFaceSearch.svelte';
    import JsonHarness from './components/JsonHarness.svelte';
    import ParamPanel from './components/ParamPanel.svelte';
    import ResultsTable from './components/ResultsTable.svelte';
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
    const hfGrowthFactor = 2.0;
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

    // ── State ─────────────────────────────────────────────────────────────────
    let modelSource = $state('local');
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

    let params = $state({
        nCtx: 4096,
        nBatch: 2048,
        nUbatch: 512,
        cacheTypeK: 'f16',
        cacheTypeV: 'f16',
        nGpuLayers: -1,
        hostRamGiB: 32,
        gpus: [],
        fitTargetMiB: 512,
        targetFreeMiB: 2048,
    });

    let status = $state('idle'); // 'idle' | 'loading-wasm' | 'running' | 'done' | 'error'
    let errorMsg = $state('');
    let result = $state(null);

    // ── Handlers ──────────────────────────────────────────────────────────────
    function handleModelSourceChange(nextSource) {
        modelSource = nextSource;
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
        modelSource = 'local';
        selectedFile = file;
        result = null;
        status = 'idle';
        errorMsg = '';
    }

    function handleHuggingFaceSelectionChange(selection) {
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

    function buildHfRangePlan(initialBytes, maxBytes, growthFactor) {
        const plan = [];
        if (!Number.isFinite(maxBytes) || maxBytes <= 0) {
            return plan;
        }

        let current = initialBytes;
        if (!Number.isFinite(current) || current <= 0 || current > maxBytes) {
            current = Math.min(1024 * 1024, maxBytes);
        }
        if (current <= 0) {
            current = 1;
        }

        const growth = Number.isFinite(growthFactor) && growthFactor > 1.0 ? growthFactor : 2.0;

        while (true) {
            plan.push({ start: 0, end: Math.trunc(current) - 1 });
            if (current >= maxBytes) {
                break;
            }

            let next = Math.ceil(current * growth);
            if (next <= current) {
                next = current + 1;
            }
            if (next > maxBytes) {
                next = maxBytes;
            }
            current = next;
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
            const parsedIndex = Number.isFinite(Number(g.index))
                ? Math.max(0, Math.trunc(Number(g.index)))
                : i;
            const parsedName = typeof g.name === 'string' ? g.name.trim() : '';
            const fallbackId = `gpu${parsedIndex}`;
            const id = parsedName.length > 0 ? parsedName : fallbackId;

            return {
                id,
                name: parsedName,
                index: parsedIndex,
                backend: 'cuda',
                free_bytes: giBToBytes(g.freeGiB),
                total_bytes: giBToBytes(g.totalGiB),
            };
        });

        const fitTargets = params.gpus.length > 0
            ? params.gpus.map(() => params.fitTargetMiB)
            : [params.fitTargetMiB];
        const freeTargets = params.gpus.length > 0
            ? params.gpus.map(() => params.targetFreeMiB)
            : [params.targetFreeMiB];

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

        const plan = buildHfRangePlan(hfInitialPrefixBytes, hfMaxPrefixBytes, hfGrowthFactor);

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
            growthFactor: hfGrowthFactor,
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

                    return {
                        ...response,
                        source: 'huggingface',
                        resolvedUrl: requestUrl,
                        requests,
                        attempts,
                    };
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

        const plan = buildHfRangePlan(hfInitialPrefixBytes, hfMaxPrefixBytes, hfGrowthFactor);

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
            growthFactor: hfGrowthFactor,
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
                    const fitFile = new File([prefixBytes], `hf_fit_prefix_${range.end + 1}.gguf`, {
                        type: 'application/octet-stream',
                    });
                    const sizeFromHeaders = parseTotalBytesFromFetchHeaders(fetchResult);
                    const sizeFromSelection = Number.isFinite(Number(selection?.fileSizeBytes))
                        ? Math.max(0, Math.trunc(Number(selection.fileSizeBytes)))
                        : 0;
                    const logicalFileSizeBytes = sizeFromHeaders > 0
                        ? sizeFromHeaders
                        : sizeFromSelection;
                    const fitInput = {
                        ...predictInput,
                        virtualFileSizeBytes: logicalFileSizeBytes > prefixBytes.byteLength
                            ? logicalFileSizeBytes
                            : undefined,
                    };

                    debugLog('hf.browserFit.attempt.fit.start', {
                        candidateIndex,
                        attemptIndex,
                        requestUrl,
                        fitFileBytes: prefixBytes.byteLength,
                        logicalFileSizeBytes,
                        logicalSizeSource: sizeFromHeaders > 0 ? 'headers' : 'hf_tree',
                    });

                    const fitResponse = await client.predictMountedFit(fitFile, fitInput);

                    debugLog('hf.browserFit.attempt.fit.result', {
                        candidateIndex,
                        attemptIndex,
                        requestUrl,
                        response: fitResponse,
                        fitStatusText: describeFitStatus(fitResponse?.fit?.status),
                    });

                    return {
                        ...fitResponse,
                        source: 'huggingface',
                        resolvedUrl: requestUrl,
                        requests,
                        attempts,
                        hfMetadata: metadataResponse.metadata ?? null,
                    };
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

        return runHfMetadataFromBrowser(client, selection);
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

        let hint = '';
        if (response?.fit?.status === 2 && sentGpuOverrides > 0 && deviceCountInResponse === 0) {
            hint = ' Hint: this may be a GPU override/device-count mismatch on the wasm backend. Try removing GPU rows and retrying.';
        }
        if (typeof response?.error === 'string' && response.error.includes('thread constructor failed: Not supported')) {
            hint += ' Hint: backend fit logs are trying to use a thread path unsupported by this wasm build. Keep VITE_DEBUG_WASM_FIT_LOGS unset.';
        }
        if (typeof response?.error === 'string' && response.error.includes('failed_to_create_fitted_context')) {
            hint += ' Hint: fit projection succeeded, but creating the actual fitted context failed in wasm (likely runtime heap allocation limits and/or batch buffer size). Try a lower n_ctx.';
        }

        return {
            message: `WASM fit failed (${fitStatusText}).${fitWarnings}${backendError}${hint}`,
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
            errorMsg = 'Validate a Hugging Face GGUF file first.';
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
                debugLog('runPrediction.hfFitFromBrowser.input', {
                    selection: hfSelection,
                    predictInput,
                });

                const predictStartedAt = performance.now();
                const res = await runHfFitFromBrowser(client, hfSelection, predictInput);
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
        running: modelSource === 'local'
            ? 'Running fit prediction…'
            : 'Downloading HF prefix and running fit prediction…',
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
        <!-- Top row: Model input + Results side-by-side -->
        <div class="top-row">
            <div class="model-col">
                <section class="panel-section">
                    <h2 class="panel-title">Model</h2>
                    <div class="source-switch" role="tablist" aria-label="Model source">
                        <button
                            class="source-btn"
                            class:active={modelSource === 'local'}
                            type="button"
                            onclick={() => handleModelSourceChange('local')}
                        >
                            Upload
                        </button>
                        <span class="source-or">OR</span>
                        <button
                            class="source-btn"
                            class:active={modelSource === 'huggingface'}
                            type="button"
                            onclick={() => handleModelSourceChange('huggingface')}
                        >
                            Search on HF
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
        </div>

        <!-- Middle: Parameters flowing left-to-right -->
        <section class="panel-section params-section">
            <h2 class="panel-title">Parameters</h2>
            <ParamPanel {params} onchange={handleParamsChange} />
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
                    <p class="hint-msg">Validate a Hugging Face GGUF file to enable prediction.</p>
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

    /* Top row: Model col (fixed) + Results panel (grows) */
    .top-row {
        display: grid;
        grid-template-columns: 1fr;
        gap: 20px;
        align-items: start;
    }

    @media (max-width: 760px) {
        .top-row {
            grid-template-columns: 1fr;
        }
    }

    .model-col {
        display: flex;
        flex-direction: column;
        gap: 14px;
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

    /* Parameters section spans the full width below */
    .params-section {
        width: 100%;
        box-sizing: border-box;
    }

    /* ── Config panel (removed — layout replaced by top-row + params-section) ── */

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
