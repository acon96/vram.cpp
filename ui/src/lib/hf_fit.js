/**
 * HF fit execution -- iterative prefix-range fetch + WASM predict loops.
 *
 * All functions are pure (no Svelte state). Callers pass stateful bits as
 * callbacks so these remain independently testable.
 */

import { gguf } from '@huggingface/gguf';
import {
    buildCanonicalHfFileUrl,
    buildHfSelectionCacheKey,
    detectShardPattern,
    fetchHfPrefixBytes,
    parseTotalBytesFromFetchHeaders,
} from './hf_utils.js';

// Minimum bytes to fetch for a stub shard header.
// Each shard GGUF header (tensor descriptors for its subset of tensors) is
// typically well under 50 KiB; 128 KiB is a generous safety margin.
const SHARD_HEADER_BYTES = 1024 * 1024;
const TENSOR_PREVIEW_COUNT = 32;

function toSafeInteger(value, fallback = 0) {
    if (typeof value === 'bigint') {
        if (value < 0 || value > BigInt(Number.MAX_SAFE_INTEGER)) {
            return fallback;
        }
        return Number(value);
    }

    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
        return fallback;
    }

    return Math.trunc(parsed);
}

function readMetadataInteger(metadata, key) {
    if (!metadata || !(key in metadata)) {
        return 0;
    }

    const parsed = toSafeInteger(metadata[key], 0);
    return parsed > 0 ? parsed : 0;
}

function extractContextLengthFromMetadata(metadata) {
    if (!metadata || typeof metadata !== 'object') {
        return 0;
    }

    const architecture = typeof metadata['general.architecture'] === 'string'
        ? metadata['general.architecture']
        : '';

    if (architecture) {
        const exact = readMetadataInteger(metadata, `${architecture}.context_length`);
        if (exact > 0) {
            return exact;
        }
    }

    return readMetadataInteger(metadata, 'context_length');
}

function extractSplitInfoFromMetadata(metadata) {
    return {
        splitCount: readMetadataInteger(metadata, 'split.count'),
        splitNo: readMetadataInteger(metadata, 'split.no'),
    };
}

function summarizeTensorInfo(tensor) {
    return {
        name: tensor?.name ?? 'unknown',
        dimensions: Array.isArray(tensor?.shape)
            ? tensor.shape.map((dim) => {
                if (typeof dim === 'bigint') {
                    return dim <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(dim) : dim.toString();
                }
                return Number.isFinite(Number(dim)) ? Math.trunc(Number(dim)) : String(dim);
            })
            : [],
        ggmlType: toSafeInteger(tensor?.dtype, tensor?.dtype ?? 0),
        dataOffset: typeof tensor?.offset === 'bigint'
            ? (tensor.offset <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(tensor.offset) : tensor.offset.toString())
            : toSafeInteger(tensor?.offset, 0),
    };
}

function buildMetadataSummary(parsed, bytesConsumed) {
    const metadata = parsed?.metadata ?? {};
    const tensorInfos = Array.isArray(parsed?.tensorInfos) ? parsed.tensorInfos : [];

    return {
        version: toSafeInteger(metadata.version, 0),
        kvCount: toSafeInteger(metadata.kv_count, 0),
        tensorCount: toSafeInteger(metadata.tensor_count, tensorInfos.length),
        bytesConsumed,
        tensors: tensorInfos.slice(0, TENSOR_PREVIEW_COUNT).map(summarizeTensorInfo),
    };
}

function parseRangeHeader(rangeValue) {
    const match = /^bytes=(\d+)-(\d+)$/i.exec(String(rangeValue || '').trim());
    if (!match) {
        return { start: 0, end: 0 };
    }

    return {
        start: Number.parseInt(match[1], 10),
        end: Number.parseInt(match[2], 10),
    };
}

function buildTrackedFetch(requests, attempts, token, includeAuth) {
    return async (input, init = {}) => {
        const requestUrl = typeof input === 'string' ? input : input?.url;
        const headers = new Headers(init?.headers ?? (typeof input === 'object' ? input?.headers : undefined) ?? undefined);

        if (includeAuth && token) {
            headers.set('Authorization', `Bearer ${token}`);
        }

        const rangeValue = headers.get('Range') ?? '';
        const { start, end } = parseRangeHeader(rangeValue);
        requests.push({
            url: requestUrl,
            start,
            end,
            headers: [
                ...(rangeValue ? [{ name: 'Range', value: rangeValue }] : []),
                ...(headers.has('Accept') ? [{ name: 'Accept', value: headers.get('Accept') }] : []),
                ...(headers.has('Authorization') ? [{ name: 'Authorization', value: 'Bearer ***' }] : []),
            ],
        });

        const response = await fetch(requestUrl, {
            ...init,
            headers,
            redirect: 'follow',
        });

        attempts.push({
            start,
            end,
            httpStatus: response.status,
            finalUrl: response.url || requestUrl,
            contentRange: response.headers.get('content-range') || '',
            contentLength: response.headers.get('content-length') || '',
            acceptRanges: response.headers.get('accept-ranges') || '',
        });

        return response;
    };
}

async function parseRemoteGguf(selection, requestUrl, logger) {
    const requests = [];
    const attempts = [];
    const token = (selection?.token || '').trim();
    const includeAuth = requestUrl.startsWith('https://huggingface.co/');
    const trackedFetch = buildTrackedFetch(requests, attempts, token, includeAuth);

    logger?.log('hf.gguf.parse.start', {
        requestUrl,
        includeAuth,
    });

    const parsed = await gguf(requestUrl, {
        typedMetadata: true,
        fetch: trackedFetch,
    });

    const bytesConsumed = toSafeInteger(parsed?.tensorDataOffset, 0);
    if (bytesConsumed <= 0) {
        throw new Error('gguf_tensor_data_offset_invalid');
    }

    const prefixFetchResult = await fetchHfPrefixBytes(
        requestUrl,
        0,
        bytesConsumed - 1,
        token,
        includeAuth,
    );

    requests.push({
        url: requestUrl,
        start: 0,
        end: bytesConsumed - 1,
        headers: [
            { name: 'Range', value: `bytes=0-${bytesConsumed - 1}` },
            { name: 'Accept', value: 'application/octet-stream' },
            ...(includeAuth && token ? [{ name: 'Authorization', value: 'Bearer ***' }] : []),
        ],
    });

    attempts.push({
        start: 0,
        end: bytesConsumed - 1,
        httpStatus: prefixFetchResult.httpStatus,
        finalUrl: prefixFetchResult.finalUrl,
        contentRange: prefixFetchResult.contentRange,
        contentLength: prefixFetchResult.contentLength,
        acceptRanges: prefixFetchResult.acceptRanges,
        fetchedChunkBytes: prefixFetchResult.bytes.byteLength,
        purpose: 'stub-prefix',
    });

    logger?.log('hf.gguf.parse.done', {
        requestUrl,
        bytesConsumed,
        tensorCount: parsed?.tensorInfos?.length ?? 0,
    });

    return {
        parsed,
        requests,
        attempts,
        prefixFetchResult,
        prefixBytes: prefixFetchResult.bytes,
        metadataResponse: {
            ok: true,
            source: 'huggingface',
            resolvedUrl: prefixFetchResult.finalUrl || requestUrl,
            metadata: buildMetadataSummary(parsed, bytesConsumed),
        },
    };
}

/**
 * Assemble the cached prefetch result object from a successful metadata parse
 * attempt.  This is stored in App state and reused for the actual fit call.
 */
export function buildPreparedHfFitResult(
    selection, requestUrl, requests, attempts,
    prefixBytes, fetchResult, metadataResponse, parsedMetadata,
) {
    const sizeFromHeaders   = parseTotalBytesFromFetchHeaders(fetchResult);
    const sizeFromSelection = Number.isFinite(Number(selection?.fileSizeBytes))
        ? Math.max(0, Math.trunc(Number(selection.fileSizeBytes))) : 0;
    const logicalFileSizeBytes = sizeFromHeaders > 0 ? sizeFromHeaders : sizeFromSelection;
    const contextLength  = extractContextLengthFromMetadata(parsedMetadata);
    const splitInfo      = extractSplitInfoFromMetadata(parsedMetadata);
    // Preserve the original HF filename so shard naming is intact when mounting.
    const originalFileName = String(selection?.file || '').split('/').filter(Boolean).pop() || 'model.gguf';

    return {
        ...metadataResponse,
        source: 'huggingface',
        resolvedUrl: requestUrl,
        requests,
        attempts,
        prefixBytes,
        logicalFileSizeBytes,
        contextLength,
        splitCount: splitInfo.splitCount,
        splitNo:    splitInfo.splitNo,
        originalFileName,
        cacheKey: buildHfSelectionCacheKey({ ...selection, resolvedUrl: requestUrl }),
        // shardHeaders populated by fetchShardHeadersIfNeeded
        shardHeaders: [],
    };
}

/**
 * If the prepared fit describes a sharded model (split.count > 1) fetch the
 * GGUF headers of all remaining shards and attach them.
 *
 * Each shard header only needs to be large enough for gguf_init_from_file to
 * read the tensor-descriptor section. llama.cpp opens shards with no_alloc=true
 * so no tensor data is ever read from the stub files.
 *
 * @param {object} preparedFit
 * @param {object} selection  - { token }
 * @param {{ log: Function, error: Function }} [logger]
 * @returns {Promise<object>} preparedFit with shardHeaders populated
 */
async function fetchShardHeadersIfNeeded(preparedFit, selection, logger) {
    const { splitCount, splitNo } = preparedFit;

    if (splitCount <= 1 || splitNo !== 0) return preparedFit;

    // Detect the shard pattern from the original selection filename, NOT from
    // the resolvedUrl. Pre-signed / XetHub URLs encode the filename inside query
    // params so buildShardUrl can't rewrite them; canonical HF URLs are reliable.
    const selectionFile = String(selection?.file || '');
    const baseFileName  = selectionFile.split('/').filter(Boolean).pop() || '';
    const pattern       = detectShardPattern(baseFileName);
    if (!pattern) {
        logger?.error('hf.shardHeaders.noPattern', { baseFileName, splitCount });
        return preparedFit;
    }

    // Directory prefix inside the repo (e.g. "models/" or "").
    const dirParts = selectionFile.split('/').filter(Boolean).slice(0, -1);

    // Fetch the same byte count as the primary shard needed.  Every shard
    // duplicates the full KV section, so the primary's byte count is a safe
    // lower bound for each stub.
    const stubFetchBytes = preparedFit.prefixBytes.byteLength;
    const token = (selection?.token || '').trim();

    logger?.log('hf.shardHeaders.start', {
        totalShards: splitCount,
        pattern,
        baseFileName,
        stubFetchBytes,
    });

    const shardFetches = [];
    for (let n = 2; n <= splitCount; n++) {
        const shardName = baseFileName.replace(
            /(-)(\d{5})(-of-\d{5}\.gguf)$/i,
            `$1${String(n).padStart(5, '0')}$3`,
        );
        const shardFilePath = dirParts.length > 0 ? `${dirParts.join('/')}/${shardName}` : shardName;
        const shardUrl = buildCanonicalHfFileUrl({ ...selection, file: shardFilePath });
        if (!shardUrl) {
            shardFetches.push(Promise.resolve({ shardNo: n, bytes: null, ok: false, error: 'could_not_build_url' }));
            continue;
        }
        shardFetches.push(
            fetchHfPrefixBytes(shardUrl, 0, stubFetchBytes - 1, token, true)
                .then((result) => ({
                    shardNo: n,
                    bytes: result.bytes,
                    ok: true,
                    logicalSizeBytes: parseTotalBytesFromFetchHeaders(result),
                }))
                .catch((err) => ({ shardNo: n, bytes: null, ok: false, error: err?.message ?? String(err) })),
        );
    }

    const results = await Promise.all(shardFetches);
    const shardHeaders = [];
    for (const r of results) {
        if (r.ok && r.bytes != null) {
            shardHeaders.push({ shardNo: r.shardNo, bytes: r.bytes, logicalSizeBytes: r.logicalSizeBytes || 0 });
            logger?.log('hf.shardHeaders.fetched', {
                shardNo: r.shardNo,
                bytes: r.bytes.byteLength,
                logicalSizeBytes: r.logicalSizeBytes,
            });
        } else {
            logger?.error('hf.shardHeaders.fetchFailed', { shardNo: r.shardNo, error: r.error });
        }
    }

    return { ...preparedFit, shardHeaders };
}

/**
 * Mount the cached prefix bytes in the WASM FS and execute a fit predict.
 * If preparedFit.shardHeaders is non-empty the stub shard files are mounted
 * alongside the primary file before the fit is run.
 */
export async function runFitFromPreparedPrefix(client, preparedFit, predictInput, logger) {
    const fileName = preparedFit.originalFileName || 'hf_cached_prefix.gguf';
    const fitFile  = new File([preparedFit.prefixBytes], fileName, { type: 'application/octet-stream' });

    // Build shardFiles array for the worker.  Each entry has the 1-based
    // shardNo, the raw Uint8Array, and the logical file size so the worker
    // can apply the sparse-read trick to each stub.
    const shardFiles = (preparedFit.shardHeaders || []).map((sh) => ({
        shardNo: sh.shardNo,
        bytes: sh.bytes,
        logicalSizeBytes: sh.logicalSizeBytes || 0,
    }));

    const fitInput = {
        ...predictInput,
        virtualFileSizeBytes: preparedFit.logicalFileSizeBytes > preparedFit.prefixBytes.byteLength
            ? preparedFit.logicalFileSizeBytes : undefined,
        shardFiles,
    };

    logger?.log('hf.cachedFit.start', {
        resolvedUrl:          preparedFit.resolvedUrl,
        fitFileBytes:         preparedFit.prefixBytes.byteLength,
        logicalFileSizeBytes: preparedFit.logicalFileSizeBytes,
        shardCount:           shardFiles.length,
    });

    const fitResponse = await client.predictMountedFit(fitFile, fitInput);
    return {
        ...fitResponse,
        source:     'huggingface',
        resolvedUrl: preparedFit.resolvedUrl,
        requests:   preparedFit.requests,
        attempts:   preparedFit.attempts,
        hfMetadata: preparedFit.metadata ?? null,
    };
}

/**
 * Parse Hugging Face-hosted GGUF metadata using @huggingface/gguf, then fetch
 * exactly the raw header bytes needed for the later mounted-file fit path.
 *
 * @param {object} client - predictor worker client
 * @param {object} selection - { repo, file, revision, token, resolvedUrl, ... }
 * @param {object} opts
 * @param {{ log: Function, error: Function }} [opts.logger]
 */
export async function runHfMetadataFromBrowser(client, selection, { logger }) {
    void client;

    const resolvedUrl  = (selection?.resolvedUrl || '').trim();
    const canonicalUrl = buildCanonicalHfFileUrl(selection || {});

    if (!resolvedUrl && !canonicalUrl) {
        return { ok: false, source: 'huggingface', error: 'invalid_huggingface_model_descriptor' };
    }

    const candidateUrls = [];
    if (resolvedUrl) candidateUrls.push(resolvedUrl);
    if (canonicalUrl && canonicalUrl !== resolvedUrl) candidateUrls.push(canonicalUrl);

    logger?.log('hf.browserMetadata.start', {
        repo: selection?.repo || '',
        file: selection?.file || '',
        candidateUrls,
    });

    let lastFailure = {
        ok: false, source: 'huggingface', error: 'hf_metadata_parse_failed', detail: 'no_candidate_urls',
    };

    for (let ci = 0; ci < candidateUrls.length; ci++) {
        const requestUrl = candidateUrls[ci];
        try {
            const {
                parsed,
                requests,
                attempts,
                prefixFetchResult,
                prefixBytes,
                metadataResponse,
            } = await parseRemoteGguf(selection, requestUrl, logger);
            const effectiveUrl = prefixFetchResult.finalUrl || requestUrl;
            const prepared = buildPreparedHfFitResult(
                selection,
                effectiveUrl,
                requests,
                attempts,
                prefixBytes,
                prefixFetchResult,
                metadataResponse,
                parsed?.metadata,
            );
            return fetchShardHeadersIfNeeded(prepared, selection, logger);
        } catch (error) {
            lastFailure = {
                ok: false,
                source: 'huggingface',
                error: 'hf_metadata_parse_failed',
                detail: error?.message ?? String(error),
                resolvedUrl: requestUrl,
            };
            logger?.error('hf.browserMetadata.parse.error', {
                requestUrl,
                error: lastFailure.detail,
            });
        }
    }

    logger?.error('hf.browserMetadata.failed', lastFailure);
    return lastFailure;
}

/**
 * Same as runHfMetadataFromBrowser but continues to run the fit after metadata
 * is successfully parsed.
 *
 * @param {object} client
 * @param {object} selection
 * @param {object} predictInput
 * @param {object} opts
 * @param {(preparedFit: object) => void} opts.onPreparedFit  callback to cache the prepared fit in app state
 * @param {{ log: Function, error: Function }} [opts.logger]
 */
export async function runHfFitFromBrowser(client, selection, predictInput, { onPreparedFit, logger }) {
    const resolvedUrl  = (selection?.resolvedUrl || '').trim();
    const canonicalUrl = buildCanonicalHfFileUrl(selection || {});

    if (!resolvedUrl && !canonicalUrl) {
        return { ok: false, source: 'huggingface', error: 'invalid_huggingface_model_descriptor' };
    }

    const candidateUrls = [];
    if (resolvedUrl) candidateUrls.push(resolvedUrl);
    if (canonicalUrl && canonicalUrl !== resolvedUrl) candidateUrls.push(canonicalUrl);

    logger?.log('hf.browserFit.start', {
        repo: selection?.repo || '',
        file: selection?.file || '',
        candidateUrls, predictInput,
    });

    let lastFailure = {
        ok: false, source: 'huggingface', error: 'hf_metadata_parse_failed', detail: 'no_candidate_urls',
    };

    for (let ci = 0; ci < candidateUrls.length; ci++) {
        const requestUrl = candidateUrls[ci];
        try {
            const {
                parsed,
                requests,
                attempts,
                prefixFetchResult,
                prefixBytes,
                metadataResponse,
            } = await parseRemoteGguf(selection, requestUrl, logger);

            let preparedFit = buildPreparedHfFitResult(
                selection,
                prefixFetchResult.finalUrl || requestUrl,
                requests,
                attempts,
                prefixBytes,
                prefixFetchResult,
                metadataResponse,
                parsed?.metadata,
            );
            preparedFit = await fetchShardHeadersIfNeeded(preparedFit, selection, logger);
            onPreparedFit?.(preparedFit);

            logger?.log('hf.browserFit.fit.start', {
                fitFileBytes: preparedFit.prefixBytes.byteLength,
                logicalFileSizeBytes: preparedFit.logicalFileSizeBytes,
                shardCount: preparedFit.shardHeaders?.length ?? 0,
            });

            const fitResponse = await runFitFromPreparedPrefix(client, preparedFit, predictInput, logger);
            logger?.log('hf.browserFit.fit.done', { ok: fitResponse?.ok });
            return fitResponse;
        } catch (error) {
            lastFailure = {
                ok: false,
                source: 'huggingface',
                error: 'hf_metadata_parse_failed',
                detail: error?.message ?? String(error),
                resolvedUrl: requestUrl,
            };
            logger?.error('hf.browserFit.parse.error', {
                requestUrl,
                error: lastFailure.detail,
            });
        }
    }

    logger?.error('hf.browserFit.failed', lastFailure);
    return lastFailure;
}
