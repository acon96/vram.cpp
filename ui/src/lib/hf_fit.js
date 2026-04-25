/**
 * HF fit execution -- iterative prefix-range fetch + WASM predict loops.
 *
 * All functions are pure (no Svelte state). Callers pass stateful bits as
 * callbacks so these remain independently testable.
 */

import { extractContextLengthFromPrefix, parseGgufSplitInfo } from './gguf_utils.js';
import {
    buildCanonicalHfFileUrl,
    buildHfRangePlan,
    buildHfSelectionCacheKey,
    buildShardUrl,
    detectShardPattern,
    fetchHfPrefixBytes,
    parseTotalBytesFromFetchHeaders,
} from './hf_utils.js';

// Minimum bytes to fetch for a stub shard header.
// Each shard GGUF header (tensor descriptors for its subset of tensors) is
// typically well under 50 KiB; 128 KiB is a generous safety margin.
const SHARD_HEADER_BYTES = 1024 * 1024;

/**
 * Assemble the cached prefetch result object from a successful metadata parse
 * attempt.  This is stored in App state and reused for the actual fit call.
 */
export function buildPreparedHfFitResult(
    selection, requestUrl, requests, attempts,
    prefixBytes, fetchResult, metadataResponse,
) {
    const sizeFromHeaders   = parseTotalBytesFromFetchHeaders(fetchResult);
    const sizeFromSelection = Number.isFinite(Number(selection?.fileSizeBytes))
        ? Math.max(0, Math.trunc(Number(selection.fileSizeBytes))) : 0;
    const logicalFileSizeBytes = sizeFromHeaders > 0 ? sizeFromHeaders : sizeFromSelection;
    const contextLength  = extractContextLengthFromPrefix(prefixBytes);
    const splitInfo      = parseGgufSplitInfo(prefixBytes);
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

// ---------------------------------------------------------------------------
// Internal helper -- concatenate two Uint8Arrays without extra copies.
// ---------------------------------------------------------------------------
function concatBytes(a, b) {
    if (a.byteLength === 0) return b;
    if (b.byteLength === 0) return a;
    const out = new Uint8Array(a.byteLength + b.byteLength);
    out.set(a, 0);
    out.set(b, a.byteLength);
    return out;
}

/**
 * Progressively fetch HF file prefix bytes and parse GGUF metadata until
 * the parser is satisfied or we exhaust the range plan.
 *
 * Each iteration fetches ONLY the new bytes for that step (the plan now
 * returns incremental ranges). Bytes are accumulated in memory so the WASM
 * parser always receives the full prefix from byte 0.
 *
 * @param {object} client - predictor worker client
 * @param {object} selection - { repo, file, revision, token, resolvedUrl, ... }
 * @param {object} opts
 * @param {{ initial: number, max: number, step: number }} opts.rangeConfig
 * @param {(prefixByteCount: number) => object} opts.buildMetadataRequest
 * @param {{ log: Function, error: Function }} [opts.logger]
 */
export async function runHfMetadataFromBrowser(client, selection, { rangeConfig, buildMetadataRequest, logger }) {
    const resolvedUrl  = (selection?.resolvedUrl || '').trim();
    const canonicalUrl = buildCanonicalHfFileUrl(selection || {});

    if (!resolvedUrl && !canonicalUrl) {
        return { ok: false, source: 'huggingface', error: 'invalid_huggingface_model_descriptor' };
    }

    const plan = buildHfRangePlan(rangeConfig.initial, rangeConfig.max, rangeConfig.step);
    const candidateUrls = [];
    if (resolvedUrl) candidateUrls.push(resolvedUrl);
    if (canonicalUrl && canonicalUrl !== resolvedUrl) candidateUrls.push(canonicalUrl);

    logger?.log('hf.browserMetadata.start', {
        repo: selection?.repo || '',
        file: selection?.file || '',
        candidateUrls,
        plan,
    });

    let lastFailure = {
        ok: false, source: 'huggingface', error: 'hf_range_fetch_failed', detail: 'no_candidate_urls',
    };

    for (let ci = 0; ci < candidateUrls.length; ci++) {
        const requestUrl = candidateUrls[ci];
        const includeAuth = requestUrl.startsWith('https://huggingface.co/');
        const requests = [];
        const attempts = [];
        let lastErrorResponse = null;
        let fetchFailure = null;
        let accumulatedBytes = new Uint8Array(0);
        let lastFetchResult = null;

        for (let ai = 0; ai < plan.length; ai++) {
            const range = plan[ai];
            requests.push({
                url: requestUrl,
                start: range.start,
                end: range.end,
                headers: [
                    { name: 'Range',  value: `bytes=${range.start}-${range.end}` },
                    { name: 'Accept', value: 'application/octet-stream' },
                    ...(includeAuth && (selection?.token || '').trim()
                        ? [{ name: 'Authorization', value: 'Bearer ***' }] : []),
                ],
            });

            let fetchResult;
            try {
                fetchResult = await fetchHfPrefixBytes(
                    requestUrl, range.start, range.end,
                    selection?.token || '', includeAuth,
                );
            } catch (error) {
                logger?.error('hf.browserMetadata.fetch.error', { ai, error: error?.message ?? String(error) });
                fetchFailure = {
                    ok: false, source: 'huggingface', error: 'hf_range_fetch_failed',
                    detail: error?.message ?? String(error), resolvedUrl: requestUrl, requests, attempts,
                };
                break;
            }

            // Accumulate: only new bytes are received per step.
            accumulatedBytes = concatBytes(accumulatedBytes, fetchResult.bytes);
            lastFetchResult  = fetchResult;

            const localRequest = buildMetadataRequest(accumulatedBytes.byteLength);
            const localFile = new File(
                [accumulatedBytes],
                `hf_prefix_${accumulatedBytes.byteLength}.gguf`,
                { type: 'application/octet-stream' },
            );
            const response = await client.predictMountedJson(localFile, localRequest);

            attempts.push({
                ai, start: range.start, end: range.end,
                fetchedChunkBytes: fetchResult.bytes.byteLength,
                totalAccumulatedBytes: accumulatedBytes.byteLength,
                httpStatus: fetchResult.httpStatus,
                finalUrl:   fetchResult.finalUrl,
                contentRange: fetchResult.contentRange,
                parserOk:   response?.ok === true,
                parserError: response?.error || '',
                minimumRequiredBytes: response?.minimumRequiredBytes || 0,
                bytesConsumed: response?.metadata?.bytesConsumed || 0,
            });

            if (response?.ok === true) {
                logger?.log('hf.browserMetadata.success', {
                    ai, totalBytes: accumulatedBytes.byteLength,
                });
                const prepared = buildPreparedHfFitResult(
                    selection, requestUrl, requests, attempts,
                    accumulatedBytes, lastFetchResult, response,
                );
                return fetchShardHeadersIfNeeded(prepared, selection, logger);
            }

            lastErrorResponse = response;
            if (response?.error !== 'insufficient_prefix_bytes') {
                logger?.error('hf.browserMetadata.nonPrefixError', { ai, response });
                return { ...response, source: 'huggingface', resolvedUrl: requestUrl, requests, attempts };
            }
        }

        if (fetchFailure != null) { lastFailure = fetchFailure; continue; }

        lastFailure = {
            ok: false, source: 'huggingface',
            error: lastErrorResponse?.error || 'insufficient_prefix_bytes',
            detail: '',
            minimumRequiredBytes: lastErrorResponse?.minimumRequiredBytes || 0,
            resolvedUrl: requestUrl, requests, attempts,
        };
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
 * @param {{ initial: number, max: number, step: number }} opts.rangeConfig
 * @param {(prefixByteCount: number) => object} opts.buildMetadataRequest
 * @param {(preparedFit: object) => void} opts.onPreparedFit  callback to cache the prepared fit in app state
 * @param {{ log: Function, error: Function }} [opts.logger]
 */
export async function runHfFitFromBrowser(client, selection, predictInput, { rangeConfig, buildMetadataRequest, onPreparedFit, logger }) {
    const resolvedUrl  = (selection?.resolvedUrl || '').trim();
    const canonicalUrl = buildCanonicalHfFileUrl(selection || {});

    if (!resolvedUrl && !canonicalUrl) {
        return { ok: false, source: 'huggingface', error: 'invalid_huggingface_model_descriptor' };
    }

    const plan = buildHfRangePlan(rangeConfig.initial, rangeConfig.max, rangeConfig.step);
    const candidateUrls = [];
    if (resolvedUrl) candidateUrls.push(resolvedUrl);
    if (canonicalUrl && canonicalUrl !== resolvedUrl) candidateUrls.push(canonicalUrl);

    logger?.log('hf.browserFit.start', {
        repo: selection?.repo || '',
        file: selection?.file || '',
        candidateUrls, plan, predictInput,
    });

    let lastFailure = {
        ok: false, source: 'huggingface', error: 'hf_range_fetch_failed', detail: 'no_candidate_urls',
    };

    for (let ci = 0; ci < candidateUrls.length; ci++) {
        const requestUrl = candidateUrls[ci];
        const includeAuth = requestUrl.startsWith('https://huggingface.co/');
        const requests = [];
        const attempts = [];
        let lastErrorResponse = null;
        let fetchFailure = null;
        let accumulatedBytes = new Uint8Array(0);
        let lastFetchResult = null;

        for (let ai = 0; ai < plan.length; ai++) {
            const range = plan[ai];
            requests.push({
                url: requestUrl,
                start: range.start,
                end: range.end,
                headers: [
                    { name: 'Range',  value: `bytes=${range.start}-${range.end}` },
                    { name: 'Accept', value: 'application/octet-stream' },
                    ...(includeAuth && (selection?.token || '').trim()
                        ? [{ name: 'Authorization', value: 'Bearer ***' }] : []),
                ],
            });

            let fetchResult;
            try {
                fetchResult = await fetchHfPrefixBytes(
                    requestUrl, range.start, range.end,
                    selection?.token || '', includeAuth,
                );
            } catch (error) {
                logger?.error('hf.browserFit.fetch.error', { ai, error: error?.message ?? String(error) });
                fetchFailure = {
                    ok: false, source: 'huggingface', error: 'hf_range_fetch_failed',
                    detail: error?.message ?? String(error), resolvedUrl: requestUrl, requests, attempts,
                };
                break;
            }

            accumulatedBytes = concatBytes(accumulatedBytes, fetchResult.bytes);
            lastFetchResult  = fetchResult;

            const localRequest = buildMetadataRequest(accumulatedBytes.byteLength);
            const metadataFile = new File(
                [accumulatedBytes],
                `hf_prefix_${accumulatedBytes.byteLength}.gguf`,
                { type: 'application/octet-stream' },
            );
            const metadataResponse = await client.predictMountedJson(metadataFile, localRequest);

            attempts.push({
                ai, start: range.start, end: range.end,
                fetchedChunkBytes: fetchResult.bytes.byteLength,
                totalAccumulatedBytes: accumulatedBytes.byteLength,
                httpStatus: fetchResult.httpStatus,
                finalUrl:   fetchResult.finalUrl,
                contentRange: fetchResult.contentRange,
                parserOk:   metadataResponse?.ok === true,
                parserError: metadataResponse?.error || '',
                minimumRequiredBytes: metadataResponse?.minimumRequiredBytes || 0,
                bytesConsumed: metadataResponse?.metadata?.bytesConsumed || 0,
            });

            if (metadataResponse?.ok === true) {
                let preparedFit = buildPreparedHfFitResult(
                    selection, requestUrl, requests, attempts,
                    accumulatedBytes, lastFetchResult, metadataResponse,
                );
                // Fetch remaining shard headers in parallel before caching so
                // the cached preparedFit already has shardHeaders populated.
                preparedFit = await fetchShardHeadersIfNeeded(preparedFit, selection, logger);
                onPreparedFit?.(preparedFit);

                logger?.log('hf.browserFit.fit.start', {
                    ai, fitFileBytes: preparedFit.prefixBytes.byteLength,
                    logicalFileSizeBytes: preparedFit.logicalFileSizeBytes,
                    shardCount: preparedFit.shardHeaders?.length ?? 0,
                });

                const fitResponse = await runFitFromPreparedPrefix(client, preparedFit, predictInput, logger);
                logger?.log('hf.browserFit.fit.done', { ai, ok: fitResponse?.ok });
                return fitResponse;
            }

            lastErrorResponse = metadataResponse;
            if (metadataResponse?.error !== 'insufficient_prefix_bytes') {
                logger?.error('hf.browserFit.nonPrefixError', { ai, response: metadataResponse });
                return { ...metadataResponse, source: 'huggingface', resolvedUrl: requestUrl, requests, attempts };
            }
        }

        if (fetchFailure != null) { lastFailure = fetchFailure; continue; }

        lastFailure = {
            ok: false, source: 'huggingface',
            error: lastErrorResponse?.error || 'insufficient_prefix_bytes',
            detail: '',
            minimumRequiredBytes: lastErrorResponse?.minimumRequiredBytes || 0,
            resolvedUrl: requestUrl, requests, attempts,
        };
    }

    logger?.error('hf.browserFit.failed', lastFailure);
    return lastFailure;
}
