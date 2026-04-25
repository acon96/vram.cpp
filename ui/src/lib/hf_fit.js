/**
 * HF fit execution — iterative prefix-range fetch + WASM predict loops.
 *
 * All functions are pure (no Svelte state). Callers pass stateful bits as
 * callbacks so these remain independently testable.
 */

import { extractContextLengthFromPrefix } from './gguf_utils.js';
import {
    buildCanonicalHfFileUrl,
    buildHfRangePlan,
    buildHfSelectionCacheKey,
    fetchHfPrefixBytes,
    parseTotalBytesFromFetchHeaders,
} from './hf_utils.js';

/**
 * Assemble the cached prefetch result object from a successful metadata parse
 * attempt.  This is stored in App state and reused for the actual fit call.
 */
export function buildPreparedHfFitResult(
    selection, requestUrl, requests, attempts,
    prefixBytes, fetchResult, metadataResponse,
) {
    const sizeFromHeaders  = parseTotalBytesFromFetchHeaders(fetchResult);
    const sizeFromSelection = Number.isFinite(Number(selection?.fileSizeBytes))
        ? Math.max(0, Math.trunc(Number(selection.fileSizeBytes))) : 0;
    const logicalFileSizeBytes = sizeFromHeaders > 0 ? sizeFromHeaders : sizeFromSelection;
    const contextLength  = extractContextLengthFromPrefix(prefixBytes);
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
        originalFileName,
        cacheKey: buildHfSelectionCacheKey({ ...selection, resolvedUrl: requestUrl }),
    };
}

/**
 * Mount the cached prefix bytes in the WASM FS and execute a fit predict.
 */
export async function runFitFromPreparedPrefix(client, preparedFit, predictInput, logger) {
    const fileName = preparedFit.originalFileName || 'hf_cached_prefix.gguf';
    const fitFile  = new File([preparedFit.prefixBytes], fileName, { type: 'application/octet-stream' });
    const fitInput = {
        ...predictInput,
        virtualFileSizeBytes: preparedFit.logicalFileSizeBytes > preparedFit.prefixBytes.byteLength
            ? preparedFit.logicalFileSizeBytes : undefined,
    };

    logger?.log('hf.cachedFit.start', {
        resolvedUrl: preparedFit.resolvedUrl,
        fitFileBytes: preparedFit.prefixBytes.byteLength,
        logicalFileSizeBytes: preparedFit.logicalFileSizeBytes,
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

/**
 * Progressively fetch HF file prefix bytes and parse GGUF metadata until
 * the parser is satisfied or we exhaust the range plan.
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

            const prefixBytes = fetchResult.bytes;
            const localRequest = buildMetadataRequest(prefixBytes.byteLength);
            const localFile = new File([prefixBytes], `hf_prefix_${range.end + 1}.gguf`, { type: 'application/octet-stream' });
            const response = await client.predictMountedJson(localFile, localRequest);

            attempts.push({
                ai, start: range.start, end: range.end,
                fetchedBytes: prefixBytes.byteLength,
                httpStatus: fetchResult.httpStatus,
                finalUrl: fetchResult.finalUrl,
                contentRange: fetchResult.contentRange,
                parserOk: response?.ok === true,
                parserError: response?.error || '',
                minimumRequiredBytes: response?.minimumRequiredBytes || 0,
                bytesConsumed: response?.metadata?.bytesConsumed || 0,
            });

            if (response?.ok === true) {
                logger?.log('hf.browserMetadata.success', { ai, fetchedBytes: prefixBytes.byteLength });
                return buildPreparedHfFitResult(
                    selection, requestUrl, requests, attempts, prefixBytes, fetchResult, response,
                );
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

            const prefixBytes = fetchResult.bytes;
            const localRequest = buildMetadataRequest(prefixBytes.byteLength);
            const metadataFile = new File([prefixBytes], `hf_prefix_${range.end + 1}.gguf`, { type: 'application/octet-stream' });
            const metadataResponse = await client.predictMountedJson(metadataFile, localRequest);

            attempts.push({
                ai, start: range.start, end: range.end,
                fetchedBytes: prefixBytes.byteLength,
                httpStatus: fetchResult.httpStatus,
                finalUrl: fetchResult.finalUrl,
                contentRange: fetchResult.contentRange,
                parserOk: metadataResponse?.ok === true,
                parserError: metadataResponse?.error || '',
                minimumRequiredBytes: metadataResponse?.minimumRequiredBytes || 0,
                bytesConsumed: metadataResponse?.metadata?.bytesConsumed || 0,
            });

            if (metadataResponse?.ok === true) {
                const preparedFit = buildPreparedHfFitResult(
                    selection, requestUrl, requests, attempts, prefixBytes, fetchResult, metadataResponse,
                );
                onPreparedFit?.(preparedFit);

                logger?.log('hf.browserFit.fit.start', {
                    ai, fitFileBytes: preparedFit.prefixBytes.byteLength,
                    logicalFileSizeBytes: preparedFit.logicalFileSizeBytes,
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
