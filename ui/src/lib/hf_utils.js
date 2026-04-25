/**
 * Hugging Face URL, range-plan, and byte-fetch utilities.
 */

export function encodeRepoPath(repo) {
    return String(repo || '').split('/').filter(Boolean).map(encodeURIComponent).join('/');
}

export function encodePathPreservingSlashes(path) {
    return String(path || '').split('/').map(encodeURIComponent).join('/');
}

/** Build a canonical HF resolve URL from a selection object { repo, file, revision }. */
export function buildCanonicalHfFileUrl(selection) {
    const encodedRepo     = encodeRepoPath(selection?.repo || '');
    const encodedFile     = encodePathPreservingSlashes(selection?.file || '');
    const encodedRevision = encodeURIComponent((selection?.revision || 'main').trim() || 'main');
    if (!encodedRepo || !encodedFile) return '';
    return `https://huggingface.co/${encodedRepo}/resolve/${encodedRevision}/${encodedFile}`;
}

/** Stable cache key for a selection object. */
export function buildHfSelectionCacheKey(selection) {
    const repo     = String(selection?.repo     || '').trim();
    const file     = String(selection?.file     || '').trim();
    const revision = String(selection?.revision || 'main').trim() || 'main';
    const token    = String(selection?.token    || '').trim();
    return `${repo}||${file}||${revision}||${token}`;
}

/**
 * Build a list of byte-range requests growing from initialBytes to maxBytes
 * in steps of stepBytes.  Returns [{ start, end }, ...].
 */
export function buildHfRangePlan(initialBytes, maxBytes, stepBytes) {
    const plan = [];
    if (!Number.isFinite(maxBytes) || maxBytes <= 0) return plan;
    const first = Number.isFinite(initialBytes) && initialBytes > 0
        ? Math.trunc(initialBytes)
        : Math.min(2 * 1024 * 1024, Math.trunc(maxBytes));
    const step = Number.isFinite(stepBytes) && stepBytes > 0 ? Math.trunc(stepBytes) : first;
    if (first <= 0 || step <= 0) return plan;
    for (let cur = first; cur <= maxBytes; cur += step) {
        plan.push({ start: 0, end: cur - 1 });
    }
    const lastEnd = plan.length > 0 ? plan[plan.length - 1].end : -1;
    if (lastEnd + 1 < maxBytes) plan.push({ start: 0, end: Math.trunc(maxBytes) - 1 });
    return plan;
}

/**
 * Fetch a byte range from a URL.  Throws a plain-object error on failure.
 * Returns { bytes, httpStatus, redirected, finalUrl, contentRange, contentLength, acceptRanges }.
 */
export async function fetchHfPrefixBytes(url, start, end, token, includeAuthHeader) {
    const headers = {
        Accept: 'application/octet-stream',
        Range: `bytes=${start}-${end}`,
    };
    const trimmedToken = (token || '').trim();
    if (includeAuthHeader && trimmedToken) {
        headers.Authorization = `Bearer ${trimmedToken}`;
    }

    const response = await fetch(url, { method: 'GET', headers, redirect: 'follow' });
    const contentRange  = response.headers.get('content-range')  || '';
    const contentLength = response.headers.get('content-length') || '';
    const acceptRanges  = response.headers.get('accept-ranges')  || '';

    if (!response.ok) {
        throw {
            message: `hf_range_fetch_failed_http_${response.status}`,
            httpStatus: response.status,
            finalUrl: response.url || url,
            contentRange, contentLength,
        };
    }

    const buffer = await response.arrayBuffer();
    if (buffer.byteLength === 0) {
        throw {
            message: 'empty_response',
            httpStatus: response.status,
            finalUrl: response.url || url,
            contentRange, contentLength,
        };
    }

    return {
        bytes: new Uint8Array(buffer),
        httpStatus: response.status,
        redirected: response.redirected === true,
        finalUrl: response.url || url,
        contentRange, contentLength, acceptRanges,
    };
}

/**
 * Extract the logical total file size from Content-Range or Content-Length headers
 * returned by fetchHfPrefixBytes.
 */
export function parseTotalBytesFromFetchHeaders(fetchResult) {
    const rangeHeader = String(fetchResult?.contentRange || '').trim();
    if (rangeHeader) {
        const slash = rangeHeader.lastIndexOf('/');
        if (slash >= 0) {
            const part = rangeHeader.slice(slash + 1).trim();
            if (part !== '*') {
                const n = Number(part);
                if (Number.isFinite(n) && n > 0) return Math.trunc(n);
            }
        }
    }
    const lenHeader = String(fetchResult?.contentLength || '').trim();
    if (lenHeader) {
        const n = Number(lenHeader);
        if (Number.isFinite(n) && n > 0) return Math.trunc(n);
    }
    return 0;
}
