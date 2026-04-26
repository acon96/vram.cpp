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

/** Build ordered candidate URLs, preferring explicit resolvedUrl then canonical URL. */
export function buildHfCandidateUrls(selection) {
    const resolvedUrl = String(selection?.resolvedUrl || '').trim();
    const canonicalUrl = buildCanonicalHfFileUrl(selection || {});
    const candidates = [];

    if (resolvedUrl) {
        candidates.push(resolvedUrl);
    }
    if (canonicalUrl && canonicalUrl !== resolvedUrl) {
        candidates.push(canonicalUrl);
    }

    return candidates;
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
 * in steps of stepBytes.  Each entry describes only the NEW bytes to fetch
 * for that step (incremental, not cumulative from 0).
 * The caller is responsible for concatenating received chunks.
 * Returns [{ start, end }, ...].
 */
export function buildHfRangePlan(initialBytes, maxBytes, stepBytes) {
    const plan = [];
    if (!Number.isFinite(maxBytes) || maxBytes <= 0) return plan;
    const first = Number.isFinite(initialBytes) && initialBytes > 0
        ? Math.trunc(initialBytes)
        : Math.min(2 * 1024 * 1024, Math.trunc(maxBytes));
    const step = Number.isFinite(stepBytes) && stepBytes > 0 ? Math.trunc(stepBytes) : first;
    if (first <= 0 || step <= 0) return plan;
    // First step: always start from 0
    plan.push({ start: 0, end: Math.min(first, maxBytes) - 1 });
    // Subsequent steps: fetch only the new chunk
    for (let cur = first + step; cur <= maxBytes; cur += step) {
        plan.push({ start: plan[plan.length - 1].end + 1, end: Math.min(cur, maxBytes) - 1 });
    }
    const lastEnd = plan.length > 0 ? plan[plan.length - 1].end : -1;
    if (lastEnd + 1 < maxBytes) {
        plan.push({ start: lastEnd + 1, end: Math.trunc(maxBytes) - 1 });
    }
    return plan;
}

/**
 * Detect a shard pattern in a filename or URL path component.
 * Returns { prefix, shardNo, totalShards } (1-based shardNo) or null.
 * Example: "model-00001-of-00007.gguf" → { prefix: "model", shardNo: 1, totalShards: 7 }
 */
export function detectShardPattern(filenameOrUrl) {
    const filename = String(filenameOrUrl || '').split('/').filter(Boolean).pop() || '';
    const match = /^(.+)-(\d{5})-of-(\d{5})\.gguf$/i.exec(filename);
    if (!match) return null;
    return {
        prefix: match[1],
        shardNo: parseInt(match[2], 10),
        totalShards: parseInt(match[3], 10),
    };
}

/**
 * Replace the shard number in a URL to target a different shard.
 * Operates on the last path component only.
 */
export function buildShardUrl(primaryUrl, targetShardNo, totalShards) {
    const n = String(targetShardNo).padStart(5, '0');
    const t = String(totalShards).padStart(5, '0');
    return String(primaryUrl).replace(
        /-(\d{5})-of-(\d{5})\.gguf(\?.*)?$/i,
        `-${n}-of-${t}.gguf$3`,
    );
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
