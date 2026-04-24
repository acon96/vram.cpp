/** Format bytes as a human-readable MiB/GiB string. */
export function formatMiB(mib) {
    if (mib == null) return '—';
    if (mib >= 1024) {
        return `${(mib / 1024).toFixed(2)} GiB`;
    }
    return `${mib} MiB`;
}

/** Format bytes as MiB (integer) */
export function bytesToMiB(bytes) {
    return Math.round(bytes / (1024 * 1024));
}

/** Format bytes as GiB (2 decimal places) */
export function bytesToGiB(bytes) {
    return (bytes / (1024 * 1024 * 1024)).toFixed(2);
}

/** Parse a GiB string input into bytes */
export function giBToBytes(gib) {
    const n = parseFloat(gib);
    if (!isFinite(n) || n < 0) return 0;
    return Math.round(n * 1024 * 1024 * 1024);
}

/** Display a percentage (0-100) */
export function formatPct(used, total) {
    if (!total) return '';
    return `${Math.round((used / total) * 100)}%`;
}
