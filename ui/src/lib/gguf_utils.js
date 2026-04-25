/**
 * GGUF binary prefix parser — minimal reader to extract context_length
 * from the KV section of a partial GGUF file fetch.
 */

const T_UINT8   = 0;
const T_INT8    = 1;
const T_UINT16  = 2;
const T_INT16   = 3;
const T_UINT32  = 4;
const T_INT32   = 5;
const T_FLOAT32 = 6;
const T_BOOL    = 7;
const T_STRING  = 8;
const T_ARRAY   = 9;
const T_UINT64  = 10;
const T_INT64   = 11;
const T_FLOAT64 = 12;

function scalarByteSize(t) {
    if (t === T_UINT8  || t === T_INT8  || t === T_BOOL)    return 1;
    if (t === T_UINT16 || t === T_INT16)                     return 2;
    if (t === T_UINT32 || t === T_INT32 || t === T_FLOAT32)  return 4;
    if (t === T_UINT64 || t === T_INT64 || t === T_FLOAT64)  return 8;
    return 0;
}

function dv(bytes) {
    return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function readU32(bytes, pos) {
    return pos + 4 <= bytes.length ? dv(bytes).getUint32(pos, true) : null;
}

function readI32(bytes, pos) {
    return pos + 4 <= bytes.length ? dv(bytes).getInt32(pos, true) : null;
}

function readU64(bytes, pos) {
    if (pos + 8 > bytes.length) return null;
    const d = dv(bytes);
    const lo = d.getUint32(pos, true);
    const hi = d.getUint32(pos + 4, true);
    const v = hi * 4294967296 + lo;
    return Number.isSafeInteger(v) ? v : null;
}

function readI64(bytes, pos) {
    if (pos + 8 > bytes.length) return null;
    const d = dv(bytes);
    const lo = d.getUint32(pos, true);
    const hi = d.getInt32(pos + 4, true);
    const v = hi * 4294967296 + lo;
    return Number.isSafeInteger(v) ? v : null;
}

function readF32(bytes, pos) {
    return pos + 4 <= bytes.length ? dv(bytes).getFloat32(pos, true) : null;
}

function readF64(bytes, pos) {
    return pos + 8 <= bytes.length ? dv(bytes).getFloat64(pos, true) : null;
}

function readString(bytes, pos) {
    const len = readU64(bytes, pos);
    if (len == null) return null;
    const start = pos + 8;
    const end = start + len;
    if (end > bytes.length) return null;
    return { value: new TextDecoder().decode(bytes.subarray(start, end)), next: end };
}

function skipValue(bytes, pos, t) {
    if (t === T_STRING) {
        const s = readString(bytes, pos);
        return s == null ? null : s.next;
    }
    if (t === T_ARRAY) {
        const elemType = readU32(bytes, pos);
        const count    = readU64(bytes, pos + 4);
        if (elemType == null || count == null || elemType === T_ARRAY) return null;
        let cur = pos + 12;
        if (elemType === T_STRING) {
            for (let i = 0; i < count; i++) {
                const s = readString(bytes, cur);
                if (s == null) return null;
                cur = s.next;
            }
            return cur;
        }
        const sz = scalarByteSize(elemType);
        if (sz <= 0) return null;
        const total = sz * count;
        if (!Number.isSafeInteger(total)) return null;
        const end = cur + total;
        return end <= bytes.length ? end : null;
    }
    const sz = scalarByteSize(t);
    if (sz <= 0) return null;
    const end = pos + sz;
    return end <= bytes.length ? end : null;
}

function readNumeric(bytes, pos, t) {
    const d = () => dv(bytes);
    if (t === T_UINT8)   return bytes[pos];
    if (t === T_INT8)    return (bytes[pos] << 24) >> 24;
    if (t === T_UINT16)  return d().getUint16(pos, true);
    if (t === T_INT16)   return d().getInt16(pos, true);
    if (t === T_UINT32)  return readU32(bytes, pos);
    if (t === T_INT32)   return readI32(bytes, pos);
    if (t === T_FLOAT32) return readF32(bytes, pos);
    if (t === T_BOOL)    return bytes[pos] !== 0 ? 1 : 0;
    if (t === T_UINT64)  return readU64(bytes, pos);
    if (t === T_INT64)   return readI64(bytes, pos);
    if (t === T_FLOAT64) return readF64(bytes, pos);
    return null;
}

/**
 * Scan the GGUF KV section of a partial prefix buffer and return the
 * model's native context_length, or 0 if it cannot be determined.
 */
export function extractContextLengthFromPrefix(prefixBytes) {
    const bytes = prefixBytes instanceof Uint8Array ? prefixBytes : new Uint8Array(prefixBytes || []);
    if (bytes.length < 24) return 0;
    if (bytes[0] !== 0x47 || bytes[1] !== 0x47 || bytes[2] !== 0x55 || bytes[3] !== 0x46) return 0;

    let pos = 4;
    const version = readU32(bytes, pos);
    if (version == null || version < 2 || version > 3) return 0;
    pos += 4;

    const kvCount = readU64(bytes, pos + 8);
    if (kvCount == null) return 0;
    pos += 16;

    let architecture = '';
    const candidates = [];

    for (let i = 0; i < kvCount; i++) {
        const key = readString(bytes, pos);
        if (key == null) return 0;
        pos = key.next;

        const t = readU32(bytes, pos);
        if (t == null) return 0;
        pos += 4;

        if (key.value === 'general.architecture' && t === T_STRING) {
            const s = readString(bytes, pos);
            if (s == null) return 0;
            architecture = s.value;
            pos = s.next;
            continue;
        }

        if (key.value === 'context_length' || key.value.endsWith('.context_length')) {
            const num  = readNumeric(bytes, pos, t);
            const next = skipValue(bytes, pos, t);
            if (next == null) return 0;
            pos = next;
            if (num != null && Number.isFinite(Number(num))) {
                const ctx = Math.trunc(Number(num));
                if (ctx > 0) candidates.push({ key: key.value, value: ctx });
            }
            continue;
        }

        const next = skipValue(bytes, pos, t);
        if (next == null) return 0;
        pos = next;
    }

    if (candidates.length === 0) return 0;
    if (architecture) {
        const exact = candidates.find((c) => c.key === `${architecture}.context_length`);
        if (exact) return exact.value;
    }
    return candidates[0].value;
}

/**
 * Scan the GGUF KV section and return split metadata.
 * Returns { splitCount, splitNo } where both default to 0 if not present.
 */
export function parseGgufSplitInfo(prefixBytes) {
    const bytes = prefixBytes instanceof Uint8Array ? prefixBytes : new Uint8Array(prefixBytes || []);
    const result = { splitCount: 0, splitNo: 0 };
    if (bytes.length < 24) return result;
    if (bytes[0] !== 0x47 || bytes[1] !== 0x47 || bytes[2] !== 0x55 || bytes[3] !== 0x46) return result;

    let pos = 4;
    const version = readU32(bytes, pos);
    if (version == null || version < 2 || version > 3) return result;
    pos += 4;

    const kvCount = readU64(bytes, pos + 8);
    if (kvCount == null) return result;
    pos += 16;

    let found = 0;
    for (let i = 0; i < kvCount && found < 2; i++) {
        const key = readString(bytes, pos);
        if (key == null) return result;
        pos = key.next;

        const t = readU32(bytes, pos);
        if (t == null) return result;
        pos += 4;

        if (key.value === 'split.count' || key.value === 'split.no') {
            const num = readNumeric(bytes, pos, t);
            const next = skipValue(bytes, pos, t);
            if (next == null) return result;
            pos = next;
            if (num != null && Number.isFinite(Number(num))) {
                if (key.value === 'split.count') { result.splitCount = Math.trunc(Number(num)); found++; }
                if (key.value === 'split.no')    { result.splitNo    = Math.trunc(Number(num)); found++; }
            }
            continue;
        }

        const next = skipValue(bytes, pos, t);
        if (next == null) return result;
        pos = next;
    }

    return result;
}
