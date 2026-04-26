import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const thisFile = fileURLToPath(import.meta.url);
const thisDir = path.dirname(thisFile);
const repoRoot = path.resolve(thisDir, '..', '..');

const host = process.env.WASM_ASSETS_HOST ?? '127.0.0.1';
const port = Number.parseInt(process.env.WASM_ASSETS_PORT ?? '8123', 10);
const mountPath = normalizeMountPath(process.env.WASM_ASSETS_PATH ?? '/assets');

const wasmBuildDir = path.resolve(repoRoot, process.env.WASM_BUILD_DIR ?? 'build-wasm');

if (!Number.isFinite(port) || port <= 0 || port > 65535) {
    throw new Error(`Invalid WASM_ASSETS_PORT value: ${process.env.WASM_ASSETS_PORT ?? '(unset)'}`);
}

const routeToFile = {
    [`${mountPath}/vram_predictor.js`]: path.join(wasmBuildDir, 'vram_predictor.js'),
    [`${mountPath}/vram_predictor.wasm`]: path.join(wasmBuildDir, 'vram_predictor.wasm'),
};

function normalizeMountPath(inputPath) {
    const prefixed = inputPath.startsWith('/') ? inputPath : `/${inputPath}`;
    const withoutTrailingSlash = prefixed.replace(/\/+$/, '');
    return withoutTrailingSlash.length > 0 ? withoutTrailingSlash : '/assets';
}

function setCorsHeaders(response) {
    response.setHeader('Access-Control-Allow-Origin', '*');
    response.setHeader('Access-Control-Allow-Methods', 'GET,HEAD,OPTIONS');
    response.setHeader('Access-Control-Allow-Headers', 'Content-Type, Origin, Accept');
}

function mimeTypeFor(filePath) {
    if (filePath.endsWith('.js')) {
        return 'text/javascript; charset=utf-8';
    }
    if (filePath.endsWith('.wasm')) {
        return 'application/wasm';
    }
    return 'application/octet-stream';
}

function writeTextResponse(response, statusCode, message) {
    response.writeHead(statusCode, {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'no-store',
    });
    response.end(message);
}

const server = createServer(async (request, response) => {
    setCorsHeaders(response);

    if (!request.url) {
        writeTextResponse(response, 400, 'Missing request URL');
        return;
    }

    if (request.method === 'OPTIONS') {
        response.writeHead(204);
        response.end();
        return;
    }

    if (request.method !== 'GET' && request.method !== 'HEAD') {
        writeTextResponse(response, 405, `Method not allowed: ${request.method}`);
        return;
    }

    const requestUrl = new URL(request.url, `http://${host}:${port}`);
    const filePath = routeToFile[requestUrl.pathname];

    if (!filePath) {
        writeTextResponse(response, 404, `Unknown asset path: ${requestUrl.pathname}`);
        return;
    }

    try {
        const fileInfo = await stat(filePath);
        response.writeHead(200, {
            'Content-Type': mimeTypeFor(filePath),
            'Content-Length': fileInfo.size,
            'Cache-Control': 'no-store',
        });

        if (request.method === 'HEAD') {
            response.end();
            return;
        }

        createReadStream(filePath).pipe(response);
    } catch (_error) {
        const relativePath = path.relative(repoRoot, filePath);
        writeTextResponse(response, 404, `Asset not found. Build artifacts expected at: ${relativePath}`);
    }
});

server.listen(port, host, () => {
    const baseUrl = `http://${host}:${port}${mountPath}`;
    console.log(`[assets] serving VRAM wasm assets at ${baseUrl}`);
    console.log(`[assets] wasm build dir: ${path.relative(repoRoot, wasmBuildDir)}`);
});

process.on('SIGINT', () => {
    server.close(() => {
        process.exit(0);
    });
});
