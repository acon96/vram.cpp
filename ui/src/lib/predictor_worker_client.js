/** @typedef {{ wasmJsUrl: string, mountRoot: string }} WorkerConfig */

/** @type {Promise<ReturnType<typeof createWorkerClient>> | null} */
let workerClientPromise = null;

/** @param {string} urlLike */
function toAbsoluteUrl(urlLike) {
    return new URL(urlLike, window.location.href).toString();
}

/** @param {WorkerConfig} config
 *  @param {boolean} debugEnabled */
function createWorkerClient(config, debugEnabled) {
    const worker = new Worker(new URL('./predictor_fit_worker.js', import.meta.url), {
        type: 'classic',
    });

    worker.postMessage({
        type: 'set-debug',
        enabled: debugEnabled,
    });

    let nextJobId = 1;
    const pending = new Map();

    /** @param {string} message */
    const rejectAllPending = (message) => {
        for (const { reject } of pending.values()) {
            reject(new Error(message));
        }
        pending.clear();
    };

    worker.onmessage = (event) => {
        const message = event.data || {};
        if (message.type !== 'result' && message.type !== 'error') {
            return;
        }

        const pendingJob = pending.get(message.jobId);
        if (!pendingJob) {
            return;
        }

        pending.delete(message.jobId);

        if (message.type === 'error') {
            pendingJob.reject(new Error(message.error || 'worker_job_failed'));
            return;
        }

        pendingJob.resolve(message.result);
    };

    worker.onerror = (event) => {
        rejectAllPending(`predictor_worker_error: ${event.message || 'unknown_error'}`);
    };

    /**
     * @param {string} type
     * @param {Record<string, unknown>} payload
     * @param {Transferable[]} [transferList]
     */
    const sendRequest = (type, payload, transferList = []) => {
        const jobId = nextJobId++;
        return new Promise((resolve, reject) => {
            pending.set(jobId, { resolve, reject });
            worker.postMessage({
                type,
                jobId,
                config,
                ...payload,
            }, transferList);
        });
    };

    return {
        async getSystemInfo() {
            return sendRequest('get-system-info', {});
        },

        /** @param {File|Blob} file
         *  @param {Record<string, unknown>} options */
        async predictMountedFit(file, options) {
            if (file == null || typeof file.arrayBuffer !== 'function') {
                throw new Error('predictMountedFit requires a browser File or Blob-like object');
            }

            const fileBuffer = await file.arrayBuffer();
            const fileName = 'name' in file && typeof file.name === 'string'
                ? file.name
                : 'model.gguf';
            return sendRequest('predict', {
                fileName,
                fileBuffer,
                options,
            }, [fileBuffer]);
        },

        /** @param {File|Blob|null} file
         *  @param {Record<string, unknown>} request */
        async predictMountedJson(file, request) {
            let fileBuffer;
            let fileName = 'model.gguf';
            const transferList = [];

            if (file != null) {
                if (typeof file.arrayBuffer !== 'function') {
                    throw new Error('predictMountedJson requires a browser File or Blob-like object when a file is provided');
                }

                fileBuffer = await file.arrayBuffer();
                fileName = 'name' in file && typeof file.name === 'string'
                    ? file.name
                    : fileName;
                transferList.push(fileBuffer);
            }

            return sendRequest('predict-json', {
                fileName,
                fileBuffer,
                request,
            }, transferList);
        },

        terminate() {
            rejectAllPending('predictor_worker_terminated');
            worker.terminate();
        },
    };
}

/**
 * @param {{ wasmJsUrl: string, mountRoot?: string, debugEnabled?: boolean }} options
 */
export async function initPredictorWorker({ wasmJsUrl, mountRoot = '/models', debugEnabled = false }) {
    if (workerClientPromise) {
        return workerClientPromise;
    }

    workerClientPromise = Promise.resolve(
        createWorkerClient({
            wasmJsUrl: toAbsoluteUrl(wasmJsUrl),
            mountRoot,
        }, debugEnabled)
    );

    return workerClientPromise;
}

export function resetPredictorWorker() {
    workerClientPromise = null;
}
