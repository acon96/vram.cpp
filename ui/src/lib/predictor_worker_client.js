let workerClientPromise = null;

function toAbsoluteUrl(urlLike) {
    return new URL(urlLike, window.location.href).toString();
}

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

        async predictMountedFit(file, options) {
            if (file == null || typeof file.arrayBuffer !== 'function') {
                throw new Error('predictMountedFit requires a browser File or Blob-like object');
            }

            const fileBuffer = await file.arrayBuffer();
            return sendRequest('predict', {
                fileName: file?.name || 'model.gguf',
                fileBuffer,
                options,
            }, [fileBuffer]);
        },

        terminate() {
            rejectAllPending('predictor_worker_terminated');
            worker.terminate();
        },
    };
}

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
