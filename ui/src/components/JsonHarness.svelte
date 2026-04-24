<script>
    import { initPredictorWorker } from '../lib/predictor_worker_client.js';

    const defaultRequest = JSON.stringify({
        mode: 'fit',
        model: {
            source: 'local',
            path: '__MOUNTED_MODEL__',
        },
        runtime: {
            n_ctx: 131072,
            n_batch: 2048,
            n_ubatch: 512,
            n_gpu_layers: -1,
            cache_type_k: 'f16',
            cache_type_v: 'f16',
        },
        device: {
            host_ram_bytes: 274877906944,
            fit_target_mib: [512],
            target_free_mib: [2048],
            gpus: [
                {
                    id: 'gpu0',
                    name: 'GPU 0',
                    backend: 'cuda',
                    free_bytes: 85899345920,
                    total_bytes: 85899345920,
                },
            ],
        },
        fit: {
            min_ctx: 512,
            execute_in_process: true,
            show_fit_logs: false,
        },
    }, null, 2);

    let {
        wasmJsUrl,
        debugEnabled = false,
    } = $props();

    let selectedFile = $state(null);
    let requestJson = $state(defaultRequest);
    let responseJson = $state('');
    let status = $state('idle');
    let errorMsg = $state('');

    function handleFileChange(event) {
        const [file] = event.currentTarget.files ?? [];
        selectedFile = file ?? null;
    }

    async function runJsonHarness() {
        errorMsg = '';
        responseJson = '';
        status = 'loading';

        let request;
        try {
            request = JSON.parse(requestJson);
        } catch (error) {
            status = 'error';
            errorMsg = `Invalid JSON: ${error.message}`;
            return;
        }

        if (request?.model?.source === 'local' && selectedFile == null) {
            status = 'error';
            errorMsg = 'Select a local GGUF file before submitting a local-mode request.';
            return;
        }

        try {
            const client = await initPredictorWorker({
                wasmJsUrl,
                debugEnabled,
            });

            const response = await client.predictMountedJson(selectedFile, request);
            responseJson = JSON.stringify(response, null, 2);
            status = response?.ok === false ? 'error' : 'done';
            if (response?.ok === false) {
                errorMsg = response?.error ?? 'Prediction returned ok=false';
            }
        } catch (error) {
            status = 'error';
            errorMsg = error?.message ?? String(error);
        }
    }
</script>

<section class="harness-shell">
    <header class="harness-header">
        <div>
            <p class="eyebrow">Browser Harness</p>
            <h1>Raw JSON fit runner</h1>
        </div>
        <p class="hint">Upload a GGUF once, paste a request body, and submit it directly through the WASM worker. For local mode, the uploaded file path replaces <code>model.path</code> during execution.</p>
    </header>

    <div class="harness-grid">
        <section class="card">
            <label class="field-label" for="harness-file">Model file</label>
            <input id="harness-file" type="file" accept=".gguf,application/octet-stream" onchange={handleFileChange} />
            {#if selectedFile}
                <p class="file-meta">{selectedFile.name} · {selectedFile.size.toLocaleString()} bytes</p>
            {/if}

            <label class="field-label" for="request-json">Request JSON</label>
            <textarea id="request-json" bind:value={requestJson} spellcheck="false"></textarea>

            <button class="submit-btn" type="button" onclick={runJsonHarness} disabled={status === 'loading'}>
                {status === 'loading' ? 'Running…' : 'Submit JSON'}
            </button>

            {#if errorMsg}
                <p class="error-msg">{errorMsg}</p>
            {/if}
        </section>

        <section class="card">
            <div class="response-header">
                <label class="field-label" for="response-json">Response</label>
                <span class:status-ok={status === 'done'} class:status-error={status === 'error'}>{status}</span>
            </div>
            <textarea id="response-json" readonly value={responseJson}></textarea>
        </section>
    </div>
</section>

<style>
    .harness-shell {
        display: grid;
        gap: 20px;
    }

    .harness-header {
        display: grid;
        gap: 10px;
        padding: 24px;
        border: 1px solid var(--border);
        border-radius: 20px;
        background: rgba(255, 250, 244, 0.9);
    }

    .eyebrow {
        margin: 0;
        font-size: 0.8rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
    }

    h1 {
        margin: 0;
        font-size: clamp(1.8rem, 3vw, 2.5rem);
    }

    .hint {
        margin: 0;
        color: var(--muted);
        line-height: 1.5;
    }

    .harness-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 20px;
    }

    .card {
        display: grid;
        gap: 12px;
        padding: 20px;
        border: 1px solid var(--border);
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.8);
        min-height: 0;
    }

    .field-label {
        font-size: 0.9rem;
        font-weight: 700;
    }

    textarea {
        min-height: 420px;
        width: 100%;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px;
        font: 0.9rem/1.5 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
        resize: vertical;
        background: rgba(248, 244, 238, 0.95);
        color: var(--text);
    }

    input[type='file'] {
        padding: 10px;
        border: 1px dashed var(--border);
        border-radius: 14px;
        background: rgba(248, 244, 238, 0.95);
    }

    .file-meta {
        margin: 0;
        color: var(--muted);
        font-size: 0.9rem;
    }

    .submit-btn {
        justify-self: start;
        border: 0;
        border-radius: 999px;
        padding: 12px 20px;
        background: #123524;
        color: #fffaf4;
        font-weight: 700;
        cursor: pointer;
    }

    .submit-btn:disabled {
        opacity: 0.7;
        cursor: progress;
    }

    .response-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
    }

    .status-ok,
    .status-error {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .status-ok {
        color: #1f7a44;
    }

    .status-error {
        color: #a33030;
    }

    .error-msg {
        margin: 0;
        color: #a33030;
    }

    @media (max-width: 960px) {
        .harness-grid {
            grid-template-columns: 1fr;
        }

        textarea {
            min-height: 300px;
        }
    }
</style>