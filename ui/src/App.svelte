<script>
    import FileUpload from './components/FileUpload.svelte';
    import ParamPanel from './components/ParamPanel.svelte';
    import ResultsTable from './components/ResultsTable.svelte';
    import { initPredictor } from './lib/predictor.js';
    import { giBToBytes } from './lib/format.js';

    // ── WASM paths ────────────────────────────────────────────────────────────
    // VITE_WASM_BASE_URL accepts a full URL (for cross-port local assets)
    // or a relative path (for static GitHub Pages deployments).
    const configuredWasmBase = import.meta.env.VITE_WASM_BASE_URL ?? './wasm/';
    const wasmBaseUrl = new URL(configuredWasmBase, window.location.href);
    const wasmJsUrl = new URL('vram_predictor_wasm.js', wasmBaseUrl).toString();
    const browserHelperUrl = new URL('vram_predictor_browser.js', wasmBaseUrl).toString();

    const wasmDebugEnabled = import.meta.env.VITE_DEBUG_WASM === '1' || import.meta.env.DEV;
    const wasmFitLogsEnabled = import.meta.env.VITE_DEBUG_WASM_FIT_LOGS === '1';
    globalThis.__VRAM_DEBUG__ = wasmDebugEnabled;

    function debugLog(event, payload) {
        if (!wasmDebugEnabled) return;
        console.log(`[vram-ui] ${event}`, payload);
    }

    function debugError(event, payload) {
        if (!wasmDebugEnabled) return;
        console.error(`[vram-ui] ${event}`, payload);
    }

    function describeFitStatus(statusCode) {
        if (statusCode === 0) {
            return 'fit_success';
        }
        if (statusCode === 1) {
            return 'fit_failure_no_viable_allocation';
        }
        if (statusCode === 2) {
            return 'fit_error_hard_failure';
        }
        return `fit_status_unknown_${statusCode}`;
    }

    debugLog('config', {
        wasmDebugEnabled,
        wasmFitLogsEnabled,
        configuredWasmBase,
        wasmJsUrl,
        browserHelperUrl,
    });

    // ── State ─────────────────────────────────────────────────────────────────
    let selectedFile = $state(null);

    let params = $state({
        nCtx: 4096,
        nBatch: 2048,
        nUbatch: 512,
        cacheTypeK: 'f16',
        cacheTypeV: 'f16',
        nGpuLayers: 0,
        hostRamGiB: 32,
        gpus: [],
        fitTargetMiB: 512,
        targetFreeMiB: 2048,
    });

    let status = $state('idle'); // 'idle' | 'loading-wasm' | 'running' | 'done' | 'error'
    let errorMsg = $state('');
    let result = $state(null);

    // ── Handlers ──────────────────────────────────────────────────────────────
    function handleFile(file) {
        debugLog('handleFile', {
            name: file?.name,
            sizeBytes: file?.size,
            type: file?.type,
        });
        selectedFile = file;
        result = null;
        status = 'idle';
        errorMsg = '';
    }

    function handleParamsChange(p) {
        debugLog('handleParamsChange', p);
        params = p;
    }

    async function runPrediction() {
        if (!selectedFile) {
            errorMsg = 'Please select a GGUF file first.';
            status = 'error';
            return;
        }

        status = 'loading-wasm';
        errorMsg = '';
        result = null;

        let client;
        try {
            debugLog('runPrediction.initPredictor.start', {
                wasmJsUrl,
                browserHelperUrl,
            });

            client = await initPredictor({ wasmJsUrl, browserHelperUrl });

            try {
                const systemInfo = client.getSystemInfo();
                debugLog('runPrediction.systemInfo', systemInfo);
            } catch (systemInfoError) {
                debugError('runPrediction.systemInfo.error', { systemInfoError });
            }
        } catch (err) {
            status = 'error';
            errorMsg = `Failed to load WASM module: ${err.message}`;
            debugError('runPrediction.initPredictor.error', { err });
            return;
        }

        status = 'running';
        try {
            const mountStartedAt = performance.now();
            const virtualPath = await client.mountBrowserFile(selectedFile);
            const mountElapsedMs = Math.round((performance.now() - mountStartedAt) * 100) / 100;

            debugLog('runPrediction.mount.complete', {
                virtualPath,
                mountElapsedMs,
            });

            const gpus = params.gpus.map((g, i) => ({
                id: `gpu${i}`,
                free_bytes: giBToBytes(g.freeGiB),
                total_bytes: giBToBytes(g.totalGiB),
            }));

            const fitTargets = params.gpus.length > 0
                ? params.gpus.map(() => params.fitTargetMiB)
                : [params.fitTargetMiB];
            const freeTargets = params.gpus.length > 0
                ? params.gpus.map(() => params.targetFreeMiB)
                : [params.targetFreeMiB];

            const predictInput = {
                modelPath: virtualPath,
                hostRamBytes: giBToBytes(params.hostRamGiB),
                fitTargetMiB: fitTargets,
                targetFreeMiB: freeTargets,
                gpus,
                nCtx: params.nCtx,
                nBatch: params.nBatch,
                nUbatch: params.nUbatch,
                nGpuLayers: params.nGpuLayers,
                cacheTypeK: params.cacheTypeK,
                cacheTypeV: params.cacheTypeV,
                minCtx: 512,
                // Keep llama/common fit logs opt-in only in wasm. Some logging
                // paths can try to spawn threads that are unavailable.
                showFitLogs: wasmFitLogsEnabled,
            };

            debugLog('runPrediction.predictMountedFit.input', predictInput);

            const predictStartedAt = performance.now();
            let res = client.predictMountedFit(predictInput);
            let predictElapsedMs = Math.round((performance.now() - predictStartedAt) * 100) / 100;

            const hasThreadCtorError = res?.ok === false
                && typeof res?.error === 'string'
                && res.error.includes('thread constructor failed: Not supported');

            if (hasThreadCtorError && predictInput.showFitLogs) {
                debugError('runPrediction.predictMountedFit.threadLoggingRetry', {
                    firstResponse: res,
                    retryWithShowFitLogs: false,
                });

                const retryStartedAt = performance.now();
                res = client.predictMountedFit({
                    ...predictInput,
                    showFitLogs: false,
                });
                const retryElapsedMs = Math.round((performance.now() - retryStartedAt) * 100) / 100;
                predictElapsedMs = Math.round((predictElapsedMs + retryElapsedMs) * 100) / 100;

                debugLog('runPrediction.predictMountedFit.retryOutput', {
                    retryElapsedMs,
                    response: res,
                });
            }

            debugLog('runPrediction.predictMountedFit.output', {
                predictElapsedMs,
                response: res,
                fitStatusText: describeFitStatus(res?.fit?.status),
            });

            result = res;
            if (res?.ok === false) {
                status = 'error';
                const fitStatusText = describeFitStatus(res?.fit?.status);
                const fitWarnings = Array.isArray(res?.fit?.warnings) && res.fit.warnings.length > 0
                    ? ` warnings=${res.fit.warnings.join(',')}`
                    : '';
                const backendError = typeof res?.error === 'string' ? ` backendError=${res.error}` : '';
                const deviceCountInResponse = Array.isArray(res?.fit?.breakdown?.devices)
                    ? res.fit.breakdown.devices.length
                    : 0;
                const sentGpuOverrides = Array.isArray(predictInput.gpus)
                    ? predictInput.gpus.length
                    : 0;

                let hint = '';
                if (res?.fit?.status === 2 && sentGpuOverrides > 0 && deviceCountInResponse === 0) {
                    hint = ' Hint: this may be a GPU override/device-count mismatch on the wasm backend. Try removing GPU rows and retrying.';
                }
                if (typeof res?.error === 'string' && res.error.includes('thread constructor failed: Not supported')) {
                    hint += ' Hint: backend fit logs are trying to use a thread path unsupported by this wasm build. Keep VITE_DEBUG_WASM_FIT_LOGS unset.';
                }
                if (typeof res?.error === 'string' && res.error.includes('failed_to_create_fitted_context')) {
                    hint += ' Hint: fit projection succeeded, but creating the actual fitted context failed in wasm (likely runtime heap allocation limits and/or batch buffer size). Try a lower n_ctx.';
                }

                errorMsg = `WASM fit failed (${fitStatusText}).${fitWarnings}${backendError}${hint}`;
                debugError('runPrediction.predictMountedFit.failed', {
                    fitStatus: res?.fit?.status,
                    fitStatusText,
                    warnings: res?.fit?.warnings,
                    sentGpuOverrides,
                    deviceCountInResponse,
                    response: res,
                });
            } else {
                status = 'done';
            }

            // Clean up the mounted file from WASM virtual FS
            try {
                client.unmountFile(virtualPath);
            } catch (unmountError) {
                debugError('runPrediction.unmount.error', {
                    virtualPath,
                    unmountError,
                });
            }
        } catch (err) {
            status = 'error';
            errorMsg = err.message ?? String(err);
            debugError('runPrediction.error', { err });
        }
    }

    const isRunning = $derived(status === 'loading-wasm' || status === 'running');
    const statusLabel = $derived({
        idle: '',
        'loading-wasm': 'Loading WASM module…',
        running: 'Running fit prediction…',
        done: '',
        error: '',
    }[status] ?? '');
</script>

<div class="app-shell">
    <header class="app-header">
        <div class="header-inner">
            <div class="logo-row">
                <span class="logo-mark">⬛</span>
                <span class="logo-text">vram.cpp</span>
            </div>
            <p class="tagline">Predict LLM VRAM usage using llama.cpp fit logic</p>
        </div>
    </header>

    <main class="app-main">
        <!-- Left: Config panel -->
        <aside class="config-panel">
            <section class="panel-section">
                <h2 class="panel-title">Model</h2>
                <FileUpload onfile={handleFile} />
            </section>

            <section class="panel-section">
                <h2 class="panel-title">Parameters</h2>
                <ParamPanel {params} onchange={handleParamsChange} />
            </section>

            <div class="action-row">
                <button
                    class="run-btn"
                    type="button"
                    onclick={runPrediction}
                    disabled={isRunning || !selectedFile}
                >
                    {#if isRunning}
                        <span class="spinner" aria-hidden="true"></span>
                        {statusLabel}
                    {:else}
                        ▶ Predict VRAM
                    {/if}
                </button>

                {#if status === 'error'}
                    <p class="error-msg">{errorMsg}</p>
                {/if}
            </div>
        </aside>

        <!-- Right: Results panel -->
        <section class="results-panel">
            <h2 class="panel-title">
                Memory Breakdown
                {#if status === 'done'}
                    <span class="ok-badge">✓</span>
                {/if}
            </h2>

            {#if isRunning}
                <div class="loading-state">
                    <span class="spinner lg" aria-hidden="true"></span>
                    <p>{statusLabel}</p>
                </div>
            {:else}
                <ResultsTable {result} />
            {/if}
        </section>
    </main>

    <footer class="app-footer">
        <span>
            Powered by <a href="https://github.com/ggml-org/llama.cpp" target="_blank" rel="noreferrer">llama.cpp</a>
            fit logic compiled to WebAssembly.
        </span>
    </footer>
</div>

<style>
    .app-shell {
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        background: var(--bg);
    }

    /* ── Header ── */
    .app-header {
        background: var(--header-bg);
        border-bottom: 1px solid var(--border);
        padding: 0 24px;
    }

    .header-inner {
        max-width: 1280px;
        margin: 0 auto;
        padding: 16px 0;
        display: flex;
        align-items: baseline;
        gap: 20px;
        flex-wrap: wrap;
    }

    .logo-row {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .logo-mark {
        font-size: 1.4rem;
    }

    .logo-text {
        font-size: 1.3rem;
        font-weight: 700;
        font-family: var(--mono);
        color: var(--text-primary);
    }

    .tagline {
        margin: 0;
        font-size: 0.88rem;
        color: var(--text-muted);
    }

    /* ── Main layout ── */
    .app-main {
        flex: 1;
        max-width: 1280px;
        width: 100%;
        margin: 0 auto;
        padding: 24px;
        display: grid;
        grid-template-columns: 360px 1fr;
        gap: 24px;
        box-sizing: border-box;
        align-items: start;
    }

    @media (max-width: 900px) {
        .app-main {
            grid-template-columns: 1fr;
        }
    }

    /* ── Config panel ── */
    .config-panel {
        display: flex;
        flex-direction: column;
        gap: 20px;
        position: sticky;
        top: 24px;
    }

    @media (max-width: 900px) {
        .config-panel {
            position: static;
        }
    }

    .panel-section {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 18px;
        display: flex;
        flex-direction: column;
        gap: 14px;
    }

    .panel-title {
        margin: 0;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Results panel ── */
    .results-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 18px;
        display: flex;
        flex-direction: column;
        gap: 16px;
        min-height: 320px;
    }

    /* ── Action row ── */
    .action-row {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .run-btn {
        width: 100%;
        padding: 13px;
        background: var(--accent);
        color: var(--on-accent);
        border: none;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        transition: opacity 0.2s, transform 0.1s;
    }

    .run-btn:hover:not(:disabled) {
        opacity: 0.88;
    }

    .run-btn:active:not(:disabled) {
        transform: scale(0.98);
    }

    .run-btn:disabled {
        opacity: 0.45;
        cursor: not-allowed;
    }

    .error-msg {
        margin: 0;
        font-size: 0.85rem;
        color: var(--error);
        padding: 8px 12px;
        background: var(--error-bg);
        border-radius: 6px;
    }

    /* ── Loading state ── */
    .loading-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        flex: 1;
        gap: 14px;
        color: var(--text-muted);
        padding: 48px 0;
    }

    .loading-state p {
        margin: 0;
        font-size: 0.9rem;
    }

    /* ── Spinner ── */
    .spinner {
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid transparent;
        border-top-color: currentColor;
        border-right-color: currentColor;
        border-radius: 50%;
        animation: spin 0.7s linear infinite;
    }

    .spinner.lg {
        width: 28px;
        height: 28px;
        border-width: 3px;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .ok-badge {
        display: inline-flex;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: var(--accent);
        color: var(--on-accent);
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
    }

    /* ── Footer ── */
    .app-footer {
        border-top: 1px solid var(--border);
        padding: 14px 24px;
        text-align: center;
        font-size: 0.8rem;
        color: var(--text-muted);
    }

    .app-footer a {
        color: var(--accent);
        text-decoration: none;
    }

    .app-footer a:hover {
        text-decoration: underline;
    }
</style>
