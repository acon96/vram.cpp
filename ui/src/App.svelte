<script>
    import FileUpload from './components/FileUpload.svelte';
    import HardwarePanel from './components/HardwarePanel.svelte';
    import HuggingFaceSearch from './components/HuggingFaceSearch.svelte';
    import JsonHarness from './components/JsonHarness.svelte';
    import ResultsTable from './components/ResultsTable.svelte';
    import RuntimePanel from './components/RuntimePanel.svelte';
    import { initPredictorWorker, resetPredictorWorker } from './lib/predictor_worker_client.js';
    import { giBToBytes } from './lib/format.js';
    import { buildHfSelectionCacheKey } from './lib/hf_utils.js';
    import { runHfMetadataFromBrowser, runHfFitFromBrowser, runFitFromPreparedPrefix } from './lib/hf_fit.js';

    // ── WASM config ───────────────────────────────────────────────────────────
    const configuredWasmBase = import.meta.env.VITE_WASM_BASE_URL ?? './wasm/';
    const wasmBaseUrl        = new URL(configuredWasmBase, window.location.href);
    const wasmJsUrl          = new URL('vram_predictor.js', wasmBaseUrl).toString();
    const wasmDebugEnabled   = import.meta.env.VITE_DEBUG_WASM === '1' || import.meta.env.DEV;
    const wasmFitLogsEnabled = import.meta.env.VITE_DEBUG_WASM_FIT_LOGS === '1';
    const llamaVersion       = import.meta.env.VITE_LLAMA_VERSION ?? 'dev';
    const appView            = new URLSearchParams(window.location.search).get('view') ?? 'app';
    const isJsonHarnessView  = appView === 'harness';

    const hardwareConfigStorageKey = 'vram_cpp_hardware_config_v1';

    globalThis.__VRAM_DEBUG__ = wasmDebugEnabled;

    const logger = {
        log:   (event, payload) => { if (wasmDebugEnabled) console.log(`[vram-ui] ${event}`, payload); },
        error: (event, payload) => { if (wasmDebugEnabled) console.error(`[vram-ui] ${event}`, payload); },
    };

    function describeFitStatus(statusCode) {
        if (statusCode === 0) return 'fit_success';
        if (statusCode === 1) return 'fit_failure_no_viable_allocation';
        if (statusCode === 2) return 'fit_error_hard_failure';
        return `fit_status_unknown_${statusCode}`;
    }

    logger.log('config', { wasmDebugEnabled, wasmFitLogsEnabled, configuredWasmBase, wasmJsUrl });

    function normalizeHardwareGpu(input, fallbackIndex) {
        const totalGiB  = Number.isFinite(Number(input?.totalGiB)) ? Math.max(0.5, Number(input.totalGiB)) : 8;
        const bufferMiB = Number.isFinite(Number(input?.bufferMiB)) ? Math.max(0, Math.trunc(Number(input.bufferMiB))) : 512;
        const backendRaw = typeof input?.backend === 'string' ? input.backend.toLowerCase() : 'cuda';
        const backend = ['cuda', 'metal', 'vulkan', 'generic'].includes(backendRaw) ? backendRaw : 'cuda';
        return {
            name: typeof input?.name === 'string' && input.name.trim().length > 0 ? input.name : `GPU ${fallbackIndex}`,
            totalGiB,
            bufferMiB,
            backend,
        };
    }

    function readStoredHardwareConfig() {
        try {
            const raw = window?.localStorage?.getItem(hardwareConfigStorageKey);
            if (!raw) return {};
            const parsed = JSON.parse(raw);
            if (parsed == null || typeof parsed !== 'object') return {};
            const gpus = Array.isArray(parsed.gpus)
                ? parsed.gpus.slice(0, 6).map((g, i) => normalizeHardwareGpu(g, i))
                : undefined;
            return {
                hostRamGiB: Number.isFinite(Number(parsed.hostRamGiB)) ? Math.max(1, Number(parsed.hostRamGiB)) : undefined,
                gpus,
            };
        } catch { return {}; }
    }

    const defaultParams = {
        nCtx: 4096,
        nBatch: 2048,
        nUbatch: 512,
        cacheTypeK: 'f16',
        cacheTypeV: 'f16',
        nGpuLayers: -1,
        splitMode: 'layer',
        hostRamGiB: 32,
        gpus: [],
    };
    const storedHardwareConfig = readStoredHardwareConfig();

    // ── State ─────────────────────────────────────────────────────────────────
    let modelSource = $state('huggingface');
    let selectedFile = $state(null);
    let hfSelection = $state({
        repo: '',
        file: '',
        fileSizeBytes: 0,
        revision: 'main',
        token: '',
        resolvedUrl: '',
        validated: false,
        response: null,
        metadata: null,
        error: '',
    });
    let hfPreparedFit = $state(null);

    let params = $state({
        ...defaultParams,
        hostRamGiB: storedHardwareConfig.hostRamGiB ?? defaultParams.hostRamGiB,
        gpus: Array.isArray(storedHardwareConfig.gpus)
            ? storedHardwareConfig.gpus
            : defaultParams.gpus,
    });

    let status = $state('idle'); // 'idle' | 'loading-wasm' | 'running' | 'done' | 'error'
    let errorMsg = $state('');
    let result = $state(null);
    let fitProgress = $state({ attempt: 0, nCtx: null, nGpuLayers: null, lastLine: '' });
    let fitLogLines = $state([]);
    let activeWorkerClient = $state(null);

    // ── Handlers ──────────────────────────────────────────────────────────────
    function handleModelSourceChange(nextSource) {
        modelSource = nextSource;
        if (nextSource !== 'huggingface') hfPreparedFit = null;
        result = null; status = 'idle'; errorMsg = '';
    }

    function handleFile(file) {
        logger.log('handleFile', { name: file?.name, size: file?.size });
        hfPreparedFit = null;
        modelSource = 'local';
        selectedFile = file;
        result = null; status = 'idle'; errorMsg = '';
    }

    function handleHuggingFaceSelectionChange(selection) {
        const nextCacheKey = buildHfSelectionCacheKey(selection);
        if (selection?.validated !== true || hfPreparedFit?.cacheKey !== nextCacheKey) {
            hfPreparedFit = null;
        }
        hfSelection = {
            repo: selection?.repo ?? '', file: selection?.file ?? '',
            fileSizeBytes: Number.isFinite(Number(selection?.fileSizeBytes))
                ? Math.max(0, Math.trunc(Number(selection.fileSizeBytes))) : 0,
            revision: selection?.revision ?? 'main', token: selection?.token ?? '',
            resolvedUrl: selection?.resolvedUrl ?? '', validated: selection?.validated === true,
            response: selection?.response ?? null, metadata: selection?.metadata ?? null,
            error: selection?.error ?? '',
        };
        result = null; status = 'idle'; errorMsg = '';
    }

    function handleParamsChange(p) {
        logger.log('handleParamsChange', p);
        params = p;
    }

    $effect(() => {
        try {
            const cfg = {
                hostRamGiB: params.hostRamGiB,
                gpus: Array.isArray(params.gpus)
                    ? params.gpus.slice(0, 6).map((g, i) => normalizeHardwareGpu(g, i))
                    : [],
            };
            window?.localStorage?.setItem(hardwareConfigStorageKey, JSON.stringify(cfg));
        } catch { /* ignore */ }
    });

    // ── Predict input builders ────────────────────────────────────────────────
    function buildPredictFitInput() {
        const gpus = params.gpus.map((g, i) => {
            const name = typeof g.name === 'string' ? g.name.trim() : '';
            return {
                id: name || `gpu${i}`, name, index: i, backend: g.backend ?? 'cuda',
                // free_bytes == total_bytes: reserve policy is controlled by target_free_mib
                free_bytes:  giBToBytes(g.totalGiB),
                total_bytes: giBToBytes(g.totalGiB),
            };
        });
        const bufferTargets = params.gpus.map((g) => g.bufferMiB ?? 512);
        const zeroMargins = bufferTargets.map(() => 0);
        return {
            hostRamBytes: giBToBytes(params.hostRamGiB),
            // Keep free should map to one backend knob. Leave fit_target at 0 so it
            // does not stack on top of target_free_mib.
            fitTargetMiB:  bufferTargets.length > 0 ? zeroMargins : [512],
            targetFreeMiB: bufferTargets.length > 0 ? bufferTargets : [],
            gpus,
            nCtx: params.nCtx, nBatch: params.nBatch, nUbatch: params.nUbatch,
            nGpuLayers: params.nGpuLayers,
            splitMode: params.splitMode,
            cacheTypeK: params.cacheTypeK, cacheTypeV: params.cacheTypeV,
            // When nCtxAuto is enabled, the user's nCtx value becomes a minimum —
            // llama.cpp may use a larger context if memory allows.
            minCtx: params.nCtxAuto ? params.nCtx : 0,
            // Enable fit logs for worker-side progress parsing; worker fallback disables if unsupported.
            showFitLogs: true,
        };
    }

    function handleFitProgress(progress) {
        const nextLine = typeof progress?.lastLine === 'string' ? progress.lastLine.trim() : '';
        if (nextLine.length > 0) {
            const lastLine = fitLogLines.length > 0 ? fitLogLines[fitLogLines.length - 1] : '';
            if (nextLine !== lastLine) {
                fitLogLines = [...fitLogLines, nextLine].slice(-120);
            }
        }

        fitProgress = {
            attempt: Number.isFinite(Number(progress?.attempt)) ? Math.max(0, Math.trunc(Number(progress.attempt))) : fitProgress.attempt,
            nCtx: Number.isFinite(Number(progress?.nCtx)) ? Math.max(0, Math.trunc(Number(progress.nCtx))) : fitProgress.nCtx,
            nGpuLayers: Number.isFinite(Number(progress?.nGpuLayers)) ? Math.max(0, Math.trunc(Number(progress.nGpuLayers))) : fitProgress.nGpuLayers,
            lastLine: typeof progress?.lastLine === 'string' ? progress.lastLine : fitProgress.lastLine,
        };
    }

    function cancelPrediction() {
        if (!isRunning || activeWorkerClient == null) {
            return;
        }

        try {
            activeWorkerClient.cancelActiveJob();
        } catch (err) {
            logger.error('cancelPrediction.error', { err });
        }

        resetPredictorWorker();
        activeWorkerClient = null;
        status = 'idle';
        errorMsg = '';
        fitProgress = { attempt: 0, nCtx: null, nGpuLayers: null, lastLine: '' };
        fitLogLines = [];
    }

    // ── HF validate (metadata-only pass) ────────────────────────────────────
    async function validateHfSelection(selection, onStatusUpdate) {
        const client = await initPredictorWorker({ wasmJsUrl, debugEnabled: wasmDebugEnabled });
        const response = await runHfMetadataFromBrowser(client, selection, { logger, onStatusUpdate });
        if (response?.ok === true && response?.prefixBytes instanceof Uint8Array) {
            hfPreparedFit = response;
            if (Number.isFinite(Number(response.contextLength)) && Number(response.contextLength) > 0) {
                params = { ...params, nCtx: Math.max(1, Math.trunc(Number(response.contextLength))) };
            }
            const { prefixBytes, logicalFileSizeBytes, contextLength, originalFileName, cacheKey, ...forUi } = response;
            return forUi;
        }
        return response;
    }


    function buildFitFailureDetails(response, predictInput) {
        const fitStatusText = describeFitStatus(response?.fit?.status);
        const fitWarnings = Array.isArray(response?.fit?.warnings) && response.fit.warnings.length > 0
            ? ` warnings=${response.fit.warnings.join(',')}`
            : '';
        const backendError = typeof response?.error === 'string' ? ` backendError=${response.error}` : '';
        const deviceCountInResponse = Array.isArray(response?.fit?.breakdown?.devices)
            ? response.fit.breakdown.devices.length
            : 0;
        const sentGpuOverrides = Array.isArray(predictInput?.gpus)
            ? predictInput.gpus.length
            : 0;

        return {
            message: `WASM fit failed (${fitStatusText}).${fitWarnings}${backendError}`,
            diagnostics: {
                fitStatus: response?.fit?.status,
                fitStatusText,
                warnings: response?.fit?.warnings,
                sentGpuOverrides,
                deviceCountInResponse,
                response,
            },
        };
    }

    async function runPrediction() {
        if (modelSource === 'local') {
            if (!selectedFile) { errorMsg = 'Please select a GGUF file first.'; status = 'error'; return; }
        } else if (!hfSelection.validated) {
            errorMsg = 'Please select a valid Hugging Face model first.'; status = 'error'; return;
        }

        status = 'loading-wasm'; errorMsg = ''; result = null;
        fitProgress = { attempt: 0, nCtx: null, nGpuLayers: null, lastLine: '' };
        fitLogLines = [];

        let client;
        try {
            client = await initPredictorWorker({ wasmJsUrl, debugEnabled: wasmDebugEnabled });
            activeWorkerClient = client;
            try { logger.log('systemInfo', await client.getSystemInfo()); } catch { /* optional */ }
        } catch (err) {
            status = 'error'; errorMsg = `Failed to load WASM module: ${err.message}`;
            logger.error('runPrediction.init.error', { err }); return;
        }

        status = 'running';
        try {
            const predictInput = buildPredictFitInput();
            fitProgress = {
                attempt: 0,
                nCtx: predictInput.nCtx,
                nGpuLayers: predictInput.nGpuLayers,
                lastLine: 'fit started',
            };

            if (modelSource === 'huggingface') {
                const cacheKey   = buildHfSelectionCacheKey(hfSelection);
                const hasCached  = hfPreparedFit?.cacheKey === cacheKey && hfPreparedFit?.prefixBytes instanceof Uint8Array;
                logger.log('runPrediction.hf.start', { hasCached, predictInput });

                const t0  = performance.now();
                const res = hasCached
                    ? await runFitFromPreparedPrefix(client, hfPreparedFit, predictInput, logger, handleFitProgress)
                    : await runHfFitFromBrowser(client, hfSelection, predictInput, /** @type {any} */ ({
                        onPreparedFit: (pf) => { hfPreparedFit = pf; },
                        logger,
                        onProgress: handleFitProgress,
                    }));
                logger.log('runPrediction.hf.done', { ms: Math.round((performance.now() - t0) * 10) / 10, ok: res?.ok });

                result = res;
                if (res?.ok === false) {
                    status = 'error';
                    if (res?.fit != null) { errorMsg = buildFitFailureDetails(res, predictInput).message; }
                    else { errorMsg = `HF fit failed: ${res?.error ?? 'unknown_error'}${res?.detail ? ` (${res.detail})` : ''}`; }
                } else { status = 'done'; }
                return;
            }

            const t0  = performance.now();
            const res = await client.predictMountedFit(selectedFile, predictInput, { onProgress: handleFitProgress });
            logger.log('runPrediction.local.done', { ms: Math.round((performance.now() - t0) * 10) / 10, ok: res?.ok });

            result = res;
            if (res?.ok === false) {
                status = 'error'; errorMsg = buildFitFailureDetails(res, predictInput).message;
            } else { status = 'done'; }
        } catch (err) {
            const message = err?.message ?? String(err);
            if (message.includes('predictor_worker_cancelled')) {
                status = 'idle';
                errorMsg = '';
                logger.log('runPrediction.cancelled', {});
            } else {
                status = 'error'; errorMsg = message;
                logger.error('runPrediction.error', { err });
            }
        } finally {
            activeWorkerClient = null;
        }
    }

    const isRunning   = $derived(status === 'loading-wasm' || status === 'running');
    const canRun      = $derived(modelSource === 'local' ? selectedFile != null : hfSelection.validated === true);
    const statusLabel = $derived({ idle: '', 'loading-wasm': 'Preparing WASM worker…', running: 'Running fit prediction…', done: '', error: '' }[status] ?? '');
    const progressLabel = $derived((() => {
        if (status !== 'running') return '';
        const parts = [];
        if ((fitProgress?.attempt ?? 0) > 0) parts.push(`attempt ${fitProgress.attempt}`);
        if (Number.isFinite(Number(fitProgress?.nCtx))) parts.push(`n_ctx ${fitProgress.nCtx}`);
        if (Number.isFinite(Number(fitProgress?.nGpuLayers))) parts.push(`n_gpu_layers ${fitProgress.nGpuLayers}`);
        return parts.join(' • ');
    })());
</script>

<div class="app-shell">
    <header class="app-header">
        <div class="header-inner">
            <span class="logo-text">vram.cpp</span>
            <p class="tagline">Predict LLM VRAM usage using llama.cpp fit logic</p>
        </div>
    </header>

    <main class="app-main">
        {#if isJsonHarnessView}
            <JsonHarness wasmJsUrl={wasmJsUrl} debugEnabled={wasmDebugEnabled} />
        {:else}

        <!-- Row 1: Model input + Runtime params -->
        <div class="model-runtime-row">
            <section class="panel-section model-col">
                <h2 class="panel-title">Model</h2>
                <div class="source-switch" role="tablist" aria-label="Model source">
                    <button class="source-btn" class:active={modelSource === 'huggingface'} type="button"
                        onclick={() => handleModelSourceChange('huggingface')}>HuggingFace</button>
                    <span class="source-or">OR</span>
                    <button class="source-btn" class:active={modelSource === 'local'} type="button"
                        onclick={() => handleModelSourceChange('local')}>Upload</button>
                </div>
                <div class="model-input-area">
                    {#if modelSource === 'local'}
                        <FileUpload onfile={handleFile} />
                    {:else}
                        <HuggingFaceSearch
                            onselectionchange={handleHuggingFaceSelectionChange}
                            onvalidate={validateHfSelection}
                        />
                    {/if}
                </div>
            </section>

            <section class="panel-section runtime-col">
                <h2 class="panel-title">Runtime</h2>
                <RuntimePanel {params} onchange={handleParamsChange} />
            </section>
        </div>

        <!-- Row 2: Hardware config -->
        <section class="panel-section hardware-section">
            <h2 class="panel-title">Hardware</h2>
            <HardwarePanel {params} onchange={handleParamsChange} />
        </section>

        <!-- Row 3: Run + Results -->
        <section class="results-section">
            <div class="action-row">
                <button class="run-btn" type="button" onclick={runPrediction} disabled={isRunning || !canRun}>
                    {#if isRunning}
                        <span class="spinner" aria-hidden="true"></span>
                        {statusLabel}
                    {:else}
                        ▶ Predict VRAM
                    {/if}
                </button>
                {#if isRunning}
                    <button class="cancel-btn" type="button" onclick={cancelPrediction}>Cancel</button>
                {/if}
                {#if progressLabel}
                    <p class="hint-msg progress-msg">{progressLabel}</p>
                {/if}
                {#if modelSource === 'huggingface' && !hfSelection.validated}
                    <p class="hint-msg">Select a Hugging Face model to enable fitting.</p>
                {:else if modelSource === 'local' && selectedFile == null}
                    <p class="hint-msg">Upload a GGUF file to enable fitting.</p>
                {/if}
                {#if status === 'error'}
                    <p class="error-msg">{errorMsg}</p>
                {/if}
            </div>

            <section class="results-panel">
                <h2 class="panel-title">
                    Memory Breakdown
                    {#if status === 'done'}<span class="ok-badge">✓</span>{/if}
                </h2>
                {#if isRunning}
                    <div class="loading-state">
                        <span class="spinner lg" aria-hidden="true"></span>
                        <p>{statusLabel}</p>
                        {#if progressLabel}
                            <p class="progress-inline">{progressLabel}</p>
                        {/if}
                        {#if fitLogLines.length > 0}
                            <div class="fit-log-view" role="log" aria-live="polite">
                                {#each fitLogLines as line}
                                    <div class="fit-log-line">{line}</div>
                                {/each}
                            </div>
                        {/if}
                    </div>
                {:else}
                    <ResultsTable {result} />
                {/if}
            </section>
        </section>
        {/if}
    </main>

    <footer class="app-footer">
        <span>Powered by <a href="https://github.com/ggml-org/llama.cpp" target="_blank" rel="noreferrer">llama.cpp</a> compiled to WebAssembly.</span>
        <span class="footer-version">version {llamaVersion}</span>
    </footer>
</div>

<style>
    .app-shell {
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        background: var(--bg);
    }

    /* -- Header -- */
    .app-header {
        background: var(--header-bg);
        border-bottom: 1px solid var(--border);
        padding: 0 24px;
    }

    .header-inner {
        max-width: 1280px;
        margin: 0 auto;
        padding: 10px 0;
        display: flex;
        align-items: baseline;
        gap: 16px;
        flex-wrap: wrap;
    }

    .logo-text {
        font-size: 1.1rem;
        font-weight: 700;
        font-family: var(--mono);
        color: var(--text-header);
    }

    .tagline {
        margin: 0;
        font-size: 0.85rem;
        color: var(--text-muted);
    }

    /* -- Main layout -- */
    .app-main {
        flex: 1;
        max-width: 1280px;
        width: 100%;
        margin: 0 auto;
        padding: 20px 24px;
        display: flex;
        flex-direction: column;
        gap: 16px;
        box-sizing: border-box;
    }

    /* Row 1: fixed height */
    .model-runtime-row {
        display: grid;
        grid-template-columns: minmax(320px, 1fr) minmax(320px, 1fr);
        gap: 16px;
        align-items: stretch;
        height: 400px;
    }

    @media (max-width: 800px) {
        .model-runtime-row { grid-template-columns: 1fr; height: auto; }
    }

    .model-col {
        height: 100%;
        overflow: hidden;
    }

    .model-input-area {
        flex: 1;
        overflow-y: auto;
        min-height: 0;
    }

    .runtime-col {
        height: 100%;
        overflow-y: auto;
    }

    .hardware-section { width: 100%; box-sizing: border-box; }

    /* -- Source switch -- */
    .source-switch {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        gap: 8px;
        flex-shrink: 0;
    }

    .source-btn {
        border: 1px solid var(--border);
        background: var(--surface-raised);
        color: var(--text-primary);
        border-radius: 10px;
        padding: 8px 12px;
        font-size: 0.88rem;
        font-weight: 600;
        cursor: pointer;
        font-family: inherit;
        transition: background 0.15s;
    }

    .source-btn.active {
        background: var(--accent);
        border-color: transparent;
        color: var(--on-accent);
    }

    .source-or {
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        font-family: var(--mono);
        text-align: center;
    }

    /* -- Generic panel -- */
    .panel-section {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    .panel-title {
        margin: 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 8px;
        flex-shrink: 0;
    }

    /* -- Results -- */
    .results-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 14px;
        min-height: 280px;
    }

    .action-row {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding-bottom: 16px;
    }

    .run-btn {
        width: 100%;
        padding: 12px;
        background: var(--accent);
        color: var(--on-accent);
        border: none;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        font-family: inherit;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        transition: opacity 0.2s, transform 0.1s;
    }

    .run-btn:hover:not(:disabled) { opacity: 0.88; }
    .run-btn:active:not(:disabled) { transform: scale(0.98); }
    .run-btn:disabled { opacity: 0.45; cursor: not-allowed; }

    .cancel-btn {
        width: 100%;
        padding: 10px;
        background: transparent;
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: 10px;
        font-size: 0.92rem;
        font-weight: 600;
        cursor: pointer;
        font-family: inherit;
    }

    .cancel-btn:hover { background: var(--surface-raised); }

    .error-msg {
        margin: 0;
        font-size: 0.85rem;
        color: var(--error);
        padding: 8px 12px;
        background: var(--error-bg);
        border-radius: 6px;
    }

    .hint-msg {
        margin: 0;
        font-size: 0.82rem;
        color: var(--text-muted);
        padding: 8px 12px;
        border: 1px dashed var(--border);
        border-radius: 6px;
    }

    .progress-msg {
        font-family: var(--mono);
    }

    .loading-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        flex: 1;
        gap: 14px;
        color: var(--text-muted);
        padding: 40px 0;
    }

    .loading-state p { margin: 0; font-size: 0.9rem; }

    .loading-state .progress-inline {
        font-family: var(--mono);
        font-size: 0.82rem;
    }

    .fit-log-view {
        width: 100%;
        max-height: 180px;
        overflow: auto;
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--surface-raised);
        padding: 8px;
        box-sizing: border-box;
        text-align: left;
    }

    .fit-log-line {
        font-family: var(--mono);
        font-size: 0.74rem;
        color: var(--text-secondary);
        white-space: pre-wrap;
        word-break: break-word;
        line-height: 1.35;
    }

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

    .spinner.lg { width: 28px; height: 28px; border-width: 3px; }

    @keyframes spin { to { transform: rotate(360deg); } }

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

    /* -- Footer -- */
    .app-footer {
        border-top: 1px solid var(--border);
        padding: 12px 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
        font-size: 0.78rem;
        color: var(--text-muted);
    }

    .footer-version {
        font-family: var(--mono);
        color: var(--text-secondary);
    }

    .app-footer a { color: var(--accent); text-decoration: none; }
    .app-footer a:hover { text-decoration: underline; }
</style>
