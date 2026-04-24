<script>
    import FileUpload from './components/FileUpload.svelte';
    import ParamPanel from './components/ParamPanel.svelte';
    import ResultsTable from './components/ResultsTable.svelte';
    import { initPredictor } from './lib/predictor.js';
    import { giBToBytes } from './lib/format.js';

    // ── WASM paths ────────────────────────────────────────────────────────────
    // These can be overridden at build time via VITE_WASM_BASE_URL env var.
    // For local dev pointing at build-wasm-vendor/, set the env var or adjust.
    const wasmBase = import.meta.env.VITE_WASM_BASE_URL ?? './wasm';
    const wasmJsUrl = `${wasmBase}/vram_predictor_wasm.js`;
    const browserHelperUrl = `${wasmBase}/vram_predictor_browser.js`;

    // ── State ─────────────────────────────────────────────────────────────────
    let selectedFile = $state(null);

    let params = $state({
        nCtx: 4096,
        cacheTypeK: 'f16',
        cacheTypeV: 'f16',
        nGpuLayers: -1,
        hostRamGiB: 32,
        gpus: [{ totalGiB: 8, freeGiB: 8 }],
        fitTargetMiB: 512,
        targetFreeMiB: 2048,
    });

    let status = $state('idle'); // 'idle' | 'loading-wasm' | 'running' | 'done' | 'error'
    let errorMsg = $state('');
    let result = $state(null);

    // ── Handlers ──────────────────────────────────────────────────────────────
    function handleFile(file) {
        selectedFile = file;
        result = null;
        status = 'idle';
        errorMsg = '';
    }

    function handleParamsChange(p) {
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
            client = await initPredictor({ wasmJsUrl, browserHelperUrl });
        } catch (err) {
            status = 'error';
            errorMsg = `Failed to load WASM module: ${err.message}`;
            return;
        }

        status = 'running';
        try {
            const virtualPath = await client.mountBrowserFile(selectedFile);

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

            const res = client.predictMountedFit({
                modelPath: virtualPath,
                hostRamBytes: giBToBytes(params.hostRamGiB),
                fitTargetMiB: fitTargets,
                targetFreeMiB: freeTargets,
                gpus,
                nCtx: params.nCtx,
                nGpuLayers: params.nGpuLayers,
                cacheTypeK: params.cacheTypeK,
                cacheTypeV: params.cacheTypeV,
                minCtx: 512,
            });

            result = res;
            status = 'done';

            // Clean up the mounted file from WASM virtual FS
            try { client.unmountFile(virtualPath); } catch (_) {}
        } catch (err) {
            status = 'error';
            errorMsg = err.message ?? String(err);
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
