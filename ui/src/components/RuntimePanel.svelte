<script>
    /**
     * RuntimePanel — inference runtime configuration.
     *
     * @typedef {object} Params
     * @property {number} nCtx
     * @property {boolean} nCtxAuto
     * @property {number} nBatch
     * @property {number} nUbatch
     * @property {string} cacheTypeK
     * @property {string} cacheTypeV
     * @property {number} nGpuLayers
        * @property {string} splitMode
     */

    /** @type {{ params: Params, onchange: (p: Params) => void }} */
    let { params, onchange } = $props();

    const KV_TYPES = ['f16', 'q8_0', 'q4_0'];
    const SPLIT_MODES = [
        { value: 'layer', label: 'layer' },
        { value: 'row', label: 'row' },
        { value: 'tensor', label: 'tensor' },
    ];

    // Snap points for the context slider — powers of 2 from 512 up to 256k, plus 400k.
    const CTX_SNAP_POINTS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 409600];

    function ctxToSliderIndex(ctx) {
        let closest = 0;
        let minDist = Math.abs(CTX_SNAP_POINTS[0] - ctx);
        for (let i = 1; i < CTX_SNAP_POINTS.length; i++) {
            const dist = Math.abs(CTX_SNAP_POINTS[i] - ctx);
            if (dist < minDist) {
                minDist = dist;
                closest = i;
            }
        }
        return closest;
    }

    function onCtxSliderChange(e) {
        const idx = parseInt(e.currentTarget.value, 10);
        update({ nCtx: CTX_SNAP_POINTS[idx] });
    }

    function onCtxTextChange(e) {
        const val = parseInt(e.currentTarget.value, 10);
        if (!Number.isNaN(val) && val > 0) {
            update({ nCtx: val });
        }
    }

    function onAutoCtxChange(e) {
        update({ nCtxAuto: e.currentTarget.checked });
    }

    function update(patch) {
        onchange({ ...params, ...patch });
    }

    function onBatchChange(value) {
        const next = Math.max(1, parseInt(value, 10) || 1);
        update({
            nBatch: next,
            nUbatch: Math.min(params.nUbatch, next),
        });
    }

    function onUbatchChange(value) {
        const next = Math.max(1, parseInt(value, 10) || 1);
        update({ nUbatch: Math.min(next, params.nBatch) });
    }

    function onGpuLayersChange(value) {
        const parsed = parseInt(value, 10);
        if (Number.isNaN(parsed)) {
            return;
        }
        update({ nGpuLayers: Math.max(-1, parsed) });
    }
</script>

<div class="runtime-panel">
    <div class="field-row">
        <div class="field">
            <label for="n-ctx">Context length (n_ctx)</label>
            <div class="ctx-row">
                <input
                    id="n-ctx"
                    type="range"
                    min="0"
                    max={CTX_SNAP_POINTS.length - 1}
                    step="1"
                    value={ctxToSliderIndex(params.nCtx)}
                    oninput={onCtxSliderChange}
                />
            </div>
            <div class="ctx-marks">
                {#each CTX_SNAP_POINTS as pt}
                    <span>{pt >= 1024 ? `${pt / 1024}k` : pt}</span>
                {/each}
            </div>
        </div>
        <div class="small-field">
            <input
                class="ctx-text"
                type="number"
                min="1"
                max="409600"
                step="1"
                value={params.nCtx}
                oninput={onCtxTextChange}
            />
            <div class="field-row">
                <span class="hint" title="When enabled, the n_ctx value is treated as a minimum — llama.cpp may use a larger context size if memory allows">Min Ctx?</span>
                <input
                    class="ctx-checkbox"
                    type="checkbox"
                    checked={params.nCtxAuto}
                    oninput={onAutoCtxChange}
                />
            </div>
        </div>
    </div>

    <div class="field-row">
        <div class="field">
            <label for="cache-k">K-cache type</label>
            <select
                id="cache-k"
                value={params.cacheTypeK}
                onchange={(e) => update({ cacheTypeK: e.currentTarget.value })}
            >
                {#each KV_TYPES as t}
                    <option value={t}>{t}</option>
                {/each}
            </select>
        </div>
        <div class="field">
            <label for="cache-v">V-cache type</label>
            <select
                id="cache-v"
                value={params.cacheTypeV}
                onchange={(e) => update({ cacheTypeV: e.currentTarget.value })}
            >
                {#each KV_TYPES as t}
                    <option value={t}>{t}</option>
                {/each}
            </select>
        </div>
    </div>

    <div class="field-row">
        <div class="field">
            <label for="gpu-layers">GPU layers (n_gpu_layers)</label>
            <input
                id="gpu-layers"
                type="number"
                min="-1"
                step="1"
                value={params.nGpuLayers}
                oninput={(e) => onGpuLayersChange(e.currentTarget.value)}
            />
            <span class="hint">{params.nGpuLayers === -1 ? 'all layers on GPU' : params.nGpuLayers === 0 ? 'CPU only' : `${params.nGpuLayers} layers`}</span>
        </div>
        <div class="field">
            <label for="split-mode">Split mode</label>
            <select
                id="split-mode"
                value={params.splitMode ?? 'layer'}
                onchange={(e) => update({ splitMode: e.currentTarget.value })}
            >
                {#each SPLIT_MODES as mode}
                    <option value={mode.value}>{mode.label}</option>
                {/each}
            </select>
        </div>
    </div>

    <div class="field">
        
        
    </div>

    <div class="field-row">
        <div class="field">
            <label for="n-batch">Batch size (n_batch)</label>
            <input
                id="n-batch"
                type="number"
                min="1"
                step="1"
                value={params.nBatch}
                oninput={(e) => onBatchChange(e.currentTarget.value)}
            />
        </div>
        <div class="field">
            <label for="n-ubatch">Micro-batch (n_ubatch)</label>
            <input
                id="n-ubatch"
                type="number"
                min="1"
                max={params.nBatch}
                step="1"
                value={params.nUbatch}
                oninput={(e) => onUbatchChange(e.currentTarget.value)}
            />
        </div>
    </div>
</div>

<style>
    .runtime-panel {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    .field {
        display: flex;
        flex-direction: column;
        gap: 4px;
        flex: 1;
    }

    .small-field {
        display: flex;
        flex-direction: column;
        gap: 4px;
        width: 100px;
    }

    .field-row {
        display: flex;
        gap: 12px;
        flex-direction: row;
    }

    label {
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    input[type='range'] {
        flex: 1;
    }

    .input-row {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .ctx-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-left: 0px;
        padding-right: 6px;
    }

    .ctx-row input[type='range'] {
        flex: 1;
    }

    .ctx-text {
        width: 90px !important;
        flex-shrink: 0;
        font-family: var(--text-muted);
        font-size: 0.85rem !important;
        margin-bottom: 10px;
    }

    .ctx-marks {
        display: flex;
        justify-content: space-between;
        margin-top: 2px;
        padding: 0 2px;
    }

    .ctx-marks span {
        font-size: 0.65rem;
        color: var(--text-muted);
        font-family: var(--mono);
    }

    .ctx-checkbox {
        padding-top: 20px;
    }

    .hint {
        font-size: 0.75rem;
        color: var(--text-muted);
        line-height: 2;
    }
</style>
