<script>
    /**
     * ParamPanel — runtime & device configuration for the fit predictor.
     *
     * @typedef {object} Params
     * @property {number}   nCtx
     * @property {number}   nBatch
     * @property {number}   nUbatch
     * @property {string}   cacheTypeK
     * @property {string}   cacheTypeV
     * @property {number}   nGpuLayers
     * @property {number}   hostRamGiB
     * @property {Array<{name?: string, index?: number, totalGiB: number, freeGiB: number}>} gpus
     * @property {number}   fitTargetMiB
     * @property {number}   targetFreeMiB
     */

    /** @type {{ params: Params, onchange: (p: Params) => void }} */
    let { params, onchange } = $props();

    const KV_TYPES = ['f32', 'f16', 'bf16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'iq4_nl'];

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

    function update(patch) {
        onchange({ ...params, ...patch });
    }

    function updateGpu(index, patch) {
        const gpus = params.gpus.map((g, i) => (i === index ? { ...g, ...patch } : g));
        onchange({ ...params, gpus });
    }

    function addGpu() {
        const nextIndex = params.gpus.length;
        onchange({
            ...params,
            gpus: [...params.gpus, {
                name: `GPU ${nextIndex}`,
                index: nextIndex,
                totalGiB: 8,
                freeGiB: 8,
            }],
        });
    }

    function removeGpu(index) {
        onchange({ ...params, gpus: params.gpus.filter((_, i) => i !== index) });
    }

    function onGpuTotalChange(index, value) {
        const total = parseFloat(value) || 0;
        const gpu = params.gpus[index];
        updateGpu(index, { totalGiB: total, freeGiB: Math.min(gpu.freeGiB, total) });
    }

    function onGpuNameChange(index, value) {
        updateGpu(index, { name: value });
    }

    function onGpuIndexChange(index, value) {
        const next = parseInt(value, 10);
        if (!Number.isNaN(next) && next >= 0) {
            updateGpu(index, { index: next });
        }
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

<div class="param-panel">
    <!-- Runtime -->
    <section class="runtime-section">
        <h3>Runtime</h3>

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
                <input
                    class="ctx-text"
                    type="number"
                    min="1"
                    max="409600"
                    step="1"
                    value={params.nCtx}
                    oninput={onCtxTextChange}
                />
            </div>
            <div class="ctx-marks">
                {#each CTX_SNAP_POINTS as pt}
                    <span>{pt >= 1024 ? `${pt / 1024}k` : pt}</span>
                {/each}
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

        <div class="field">
            <label for="gpu-layers">GPU layers (n_gpu_layers)</label>
            <div class="input-row">
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
                <label for="n-ubatch">Micro-batch size (n_ubatch)</label>
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
    </section>

    <!-- Host Memory -->
    <section class="host-memory-section">
        <h3>Host Memory</h3>
        <div class="field">
            <label for="host-ram">System RAM (GiB)</label>
            <input
                id="host-ram"
                type="number"
                min="1"
                step="1"
                value={params.hostRamGiB}
                oninput={(e) => update({ hostRamGiB: parseFloat(e.currentTarget.value) || 0 })}
            />
        </div>
    </section>

    <!-- Fit Targets -->
    <section class="fit-targets-section">
        <h3>Fit Targets</h3>
        <div class="field">
            <label for="fit-target">Fit target (MiB)</label>
            <input
                id="fit-target"
                type="number"
                min="0"
                step="128"
                value={params.fitTargetMiB}
                oninput={(e) => update({ fitTargetMiB: parseInt(e.currentTarget.value) || 0 })}
            />
            <span class="field-hint">Margin to keep free per GPU during fit</span>
        </div>
        <div class="field">
            <label for="target-free">Target free (MiB)</label>
            <input
                id="target-free"
                type="number"
                min="0"
                step="256"
                value={params.targetFreeMiB}
                oninput={(e) => update({ targetFreeMiB: parseInt(e.currentTarget.value) || 0 })}
            />
            <span class="field-hint">Desired free memory after fit adjustments</span>
        </div>
    </section>

    <!-- GPU Devices -->
    <section class="gpus-section">
        <div class="section-header">
            <h3>GPU Devices</h3>
            {#if params.gpus.length < 4}
                <button type="button" class="add-btn" onclick={addGpu}>+ Add GPU</button>
            {:else}
                <span class="hint">Maximum of 4 GPUs supported</span>
            {/if}
            
        </div>
        <div class="gpus-list">
        {#if params.gpus.length === 0}
            <p class="muted">No GPUs — CPU-only mode. <button type="button" class="link-btn" onclick={addGpu}>Add a GPU</button></p>
        {/if}
            {#each params.gpus as gpu, i}
            <div class="gpu-card">
                <div class="gpu-header">
                    <span class="gpu-label">GPU {i}</span>
                    <button type="button" class="remove-btn" onclick={() => removeGpu(i)} aria-label="Remove GPU {i}">✕</button>
                </div>

                <div class="field-row">
                    <div class="field">
                        <label for="gpu-name-{i}">Display name</label>
                        <input
                            id="gpu-name-{i}"
                            type="text"
                            value={gpu.name ?? `GPU ${i}`}
                            oninput={(e) => onGpuNameChange(i, e.currentTarget.value)}
                        />
                    </div>
                    <div class="field">
                        <label for="gpu-index-{i}">Device index</label>
                        <input
                            id="gpu-index-{i}"
                            type="number"
                            min="0"
                            step="1"
                            value={gpu.index ?? i}
                            oninput={(e) => onGpuIndexChange(i, e.currentTarget.value)}
                        />
                    </div>
                </div>

                <div class="field-row">
                    <div class="field">
                        <label for="gpu-total-{i}">Total VRAM (GiB)</label>
                        <input
                            id="gpu-total-{i}"
                            type="number"
                            min="0.5"
                            step="0.5"
                            value={gpu.totalGiB}
                            oninput={(e) => onGpuTotalChange(i, e.currentTarget.value)}
                        />
                    </div>
                    <div class="field">
                        <label for="gpu-free-{i}">Free VRAM (GiB)</label>
                        <input
                            id="gpu-free-{i}"
                            type="number"
                            min="0"
                            max={gpu.totalGiB}
                            step="0.5"
                            value={gpu.freeGiB}
                            oninput={(e) => updateGpu(i, { freeGiB: parseFloat(e.currentTarget.value) || 0 })}
                        />
                    </div>
                </div>
            </div>
            {/each}
        </div>
    </section>

</div>

<style>
    .param-panel {
        display: flex;
        flex-direction: column;
        max-height: 60vh;
        flex-wrap: wrap;
        gap: 24px;
        align-items: start;
    }

    section {
        display: flex;
        flex-direction: column;
        gap: 12px;
        flex: 1;
        min-width: 220px;
        min-height: 200px;
    }

    h3 {
        margin: 0;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        border-bottom: 1px solid var(--border);
        padding-bottom: 6px;
    }

    .section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid var(--border);
        padding-bottom: 6px;
    }

    .section-header h3 {
        border-bottom: none;
        padding-bottom: 0;
    }

    .field {
        display: flex;
        flex-direction: column;
        gap: 4px;
        flex: 1;
    }

    .field-row {
        display: flex;
        gap: 12px;
    }

    label {
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    input[type='number'],
    input[type='text'],
    select {
        width: 100%;
        box-sizing: border-box;
        padding: 7px 10px;
        border: 1px solid var(--border);
        border-radius: 6px;
        background: var(--input-bg);
        color: var(--text-primary);
        font-size: 0.9rem;
        font-family: inherit;
        transition: border-color 0.15s;
    }

    input:focus,
    select:focus {
        outline: none;
        border-color: var(--accent);
    }

    input[type='range'] {
        flex: 1;
        accent-color: var(--accent);
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
    }

    .ctx-row input[type='range'] {
        flex: 1;
    }

    .ctx-text {
        width: 90px !important;
        flex-shrink: 0;
        font-family: var(--mono);
        font-size: 0.85rem !important;
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

    .hint {
        font-size: 0.75rem;
        color: var(--text-muted);
    }

    .field-hint {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 2px;
    }

    .gpus-list {
        display: flex;
        gap: 12px;
        flex-direction: column;
        overflow-y: scroll;
        max-height: 400px;
        flex: 1;
    }

    .gpu-card {
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px;
        background: var(--surface-raised);
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .gpu-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .gpu-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-secondary);
    }

    .add-btn {
        background: none;
        border: 1px solid var(--accent);
        color: var(--accent);
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 0.8rem;
        cursor: pointer;
        transition: background 0.15s;
    }

    .add-btn:hover {
        background: var(--accent-subtle);
    }

    .remove-btn {
        background: none;
        border: none;
        color: var(--text-muted);
        cursor: pointer;
        font-size: 0.9rem;
        padding: 2px 6px;
        border-radius: 4px;
        transition: color 0.15s;
    }

    .remove-btn:hover {
        color: var(--error);
    }

    .link-btn {
        background: none;
        border: none;
        color: var(--accent);
        cursor: pointer;
        text-decoration: underline;
        padding: 0;
        font-size: inherit;
    }

    .muted {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin: 0;
    }

    .runtime-section {
        min-height: 100%;
    }

    .host-memory-section {
        flex-grow: 0;
        min-height: 0;
        border-left: 1px solid var(--border);
        padding-left: 16px;
    }

    .fit-targets-section {
        border-left: 1px solid var(--border);
        padding-left: 16px;
    }

    .gpus-section {
        min-width: 450px;
    }

    @media (max-width: 980px) {
        .host-memory-section {
            padding-right: 0;
        }

        .fit-targets-section {
            border-left: none;
            padding-left: 0;
        }
    }
</style>
