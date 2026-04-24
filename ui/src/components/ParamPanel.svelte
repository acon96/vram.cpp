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
</script>

<div class="param-panel">
    <!-- Runtime -->
    <section>
        <h3>Runtime</h3>

        <div class="field">
            <label for="n-ctx">Context length (n_ctx)</label>
            <div class="input-row">
                <input
                    id="n-ctx"
                    type="range"
                    min="512"
                    max="131072"
                    step="512"
                    value={params.nCtx}
                    oninput={(e) => update({ nCtx: parseInt(e.currentTarget.value) })}
                />
                <span class="value-badge">{params.nCtx.toLocaleString()}</span>
            </div>
        </div>

        <div class="field-row">
            <div class="field">
                <label for="cache-k">KV cache K type</label>
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
                <label for="cache-v">KV cache V type</label>
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
                    oninput={(e) => update({ nGpuLayers: parseInt(e.currentTarget.value) })}
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
    <section>
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

    <!-- GPU Devices -->
    <section>
        <div class="section-header">
            <h3>GPU Devices</h3>
            {#if params.gpus.length < 4}
                <button type="button" class="add-btn" onclick={addGpu}>+ Add GPU</button>
            {/if}
        </div>

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
    </section>

    <!-- Fit Targets -->
    <section>
        <h3>Fit Targets</h3>
        <div class="field-row">
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
        </div>
    </section>
</div>

<style>
    .param-panel {
        display: flex;
        flex-direction: column;
        gap: 24px;
    }

    section {
        display: flex;
        flex-direction: column;
        gap: 12px;
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

    .value-badge {
        font-size: 0.85rem;
        font-family: var(--mono);
        color: var(--text-primary);
        min-width: 60px;
        text-align: right;
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
</style>
