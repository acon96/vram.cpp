<script>
    /**
     * HardwarePanel — host RAM and GPU device configuration.
     *
     * GPU objects: { name?: string, totalGiB: number, freeGiB: number, bufferMiB: number }
     * - totalGiB:  total VRAM on the device
     * - freeGiB:   available VRAM before the LLM is loaded (after OS/driver overhead)
     * - bufferMiB: keep-free margin during and after fit (fed to fit engine as
     *              both fit_target_mib and target_free_mib)
     *
     * @typedef {object} Params
     * @property {number} hostRamGiB
     * @property {Array<{name?: string, totalGiB: number, freeGiB: number, bufferMiB: number}>} gpus
     */

    /** @type {{ params: Params, onchange: (p: Params) => void }} */
    let { params, onchange } = $props();

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
                totalGiB: 8,
                freeGiB: 8,
                bufferMiB: 512,
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
</script>

<div class="hardware-panel">
    <!-- Host Memory -->
    <section class="host-section">
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
    <section class="gpus-section">
        <div class="section-header">
            <h3>GPU Devices</h3>
            {#if params.gpus.length < 4}
                <button type="button" class="add-btn" onclick={addGpu}>+ Add GPU</button>
            {:else}
                <span class="hint">Maximum 4 GPUs</span>
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

                    <div class="field">
                        <label for="gpu-name-{i}">Display name</label>
                        <input
                            id="gpu-name-{i}"
                            type="text"
                            value={gpu.name ?? `GPU ${i}`}
                            oninput={(e) => updateGpu(i, { name: e.currentTarget.value })}
                        />
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
                            <label for="gpu-free-{i}">Available VRAM (GiB)</label>
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
                        <div class="field">
                            <label for="gpu-buffer-{i}">Keep free (MiB)</label>
                            <input
                                id="gpu-buffer-{i}"
                                type="number"
                                min="0"
                                step="128"
                                value={gpu.bufferMiB ?? 512}
                                oninput={(e) => updateGpu(i, { bufferMiB: parseInt(e.currentTarget.value) || 0 })}
                            />
                        </div>
                    </div>
                </div>
            {/each}
        </div>
    </section>
</div>

<style>
    .hardware-panel {
        display: flex;
        gap: 24px;
        align-items: start;
        flex-wrap: wrap;
    }

    section {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    .host-section {
        min-width: 160px;
        flex-shrink: 0;
    }

    .gpus-section {
        flex: 1;
        min-width: 300px;
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
    input[type='text'] {
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

    input:focus {
        outline: none;
        border-color: var(--accent);
    }

    .gpus-list {
        display: flex;
        gap: 12px;
        flex-direction: column;
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

    .hint {
        font-size: 0.75rem;
        color: var(--text-muted);
    }
</style>
