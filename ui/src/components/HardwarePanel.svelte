<script>
    /**
     * HardwarePanel — host RAM slider + horizontal GPU device cards.
     *
     * GPU shape: { name?: string, totalGiB: number, bufferMiB: number }
     *   - totalGiB:  total VRAM on the device
        *   - bufferMiB: memory to keep free (fed to fit engine as target_free_mib;
        *                fit_target_mib stays zero so reserve is not double-counted)
     */

    /** @type {{ params: { hostRamGiB: number, gpus: Array<{name?: string, totalGiB: number, bufferMiB: number, backend?: string}> }, onchange: Function }} */
    let { params, onchange } = $props();

    const RAM_SNAPS = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 768, 1024, 1536, 2048];
    const MAX_GPUS  = 8;
    const BACKEND_OPTIONS = ['cuda', 'metal', 'vulkan', 'generic'];
    const GPU_PRESETS = [
        { id: 'custom', label: 'Custom values' },
        { id: 'rtx-3060', label: 'NVIDIA RTX 3060', name: 'RTX 3060', totalGiB: 12, bufferMiB: 768, backend: 'cuda' },
        { id: 'rtx-3090', label: 'NVIDIA RTX 3090', name: 'RTX 3090', totalGiB: 24, bufferMiB: 1024, backend: 'cuda' },
        { id: 'rtx-4090', label: 'NVIDIA RTX 4090', name: 'RTX 4090', totalGiB: 24, bufferMiB: 1024, backend: 'cuda' },
        { id: 'a100-80', label: 'NVIDIA A100 (80GB)', name: 'A100 80GB', totalGiB: 80, bufferMiB: 2048, backend: 'cuda' },
        { id: 'a100-40', label: 'NVIDIA A100 (40GB)', name: 'A100 40GB', totalGiB: 40, bufferMiB: 1024, backend: 'cuda' },
        { id: 'h100-80', label: 'NVIDIA H100', name: 'H100 80GB', totalGiB: 80, bufferMiB: 2048, backend: 'cuda' },
        { id: 'm2-max', label: 'Apple M2 Max', name: 'M2 Max', totalGiB: 30, bufferMiB: 1024, backend: 'metal' },
        { id: 'm3-max', label: 'Apple M3 Max', name: 'M3 Max', totalGiB: 40, bufferMiB: 1024, backend: 'metal' },
        { id: 'rx-7900-xtx', label: 'AMD RX 7900 XTX', name: 'RX 7900 XTX', totalGiB: 24, bufferMiB: 1024, backend: 'vulkan' },
    ];

    function ramToSliderIndex(gib) {
        let best = 0;
        let bestDist = Math.abs(RAM_SNAPS[0] - gib);
        for (let i = 1; i < RAM_SNAPS.length; i++) {
            const d = Math.abs(RAM_SNAPS[i] - gib);
            if (d < bestDist) { bestDist = d; best = i; }
        }
        return best;
    }

    function update(patch) { onchange({ ...params, ...patch }); }

    function updateGpu(i, patch) {
        const gpus = params.gpus.map((g, idx) => idx === i ? { ...g, ...patch } : g);
        onchange({ ...params, gpus });
    }

    function addGpu() {
        if (params.gpus.length >= MAX_GPUS) return;
        const n = params.gpus.length;
        onchange({ ...params, gpus: [...params.gpus, { name: `GPU ${n}`, totalGiB: 8, bufferMiB: 512, backend: 'cuda' }] });
    }

    function removeGpu(i) {
        onchange({ ...params, gpus: params.gpus.filter((_, idx) => idx !== i) });
    }

    function applyPreset(i, presetId) {
        if (presetId === 'custom') return;
        const preset = GPU_PRESETS.find((p) => p.id === presetId);
        if (!preset) return;
        updateGpu(i, {
            name: preset.name,
            totalGiB: preset.totalGiB,
            bufferMiB: preset.bufferMiB,
            backend: preset.backend,
        });
    }
</script>

<div class="hardware-panel">

    <div class="ram-row">
        <span class="hw-label">System RAM</span>
        <input
            class="ram-slider"
            type="range"
            min="0"
            max={RAM_SNAPS.length - 1}
            step="1"
            value={ramToSliderIndex(params.hostRamGiB)}
            oninput={(e) => update({ hostRamGiB: RAM_SNAPS[parseInt(e.currentTarget.value, 10)] })}
        />
        <div id="host-ram">
        <input
            class="ctx-text"
            type="number"
            min="8"
            max={RAM_SNAPS[RAM_SNAPS.length - 1]}
            step="1"
            value={params.hostRamGiB}
            oninput={(e) => update({ hostRamGiB: parseInt(e.currentTarget.value, 10) || 0 })}
        />
        </div>
    </div>

    <div class="gpu-section">
        <div class="gpu-section-header">
            <span class="hw-label">GPU Devices</span>
            {#if params.gpus.length < MAX_GPUS}
                <button type="button" class="add-gpu-btn" onclick={addGpu}>+ Add GPU</button>
            {:else}
                <span class="gpu-limit-hint">Max {MAX_GPUS}</span>
            {/if}
        </div>

        <div class="gpu-track">
            {#if params.gpus.length === 0}
                <div class="gpu-empty">
                    CPU-only — <button type="button" class="link-btn" onclick={addGpu}>add a GPU</button>
                </div>
            {/if}
            {#each params.gpus as gpu, i}
                <div class="gpu-card">
                    <div class="gpu-card-header">
                        <span class="gpu-index">GPU {i}</span>
                        <button type="button" class="remove-btn" onclick={() => removeGpu(i)} aria-label="Remove GPU {i}">✕</button>
                    </div>
                    <input
                        class="gpu-name-input"
                        type="text"
                        placeholder="Name (optional)"
                        value={gpu.name ?? `GPU ${i}`}
                        oninput={(e) => updateGpu(i, { name: e.currentTarget.value })}
                    />
                    <div class="gpu-fields">
                        <div class="gpu-field">
                            <label for="gpu-preset-{i}">Preset</label>
                            <select
                                id="gpu-preset-{i}"
                                value="custom"
                                onchange={(e) => applyPreset(i, e.currentTarget.value)}
                            >
                                {#each GPU_PRESETS as preset}
                                    <option value={preset.id}>{preset.label}</option>
                                {/each}
                            </select>
                        </div>
                        <div class="gpu-field">
                            <label for="gpu-backend-{i}">Backend</label>
                            <select
                                id="gpu-backend-{i}"
                                value={gpu.backend ?? 'cuda'}
                                onchange={(e) => updateGpu(i, { backend: e.currentTarget.value })}
                            >
                                {#each BACKEND_OPTIONS as backend}
                                    <option value={backend}>{backend}</option>
                                {/each}
                            </select>
                        </div>
                        <div class="gpu-field">
                            <label for="gpu-vram-{i}">VRAM (GiB)</label>
                            <input
                                id="gpu-vram-{i}"
                                type="number"
                                min="0.5"
                                step="0.5"
                                value={gpu.totalGiB}
                                oninput={(e) => updateGpu(i, { totalGiB: parseFloat(e.currentTarget.value) || 0 })}
                            />
                        </div>
                        <div class="gpu-field">
                            <label for="gpu-buf-{i}">Keep free (MiB)</label>
                            <input
                                id="gpu-buf-{i}"
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
    </div>

</div>

<style>
    .hardware-panel {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .ram-row {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .hw-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--text-muted);
        white-space: nowrap;
        flex-shrink: 0;
        min-width: 88px;
    }

    .ram-slider {
        flex: 1;
        accent-color: var(--accent);
    }

    .gpu-section {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }

    .gpu-section-header {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .gpu-limit-hint {
        font-size: 0.75rem;
        color: var(--text-muted);
    }

    .add-gpu-btn {
        background: none;
        border: 1px solid var(--accent);
        color: var(--accent);
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-family: inherit;
        cursor: pointer;
        transition: background 0.15s;
    }

    .add-gpu-btn:hover { background: var(--accent-subtle); }

    .gpu-track {
        display: flex;
        flex-direction: row;
        gap: 8px;
        overflow-x: auto;
        padding-bottom: 4px;
        scrollbar-width: thin;
        scrollbar-color: var(--border) transparent;
    }

    .gpu-track::-webkit-scrollbar { height: 4px; }
    .gpu-track::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

    .gpu-empty {
        font-size: 0.82rem;
        color: var(--text-muted);
        padding: 4px 0;
        white-space: nowrap;
    }

    .gpu-card {
        flex-shrink: 0;
        width: 196px;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 8px;
        background: var(--surface-raised);
        display: flex;
        flex-direction: column;
        gap: 6px;
    }

    .gpu-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .gpu-index {
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .gpu-name-input {
        width: 100%;
        box-sizing: border-box;
        padding: 4px 7px;
        border: 1px solid var(--border);
        border-radius: 5px;
        background: var(--input-bg);
        color: var(--text-primary);
        font-size: 0.82rem;
        font-family: inherit;
    }

    .gpu-name-input:focus { outline: none; border-color: var(--accent); }

    .gpu-fields {
        display: flex;
        flex-direction: column;
        gap: 5px;
    }

    .gpu-field {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }

    .gpu-field label {
        font-size: 0.72rem;
        color: var(--text-muted);
    }

    .gpu-field input {
        width: 100%;
        box-sizing: border-box;
        padding: 4px 7px;
        border: 1px solid var(--border);
        border-radius: 5px;
        background: var(--input-bg);
        color: var(--text-primary);
        font-size: 0.85rem;
        font-family: inherit;
    }

    .gpu-field select {
        width: 100%;
        box-sizing: border-box;
        padding: 4px 7px;
        border: 1px solid var(--border);
        border-radius: 5px;
        background: var(--input-bg);
        color: var(--text-primary);
        font-size: 0.85rem;
        font-family: inherit;
    }

    .gpu-field input:focus { outline: none; border-color: var(--accent); }
    .gpu-field select:focus { outline: none; border-color: var(--accent); }

    .remove-btn {
        background: none;
        border: none;
        color: var(--text-muted);
        cursor: pointer;
        font-size: 0.82rem;
        padding: 1px 4px;
        border-radius: 3px;
        line-height: 1;
        transition: color 0.15s;
    }

    .remove-btn:hover { color: var(--error); }

    .link-btn {
        background: none;
        border: none;
        color: var(--accent);
        cursor: pointer;
        text-decoration: underline;
        padding: 0;
        font-size: inherit;
        font-family: inherit;
    }

    input[type='number'] {
        width: 100px;
    }

    #host-ram::after {
        content: ' GiB';
        font-family: var(--mono);
        color: var(--text-muted);
    }
</style>
