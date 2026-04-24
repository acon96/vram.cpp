<script>
    import { formatMiB } from '../lib/format.js';

    /** @type {{ result: object }} */
    let { result } = $props();

    const fit = $derived(result?.fit);
    const breakdown = $derived(fit?.breakdown);
    const recommended = $derived(fit?.recommended);
    const memBytes = $derived(fit?.memoryBytes);
    const warnings = $derived(fit?.warnings ?? []);
    const fitStatus = $derived(fit?.status);

    function fitStatusMessage(code) {
        if (code === 0) return 'fit_success';
        if (code === 1) return 'fit_failure_no_viable_allocation';
        if (code === 2) return 'fit_error_hard_failure';
        if (code == null) return 'fit_status_unset';
        return `fit_status_unknown_${code}`;
    }

    function deviceRows(breakdown) {
        if (!breakdown) return [];
        const rows = [];
        if (breakdown.devices) {
            for (const d of breakdown.devices) {
                rows.push({ ...d, type: 'GPU' });
            }
        }
        if (breakdown.host) {
            rows.push({ ...breakdown.host, type: 'Host' });
        }
        return rows;
    }

    const rows = $derived(deviceRows(breakdown));

    function usagePct(used, total) {
        if (!total) return 0;
        return Math.min(100, Math.round((used / total) * 100));
    }

    function totalUsed(row) {
        return (row.modelMiB ?? 0) + (row.contextMiB ?? 0) + (row.computeMiB ?? 0);
    }
</script>

<div class="results-table">
    <!-- Warnings -->
    {#if warnings.length > 0}
        <div class="warnings">
            {#each warnings as w}
                <div class="warning-item">⚠ {w}</div>
            {/each}
        </div>
    {/if}

    <!-- Recommendations -->
    {#if recommended}
        <div class="rec-row">
            <div class="rec-chip">
                <span class="rec-label">Recommended n_ctx</span>
                <span class="rec-value">{recommended.n_ctx?.toLocaleString() ?? '—'}</span>
            </div>
            <div class="rec-chip">
                <span class="rec-label">GPU layers</span>
                <span class="rec-value">{recommended.n_gpu_layers ?? '—'}</span>
            </div>
            {#if memBytes}
                <div class="rec-chip">
                    <span class="rec-label">Weights</span>
                    <span class="rec-value">{formatMiB(Math.round((memBytes.weights ?? 0) / (1024 * 1024)))}</span>
                </div>
                <div class="rec-chip">
                    <span class="rec-label">KV Cache</span>
                    <span class="rec-value">{formatMiB(Math.round((memBytes.kvCache ?? 0) / (1024 * 1024)))}</span>
                </div>
            {/if}
        </div>
    {/if}

    <!-- Error state -->
    {#if result?.ok === false}
        <div class="empty-state error-state">
            <p>Prediction failed: {result.error ?? fitStatusMessage(fitStatus)}</p>
            {#if fitStatus != null}
                <p class="detail">status code: {fitStatus}</p>
            {/if}
            {#if warnings.length > 0}
                <p class="detail">warnings: {warnings.join(', ')}</p>
            {/if}
            {#if result.detail}
                <p class="detail">{result.detail}</p>
            {/if}
        </div>

    <!-- Memory breakdown table -->
    {:else if rows.length > 0}
        <div class="table-scroll">
            <table>
                <thead>
                    <tr>
                        <th>Device</th>
                        <th>Model</th>
                        <th>KV Cache</th>
                        <th>Compute</th>
                        <th>Used</th>
                        <th>Total</th>
                        <th class="bar-col">Usage</th>
                    </tr>
                </thead>
                <tbody>
                    {#each rows as row}
                        {@const used = totalUsed(row)}
                        {@const pct = usagePct(used, row.totalMiB)}
                        <tr class:host-row={row.type === 'Host'}>
                            <td class="device-name">
                                <span class="device-badge" class:gpu={row.type === 'GPU'} class:host={row.type === 'Host'}>
                                    {row.type === 'GPU' ? '⬛' : '🖥'} {row.name ?? row.type}
                                </span>
                            </td>
                            <td class="num">{formatMiB(row.modelMiB)}</td>
                            <td class="num">{formatMiB(row.contextMiB)}</td>
                            <td class="num">{formatMiB(row.computeMiB)}</td>
                            <td class="num bold">{formatMiB(used)}</td>
                            <td class="num muted">{formatMiB(row.totalMiB)}</td>
                            <td class="bar-col">
                                <div class="bar-wrap">
                                    <div
                                        class="bar-fill"
                                        class:bar-warn={pct > 80}
                                        class:bar-crit={pct > 95}
                                        style="width: {pct}%"
                                    ></div>
                                    <span class="bar-pct">{pct}%</span>
                                </div>
                            </td>
                        </tr>
                    {/each}
                </tbody>
                {#if breakdown?.totals}
                    <tfoot>
                        <tr>
                            <td class="foot-label">Totals</td>
                            <td class="num">{formatMiB(breakdown.totals.modelMiB)}</td>
                            <td class="num">{formatMiB(breakdown.totals.contextMiB)}</td>
                            <td class="num">{formatMiB(breakdown.totals.computeMiB)}</td>
                            <td class="num bold">{formatMiB((breakdown.totals.modelMiB ?? 0) + (breakdown.totals.contextMiB ?? 0) + (breakdown.totals.computeMiB ?? 0))}</td>
                            <td></td>
                            <td></td>
                        </tr>
                    </tfoot>
                {/if}
            </table>
        </div>
    {:else}
        <div class="empty-state">
            <p>No results yet. Upload a model and run a prediction.</p>
        </div>
    {/if}
</div>

<style>
    .results-table {
        display: flex;
        flex-direction: column;
        gap: 16px;
    }

    .warnings {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }

    .warning-item {
        padding: 8px 12px;
        background: var(--warn-bg);
        border-left: 3px solid var(--warn);
        border-radius: 4px;
        font-size: 0.85rem;
        color: var(--warn-text);
    }

    .rec-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }

    .rec-chip {
        display: flex;
        flex-direction: column;
        padding: 10px 16px;
        background: var(--surface-raised);
        border: 1px solid var(--border);
        border-radius: 10px;
        min-width: 110px;
    }

    .rec-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
    }

    .rec-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        font-family: var(--mono);
    }

    .table-scroll {
        overflow-x: auto;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
    }

    th {
        text-align: left;
        padding: 8px 10px;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        border-bottom: 1px solid var(--border);
        white-space: nowrap;
    }

    td {
        padding: 10px 10px;
        border-bottom: 1px solid var(--border-subtle);
        color: var(--text-secondary);
    }

    .num {
        text-align: right;
        font-family: var(--mono);
        font-size: 0.85rem;
        white-space: nowrap;
    }

    .bold {
        font-weight: 600;
        color: var(--text-primary);
    }

    .muted {
        color: var(--text-muted);
    }

    .bar-col {
        width: 140px;
        min-width: 100px;
    }

    .bar-wrap {
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .bar-fill {
        height: 6px;
        border-radius: 3px;
        background: var(--accent);
        transition: width 0.4s ease;
        min-width: 2px;
    }

    .bar-fill.bar-warn {
        background: var(--warn);
    }

    .bar-fill.bar-crit {
        background: var(--error);
    }

    .bar-pct {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-family: var(--mono);
        min-width: 32px;
    }

    .host-row td {
        background: var(--surface-raised);
    }

    tfoot td {
        font-weight: 600;
        border-top: 2px solid var(--border);
        border-bottom: none;
        color: var(--text-primary);
    }

    .foot-label {
        color: var(--text-muted);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .device-name {
        white-space: nowrap;
    }

    .device-badge {
        font-size: 0.85rem;
    }

    .empty-state {
        text-align: center;
        padding: 48px 24px;
        color: var(--text-muted);
        font-size: 0.9rem;
    }

    .error-state {
        color: var(--error);
    }

    .detail {
        font-size: 0.8rem;
        margin-top: 8px;
        opacity: 0.8;
    }
</style>
