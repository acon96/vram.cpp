<script>
    /** @type {{ onfile: (file: File) => void }} */
    let { onfile } = $props();

    let dragging = $state(false);
    let fileName = $state('');
    let fileSize = $state(0);

    function handleFiles(files) {
        const file = files?.[0];
        if (!file) return;
        fileName = file.name;
        fileSize = file.size;
        onfile(file);
    }

    function onDragOver(e) {
        e.preventDefault();
        dragging = true;
    }

    function onDragLeave() {
        dragging = false;
    }

    function onDrop(e) {
        e.preventDefault();
        dragging = false;
        handleFiles(e.dataTransfer?.files);
    }

    function onInputChange(e) {
        handleFiles(e.currentTarget.files);
    }

    function formatSize(bytes) {
        if (bytes >= 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GiB`;
        if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
        return `${bytes} B`;
    }
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
    class="drop-zone"
    class:dragging
    class:has-file={!!fileName}
    ondragover={onDragOver}
    ondragleave={onDragLeave}
    ondrop={onDrop}
    role="button"
    tabindex="0"
    aria-label="Upload GGUF file"
    onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') document.getElementById('gguf-file-input')?.click(); }}
>
    {#if fileName}
        <div class="file-info">
            <span class="file-icon">📦</span>
            <div class="file-meta">
                <span class="file-name">{fileName}</span>
                <span class="file-size">{formatSize(fileSize)}</span>
            </div>
            <label class="change-btn" for="gguf-file-input">Change</label>
        </div>
    {:else}
        <div class="prompt">
            <span class="upload-icon">⬆</span>
            <p class="primary">Drop a <code>.gguf</code> file here</p>
            <p class="secondary">or</p>
            <label class="browse-btn" for="gguf-file-input">Browse files</label>
        </div>
    {/if}
    <input
        id="gguf-file-input"
        type="file"
        accept=".gguf"
        class="hidden-input"
        onchange={onInputChange}
    />
</div>

<style>
    .drop-zone {
        border: 2px dashed var(--border);
        border-radius: 12px;
        padding: 32px 24px;
        text-align: center;
        cursor: pointer;
        transition: border-color 0.2s, background-color 0.2s;
        background: var(--surface);
    }

    .drop-zone:hover,
    .drop-zone.dragging {
        border-color: var(--accent);
        background: var(--accent-subtle);
    }

    .drop-zone.has-file {
        border-style: solid;
        border-color: var(--accent);
    }

    .prompt {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 8px;
    }

    .upload-icon {
        font-size: 2rem;
        opacity: 0.5;
    }

    .primary {
        margin: 0;
        font-size: 1rem;
        color: var(--text-primary);
    }

    .secondary {
        margin: 0;
        font-size: 0.85rem;
        color: var(--text-muted);
    }

    code {
        background: var(--code-bg);
        border-radius: 4px;
        padding: 1px 5px;
        font-family: var(--mono);
        font-size: 0.9em;
    }

    .browse-btn,
    .change-btn {
        display: inline-block;
        padding: 8px 20px;
        background: var(--accent);
        color: var(--on-accent);
        border-radius: 999px;
        cursor: pointer;
        font-size: 0.9rem;
        transition: opacity 0.2s;
    }

    .browse-btn:hover,
    .change-btn:hover {
        opacity: 0.85;
    }

    .file-info {
        display: flex;
        align-items: center;
        gap: 12px;
        text-align: left;
    }

    .file-icon {
        font-size: 2rem;
        flex-shrink: 0;
    }

    .file-meta {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 2px;
        min-width: 0;
    }

    .file-name {
        font-weight: 600;
        color: var(--text-primary);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .file-size {
        font-size: 0.85rem;
        color: var(--text-muted);
    }

    .hidden-input {
        display: none;
    }
</style>
