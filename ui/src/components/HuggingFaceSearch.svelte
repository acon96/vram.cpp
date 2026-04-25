<script>
    import { onDestroy } from 'svelte';

    /**
     * @typedef {{
     *   repo: string,
     *   file: string,
      *   fileSizeBytes: number,
     *   revision: string,
     *   token: string,
        *   resolvedUrl: string,
     *   validated: boolean,
     *   response: object | null,
     *   metadata: object | null,
     *   error: string
     * }} HFSelectionState
     */

    /** @type {{
     *   onselectionchange: (state: HFSelectionState) => void,
          *   onvalidate: (selection: { repo: string, file: string, fileSizeBytes: number, revision: string, token: string, resolvedUrl: string }) => Promise<object>
     * }} */
    let { onselectionchange, onvalidate } = $props();

    const repoSearchMinChars = 2;
    const repoSearchDebounceMs = 250;

    let repoQuery = $state('');
    let repoDropdownOpen = $state(false);
    let searchResults = $state([]);
    let searching = $state(false);
    let searchError = $state('');
    let searchRequestId = 0;
    let searchTimerId = null;

    let revision = $state('main');
    let token = $state('');

    let ggufFiles = $state([]);
    let loadingFiles = $state(false);
    let filesError = $state('');
    let selectedFile = $state('');

    let validating = $state(false);
    let validationError = $state('');
    let validationResponse = $state(null);
    let resolvingUrl = $state(false);
    let resolvedUrl = $state('');
    let resolvedUrlError = $state('');
    let metadataPreviewOpen = $state(false);

    const showRepoDropdown = $derived(
        repoDropdownOpen && (
            searching
            || searchResults.length > 0
            || searchError.length > 0
            || repoQuery.trim().length > 0
        )
    );

    function normalizeRevision(value) {
        const trimmed = typeof value === 'string' ? value.trim() : '';
        return trimmed.length > 0 ? trimmed : 'main';
    }

    function encodeRepoPath(repo) {
        return String(repo || '')
            .split('/')
            .filter((segment) => segment.length > 0)
            .map((segment) => encodeURIComponent(segment))
            .join('/');
    }

    function encodePathPreservingSlashes(path) {
        return String(path || '')
            .split('/')
            .map((segment) => encodeURIComponent(segment))
            .join('/');
    }

    function buildCanonicalHfFileUrl(repo, file, revisionValue) {
        const encodedRepo = encodeRepoPath(repo);
        const encodedFile = encodePathPreservingSlashes(file);
        const encodedRevision = encodeURIComponent(normalizeRevision(revisionValue));

        if (encodedRepo.length === 0 || encodedFile.length === 0) {
            return '';
        }

        return `https://huggingface.co/${encodedRepo}/resolve/${encodedRevision}/${encodedFile}`;
    }

    function buildHeaders() {
        const headers = {
            Accept: 'application/json',
        };

        const trimmedToken = token.trim();
        if (trimmedToken.length > 0) {
            headers.Authorization = `Bearer ${trimmedToken}`;
        }

        return headers;
    }

    function buildHeadHeaders() {
        const headers = {
            Accept: 'application/octet-stream',
        };

        const trimmedToken = token.trim();
        if (trimmedToken.length > 0) {
            headers.Authorization = `Bearer ${trimmedToken}`;
        }

        return headers;
    }

    async function resolveFinalHfFileUrl(repo, file, revisionValue) {
        const canonicalUrl = buildCanonicalHfFileUrl(repo, file, revisionValue);
        if (canonicalUrl.length === 0) {
            throw new Error('invalid_huggingface_model_descriptor');
        }

        const response = await fetch(canonicalUrl, {
            method: 'HEAD',
            redirect: 'follow',
            headers: buildHeadHeaders(),
        });

        if (!response.ok) {
            throw new Error(`head_resolve_failed_http_${response.status}`);
        }

        if (typeof response.url === 'string' && response.url.length > 0) {
            return response.url;
        }

        return canonicalUrl;
    }

    function formatCount(value) {
        const parsed = Number(value);
        if (!Number.isFinite(parsed) || parsed < 0) {
            return '0';
        }
        return Math.round(parsed).toLocaleString();
    }

    function formatSize(bytes) {
        const n = Number(bytes);
        if (!Number.isFinite(n) || n <= 0) {
            return 'size unknown';
        }
        if (n >= 1024 * 1024 * 1024) {
            return `${(n / 1024 / 1024 / 1024).toFixed(2)} GiB`;
        }
        if (n >= 1024 * 1024) {
            return `${(n / 1024 / 1024).toFixed(1)} MiB`;
        }
        if (n >= 1024) {
            return `${Math.round(n / 1024)} KiB`;
        }
        return `${Math.round(n)} B`;
    }

    function emitSelection(validated, response = null, error = '') {
        if (typeof onselectionchange !== 'function') {
            return;
        }

        const selectedFileInfo = ggufFiles.find((candidate) => candidate.path === selectedFile) || null;
        const selectedFileSizeBytes = Number.isFinite(Number(selectedFileInfo?.size))
            ? Math.max(0, Math.trunc(Number(selectedFileInfo.size)))
            : 0;

        onselectionchange({
            repo: repoQuery.trim(),
            file: selectedFile,
            fileSizeBytes: selectedFileSizeBytes,
            revision: normalizeRevision(revision),
            token: token.trim(),
            resolvedUrl,
            validated,
            response,
            metadata: response?.metadata ?? null,
            error,
        });
    }

    function resetValidationState(error = '') {
        validationResponse = null;
        validationError = error;
        metadataPreviewOpen = false;
        emitSelection(false, null, error);
    }

    function clearLoadedRepoState() {
        ggufFiles = [];
        selectedFile = '';
        filesError = '';
        resolvedUrl = '';
        resolvedUrlError = '';
        resetValidationState('');
    }

    function queueSearchForInput() {
        if (searchTimerId != null) {
            clearTimeout(searchTimerId);
        }

        const trimmed = repoQuery.trim();
        if (trimmed.length < repoSearchMinChars) {
            searchResults = [];
            searchError = '';
            searching = false;
            return;
        }

        searchTimerId = setTimeout(() => {
            searchTimerId = null;
            void searchModels(trimmed);
        }, repoSearchDebounceMs);
    }

    async function searchModels(queryOverride = repoQuery.trim()) {
        const trimmed = String(queryOverride || '').trim();
        if (trimmed.length === 0) {
            searchResults = [];
            searchError = 'Enter a repository search query.';
            repoDropdownOpen = true;
            return;
        }

        const requestId = ++searchRequestId;
        searching = true;
        searchError = '';
        repoDropdownOpen = true;

        try {
            const url = new URL('https://huggingface.co/api/models');
            url.searchParams.set('search', trimmed);
            url.searchParams.set('limit', '25');

            const response = await fetch(url, {
                method: 'GET',
                headers: buildHeaders(),
            });

            if (requestId !== searchRequestId) {
                return;
            }

            if (!response.ok) {
                throw new Error(`search_failed_http_${response.status}`);
            }

            const payload = await response.json();
            const rows = Array.isArray(payload) ? payload : [];
            searchResults = rows
                .map((item) => {
                    const id = typeof item?.id === 'string'
                        ? item.id
                        : (typeof item?.modelId === 'string' ? item.modelId : '');

                    return {
                        id,
                        downloads: item?.downloads ?? 0,
                        likes: item?.likes ?? 0,
                        private: item?.private === true,
                    };
                })
                .filter((item) => item.id.length > 0);

            if (searchResults.length === 0) {
                searchError = 'No matching repositories found.';
            }
        } catch (error) {
            if (requestId !== searchRequestId) {
                return;
            }

            searchResults = [];
            searchError = error?.message ?? String(error);
        } finally {
            if (requestId === searchRequestId) {
                searching = false;
            }
        }
    }

    async function loadRepoFiles(repoInput = repoQuery) {
        const repo = String(repoInput || '').trim();
        if (repo.length === 0) {
            filesError = 'Enter or select a repository first.';
            clearLoadedRepoState();
            return;
        }

        loadingFiles = true;
        clearLoadedRepoState();

        try {
            const revisionName = normalizeRevision(revision);
            const encodedRepo = encodeRepoPath(repo);
            if (encodedRepo.length === 0) {
                throw new Error('invalid_repo_id');
            }

            const treeUrl = `https://huggingface.co/api/models/${encodedRepo}/tree/${encodeURIComponent(revisionName)}?recursive=1`;
            const response = await fetch(treeUrl, {
                method: 'GET',
                headers: buildHeaders(),
            });

            if (!response.ok) {
                throw new Error(`repo_tree_failed_http_${response.status}`);
            }

            const payload = await response.json();
            const rows = Array.isArray(payload) ? payload : [];
            const files = rows
                .map((item) => {
                    const path = typeof item?.path === 'string'
                        ? item.path
                        : (typeof item?.rfilename === 'string' ? item.rfilename : '');

                    const type = typeof item?.type === 'string' ? item.type : '';
                    return {
                        path,
                        type,
                        size: Number(item?.size) || 0,
                    };
                })
                .filter((item) => item.path.toLowerCase().endsWith('.gguf'))
                .filter((item) => item.type.length === 0 || item.type === 'file')
                .sort((a, b) => a.path.localeCompare(b.path));

            repoQuery = repo;
            ggufFiles = files;

            if (ggufFiles.length === 0) {
                filesError = 'No GGUF files were found in this repository/revision.';
                return;
            }

            if (ggufFiles.length === 1) {
                selectedFile = ggufFiles[0].path;
                resetValidationState('');
            }
        } catch (error) {
            ggufFiles = [];
            selectedFile = '';
            filesError = error?.message ?? String(error);
            resetValidationState(filesError);
        } finally {
            loadingFiles = false;
        }
    }

    async function handleRepoPick(repoId) {
        repoQuery = repoId;
        repoDropdownOpen = false;
        searchError = '';
        await loadRepoFiles(repoId);
    }

    function handleRepoInput(event) {
        repoQuery = event.currentTarget.value;
        clearLoadedRepoState();
        queueSearchForInput();
    }

    function handleRepoInputFocus() {
        repoDropdownOpen = true;
    }

    function handleRepoInputBlur() {
        // Delay closing so click handlers in the dropdown can execute.
        setTimeout(() => {
            repoDropdownOpen = false;
        }, 120);
    }

    function handleRepoInputKeydown(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            const trimmed = repoQuery.trim();
            if (trimmed.length > 0) {
                void loadRepoFiles(trimmed);
            }
        }
    }

    function handleRevisionInput(event) {
        revision = event.currentTarget.value;
        resolvedUrl = '';
        resolvedUrlError = '';
        resetValidationState('');
    }

    function handleTokenInput(event) {
        token = event.currentTarget.value;
        resolvedUrl = '';
        resolvedUrlError = '';
        resetValidationState('');
    }

    function handleFileSelection(event) {
        selectedFile = event.currentTarget.value;
        resolvedUrl = '';
        resolvedUrlError = '';
        resetValidationState('');
    }

    async function validateSelectedFile() {
        if (typeof onvalidate !== 'function') {
            validationError = 'validate_callback_missing';
            emitSelection(false, null, validationError);
            return;
        }

        const repo = repoQuery.trim();
        const file = selectedFile;
        const revisionName = normalizeRevision(revision);

        if (repo.length === 0 || file.length === 0) {
            validationError = 'Select a repository and GGUF file first.';
            emitSelection(false, null, validationError);
            return;
        }

        validating = true;
        resolvingUrl = true;
        validationError = '';
        resolvedUrlError = '';

        let resolvedUrlForRequest = '';
        try {
            resolvedUrlForRequest = await resolveFinalHfFileUrl(repo, file, revisionName);
            resolvedUrl = resolvedUrlForRequest;
        } catch (error) {
            resolvedUrl = '';
            resolvedUrlError = error?.message ?? String(error);

            // Fall back to the canonical HF resolve URL when HEAD resolution fails.
            resolvedUrlForRequest = buildCanonicalHfFileUrl(repo, file, revisionName);
        } finally {
            resolvingUrl = false;
        }

        try {
            const response = await onvalidate({
                repo,
                file,
                fileSizeBytes: Number.isFinite(Number(ggufFiles.find((candidate) => candidate.path === file)?.size))
                    ? Math.max(0, Math.trunc(Number(ggufFiles.find((candidate) => candidate.path === file)?.size)))
                    : 0,
                revision: revisionName,
                token: token.trim(),
                resolvedUrl: resolvedUrlForRequest,
            });

            if (response?.ok === true && response?.metadata != null) {
                validationResponse = response;
                validationError = '';
                emitSelection(true, response, '');
                return;
            }

            const message = response?.detail || response?.error || 'metadata_validation_failed';
            validationResponse = null;
            validationError = message;
            emitSelection(false, null, message);
        } catch (error) {
            const message = error?.message ?? String(error);
            validationResponse = null;
            validationError = message;
            emitSelection(false, null, message);
        } finally {
            validating = false;
        }
    }

    onDestroy(() => {
        if (searchTimerId != null) {
            clearTimeout(searchTimerId);
        }
    });
</script>

<div class="hf-shell">
    <div class="repo-select">
        <label for="hf-repo-input">Repository</label>
        <div class="repo-combobox">
            <input
                id="hf-repo-input"
                type="text"
                placeholder="Search or enter owner/repo"
                value={repoQuery}
                oninput={handleRepoInput}
                onfocus={handleRepoInputFocus}
                onblur={handleRepoInputBlur}
                onkeydown={handleRepoInputKeydown}
                spellcheck="false"
                autocomplete="off"
            />
            <button
                type="button"
                class="secondary"
                onclick={() => loadRepoFiles(repoQuery)}
                disabled={loadingFiles || repoQuery.trim().length === 0}
            >
                {loadingFiles ? 'Loading files...' : 'Refresh GGUF Files'}
            </button>
        </div>

        {#if showRepoDropdown}
            <div class="repo-results" role="listbox">
                {#if searchResults.length > 0}
                    {#each searchResults as item}
                        <button
                            type="button"
                            class="repo-row"
                            class:active={item.id === repoQuery.trim()}
                            onmousedown={(event) => event.preventDefault()}
                            onclick={() => handleRepoPick(item.id)}
                        >
                            <span class="repo-id">{item.id}</span>
                            <span class="repo-meta">
                                downloads {formatCount(item.downloads)} · likes {formatCount(item.likes)}
                                {item.private ? ' · private' : ''}
                            </span>
                        </button>
                    {/each}
                {:else if searching}
                    <div class="repo-empty">Searching repositories...</div>
                {:else if searchError}
                    <div class="repo-empty">{searchError}</div>
                {:else if repoQuery.trim().length > 0}
                    <button
                        type="button"
                        class="repo-row direct"
                        onmousedown={(event) => event.preventDefault()}
                        onclick={() => handleRepoPick(repoQuery.trim())}
                    >
                        <span class="repo-id">Use "{repoQuery.trim()}"</span>
                        <span class="repo-meta">Direct repository id</span>
                    </button>
                {/if}
            </div>
        {/if}
    </div>

    <div class="options-column">
        <label>
            Revision
            <input type="text" value={revision} oninput={handleRevisionInput} placeholder="main" />
        </label>
        <label>
            HF Token (optional)
            <input type="password" value={token} oninput={handleTokenInput} placeholder="hf_..." />
        </label>
    </div>

    {#if filesError}
        <p class="status error">{filesError}</p>
    {/if}

    {#if ggufFiles.length > 0}
        <label class="file-picker" for="hf-file-select">
            GGUF File
            <select id="hf-file-select" value={selectedFile} onchange={handleFileSelection}>
                <option value="">Select a file...</option>
                {#each ggufFiles as file}
                    <option value={file.path}>{file.path} ({formatSize(file.size)})</option>
                {/each}
            </select>
        </label>
    {/if}

    <div class="validate-row">
        <button
            type="button"
            class="validate-btn"
            onclick={validateSelectedFile}
            disabled={validating || resolvingUrl || repoQuery.trim().length === 0 || selectedFile.length === 0}
        >
            {validating ? 'Validating...' : 'Retrieve Tensor Info'}
        </button>
        {#if resolvingUrl}
            <span class="status">Resolving final file URL...</span>
        {/if}
        {#if validationResponse?.ok === true && validationResponse?.metadata}
            <button
                type="button"
                class="toggle-metadata-btn"
                onclick={() => { metadataPreviewOpen = !metadataPreviewOpen; }}
            >
                {metadataPreviewOpen ? '▾ Hide tensor info' : '▸ Show tensor info'}
            </button>
        {/if}
    </div>

    {#if resolvedUrlError}
        <p class="status error">URL resolution warning: {resolvedUrlError}</p>
    {/if}

    {#if validationError}
        <p class="status error">{validationError}</p>
    {/if}

    {#if metadataPreviewOpen && validationResponse?.ok === true && validationResponse?.metadata}
        <section class="metadata-preview">
            <h3>Metadata Preview</h3>
            <div class="metadata-grid">
                <div><span>Version</span><strong>{validationResponse.metadata.version ?? 'n/a'}</strong></div>
                <div><span>KV count</span><strong>{validationResponse.metadata.kvCount ?? 'n/a'}</strong></div>
                <div><span>Tensors</span><strong>{validationResponse.metadata.tensorCount ?? 'n/a'}</strong></div>
                <div><span>Prefix bytes consumed</span><strong>{formatSize(validationResponse.metadata.bytesConsumed ?? 0)}</strong></div>
            </div>
            {#if validationResponse.resolvedUrl}
                <p class="resolved-url">{validationResponse.resolvedUrl}</p>
            {/if}
        </section>
    {/if}
</div>

<style>
    .hf-shell {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    .repo-select {
        display: grid;
        gap: 8px;
        position: relative;
    }

    .repo-select > label,
    .options-grid label,
    .file-picker {
        display: flex;
        flex-direction: column;
        gap: 6px;
        font-size: 0.85rem;
        color: var(--text-primary);
        font-weight: 600;
    }

    .repo-combobox {
        display: grid;
        grid-template-columns: 1fr auto auto;
        gap: 10px;
    }

    input,
    select {
        width: 100%;
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--input-bg);
        color: var(--text-primary);
        padding: 10px;
        font: inherit;
    }

    button {
        border: 1px solid transparent;
        border-radius: 8px;
        background: var(--accent);
        color: var(--on-accent);
        font: inherit;
        font-weight: 600;
        cursor: pointer;
        padding: 10px 14px;
    }

    button.secondary {
        background: var(--surface-raised);
        border-color: var(--border);
        color: var(--text-primary);
    }

    button:disabled {
        opacity: 0.55;
        cursor: not-allowed;
    }

    .repo-results {
        max-height: 260px;
        overflow: auto;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: var(--surface-raised);
    }

    .repo-row {
        width: 100%;
        text-align: left;
        border-radius: 0;
        border: 0;
        border-bottom: 1px solid var(--border-subtle);
        background: transparent;
        color: var(--text-primary);
        padding: 10px 12px;
        display: flex;
        flex-direction: column;
        gap: 3px;
    }

    .repo-row:hover,
    .repo-row.active {
        background: var(--accent-subtle);
    }

    .repo-row:last-child {
        border-bottom: 0;
    }

    .repo-row.direct {
        border-radius: 10px;
    }

    .repo-empty {
        padding: 12px;
        font-size: 0.82rem;
        color: var(--text-muted);
    }

    .repo-id {
        font-weight: 600;
        font-family: var(--mono);
        font-size: 0.84rem;
        word-break: break-all;
    }

    .repo-meta {
        font-size: 0.78rem;
        color: var(--text-muted);
    }

    .options-column {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .validate-row {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }

    .validate-btn {
        min-width: 180px;
    }

    .status {
        margin: 0;
        font-size: 0.8rem;
    }

    .status.error {
        color: var(--error);
    }

    .toggle-metadata-btn {
        background: none;
        border: 1px solid var(--border);
        color: var(--accent);
        font-size: 0.82rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        cursor: pointer;
    }

    .toggle-metadata-btn:hover {
        background: var(--accent-subtle);
    }

    .metadata-preview {
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px;
        background: var(--surface-raised);
        display: grid;
        gap: 10px;
    }

    .metadata-preview h3 {
        margin: 0;
        font-size: 0.92rem;
        color: var(--text-primary);
    }

    .metadata-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
    }

    .metadata-grid div {
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 8px;
        display: flex;
        flex-direction: column;
        gap: 4px;
        background: var(--surface);
    }

    .metadata-grid span {
        font-size: 0.74rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .metadata-grid strong {
        font-family: var(--mono);
        font-size: 0.88rem;
        color: var(--text-primary);
    }

    .resolved-url {
        margin: 0;
        color: var(--text-muted);
        font-size: 0.74rem;
        overflow-wrap: anywhere;
    }

    @media (max-width: 900px) {
        .repo-combobox {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 760px) {
        .options-grid {
            grid-template-columns: 1fr;
        }

        .metadata-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
