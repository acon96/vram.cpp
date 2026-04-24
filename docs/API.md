# VRAM Predictor API Reference

This document defines the JSON request and response shapes used by the WASM predictor.
These contracts must stay in sync across `cpp/src/predictor_api.cpp`, the API tests in `cpp/tests/`, and browser helper assumptions in `web/vram_predictor_browser.js`.

---

## Request

```json
{
  "mode": "metadata" | "fit",
  "model": {
    "source": "local" | "huggingface",
    "path": "<string — local file path or virtual WASM FS path>",
    "huggingFace": {
      "repo": "<string>",
      "file": "<string>",
      "revision": "<string>"
    }
  },
  "runtime": {
    "n_ctx": <integer ≥ 1>,
    "n_batch": <integer ≥ 1>,
    "n_ubatch": <integer ≥ 1>,
    "cache_type_k": "<string>",
    "cache_type_v": "<string>",
    "n_gpu_layers": <integer ≥ -1>
  },
  "device": {
    "host_ram_bytes": <integer ≥ 0>,
    "fit_target_mib": [<integer ≥ 0>, ...],
    "target_free_mib": [<integer ≥ 0>, ...],
    "gpus": [
      {
        "id": "<string>",
        "name": "<string>",
        "index": <integer ≥ 0>,
        "free_bytes": <integer ≥ 0>,
        "total_bytes": <integer ≥ 0>
      }
    ]
  },
  "fit": {
    "fit_harness_binary": "<string>",
    "min_ctx": <integer ≥ 0>,
    "execute_in_process": <boolean>,
    "show_fit_logs": <boolean>
  }
}
```

### Field notes

| Field | Required | Description |
|-------|----------|-------------|
| `mode` | yes | `"metadata"` for fast header-only estimate, `"fit"` for full llama-fit projection |
| `model.source` | yes | `"local"` or `"huggingface"` |
| `model.path` | no | Local or virtual WASM FS path (used when `source = "local"`) |
| `model.huggingFace` | no | HF repo details (used when `source = "huggingface"`) |
| `runtime.n_ctx` | yes | Context window size in tokens |
| `runtime.cache_type_k` | yes | KV cache key dtype (e.g. `"f16"`, `"q8_0"`) |
| `runtime.cache_type_v` | yes | KV cache value dtype |
| `device.host_ram_bytes` | yes | Available host RAM in bytes |
| `device.fit_target_mib` | no | Per-device llama-fit margin targets in MiB (maps to `--fit-target`) |
| `device.target_free_mib` | no | Per-device desired free memory in MiB after fit adjustments |
| `device.gpus` | no | GPU device list; omit for CPU-only mode |
| `fit` | no | Options for the fit execution pass |

---

## Response

```json
{
  "ok": <boolean>,
  "error": "<string>",
  "message": "<string>",
  "detail": "<string>",
  "source": "<string>",
  "path": "<string>",
  "resolvedUrl": "<string>",
  "supportedSources": ["<string>", ...],
  "minimumRequiredBytes": <integer>,
  "requestedBytes": <integer>,
  "requests": [
    {
      "url": "<string>",
      "start": <integer>,
      "end": <integer>,
      "headers": [
        { "name": "<string>", "value": "<string>" }
      ]
    }
  ],
  "metadata": {
    "version": <integer>,
    "kvCount": <integer>,
    "tensorCount": <integer>,
    "bytesConsumed": <integer>,
    "tensorListTruncated": <boolean>,
    "tensors": [
      {
        "name": "<string>",
        "dimensions": [<integer>, ...],
        "ggmlType": <integer>,
        "dataOffset": <integer>
      }
    ]
  },
  "fit": {
    "executedInProcess": <boolean>,
    "command": {
      "binary": "<string>",
      "args": ["<string>", ...]
    },
    "targets": {
      "fitMiB": [<integer>, ...],
      "targetFreeMiB": [<integer>, ...]
    },
    "overrides": {
      "deviceFreeMiB": [<integer>, ...],
      "deviceTotalMiB": [<integer>, ...],
      "hostFreeMiB": <integer>
    },
    "showLogs": <boolean>,
    "recommended": {
      "n_ctx": <integer>,
      "n_gpu_layers": <integer>
    },
    "status": <integer>,
    "warnings": ["<string>", ...],
    "memoryBytes": {
      "weights": <integer>,
      "kvCache": <integer>,
      "device": <integer>,
      "host": <integer>
    },
    "breakdown": {
      "totals": {
        "modelMiB": <integer>,
        "contextMiB": <integer>,
        "computeMiB": <integer>
      },
      "devices": [
        {
          "name": "<string>",
          "totalMiB": <integer>,
          "freeMiB": <integer>,
          "modelMiB": <integer>,
          "contextMiB": <integer>,
          "computeMiB": <integer>,
          "unaccountedMiB": <integer>
        }
      ],
      "host": {
        "name": "<string>",
        "totalMiB": <integer>,
        "freeMiB": <integer>,
        "modelMiB": <integer>,
        "contextMiB": <integer>,
        "computeMiB": <integer>,
        "unaccountedMiB": <integer>
      }
    }
  }
}
```

### Response shape guidelines

- Keep responses compact. The `ok` field is always present.
- `fit.status` codes: `0` = success, `1` = no viable allocation, `2` = hard failure.
- `fit.breakdown.devices` is an array parallel to the `device.gpus` input array.
- `metadata` is populated only for metadata-mode or HF planning responses.
- `requests` describes the HTTP range requests issued (HF planning mode).
