#include "vram/fit_executor.h"

#include <algorithm>
#include <exception>
#include <cstdio>

#if defined(VRAM_HAS_LLAMA_FIT_EXECUTION)
#include "llama.h"
#include "llama-ext.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#endif

namespace vram {
namespace {

constexpr size_t MiB = 1024 * 1024;

uint64_t bytes_to_mib_ceil(uint64_t bytes) {
    return (bytes + MiB - 1) / MiB;
}

std::vector<uint64_t> broadcast(const std::vector<uint64_t> & values, size_t n, uint64_t fallback) {
    if (n == 0) {
        return {values.empty() ? fallback : values[0]};
    }

    std::vector<uint64_t> out(n, values.empty() ? fallback : values[0]);
    for (size_t i = 0; i < values.size() && i < n; ++i) {
        out[i] = values[i];
    }
    return out;
}

#if defined(VRAM_HAS_LLAMA_FIT_EXECUTION)
void discard_log_callback(ggml_log_level, const char *, void *) {
}

bool collect_memory_breakdown(
        const std::string & model_path,
        const llama_model_params & model_params,
        const llama_context_params & context_params,
        const common_fit_memory_override * memory_override,
        fit_execution_result & result,
        std::string & error) {
    llama_context_params adjusted_context_params = context_params;
#if defined(__EMSCRIPTEN__)
    adjusted_context_params.n_threads = 1;
    adjusted_context_params.n_threads_batch = 1;
#endif

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        error = "failed_to_load_fitted_model";
        return false;
    }

    llama_context * context = llama_init_from_model(model, adjusted_context_params);
    if (context == nullptr) {
        llama_model_free(model);
        error = "failed_to_create_fitted_context";
        return false;
    }

    const int device_count = llama_model_n_devices(model);
    const llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(context);

    std::vector<llama_memory_breakdown_data> device_mb(static_cast<size_t>(std::max(device_count, 0)));
    llama_memory_breakdown_data host_mb;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t buffer_type = buft_mb.first;
        const llama_memory_breakdown_data & mb = buft_mb.second;
        if (ggml_backend_buft_is_host(buffer_type)) {
            host_mb.model += mb.model;
            host_mb.context += mb.context;
            host_mb.compute += mb.compute;
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buffer_type);
        if (dev == nullptr) {
            continue;
        }

        for (int i = 0; i < device_count; ++i) {
            if (dev == llama_model_get_device(model, i)) {
                device_mb[static_cast<size_t>(i)].model += mb.model;
                device_mb[static_cast<size_t>(i)].context += mb.context;
                device_mb[static_cast<size_t>(i)].compute += mb.compute;
                break;
            }
        }
    }

    result.devices.clear();
    result.devices.reserve(static_cast<size_t>(std::max(device_count, 0)));
    result.totals = {};

    for (int i = 0; i < device_count; ++i) {
        ggml_backend_dev_t dev = llama_model_get_device(model, i);
        const llama_memory_breakdown_data & mb = device_mb[static_cast<size_t>(i)];

        uint64_t free_b = 0;
        uint64_t total_b = 0;
        if (memory_override != nullptr && static_cast<size_t>(i) < memory_override->n_devices) {
            free_b = static_cast<uint64_t>(memory_override->devices[i].free);
            total_b = static_cast<uint64_t>(memory_override->devices[i].total);
        } else {
            size_t free_sz = 0;
            size_t total_sz = 0;
            ggml_backend_dev_memory(dev, &free_sz, &total_sz);
            free_b = static_cast<uint64_t>(free_sz);
            total_b = static_cast<uint64_t>(total_sz);
        }

        const uint64_t self_b = static_cast<uint64_t>(mb.total());
        const uint64_t unaccounted_b = total_b >= self_b + free_b ? total_b - self_b - free_b : 0;

        fit_memory_breakdown_entry entry;
        entry.name = std::string(ggml_backend_dev_name(dev)) + " (" + ggml_backend_dev_description(dev) + ")";
        entry.total_mib = bytes_to_mib_ceil(total_b);
        entry.free_mib = bytes_to_mib_ceil(free_b);
        entry.model_mib = bytes_to_mib_ceil(static_cast<uint64_t>(mb.model));
        entry.context_mib = bytes_to_mib_ceil(static_cast<uint64_t>(mb.context));
        entry.compute_mib = bytes_to_mib_ceil(static_cast<uint64_t>(mb.compute));
        entry.unaccounted_mib = bytes_to_mib_ceil(unaccounted_b);
        result.devices.push_back(entry);

        result.totals.model_mib += entry.model_mib;
        result.totals.context_mib += entry.context_mib;
        result.totals.compute_mib += entry.compute_mib;
    }

    uint64_t host_free_b = 0;
    uint64_t host_total_b = 0;
    if (memory_override != nullptr && memory_override->override_host) {
        host_free_b = static_cast<uint64_t>(memory_override->host.free);
        host_total_b = static_cast<uint64_t>(memory_override->host.total);
    } else {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev != nullptr) {
            size_t free_sz = 0;
            size_t total_sz = 0;
            ggml_backend_dev_memory(cpu_dev, &free_sz, &total_sz);
            host_free_b = static_cast<uint64_t>(free_sz);
            host_total_b = static_cast<uint64_t>(total_sz);
        }
    }

    const uint64_t host_self_b = static_cast<uint64_t>(host_mb.total());
    const uint64_t host_unaccounted_b = host_total_b >= host_self_b + host_free_b ? host_total_b - host_self_b - host_free_b : 0;

    result.host = {};
    result.host.name = "Host";
    result.host.total_mib = bytes_to_mib_ceil(host_total_b);
    result.host.free_mib = bytes_to_mib_ceil(host_free_b);
    result.host.model_mib = bytes_to_mib_ceil(static_cast<uint64_t>(host_mb.model));
    result.host.context_mib = bytes_to_mib_ceil(static_cast<uint64_t>(host_mb.context));
    result.host.compute_mib = bytes_to_mib_ceil(static_cast<uint64_t>(host_mb.compute));
    result.host.unaccounted_mib = bytes_to_mib_ceil(host_unaccounted_b);

    result.totals.model_mib += result.host.model_mib;
    result.totals.context_mib += result.host.context_mib;
    result.totals.compute_mib += result.host.compute_mib;

    llama_free(context);
    llama_model_free(model);
    return true;
}

std::vector<uint64_t> detect_device_free_mib(const std::string & model_path, std::string & error) {
    llama_model_params mparams = llama_model_default_params();
    mparams.no_alloc = true;
    mparams.use_mmap = false;
    mparams.use_mlock = false;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        error = "failed to load model for device detection";
        return {};
    }

    const int nd = llama_model_n_devices(model);
    std::vector<uint64_t> free_mib;
    if (nd <= 0) {
        size_t free_b = 0;
        size_t total_b = 0;
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu != nullptr) {
            ggml_backend_dev_memory(cpu, &free_b, &total_b);
        }
        free_mib.push_back(static_cast<uint64_t>(free_b / MiB));
    } else {
        free_mib.reserve(static_cast<size_t>(nd));
        for (int i = 0; i < nd; ++i) {
            size_t free_b = 0;
            size_t total_b = 0;
            ggml_backend_dev_memory(llama_model_get_device(model, i), &free_b, &total_b);
            free_mib.push_back(static_cast<uint64_t>(free_b / MiB));
        }
    }

    llama_model_free(model);
    return free_mib;
}
#endif

} // namespace

bool fit_execution_available() {
#if defined(VRAM_HAS_LLAMA_FIT_EXECUTION)
    return true;
#else
    return false;
#endif
}

bool execute_fit_request(const fit_execution_request & request, fit_execution_result & result, std::string & error) {
#if !defined(VRAM_HAS_LLAMA_FIT_EXECUTION)
    (void) request;
    (void) result;
    error = "fit_execution_unavailable_in_this_build";
    return false;
#else
    std::string phase = "initialization";
    int original_common_log_verbosity = 0;
    bool common_log_verbosity_overridden = false;

    try {
    static const bool initialized = []() {
#if defined(__EMSCRIPTEN__)
        llama_log_set(discard_log_callback, nullptr);
#else
        common_init();
#endif
        llama_backend_init();
        return true;
    }();
    (void) initialized;

    phase = "prepare_params";
    llama_model_params mparams = llama_model_default_params();
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = request.n_ctx;
#if defined(__EMSCRIPTEN__)
    cparams.n_threads = 1;
    cparams.n_threads_batch = 1;
#endif
    mparams.n_gpu_layers = request.n_gpu_layers;

    std::vector<float> tensor_split(llama_max_devices(), 0.0f);
    std::vector<llama_model_tensor_buft_override> tbo(llama_max_tensor_buft_overrides());
    if (!tbo.empty()) {
        tbo[0].pattern = nullptr;
        tbo[0].buft = nullptr;
    }

    std::vector<uint64_t> margins_mib = request.fit_target_mib.empty() ? std::vector<uint64_t>{1024} : request.fit_target_mib;
    std::vector<uint64_t> warnings_target_above_source;
    std::vector<uint64_t> source_free_mib;

    if (!request.target_free_mib.empty()) {
        phase = "resolve_target_free";
        if (!request.override_device_free_mib.empty()) {
            source_free_mib = request.override_device_free_mib;
        } else {
            phase = "detect_device_free_memory";
            source_free_mib = detect_device_free_mib(request.model_path, error);
            if (source_free_mib.empty()) {
                return false;
            }
        }

        const std::vector<uint64_t> desired_free_mib = broadcast(request.target_free_mib, source_free_mib.size(), request.target_free_mib[0]);
        const std::vector<uint64_t> base_margin_mib = broadcast(margins_mib, source_free_mib.size(), margins_mib[0]);

        margins_mib.assign(source_free_mib.size(), 0);
        for (size_t i = 0; i < source_free_mib.size(); ++i) {
            if (desired_free_mib[i] > source_free_mib[i]) {
                margins_mib[i] = base_margin_mib[i];
                warnings_target_above_source.push_back(i);
            } else {
                const uint64_t delta = source_free_mib[i] - desired_free_mib[i];
                margins_mib[i] = delta + base_margin_mib[i];
            }
        }
    } else if (!request.override_device_free_mib.empty()) {
        margins_mib = broadcast(margins_mib, request.override_device_free_mib.size(), margins_mib[0]);
    }

    std::vector<common_fit_memory_override_device> override_devices;
    common_fit_memory_override memory_override = {0, nullptr, false, {0, 0}};
    bool use_memory_override = false;

    phase = "prepare_memory_override";
    if (!request.override_device_free_mib.empty()) {
        std::vector<uint64_t> totals_mib = request.override_device_total_mib;
        if (totals_mib.empty()) {
            totals_mib = request.override_device_free_mib;
        } else if (totals_mib.size() == 1 && request.override_device_free_mib.size() > 1) {
            totals_mib = broadcast(totals_mib, request.override_device_free_mib.size(), totals_mib[0]);
        } else if (totals_mib.size() != request.override_device_free_mib.size()) {
            error = "override_device_total_mib_size_mismatch";
            return false;
        }

        override_devices.resize(request.override_device_free_mib.size());
        for (size_t i = 0; i < request.override_device_free_mib.size(); ++i) {
            const uint64_t free_mib = request.override_device_free_mib[i];
            const uint64_t total_mib = std::max(totals_mib[i], free_mib);
            override_devices[i].free = static_cast<size_t>(free_mib * MiB);
            override_devices[i].total = static_cast<size_t>(total_mib * MiB);
        }

        memory_override.n_devices = override_devices.size();
        memory_override.devices = override_devices.data();
        use_memory_override = true;
    }

    if (request.has_override_host_free_mib || request.has_override_host_total_mib) {
        const uint64_t host_free_mib = request.has_override_host_free_mib
            ? request.override_host_free_mib
            : request.override_host_total_mib;
        const uint64_t host_total_mib = request.has_override_host_total_mib
            ? request.override_host_total_mib
            : host_free_mib;
        memory_override.override_host = true;
        memory_override.host.free = static_cast<size_t>(host_free_mib * MiB);
        memory_override.host.total = static_cast<size_t>(std::max(host_total_mib, host_free_mib) * MiB);
        use_memory_override = true;
    }

    std::vector<size_t> margins_bytes(margins_mib.size(), 0);
    for (size_t i = 0; i < margins_mib.size(); ++i) {
        margins_bytes[i] = static_cast<size_t>(margins_mib[i] * MiB);
    }

    ggml_log_callback original_log_callback = nullptr;
    void * original_log_user_data = nullptr;
    if (!request.show_fit_logs) {
        phase = "install_quiet_logger";
        llama_log_get(&original_log_callback, &original_log_user_data);
        llama_log_set(discard_log_callback, nullptr);
    }

#if defined(__EMSCRIPTEN__)
    if (!request.show_fit_logs) {
        phase = "suppress_common_fit_logs";
        original_common_log_verbosity = common_log_get_verbosity_thold();
        common_log_set_verbosity_thold(-1);
        common_log_verbosity_overridden = true;
    }
#endif

    phase = use_memory_override ? "run_fit_with_override" : "run_fit";
    const common_params_fit_status status = use_memory_override
        ? common_fit_params_with_memory_override(
            request.model_path.c_str(),
            &mparams,
            &cparams,
            tensor_split.data(),
            tbo.data(),
            margins_bytes.data(),
            request.min_ctx,
            GGML_LOG_LEVEL_ERROR,
            &memory_override,
            false)
        : common_fit_params(
            request.model_path.c_str(),
            &mparams,
            &cparams,
            tensor_split.data(),
            tbo.data(),
            margins_bytes.data(),
            request.min_ctx,
            GGML_LOG_LEVEL_ERROR);

    if (common_log_verbosity_overridden) {
        common_log_set_verbosity_thold(original_common_log_verbosity);
    }

    if (!request.show_fit_logs) {
        phase = "restore_logger";
        llama_log_set(original_log_callback, original_log_user_data);
    }

    result.ok = status == COMMON_PARAMS_FIT_STATUS_SUCCESS;
    result.status = static_cast<int>(status);
    result.n_ctx = cparams.n_ctx;
    result.n_gpu_layers = mparams.n_gpu_layers;
    result.fit_target_mib = margins_mib;
    result.memory_override_enabled = use_memory_override;
    result.device_free_mib = request.override_device_free_mib;
    result.device_total_mib = request.override_device_total_mib.empty()
        ? request.override_device_free_mib
        : request.override_device_total_mib;
    result.host_override_enabled = memory_override.override_host;
    result.host_free_mib = memory_override.override_host ? memory_override.host.free / MiB : 0;
    result.host_total_mib = memory_override.override_host ? memory_override.host.total / MiB : 0;
    result.totals = {};
    result.devices.clear();
    result.host = {};
    result.warnings.clear();
    for (uint64_t device_index : warnings_target_above_source) {
        result.warnings.push_back("target_free_above_source_device_" + std::to_string(device_index));
    }

    if (result.ok) {
        phase = "collect_memory_breakdown";
        const common_fit_memory_override * breakdown_override = use_memory_override ? &memory_override : nullptr;
        if (!collect_memory_breakdown(request.model_path, mparams, cparams, breakdown_override, result, error)) {
            return false;
        }
    }

    return true;
    } catch (const std::exception & exception) {
        if (common_log_verbosity_overridden) {
            common_log_set_verbosity_thold(original_common_log_verbosity);
        }
        error = "fit_execution_exception[" + phase + "]: " + exception.what();
        return false;
    } catch (...) {
        if (common_log_verbosity_overridden) {
            common_log_set_verbosity_thold(original_common_log_verbosity);
        }
        error = "fit_execution_exception[" + phase + "]: unknown_exception";
        return false;
    }
#endif
}

} // namespace vram