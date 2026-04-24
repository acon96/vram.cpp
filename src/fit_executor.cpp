#include "vram/fit_executor.h"

#include <algorithm>
#include <cstdio>

#if defined(VRAM_HAS_LLAMA_FIT_EXECUTION)
#include "llama.h"
#include "llama-ext.h"

#include "common.h"
#include "fit.h"
#endif

namespace vram {
namespace {

constexpr size_t MiB = 1024 * 1024;

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
    static const bool initialized = []() {
        common_init();
        llama_backend_init();
        return true;
    }();
    (void) initialized;

    llama_model_params mparams = llama_model_default_params();
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = request.n_ctx;
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
        if (!request.override_device_free_mib.empty()) {
            source_free_mib = request.override_device_free_mib;
        } else {
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
        llama_log_get(&original_log_callback, &original_log_user_data);
        llama_log_set(discard_log_callback, nullptr);
    }

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

    if (!request.show_fit_logs) {
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
    result.warnings.clear();
    for (uint64_t device_index : warnings_target_above_source) {
        result.warnings.push_back("target_free_above_source_device_" + std::to_string(device_index));
    }

    return true;
#endif
}

} // namespace vram