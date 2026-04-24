#include "vram/fit_executor.h"

#include <algorithm>
#include <exception>
#include <cstdio>
#include <string>

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

uint64_t bytes_to_mib_floor(uint64_t bytes) {
    return bytes / MiB;
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

std::string join_u64_csv(const std::vector<uint64_t> & values) {
    std::string out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out.push_back(',');
        }
        out += std::to_string(values[i]);
    }
    return out;
}

#if defined(VRAM_HAS_LLAMA_FIT_EXECUTION)
void discard_log_callback(ggml_log_level, const char *, void *) {
}

void emscripten_passthrough_log_callback(ggml_log_level level, const char * text, void *) {
    FILE * stream = stdout;
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_DEBUG) {
        stream = stderr;
    }
    if (text != nullptr) {
        std::fputs(text, stream);
    }
    std::fflush(stream);
}

bool collect_memory_breakdown(
        const std::string & model_path,
        const llama_model_params & model_params,
        const llama_context_params & context_params,
        const common_fit_memory_override * memory_override,
        fit_execution_result & result,
        std::string & error) {
    llama_model_params adjusted_model_params = model_params;
    llama_context_params adjusted_context_params = context_params;
#if defined(__EMSCRIPTEN__)
    adjusted_context_params.n_threads = 1;
    adjusted_context_params.n_threads_batch = 1;
#endif

    std::vector<common_fit_memory_data> memory_rows;
    std::vector<std::string> names;
    std::string collect_error;
    if (!common_fit_collect_memory_data(
            model_path.c_str(),
            &adjusted_model_params,
            &adjusted_context_params,
            GGML_LOG_LEVEL_ERROR,
            memory_override,
            memory_rows,
            names,
            collect_error,
            false)) {
        error = collect_error.empty()
            ? "failed_to_collect_fit_memory_data"
            : "failed_to_collect_fit_memory_data[" + collect_error + "]";
        return false;
    }

    if (memory_rows.empty() || memory_rows.size() != names.size()) {
        error = "invalid_fit_memory_data";
        return false;
    }

    result.devices.clear();
    result.totals = {};

    const size_t host_index = memory_rows.size() - 1;
    result.devices.reserve(host_index);

    for (size_t i = 0; i < host_index; ++i) {
        const common_fit_memory_data & row = memory_rows[i];
        const uint64_t total_b = row.total > 0 ? static_cast<uint64_t>(row.total) : 0;
        const uint64_t free_b = row.free > 0 ? static_cast<uint64_t>(row.free) : 0;
        const uint64_t model_b = static_cast<uint64_t>(row.model);
        const uint64_t context_b = static_cast<uint64_t>(row.context);
        const uint64_t compute_b = static_cast<uint64_t>(row.compute);
        const uint64_t self_b = model_b + context_b + compute_b;
        const uint64_t unaccounted_b = total_b >= self_b + free_b ? total_b - self_b - free_b : 0;

        fit_memory_breakdown_entry entry;
        entry.name = names[i];
        entry.total_mib = bytes_to_mib_floor(total_b);
        entry.free_mib = bytes_to_mib_floor(free_b);
        entry.model_mib = bytes_to_mib_floor(model_b);
        entry.context_mib = bytes_to_mib_floor(context_b);
        entry.compute_mib = bytes_to_mib_floor(compute_b);
        entry.unaccounted_mib = bytes_to_mib_floor(unaccounted_b);
        result.devices.push_back(entry);

        result.totals.model_mib += entry.model_mib;
        result.totals.context_mib += entry.context_mib;
        result.totals.compute_mib += entry.compute_mib;
    }

    const common_fit_memory_data & host_row = memory_rows[host_index];
    const uint64_t host_total_b = host_row.total > 0 ? static_cast<uint64_t>(host_row.total) : 0;
    const uint64_t host_free_b = host_row.free > 0 ? static_cast<uint64_t>(host_row.free) : 0;
    const uint64_t host_model_b = static_cast<uint64_t>(host_row.model);
    const uint64_t host_context_b = static_cast<uint64_t>(host_row.context);
    const uint64_t host_compute_b = static_cast<uint64_t>(host_row.compute);
    const uint64_t host_self_b = host_model_b + host_context_b + host_compute_b;
    const uint64_t host_unaccounted_b = host_total_b >= host_self_b + host_free_b ? host_total_b - host_self_b - host_free_b : 0;

    result.host = {};
    result.host.name = names[host_index];
    result.host.total_mib = bytes_to_mib_floor(host_total_b);
    result.host.free_mib = bytes_to_mib_floor(host_free_b);
    result.host.model_mib = bytes_to_mib_floor(host_model_b);
    result.host.context_mib = bytes_to_mib_floor(host_context_b);
    result.host.compute_mib = bytes_to_mib_floor(host_compute_b);
    result.host.unaccounted_mib = bytes_to_mib_floor(host_unaccounted_b);

    result.totals.model_mib += result.host.model_mib;
    result.totals.context_mib += result.host.context_mib;
    result.totals.compute_mib += result.host.compute_mib;

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
    ggml_log_callback original_log_callback = nullptr;
    void * original_log_user_data = nullptr;
    bool llama_logger_overridden = false;

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
    if (request.n_batch > 0) {
        cparams.n_batch = request.n_batch;
    }
    if (request.n_ubatch > 0) {
        cparams.n_ubatch = request.n_ubatch;
    }
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

    llama_log_get(&original_log_callback, &original_log_user_data);
    if (request.show_fit_logs) {
#if defined(__EMSCRIPTEN__)
        phase = "install_emscripten_passthrough_logger";
        llama_log_set(emscripten_passthrough_log_callback, nullptr);
        llama_logger_overridden = true;
        std::fprintf(stderr,
            "[vram-fit] request model=%s n_ctx=%u n_batch=%u n_ubatch=%u min_ctx=%u n_gpu_layers=%d fit_target_mib=%s target_free_mib=%s override_device_free_mib=%s override_device_total_mib=%s override_host_free_mib=%llu override_host_total_mib=%llu\n",
            request.model_path.c_str(),
            request.n_ctx,
            request.n_batch,
            request.n_ubatch,
            request.min_ctx,
            request.n_gpu_layers,
            join_u64_csv(request.fit_target_mib).c_str(),
            join_u64_csv(request.target_free_mib).c_str(),
            join_u64_csv(request.override_device_free_mib).c_str(),
            join_u64_csv(request.override_device_total_mib).c_str(),
            static_cast<unsigned long long>(request.override_host_free_mib),
            static_cast<unsigned long long>(request.override_host_total_mib));
        std::fflush(stderr);
#endif
    } else {
        phase = "install_quiet_logger";
        llama_log_set(discard_log_callback, nullptr);
        llama_logger_overridden = true;
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

    if (llama_logger_overridden) {
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
#if defined(__EMSCRIPTEN__)
            if (request.show_fit_logs) {
                std::fprintf(stderr,
                    "[vram-fit] collect_memory_breakdown failed error=%s fitted_n_ctx=%u fitted_n_gpu_layers=%d\n",
                    error.c_str(),
                    cparams.n_ctx,
                    mparams.n_gpu_layers);
                std::fflush(stderr);
            }
#endif
            return false;
        }
    }

    return true;
    } catch (const std::exception & exception) {
        if (common_log_verbosity_overridden) {
            common_log_set_verbosity_thold(original_common_log_verbosity);
        }
        if (llama_logger_overridden) {
            llama_log_set(original_log_callback, original_log_user_data);
        }
        error = "fit_execution_exception[" + phase + "]: " + exception.what();
        return false;
    } catch (...) {
        if (common_log_verbosity_overridden) {
            common_log_set_verbosity_thold(original_common_log_verbosity);
        }
        if (llama_logger_overridden) {
            llama_log_set(original_log_callback, original_log_user_data);
        }
        error = "fit_execution_exception[" + phase + "]: unknown_exception";
        return false;
    }
#endif
}

} // namespace vram