#include "vram/fit_executor.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <limits>
#include <string>
#include <vector>

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

uint64_t saturating_add_u64(uint64_t a, uint64_t b) {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
        return std::numeric_limits<uint64_t>::max();
    }
    return a + b;
}

#if defined(VRAM_HAS_LLAMA_FIT_EXECUTION)

struct fit_memory_row_bytes {
    uint64_t total = 0;
    uint64_t free = 0;
    uint64_t model = 0;
    uint64_t context = 0;
    uint64_t compute = 0;
};

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
        const std::vector<sim_device_spec> & simulated_devices,
        bool emit_debug_logs,
        bool host_override_enabled,
        uint64_t host_override_free_mib,
        uint64_t host_override_total_mib,
        fit_execution_result & result,
        std::string & error) {
    llama_model_params adjusted_model_params = model_params;
    adjusted_model_params.no_alloc = true;
    adjusted_model_params.use_mmap = false;
    adjusted_model_params.use_mlock = false;

    llama_context_params adjusted_context_params = context_params;
#if defined(__EMSCRIPTEN__)
    adjusted_context_params.n_threads = 1;
    adjusted_context_params.n_threads_batch = 1;
#endif

    llama_model * model = llama_model_load_from_file(model_path.c_str(), adjusted_model_params);
    if (model == nullptr) {
        error = "failed_to_load_model_for_memory_breakdown";
        return false;
    }

    llama_context * ctx = llama_init_from_model(model, adjusted_context_params);
    if (ctx == nullptr) {
        llama_model_free(model);
        error = "failed_to_create_context_for_memory_breakdown";
        return false;
    }

    const int nd = llama_model_n_devices(model);
    std::vector<ggml_backend_dev_t> devices;
    devices.reserve(static_cast<size_t>(std::max(0, nd)));
    for (int i = 0; i < nd; ++i) {
        devices.push_back(llama_model_get_device(model, i));
        if (emit_debug_logs) {
            std::fprintf(stderr,
                "[vram-fit] breakdown model_device[%d]=%p name=%s desc=%s\n",
                i,
                (void *) devices.back(),
                devices.back() ? ggml_backend_dev_name(devices.back()) : "<null>",
                devices.back() ? ggml_backend_dev_description(devices.back()) : "<null>");
        }
    }

    std::vector<fit_memory_row_bytes> device_rows(devices.size());
    fit_memory_row_bytes host_row = {};

    const llama_memory_breakdown breakdown = llama_get_memory_breakdown(ctx);
    for (const auto & buft_mb : breakdown) {
        ggml_backend_buffer_type_t buft = buft_mb.first;
        const llama_memory_breakdown_data & mb = buft_mb.second;

        if (ggml_backend_buft_is_host(buft)) {
            if (emit_debug_logs) {
                std::fprintf(stderr,
                    "[vram-fit] breakdown buft=%p name=%s host=1 model=%zu context=%zu compute=%zu\n",
                    (void *) buft, ggml_backend_buft_name(buft),
                    mb.model, mb.context, mb.compute);
            }
            host_row.model += static_cast<uint64_t>(mb.model);
            host_row.context += static_cast<uint64_t>(mb.context);
            host_row.compute += static_cast<uint64_t>(mb.compute);
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev == nullptr) {
            if (emit_debug_logs) {
                std::fprintf(stderr,
                    "[vram-fit] breakdown buft=%p name=%s host=0 dev=<null> model=%zu context=%zu compute=%zu\n",
                    (void *) buft, ggml_backend_buft_name(buft),
                    mb.model, mb.context, mb.compute);
            }
            continue;
        }

        bool matched = false;
        for (size_t i = 0; i < devices.size(); ++i) {
            if (devices[i] == dev) {
                device_rows[i].model += static_cast<uint64_t>(mb.model);
                device_rows[i].context += static_cast<uint64_t>(mb.context);
                device_rows[i].compute += static_cast<uint64_t>(mb.compute);
                if (emit_debug_logs) {
                    std::fprintf(stderr,
                        "[vram-fit] breakdown buft=%p name=%s host=0 dev=%p -> model_device[%zu] model=%zu context=%zu compute=%zu\n",
                        (void *) buft, ggml_backend_buft_name(buft), (void *) dev,
                        i, mb.model, mb.context, mb.compute);
                }
                matched = true;
                break;
            }
        }
        if (emit_debug_logs && !matched) {
            std::fprintf(stderr,
                "[vram-fit] breakdown buft=%p name=%s host=0 dev=%p unmatched model=%zu context=%zu compute=%zu\n",
                (void *) buft, ggml_backend_buft_name(buft), (void *) dev,
                mb.model, mb.context, mb.compute);
        }
    }

    if (host_override_enabled) {
        const uint64_t host_free_mib = host_override_free_mib > 0 ? host_override_free_mib : host_override_total_mib;
        const uint64_t host_total_mib = host_override_total_mib > 0 ? host_override_total_mib : host_free_mib;
        host_row.free = host_free_mib * MiB;
        host_row.total = std::max(host_row.free, host_total_mib * MiB);
    } else {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu != nullptr) {
            size_t host_free = 0;
            size_t host_total = 0;
            ggml_backend_dev_memory(cpu, &host_free, &host_total);
            host_row.free = static_cast<uint64_t>(host_free);
            host_row.total = static_cast<uint64_t>(host_total);
        }
    }

    for (size_t i = 0; i < device_rows.size(); ++i) {
        size_t free_b = 0;
        size_t total_b = 0;
        ggml_backend_dev_memory(devices[i], &free_b, &total_b);

        if (free_b == 0 && total_b == 0) {
            if (i < simulated_devices.size()) {
                free_b = static_cast<size_t>(simulated_devices[i].free_bytes);
                total_b = static_cast<size_t>(std::max(simulated_devices[i].total_bytes, simulated_devices[i].free_bytes));
            } else {
                free_b = static_cast<size_t>(host_row.free);
                total_b = static_cast<size_t>(host_row.total);
            }
        }

        device_rows[i].free = static_cast<uint64_t>(free_b);
        device_rows[i].total = static_cast<uint64_t>(total_b);
        if (emit_debug_logs) {
            std::fprintf(stderr,
                "[vram-fit] breakdown device[%zu] free=%zu total=%zu\n",
                i, free_b, total_b);
        }
    }

    result.devices.clear();
    result.totals = {};
    result.devices.reserve(device_rows.size());

    for (size_t i = 0; i < device_rows.size(); ++i) {
        const fit_memory_row_bytes & row = device_rows[i];
        const uint64_t total_b = row.total;
        const uint64_t free_b = row.free;
        const uint64_t model_b = row.model;
        const uint64_t context_b = row.context;
        const uint64_t compute_b = row.compute;
        const uint64_t self_b = model_b + context_b + compute_b;
        const bool overcommitted = self_b > free_b;
        const uint64_t unaccounted_b = overcommitted ? 0 : total_b - self_b - free_b;
        if (overcommitted) {
            result.warnings.push_back("memory_breakdown_overcommitted_device_" + std::to_string(i));
            if (emit_debug_logs) {
                std::fprintf(stderr,
                    "[vram-fit] breakdown overcommit device[%zu]: self=%llu free=%llu over_free=%llu total=%llu\n",
                    i,
                    static_cast<unsigned long long>(self_b),
                    static_cast<unsigned long long>(free_b),
                    static_cast<unsigned long long>(self_b - free_b),
                    static_cast<unsigned long long>(total_b));
            }
        }

        fit_memory_breakdown_entry entry;
        if (i < simulated_devices.size() && !simulated_devices[i].name.empty()) {
            entry.name = simulated_devices[i].name;
        } else if (i < devices.size() && devices[i] != nullptr) {
            entry.name = std::string(ggml_backend_dev_name(devices[i])) + " (" + ggml_backend_dev_description(devices[i]) + ")";
        } else {
            entry.name = "GPU " + std::to_string(i);
        }
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

    const uint64_t host_total_b = host_row.total;
    const uint64_t host_free_b = host_row.free;
    const uint64_t host_model_b = host_row.model;
    const uint64_t host_context_b = host_row.context;
    const uint64_t host_compute_b = host_row.compute;
    const uint64_t host_self_b = host_model_b + host_context_b + host_compute_b;
    const bool host_overcommitted = host_self_b > host_free_b;
    const uint64_t host_unaccounted_b = host_overcommitted ? 0 : host_total_b - host_self_b - host_free_b;
    if (host_overcommitted) {
        result.warnings.push_back("memory_breakdown_overcommitted_host");
        if (emit_debug_logs) {
            std::fprintf(stderr,
                "[vram-fit] breakdown overcommit host: self=%llu free=%llu over_free=%llu total=%llu\n",
                static_cast<unsigned long long>(host_self_b),
                static_cast<unsigned long long>(host_free_b),
                static_cast<unsigned long long>(host_self_b - host_free_b),
                static_cast<unsigned long long>(host_total_b));
        }
    }

    result.host = {};
    result.host.name = "Host";
    result.host.total_mib = bytes_to_mib_floor(host_total_b);
    result.host.free_mib = bytes_to_mib_floor(host_free_b);
    result.host.model_mib = bytes_to_mib_floor(host_model_b);
    result.host.context_mib = bytes_to_mib_floor(host_context_b);
    result.host.compute_mib = bytes_to_mib_floor(host_compute_b);
    result.host.unaccounted_mib = bytes_to_mib_floor(host_unaccounted_b);

    result.totals.model_mib += result.host.model_mib;
    result.totals.context_mib += result.host.context_mib;
    result.totals.compute_mib += result.host.compute_mib;

    llama_free(ctx);
    llama_model_free(model);

    return true;
}

void fill_breakdown_fallback(
        const std::vector<sim_device_spec> & simulated_devices,
        bool host_override_enabled,
        uint64_t host_override_free_mib,
        uint64_t host_override_total_mib,
        fit_execution_result & result) {
    result.totals = {};
    result.devices.clear();
    result.devices.reserve(simulated_devices.size());

    for (size_t i = 0; i < simulated_devices.size(); ++i) {
        fit_memory_breakdown_entry entry;
        entry.name = simulated_devices[i].name.empty() ? "GPU " + std::to_string(i) : simulated_devices[i].name;
        entry.total_mib = simulated_devices[i].total_bytes / MiB;
        entry.free_mib = simulated_devices[i].free_bytes / MiB;
        result.devices.push_back(entry);
    }

    result.host = {};
    result.host.name = "Host";

    if (host_override_enabled) {
        const uint64_t host_free = host_override_free_mib > 0 ? host_override_free_mib : host_override_total_mib;
        const uint64_t host_total = host_override_total_mib > 0 ? host_override_total_mib : host_free;
        result.host.free_mib = host_free;
        result.host.total_mib = std::max(host_total, host_free);
    }
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

    sim_backend simulated_backend(request.simulated_devices);
    ggml_backend_dev_t no_devices[1] = {nullptr};

    if (request.simulated_devices.empty()) {
        mparams.devices = no_devices;
    } else {
        if (!simulated_backend.valid()) {
            error = "failed_to_initialize_simulated_devices";
            return false;
        }
        mparams.devices = simulated_backend.devices();
    }

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
    std::vector<uint64_t> simulated_device_free_mib;
    std::vector<uint64_t> simulated_device_total_mib;
    simulated_device_free_mib.reserve(request.simulated_devices.size());
    simulated_device_total_mib.reserve(request.simulated_devices.size());
    for (const sim_device_spec & spec : request.simulated_devices) {
        simulated_device_free_mib.push_back(spec.free_bytes / MiB);
        simulated_device_total_mib.push_back(std::max(spec.total_bytes, spec.free_bytes) / MiB);
    }

    if (!request.target_free_mib.empty()) {
        phase = "resolve_target_free";
        if (simulated_device_free_mib.empty()) {
            error = "target_free_mib_requires_simulated_devices";
            return false;
        }

        source_free_mib = simulated_device_free_mib;

        const std::vector<uint64_t> desired_free_mib = broadcast(request.target_free_mib, source_free_mib.size(), request.target_free_mib[0]);
        const std::vector<uint64_t> base_margin_mib = broadcast(margins_mib, source_free_mib.size(), margins_mib[0]);

        margins_mib.assign(source_free_mib.size(), 0);
        for (size_t i = 0; i < source_free_mib.size(); ++i) {
            if (desired_free_mib[i] > source_free_mib[i]) {
                margins_mib[i] = base_margin_mib[i];
                warnings_target_above_source.push_back(i);
            } else {
                margins_mib[i] = saturating_add_u64(desired_free_mib[i], base_margin_mib[i]);
            }
        }
    } else if (!simulated_device_free_mib.empty()) {
        margins_mib = broadcast(margins_mib, simulated_device_free_mib.size(), margins_mib[0]);
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
            join_u64_csv(simulated_device_free_mib).c_str(),
            join_u64_csv(simulated_device_total_mib).c_str(),
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

    phase = "run_fit";
    const common_params_fit_status status = common_fit_params(
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
    result.memory_override_enabled = !request.simulated_devices.empty();
    result.device_free_mib = simulated_device_free_mib;
    result.device_total_mib = simulated_device_total_mib;
    result.host_override_enabled = request.has_override_host_free_mib || request.has_override_host_total_mib;
    result.host_free_mib = result.host_override_enabled
        ? (request.has_override_host_free_mib ? request.override_host_free_mib : request.override_host_total_mib)
        : 0;
    result.host_total_mib = result.host_override_enabled
        ? std::max(request.override_host_total_mib, result.host_free_mib)
        : 0;
    result.totals = {};
    result.devices.clear();
    result.host = {};
    result.warnings.clear();
    for (uint64_t device_index : warnings_target_above_source) {
        result.warnings.push_back("target_free_above_source_device_" + std::to_string(device_index));
    }

    if (result.ok) {
        phase = "collect_memory_breakdown";
        if (!collect_memory_breakdown(
                request.model_path,
                mparams,
                cparams,
                request.simulated_devices,
            request.show_fit_logs,
                result.host_override_enabled,
                result.host_free_mib,
                result.host_total_mib,
                result,
                error)) {
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
            result.warnings.push_back("memory_breakdown_unavailable");
            if (!error.empty()) {
                result.warnings.push_back("memory_breakdown_error[" + error + "]");
            }
            fill_breakdown_fallback(
                request.simulated_devices,
                result.host_override_enabled,
                result.host_free_mib,
                result.host_total_mib,
                result);
            error.clear();
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