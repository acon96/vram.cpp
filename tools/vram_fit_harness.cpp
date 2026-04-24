#include "llama.h"
#include "llama-ext.h"

#include "common.h"
#include "fit.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr size_t MiB = 1024 * 1024;

struct cli_args {
    std::string model_path;
    std::vector<uint64_t> fit_target_mib = {1024};
    std::vector<uint64_t> target_free_mib;
    std::vector<uint64_t> override_device_free_mib;
    std::vector<uint64_t> override_device_total_mib;
    bool has_override_host_free_mib = false;
    bool has_override_host_total_mib = false;
    uint64_t override_host_free_mib = 0;
    uint64_t override_host_total_mib = 0;
    bool show_fit_logs = false;
    uint32_t n_ctx = 4096;
    uint32_t min_ctx = 0;
    int32_t n_gpu_layers = -1;
};

void discard_log_callback(ggml_log_level, const char *, void *) {
}

bool split_csv_u64(const std::string & input, std::vector<uint64_t> & out) {
    out.clear();
    size_t start = 0;
    while (start < input.size()) {
        size_t end = input.find(',', start);
        if (end == std::string::npos) {
            end = input.size();
        }
        const std::string part = input.substr(start, end - start);
        if (part.empty()) {
            return false;
        }
        char * endp = nullptr;
        const unsigned long long v = std::strtoull(part.c_str(), &endp, 10);
        if (endp == nullptr || *endp != '\0') {
            return false;
        }
        out.push_back(static_cast<uint64_t>(v));
        start = end + 1;
    }
    return !out.empty();
}

bool parse_args(int argc, char ** argv, cli_args & out, std::string & err) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                err = std::string("missing value for ") + name;
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--model") {
            const char * v = need_value("--model");
            if (!v) {
                return false;
            }
            out.model_path = v;
            continue;
        }

        if (arg == "--fit-target-mib") {
            const char * v = need_value("--fit-target-mib");
            if (!v || !split_csv_u64(v, out.fit_target_mib)) {
                err = "invalid --fit-target-mib";
                return false;
            }
            continue;
        }

        if (arg == "--target-free-mib") {
            const char * v = need_value("--target-free-mib");
            if (!v || !split_csv_u64(v, out.target_free_mib)) {
                err = "invalid --target-free-mib";
                return false;
            }
            continue;
        }

        if (arg == "--override-device-free-mib") {
            const char * v = need_value("--override-device-free-mib");
            if (!v || !split_csv_u64(v, out.override_device_free_mib)) {
                err = "invalid --override-device-free-mib";
                return false;
            }
            continue;
        }

        if (arg == "--override-device-total-mib") {
            const char * v = need_value("--override-device-total-mib");
            if (!v || !split_csv_u64(v, out.override_device_total_mib)) {
                err = "invalid --override-device-total-mib";
                return false;
            }
            continue;
        }

        if (arg == "--override-host-free-mib") {
            const char * v = need_value("--override-host-free-mib");
            if (!v) {
                return false;
            }
            out.override_host_free_mib = static_cast<uint64_t>(std::strtoull(v, nullptr, 10));
            out.has_override_host_free_mib = true;
            continue;
        }

        if (arg == "--override-host-total-mib") {
            const char * v = need_value("--override-host-total-mib");
            if (!v) {
                return false;
            }
            out.override_host_total_mib = static_cast<uint64_t>(std::strtoull(v, nullptr, 10));
            out.has_override_host_total_mib = true;
            continue;
        }

        if (arg == "--show-fit-logs") {
            out.show_fit_logs = true;
            continue;
        }

        if (arg == "-c" || arg == "--ctx-size") {
            const char * v = need_value("--ctx-size");
            if (!v) {
                return false;
            }
            out.n_ctx = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
            continue;
        }

        if (arg == "--fit-ctx") {
            const char * v = need_value("--fit-ctx");
            if (!v) {
                return false;
            }
            out.min_ctx = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
            continue;
        }

        if (arg == "--n-gpu-layers") {
            const char * v = need_value("--n-gpu-layers");
            if (!v) {
                return false;
            }
            out.n_gpu_layers = static_cast<int32_t>(std::strtol(v, nullptr, 10));
            continue;
        }

        err = std::string("unknown argument: ") + arg;
        return false;
    }

    if (out.model_path.empty()) {
        err = "--model is required";
        return false;
    }

    return true;
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

std::vector<uint64_t> detect_device_free_mib(const std::string & model_path, std::string & err) {
    llama_model_params mparams = llama_model_default_params();
    mparams.no_alloc = true;
    mparams.use_mmap = false;
    mparams.use_mlock = false;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        err = "failed to load model for device detection";
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

} // namespace

int main(int argc, char ** argv) {
    cli_args args;
    std::string err;
    if (!parse_args(argc, argv, args, err)) {
        std::printf("{\"ok\":false,\"error\":\"%s\"}\n", err.c_str());
        return 2;
    }

    common_init();
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = args.n_ctx;
    mparams.n_gpu_layers = args.n_gpu_layers;

    std::vector<float> tensor_split(llama_max_devices(), 0.0f);
    std::vector<llama_model_tensor_buft_override> tbo(llama_max_tensor_buft_overrides());
    if (!tbo.empty()) {
        tbo[0].pattern = nullptr;
        tbo[0].buft = nullptr;
    }

    std::vector<uint64_t> margins_mib = args.fit_target_mib;
    std::vector<uint64_t> warnings_target_above_source;
    std::vector<uint64_t> source_free_mib;

    if (!args.target_free_mib.empty()) {
        if (!args.override_device_free_mib.empty()) {
            source_free_mib = args.override_device_free_mib;
        } else {
            std::string detect_err;
            source_free_mib = detect_device_free_mib(args.model_path, detect_err);
            if (source_free_mib.empty()) {
                std::printf("{\"ok\":false,\"error\":\"%s\"}\n", detect_err.c_str());
                return 3;
            }
        }

        const std::vector<uint64_t> desired_free_mib = broadcast(args.target_free_mib, source_free_mib.size(), args.target_free_mib[0]);
        const std::vector<uint64_t> base_margin_mib = broadcast(args.fit_target_mib, source_free_mib.size(), args.fit_target_mib[0]);

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
    } else if (!args.override_device_free_mib.empty()) {
        margins_mib = broadcast(args.fit_target_mib, args.override_device_free_mib.size(), args.fit_target_mib[0]);
    }

    std::vector<common_fit_memory_override_device> override_devices;
    common_fit_memory_override memory_override = {0, nullptr, false, {0, 0}};
    bool use_memory_override = false;

    if (!args.override_device_free_mib.empty()) {
        std::vector<uint64_t> totals_mib = args.override_device_total_mib;
        if (totals_mib.empty()) {
            totals_mib = args.override_device_free_mib;
        } else if (totals_mib.size() == 1 && args.override_device_free_mib.size() > 1) {
            totals_mib = broadcast(totals_mib, args.override_device_free_mib.size(), totals_mib[0]);
        } else if (totals_mib.size() != args.override_device_free_mib.size()) {
            std::printf("{\"ok\":false,\"error\":\"override_device_total_mib_size_mismatch\"}\n");
            return 4;
        }

        override_devices.resize(args.override_device_free_mib.size());
        for (size_t i = 0; i < args.override_device_free_mib.size(); ++i) {
            const uint64_t free_mib = args.override_device_free_mib[i];
            const uint64_t total_mib = std::max(totals_mib[i], free_mib);
            override_devices[i].free = static_cast<size_t>(free_mib * MiB);
            override_devices[i].total = static_cast<size_t>(total_mib * MiB);
        }

        memory_override.n_devices = override_devices.size();
        memory_override.devices = override_devices.data();
        use_memory_override = true;
    }

    if (args.has_override_host_free_mib || args.has_override_host_total_mib) {
        const uint64_t host_free_mib = args.has_override_host_free_mib
            ? args.override_host_free_mib
            : args.override_host_total_mib;
        const uint64_t host_total_mib = args.has_override_host_total_mib
            ? args.override_host_total_mib
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
    if (!args.show_fit_logs) {
        llama_log_get(&original_log_callback, &original_log_user_data);
        llama_log_set(discard_log_callback, nullptr);
    }

    const common_params_fit_status status = use_memory_override
        ? common_fit_params_with_memory_override(
            args.model_path.c_str(),
            &mparams,
            &cparams,
            tensor_split.data(),
            tbo.data(),
            margins_bytes.data(),
            args.min_ctx,
            GGML_LOG_LEVEL_ERROR,
            &memory_override,
            false)
        : common_fit_params(
            args.model_path.c_str(),
            &mparams,
            &cparams,
            tensor_split.data(),
            tbo.data(),
            margins_bytes.data(),
            args.min_ctx,
            GGML_LOG_LEVEL_ERROR);

    if (!args.show_fit_logs) {
        llama_log_set(original_log_callback, original_log_user_data);
    }

    std::printf("{\"ok\":%s,", status == COMMON_PARAMS_FIT_STATUS_SUCCESS ? "true" : "false");
    std::printf("\"status\":%d,", static_cast<int>(status));
    std::printf("\"n_ctx\":%u,\"n_gpu_layers\":%d,", cparams.n_ctx, mparams.n_gpu_layers);
    std::printf("\"fitTargetMiB\":[");
    for (size_t i = 0; i < margins_mib.size(); ++i) {
        if (i > 0) {
            std::printf(",");
        }
        std::printf("%llu", static_cast<unsigned long long>(margins_mib[i]));
    }
    std::printf("],");

    std::printf("\"warnings\":[");
    for (size_t i = 0; i < warnings_target_above_source.size(); ++i) {
        if (i > 0) {
            std::printf(",");
        }
        std::printf("\"target_free_above_source_device_%llu\"", static_cast<unsigned long long>(warnings_target_above_source[i]));
    }
    std::printf("],");

    std::printf("\"memoryOverride\":{");
    std::printf("\"enabled\":%s", use_memory_override ? "true" : "false");
    if (use_memory_override) {
        std::printf(",\"deviceFreeMiB\":[");
        for (size_t i = 0; i < override_devices.size(); ++i) {
            if (i > 0) {
                std::printf(",");
            }
            std::printf("%llu", static_cast<unsigned long long>(override_devices[i].free / MiB));
        }
        std::printf("],\"deviceTotalMiB\":[");
        for (size_t i = 0; i < override_devices.size(); ++i) {
            if (i > 0) {
                std::printf(",");
            }
            std::printf("%llu", static_cast<unsigned long long>(override_devices[i].total / MiB));
        }
        std::printf("]");

        if (memory_override.override_host) {
            std::printf(",\"hostFreeMiB\":%llu,\"hostTotalMiB\":%llu",
                static_cast<unsigned long long>(memory_override.host.free / MiB),
                static_cast<unsigned long long>(memory_override.host.total / MiB));
        }
    }
    std::printf("}}");
    std::printf("\n");

    return status == COMMON_PARAMS_FIT_STATUS_SUCCESS ? 0 : 1;
}
