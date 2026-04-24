#include "vram/fit_executor.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t MiB = 1024 * 1024;

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
    uint32_t n_batch = 0;
    uint32_t n_ubatch = 0;
    uint32_t min_ctx = 0;
    int32_t n_gpu_layers = -1;
};

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
        const unsigned long long value = std::strtoull(part.c_str(), &endp, 10);
        if (endp == nullptr || *endp != '\0') {
            return false;
        }

        out.push_back(static_cast<uint64_t>(value));
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
            const char * value = need_value("--model");
            if (!value) {
                return false;
            }
            out.model_path = value;
            continue;
        }

        if (arg == "--fit-target-mib") {
            const char * value = need_value("--fit-target-mib");
            if (!value || !split_csv_u64(value, out.fit_target_mib)) {
                err = "invalid --fit-target-mib";
                return false;
            }
            continue;
        }

        if (arg == "--target-free-mib") {
            const char * value = need_value("--target-free-mib");
            if (!value || !split_csv_u64(value, out.target_free_mib)) {
                err = "invalid --target-free-mib";
                return false;
            }
            continue;
        }

        if (arg == "--override-device-free-mib") {
            const char * value = need_value("--override-device-free-mib");
            if (!value || !split_csv_u64(value, out.override_device_free_mib)) {
                err = "invalid --override-device-free-mib";
                return false;
            }
            continue;
        }

        if (arg == "--override-device-total-mib") {
            const char * value = need_value("--override-device-total-mib");
            if (!value || !split_csv_u64(value, out.override_device_total_mib)) {
                err = "invalid --override-device-total-mib";
                return false;
            }
            continue;
        }

        if (arg == "--override-host-free-mib") {
            const char * value = need_value("--override-host-free-mib");
            if (!value) {
                return false;
            }
            out.override_host_free_mib = static_cast<uint64_t>(std::strtoull(value, nullptr, 10));
            out.has_override_host_free_mib = true;
            continue;
        }

        if (arg == "--override-host-total-mib") {
            const char * value = need_value("--override-host-total-mib");
            if (!value) {
                return false;
            }
            out.override_host_total_mib = static_cast<uint64_t>(std::strtoull(value, nullptr, 10));
            out.has_override_host_total_mib = true;
            continue;
        }

        if (arg == "--show-fit-logs") {
            out.show_fit_logs = true;
            continue;
        }

        if (arg == "--batch-size") {
            const char * value = need_value("--batch-size");
            if (!value) {
                return false;
            }
            out.n_batch = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
            continue;
        }

        if (arg == "--ubatch-size") {
            const char * value = need_value("--ubatch-size");
            if (!value) {
                return false;
            }
            out.n_ubatch = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
            continue;
        }

        if (arg == "-c" || arg == "--ctx-size") {
            const char * value = need_value("--ctx-size");
            if (!value) {
                return false;
            }
            out.n_ctx = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
            continue;
        }

        if (arg == "--fit-ctx") {
            const char * value = need_value("--fit-ctx");
            if (!value) {
                return false;
            }
            out.min_ctx = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
            continue;
        }

        if (arg == "--n-gpu-layers") {
            const char * value = need_value("--n-gpu-layers");
            if (!value) {
                return false;
            }
            out.n_gpu_layers = static_cast<int32_t>(std::strtol(value, nullptr, 10));
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

std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size());
    for (char ch : value) {
        switch (ch) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out.push_back(ch);
                break;
        }
    }
    return out;
}

void print_u64_array(const std::vector<uint64_t> & values) {
    std::printf("[");
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            std::printf(",");
        }
        std::printf("%llu", static_cast<unsigned long long>(values[i]));
    }
    std::printf("]");
}

void print_string_array(const std::vector<std::string> & values) {
    std::printf("[");
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            std::printf(",");
        }
        std::printf("\"%s\"", json_escape(values[i]).c_str());
    }
    std::printf("]");
}

void print_breakdown_entry(const vram::fit_memory_breakdown_entry & entry) {
    std::printf(
        "{\"name\":\"%s\",\"totalMiB\":%llu,\"freeMiB\":%llu,\"modelMiB\":%llu,\"contextMiB\":%llu,\"computeMiB\":%llu,\"unaccountedMiB\":%llu}",
        json_escape(entry.name).c_str(),
        static_cast<unsigned long long>(entry.total_mib),
        static_cast<unsigned long long>(entry.free_mib),
        static_cast<unsigned long long>(entry.model_mib),
        static_cast<unsigned long long>(entry.context_mib),
        static_cast<unsigned long long>(entry.compute_mib),
        static_cast<unsigned long long>(entry.unaccounted_mib));
}

} // namespace

int main(int argc, char ** argv) {
    cli_args args;
    std::string err;
    if (!parse_args(argc, argv, args, err)) {
        std::printf("{\"ok\":false,\"error\":\"%s\"}\n", err.c_str());
        return 2;
    }

    vram::fit_execution_request request;
    request.model_path = args.model_path;
    request.fit_target_mib = args.fit_target_mib;
    request.target_free_mib = args.target_free_mib;
    request.show_fit_logs = args.show_fit_logs;
    request.n_ctx = args.n_ctx;
    request.n_batch = args.n_batch;
    request.n_ubatch = args.n_ubatch;
    request.min_ctx = args.min_ctx;
    request.n_gpu_layers = args.n_gpu_layers;

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

        request.simulated_devices.reserve(args.override_device_free_mib.size());
        for (size_t i = 0; i < args.override_device_free_mib.size(); ++i) {
            vram::sim_device_spec spec;
            spec.name = "GPU " + std::to_string(i);
            spec.description = "Simulated cuda device";
            spec.free_bytes = args.override_device_free_mib[i] * MiB;
            spec.total_bytes = std::max(totals_mib[i], args.override_device_free_mib[i]) * MiB;
            spec.profile = vram::sim_backend_profile::cuda;
            request.simulated_devices.push_back(std::move(spec));
        }
    }

    request.has_override_host_free_mib = args.has_override_host_free_mib;
    request.has_override_host_total_mib = args.has_override_host_total_mib;
    request.override_host_free_mib = args.override_host_free_mib;
    request.override_host_total_mib = args.override_host_total_mib;

    vram::fit_execution_result result;
    std::string error;
    if (!vram::execute_fit_request(request, result, error)) {
        std::printf("{\"ok\":false,\"error\":\"%s\"}\n", json_escape(error.empty() ? "fit_execution_failed" : error).c_str());
        return 3;
    }

    const bool memory_override_enabled = !request.simulated_devices.empty() || request.has_override_host_free_mib || request.has_override_host_total_mib;

    std::printf("{\"ok\":%s,", result.ok ? "true" : "false");
    std::printf("\"status\":%d,", result.status);
    std::printf("\"n_ctx\":%u,\"n_gpu_layers\":%d,", result.n_ctx, result.n_gpu_layers);
    std::printf("\"fitTargetMiB\":");
    print_u64_array(result.fit_target_mib);
    std::printf(",\"warnings\":");
    print_string_array(result.warnings);
    std::printf(",\"memoryOverride\":{\"enabled\":%s", memory_override_enabled ? "true" : "false");
    if (memory_override_enabled) {
        std::printf(",\"deviceFreeMiB\":");
        print_u64_array(result.device_free_mib);
        std::printf(",\"deviceTotalMiB\":");
        print_u64_array(result.device_total_mib);
        if (result.host_override_enabled) {
            std::printf(",\"hostFreeMiB\":%llu,\"hostTotalMiB\":%llu",
                static_cast<unsigned long long>(result.host_free_mib),
                static_cast<unsigned long long>(result.host_total_mib));
        }
    }
    std::printf("},\"breakdown\":{\"totals\":{\"modelMiB\":%llu,\"contextMiB\":%llu,\"computeMiB\":%llu},\"devices\":[",
        static_cast<unsigned long long>(result.totals.model_mib),
        static_cast<unsigned long long>(result.totals.context_mib),
        static_cast<unsigned long long>(result.totals.compute_mib));
    for (size_t i = 0; i < result.devices.size(); ++i) {
        if (i > 0) {
            std::printf(",");
        }
        print_breakdown_entry(result.devices[i]);
    }
    std::printf("],\"host\":");
    print_breakdown_entry(result.host);
    std::printf("}}\n");

    return result.ok ? 0 : 1;
}
