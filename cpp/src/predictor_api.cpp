#include "vram/predictor_api.h"

#include "vram/fit_executor.h"
#include "vram/gguf_prefix_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <cstdio>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

using nlohmann::json;

constexpr uint64_t k_default_initial_prefix_bytes = 1024 * 1024;
constexpr uint64_t k_default_max_prefix_bytes = 8 * 1024 * 1024;
constexpr size_t k_max_tensor_preview = 32;

struct byte_range {
    uint64_t start = 0;
    uint64_t end = 0;
};

struct hf_range_request {
    std::string url;
    uint64_t start = 0;
    uint64_t end = 0;
    std::vector<std::pair<std::string, std::string>> headers;
};

std::string encode_url_component(const std::string & input, bool preserve_slash) {
    static const char * k_hex = "0123456789ABCDEF";
    std::string out;
    out.reserve(input.size() * 3);

    for (const unsigned char c : input) {
        const bool unreserved =
            (c >= 'A' && c <= 'Z') ||
            (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') ||
            c == '-' || c == '_' || c == '.' || c == '~' ||
            (preserve_slash && c == '/');

        if (unreserved) {
            out.push_back(static_cast<char>(c));
            continue;
        }

        out.push_back('%');
        out.push_back(k_hex[(c >> 4) & 0x0F]);
        out.push_back(k_hex[c & 0x0F]);
    }

    return out;
}

std::vector<byte_range> build_prefix_range_plan(
    uint64_t initial_bytes,
    uint64_t max_bytes,
    double growth_factor) {
    std::vector<byte_range> ranges;
    if (max_bytes == 0) {
        return ranges;
    }

    if (initial_bytes == 0 || initial_bytes > max_bytes) {
        initial_bytes = max_bytes < 1024 * 1024 ? max_bytes : 1024 * 1024;
    }

    if (initial_bytes == 0) {
        initial_bytes = 1;
    }

    if (growth_factor <= 1.0) {
        growth_factor = 2.0;
    }

    uint64_t current = initial_bytes;
    while (true) {
        ranges.push_back({0, current - 1});
        if (current >= max_bytes) {
            break;
        }

        uint64_t next = static_cast<uint64_t>(std::ceil(static_cast<double>(current) * growth_factor));
        if (next <= current) {
            next = current + 1;
        }
        if (next > max_bytes) {
            next = max_bytes;
        }
        current = next;
    }

    return ranges;
}

std::string resolve_hf_file_url(
    const std::string & repo,
    const std::string & file,
    const std::string & revision) {
    if (repo.empty() || file.empty()) {
        return "";
    }

    const std::string effective_revision = revision.empty() ? "main" : revision;
    std::string url = "https://huggingface.co/";
    url += encode_url_component(repo, false);
    url += "/resolve/";
    url += encode_url_component(effective_revision, true);
    url += "/";
    url += encode_url_component(file, true);
    return url;
}

bool get_file_size(const char * path, uint64_t & size_out) {
    FILE * fp = std::fopen(path, "rb");
    if (fp == nullptr) {
        return false;
    }

    if (std::fseek(fp, 0, SEEK_END) != 0) {
        std::fclose(fp);
        return false;
    }

    const long n = std::ftell(fp);
    std::fclose(fp);
    if (n < 0) {
        return false;
    }

    size_out = static_cast<uint64_t>(n);
    return true;
}

bool read_file_prefix(const char * path, size_t n_bytes, std::vector<uint8_t> & out) {
    FILE * fp = std::fopen(path, "rb");
    if (fp == nullptr) {
        return false;
    }

    out.resize(n_bytes);
    const size_t read_n = n_bytes == 0 ? 0 : std::fread(out.data(), 1, n_bytes, fp);
    std::fclose(fp);
    if (read_n != n_bytes) {
        return false;
    }

    return true;
}

uint64_t json_u64_or_default(const json & obj, const char * key, uint64_t fallback) {
    if (!obj.is_object() || !obj.contains(key)) {
        return fallback;
    }

    const json & value = obj[key];
    if (!value.is_number_unsigned() && !value.is_number_integer()) {
        return fallback;
    }

    const uint64_t parsed = value.get<uint64_t>();
    return parsed;
}

double json_double_or_default(const json & obj, const char * key, double fallback) {
    if (!obj.is_object() || !obj.contains(key)) {
        return fallback;
    }

    const json & value = obj[key];
    if (!value.is_number()) {
        return fallback;
    }

    return value.get<double>();
}

bool json_bool_or_default(const json & obj, const char * key, bool fallback) {
    if (!obj.is_object() || !obj.contains(key)) {
        return fallback;
    }

    const json & value = obj[key];
    if (!value.is_boolean()) {
        return fallback;
    }

    return value.get<bool>();
}

std::string json_string_or_default(const json & obj, const char * key, const char * fallback) {
    if (!obj.is_object() || !obj.contains(key)) {
        return fallback;
    }

    const json & value = obj[key];
    if (!value.is_string()) {
        return fallback;
    }

    return value.get<std::string>();
}

std::vector<uint64_t> json_u64_array_or_default(const json & obj, const char * key, const std::vector<uint64_t> & fallback) {
    if (!obj.is_object() || !obj.contains(key)) {
        return fallback;
    }

    const json & value = obj[key];
    if (!value.is_array()) {
        return fallback;
    }

    std::vector<uint64_t> out;
    out.reserve(value.size());
    for (const auto & item : value) {
        if (!item.is_number_unsigned() && !item.is_number_integer()) {
            return fallback;
        }
        out.push_back(item.get<uint64_t>());
    }

    if (out.empty()) {
        return fallback;
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

json tensor_to_json(const vram::gguf_tensor_info & tensor) {
    return {
        {"name", tensor.name},
        {"dimensions", tensor.dimensions},
        {"ggmlType", tensor.ggml_type},
        {"dataOffset", tensor.data_offset},
    };
}

json headers_to_json(const std::vector<std::pair<std::string, std::string>> & headers) {
    json out = json::array();
    for (const auto & kv : headers) {
        out.push_back({
            {"name", kv.first},
            {"value", kv.second},
        });
    }
    return out;
}

std::vector<hf_range_request> build_hf_prefix_range_requests_for_url(
    const std::string & url,
    uint64_t initial_bytes,
    uint64_t max_bytes,
    double growth_factor,
    const std::string & bearer_token,
    bool include_authorization_header) {
    std::vector<hf_range_request> requests;
    if (url.empty()) {
        return requests;
    }

    const std::vector<byte_range> plan = build_prefix_range_plan(initial_bytes, max_bytes, growth_factor);
    requests.reserve(plan.size());

    for (const byte_range & range : plan) {
        hf_range_request req;
        req.url = url;
        req.start = range.start;
        req.end = range.end;

        char range_header[64];
        const int n = std::snprintf(
            range_header,
            sizeof(range_header),
            "bytes=%llu-%llu",
            static_cast<unsigned long long>(range.start),
            static_cast<unsigned long long>(range.end));
        if (n <= 0) {
            continue;
        }

        req.headers.push_back({"Range", range_header});
        req.headers.push_back({"Accept", "application/octet-stream"});
        if (include_authorization_header && !bearer_token.empty()) {
            req.headers.push_back({"Authorization", "Bearer " + bearer_token});
        }

        requests.push_back(req);
    }

    return requests;
}

json fit_memory_entry_to_json(const vram::fit_memory_breakdown_entry & entry) {
    return {
        {"name", entry.name},
        {"totalMiB", entry.total_mib},
        {"freeMiB", entry.free_mib},
        {"modelMiB", entry.model_mib},
        {"contextMiB", entry.context_mib},
        {"computeMiB", entry.compute_mib},
        {"unaccountedMiB", entry.unaccounted_mib},
    };
}

json fit_targets_to_json(const std::vector<uint64_t> & fit_target_mib, const std::vector<uint64_t> & target_free_mib) {
    return {
        {"fitMiB", fit_target_mib},
        {"targetFreeMiB", target_free_mib},
    };
}

json fit_overrides_to_json(
    const std::vector<uint64_t> & device_free_mib,
    const std::vector<uint64_t> & device_total_mib,
    uint64_t host_free_mib) {
    return {
        {"deviceFreeMiB", device_free_mib},
        {"deviceTotalMiB", device_total_mib},
        {"hostFreeMiB", host_free_mib},
    };
}

json fit_memory_bytes_to_json(const vram::fit_execution_result & result) {
    return {
        {"weights", result.totals.model_mib * 1024ULL * 1024ULL},
        {"kvCache", result.totals.context_mib * 1024ULL * 1024ULL},
        {"device", (result.totals.model_mib + result.totals.context_mib + result.totals.compute_mib) * 1024ULL * 1024ULL},
        {"host", (result.host.model_mib + result.host.context_mib + result.host.compute_mib) * 1024ULL * 1024ULL},
    };
}

json fit_memory_breakdown_to_json(const vram::fit_execution_result & result) {
    json devices = json::array();
    for (const auto & entry : result.devices) {
        devices.push_back(fit_memory_entry_to_json(entry));
    }

    return {
        {"totals",
            {
                {"modelMiB", result.totals.model_mib},
                {"contextMiB", result.totals.context_mib},
                {"computeMiB", result.totals.compute_mib}
            }
        },
        {"devices", devices},
        {"host", fit_memory_entry_to_json(result.host)}
    };
}

} // namespace

extern "C" const char * vram_predictor_get_system_info_json(void) {
    static std::string response;
    try {
#ifdef __EMSCRIPTEN__
    const char * target = "wasm32-emscripten";
#else
    const char * target = "native-dev";
#endif

    json body = {
        {"ok", true},
        {"version", "0.1.0"},
        {"target", target},
        {"features",
            {
                {"metadataEstimator", false},
                {"llamaFitMode", vram::fit_execution_available()},
                {"hfRangeFetch", false},
                {"ggufPrefixParser", true},
                {"fitMemoryOverrideExecution", vram::fit_execution_available()}
            }
        }
    };

    response = body.dump();
    } catch (const std::exception & error) {
        response = json({
            {"ok", false},
            {"error", "system_info_exception"},
            {"message", error.what()}
        }).dump();
    } catch (...) {
        response = json({
            {"ok", false},
            {"error", "system_info_exception"},
            {"message", "unknown_exception"}
        }).dump();
    }

    return response.c_str();
}

extern "C" const char * vram_predictor_predict_json(const char * request_json) {
    static std::string response;
    try {
    if (request_json == nullptr) {
        json error = {
            {"ok", false},
            {"error", "null_request_json"}
        };
        response = error.dump();
        return response.c_str();
    }

    const json parsed = json::parse(request_json, nullptr, false);
    if (parsed.is_discarded()) {
        json error = {
            {"ok", false},
            {"error", "invalid_json"}
        };
        response = error.dump();
        return response.c_str();
    }

    if (!parsed.is_object()) {
        json error = {
            {"ok", false},
            {"error", "request_must_be_object"}
        };
        response = error.dump();
        return response.c_str();
    }

    const json model = parsed.contains("model") ? parsed["model"] : json::object();
    const json runtime = parsed.contains("runtime") ? parsed["runtime"] : json::object();
    const json device = parsed.contains("device") ? parsed["device"] : json::object();
    const json fit = parsed.contains("fit") ? parsed["fit"] : json::object();
    const std::string mode = parsed.contains("mode") && parsed["mode"].is_string()
        ? parsed["mode"].get<std::string>()
        : "metadata";
    const std::string source = model.contains("source") && model["source"].is_string()
        ? model["source"].get<std::string>()
        : "";

    if (mode == "fit") {
        if (source != "local") {
            json error = {
                {"ok", false},
                {"error", "fit_mode_currently_requires_local_model_source"}
            };
            response = error.dump();
            return response.c_str();
        }

        if (!model.contains("path") || !model["path"].is_string()) {
            json error = {
                {"ok", false},
                {"error", "model.path_required_for_fit_mode"}
            };
            response = error.dump();
            return response.c_str();
        }

        const std::string model_path = model["path"].get<std::string>();
        const std::vector<uint64_t> fit_target_mib = json_u64_array_or_default(device, "fit_target_mib", std::vector<uint64_t>{1024});
        const std::vector<uint64_t> target_free_mib = json_u64_array_or_default(device, "target_free_mib", std::vector<uint64_t>{});
        const std::string fit_target_csv = join_u64_csv(fit_target_mib);
        const std::string target_free_csv = join_u64_csv(target_free_mib);

        std::vector<uint64_t> override_device_free_mib;
        std::vector<uint64_t> override_device_total_mib;
        std::vector<vram::sim_device_spec> simulated_devices;
        if (device.contains("gpus")) {
            if (!device["gpus"].is_array()) {
                json error = {
                    {"ok", false},
                    {"error", "device.gpus_must_be_array_when_present"}
                };
                response = error.dump();
                return response.c_str();
            }

            for (const auto & gpu : device["gpus"]) {
                if (!gpu.is_object() || !gpu.contains("free_bytes") || !gpu["free_bytes"].is_number_integer()) {
                    json error = {
                        {"ok", false},
                        {"error", "device.gpus[].free_bytes_required_for_fit_override"}
                    };
                    response = error.dump();
                    return response.c_str();
                }

                const uint64_t free_bytes = gpu["free_bytes"].get<uint64_t>();
                override_device_free_mib.push_back(free_bytes / (1024 * 1024));

                vram::sim_backend_profile backend_profile = vram::sim_backend_profile::cuda;
                if (gpu.contains("backend")) {
                    if (!gpu["backend"].is_string()) {
                        json error = {
                            {"ok", false},
                            {"error", "device.gpus[].backend_must_be_string_when_present"}
                        };
                        response = error.dump();
                        return response.c_str();
                    }

                    const std::string backend_name = gpu["backend"].get<std::string>();
                    if (!vram::parse_sim_backend_profile(backend_name, backend_profile)) {
                        json error = {
                            {"ok", false},
                            {"error", "device.gpus[].backend_invalid"},
                            {"value", backend_name}
                        };
                        response = error.dump();
                        return response.c_str();
                    }
                }

                std::string label;
                if (gpu.contains("name") && gpu["name"].is_string()) {
                    label = gpu["name"].get<std::string>();
                }
                if (label.empty() && gpu.contains("id") && gpu["id"].is_string()) {
                    label = gpu["id"].get<std::string>();
                }
                if (label.empty()) {
                    label = "GPU " + std::to_string(simulated_devices.size());
                }
                if (gpu.contains("index") && gpu["index"].is_number_integer()) {
                    const int64_t gpu_index = gpu["index"].get<int64_t>();
                    if (gpu_index >= 0) {
                        label += " [index " + std::to_string(gpu_index) + "]";
                    }
                }

                uint64_t total_bytes = free_bytes;

                if (gpu.contains("total_bytes") && gpu["total_bytes"].is_number_integer()) {
                    total_bytes = gpu["total_bytes"].get<uint64_t>();
                    override_device_total_mib.push_back(total_bytes / (1024 * 1024));
                }

                vram::sim_device_spec sim_device;
                sim_device.name = label;
                sim_device.description = std::string("Simulated ") + vram::sim_backend_profile_name(backend_profile) + " device";
                sim_device.free_bytes = free_bytes;
                sim_device.total_bytes = std::max(total_bytes, free_bytes);
                sim_device.profile = backend_profile;
                simulated_devices.push_back(sim_device);
            }
        }

        if (!override_device_total_mib.empty() && override_device_total_mib.size() != override_device_free_mib.size()) {
            json error = {
                {"ok", false},
                {"error", "device.gpus[].total_bytes_must_be_present_for_all_or_none"}
            };
            response = error.dump();
            return response.c_str();
        }

        const uint64_t override_host_free_mib = json_u64_or_default(device, "host_ram_bytes", 0) / (1024 * 1024);
        const bool fit_execution_enabled = vram::fit_execution_available();
        const bool execute_in_process = fit_execution_enabled && json_bool_or_default(fit, "execute_in_process", false);

        std::vector<std::string> args;
        args.push_back("--model");
        args.push_back(model_path);
        args.push_back("--fit-target-mib");
        args.push_back(fit_target_csv);

        if (!target_free_mib.empty()) {
            args.push_back("--target-free-mib");
            args.push_back(target_free_csv);
        }

        if (!override_device_free_mib.empty()) {
            args.push_back("--override-device-free-mib");
            args.push_back(join_u64_csv(override_device_free_mib));
        }

        if (!override_device_total_mib.empty()) {
            args.push_back("--override-device-total-mib");
            args.push_back(join_u64_csv(override_device_total_mib));
        }

        if (override_host_free_mib > 0) {
            args.push_back("--override-host-free-mib");
            args.push_back(std::to_string(override_host_free_mib));
        }

        const uint64_t fit_ctx_min = json_u64_or_default(fit, "min_ctx", 0);
        if (fit_ctx_min > 0) {
            args.push_back("--fit-ctx");
            args.push_back(std::to_string(fit_ctx_min));
        }

        const uint64_t n_ctx = json_u64_or_default(runtime, "n_ctx", 0);
        if (n_ctx > 0) {
            args.push_back("-c");
            args.push_back(std::to_string(n_ctx));
        }

        const uint64_t n_batch = json_u64_or_default(runtime, "n_batch", 0);
        if (n_batch > 0) {
            args.push_back("--batch-size");
            args.push_back(std::to_string(n_batch));
        }

        const uint64_t n_ubatch = json_u64_or_default(runtime, "n_ubatch", 0);
        if (n_ubatch > 0) {
            args.push_back("--ubatch-size");
            args.push_back(std::to_string(n_ubatch));
        }

        if (runtime.contains("n_gpu_layers") && runtime["n_gpu_layers"].is_number_integer()) {
            args.push_back("--n-gpu-layers");
            args.push_back(std::to_string(runtime["n_gpu_layers"].get<int64_t>()));
        }

        const bool show_fit_logs = json_bool_or_default(fit, "show_fit_logs", false);
        if (show_fit_logs) {
            args.push_back("--show-fit-logs");
        }

        const std::string harness_bin = json_string_or_default(fit, "fit_harness_binary", "vram_fit_harness");

        if (execute_in_process) {
            vram::fit_execution_request exec_request;
            exec_request.model_path = model_path;
            exec_request.fit_target_mib = fit_target_mib;
            exec_request.target_free_mib = target_free_mib;
            exec_request.simulated_devices = simulated_devices;
            exec_request.has_override_host_free_mib = override_host_free_mib > 0;
            exec_request.has_override_host_total_mib = override_host_free_mib > 0;
            exec_request.override_host_free_mib = override_host_free_mib;
            exec_request.override_host_total_mib = override_host_free_mib;
            exec_request.show_fit_logs = show_fit_logs;
            exec_request.min_ctx = static_cast<uint32_t>(fit_ctx_min);
            exec_request.n_ctx = static_cast<uint32_t>(n_ctx > 0 ? n_ctx : 4096);
            exec_request.n_batch = static_cast<uint32_t>(n_batch);
            exec_request.n_ubatch = static_cast<uint32_t>(n_ubatch);
            exec_request.n_gpu_layers = runtime.contains("n_gpu_layers") && runtime["n_gpu_layers"].is_number_integer()
                ? static_cast<int32_t>(runtime["n_gpu_layers"].get<int64_t>())
                : -1;

            vram::fit_execution_result exec_result;
            std::string exec_error;
            if (!vram::execute_fit_request(exec_request, exec_result, exec_error)) {
                json error = {
                    {"ok", false},
                    {"error", exec_error.empty() ? "fit_execution_failed" : exec_error}
                };
                response = error.dump();
                return response.c_str();
            }

            json body = {
                {"ok", exec_result.ok},
                {"fit",
                    {
                        {"executedInProcess", true},
                        {"targets", fit_targets_to_json(exec_result.fit_target_mib, target_free_mib)},
                        {"overrides", fit_overrides_to_json(exec_result.device_free_mib, exec_result.device_total_mib, exec_result.host_free_mib)},
                        {"recommended",
                            {
                                {"n_ctx", exec_result.n_ctx},
                                {"n_gpu_layers", exec_result.n_gpu_layers}
                            }
                        },
                        {"status", exec_result.status},
                        {"warnings", exec_result.warnings},
                        {"memoryBytes", fit_memory_bytes_to_json(exec_result)},
                        {"breakdown", fit_memory_breakdown_to_json(exec_result)}
                    }
                }
            };
            response = body.dump();
            return response.c_str();
        }

        json body = {
            {"ok", true},
            {"fit",
                {
                    {"executedInProcess", false},
                    {"command",
                        {
                            {"binary", harness_bin},
                            {"args", args}
                        }
                    },
                    {"targets", fit_targets_to_json(fit_target_mib, target_free_mib)},
                    {"overrides", fit_overrides_to_json(override_device_free_mib, override_device_total_mib, override_host_free_mib)},
                    {"showLogs", show_fit_logs}
                }
            }
        };
        response = body.dump();
        return response.c_str();
    }

    if (source != "local") {
        if (source == "huggingface") {
            if (!model.contains("huggingFace") || !model["huggingFace"].is_object()) {
                json error = {
                    {"ok", false},
                    {"error", "model.huggingFace_required_for_huggingface_source"}
                };
                response = error.dump();
                return response.c_str();
            }

            const json hf = model["huggingFace"];
            const std::string repo = hf.contains("repo") && hf["repo"].is_string() ? hf["repo"].get<std::string>() : "";
            const std::string file = hf.contains("file") && hf["file"].is_string() ? hf["file"].get<std::string>() : "";
            const std::string revision = hf.contains("revision") && hf["revision"].is_string() ? hf["revision"].get<std::string>() : "";
            const std::string token = hf.contains("token") && hf["token"].is_string() ? hf["token"].get<std::string>() : "";
            const std::string resolved_url_override = hf.contains("resolvedUrl") && hf["resolvedUrl"].is_string()
                ? hf["resolvedUrl"].get<std::string>()
                : "";

            const std::string url = resolved_url_override.empty()
                ? resolve_hf_file_url(repo, file, revision)
                : resolved_url_override;
            if (url.empty()) {
                json error = {
                    {"ok", false},
                    {"error", "invalid_huggingface_model_descriptor"}
                };
                response = error.dump();
                return response.c_str();
            }

            const json fetch = parsed.contains("fetch") ? parsed["fetch"] : json::object();
            const uint64_t initial_bytes = json_u64_or_default(fetch, "initial_bytes", k_default_initial_prefix_bytes);
            const uint64_t max_bytes = json_u64_or_default(fetch, "max_bytes", k_default_max_prefix_bytes);
            const double growth_factor = json_double_or_default(fetch, "growth_factor", 2.0);
            const bool execute_remote = json_bool_or_default(fetch, "execute_remote", false);

            std::vector<hf_range_request> requests;
            if (!resolved_url_override.empty()) {
                const bool include_authorization_header = resolved_url_override.rfind("https://huggingface.co/", 0) == 0;
                requests = build_hf_prefix_range_requests_for_url(
                    resolved_url_override,
                    initial_bytes,
                    max_bytes,
                    growth_factor,
                    token,
                    include_authorization_header);
            } else {
                requests = build_hf_prefix_range_requests_for_url(
                    url,
                    initial_bytes,
                    max_bytes,
                    growth_factor,
                    token,
                    true);
            }
            json planned = json::array();
            for (const auto & req : requests) {
                planned.push_back({
                    {"url", req.url},
                    {"start", req.start},
                    {"end", req.end},
                    {"headers", headers_to_json(req.headers)},
                });
            }

            if (execute_remote) {
                json error = {
                    {"ok", false},
                    {"source", "huggingface"},
                    {"error", "remote_hf_fetch_removed_use_js_client"},
                    {"resolvedUrl", url},
                    {"requests", planned}
                };
                response = error.dump();
                return response.c_str();
            }

            json body = {
                {"ok", true},
                {"source", "huggingface"},
                {"resolvedUrl", url},
                {"requests", planned}
            };
            response = body.dump();
            return response.c_str();
        }

        json body = {
            {"ok", false},
            {"error", "unsupported_model_source"},
            {"supportedSources", {"local", "huggingface"}}
        };
        response = body.dump();
        return response.c_str();
    }

    if (!model.contains("path") || !model["path"].is_string()) {
        json error = {
            {"ok", false},
            {"error", "model.path_required_for_local_source"}
        };
        response = error.dump();
        return response.c_str();
    }

    const std::string path = model["path"].get<std::string>();
    const json fetch = parsed.contains("fetch") ? parsed["fetch"] : json::object();
    const uint64_t initial_bytes = json_u64_or_default(fetch, "initial_bytes", k_default_initial_prefix_bytes);
    const uint64_t max_bytes = json_u64_or_default(fetch, "max_bytes", k_default_max_prefix_bytes);
    const double growth_factor = json_double_or_default(fetch, "growth_factor", 2.0);

    uint64_t file_size = 0;
    if (!get_file_size(path.c_str(), file_size)) {
        json error = {
            {"ok", false},
            {"error", "file_open_failed"},
            {"path", path}
        };
        response = error.dump();
        return response.c_str();
    }

    if (file_size == 0) {
        json error = {
            {"ok", false},
            {"error", "file_empty"},
            {"path", path}
        };
        response = error.dump();
        return response.c_str();
    }

    const std::vector<byte_range> plan = build_prefix_range_plan(initial_bytes, max_bytes, growth_factor);
    if (plan.empty()) {
        json error = {
            {"ok", false},
            {"error", "invalid_fetch_plan"}
        };
        response = error.dump();
        return response.c_str();
    }

    uint64_t min_required = 0;
    std::string last_error;

    for (const byte_range & range : plan) {
        const uint64_t capped_end = std::min(range.end, file_size - 1);
        const uint64_t requested_bytes_u64 = capped_end + 1;
        const size_t requested_bytes = static_cast<size_t>(requested_bytes_u64);

        std::vector<uint8_t> prefix;
        if (!read_file_prefix(path.c_str(), requested_bytes, prefix)) {
            json error = {
                {"ok", false},
                {"error", "file_read_failed"},
                {"requestedBytes", requested_bytes_u64},
                {"path", path}
            };
            response = error.dump();
            return response.c_str();
        }

        const auto parsed_prefix = vram::parse_gguf_prefix(prefix.data(), prefix.size(), k_max_tensor_preview);

        if (parsed_prefix.status == vram::gguf_prefix_parse_status::complete) {
            json tensors = json::array();
            for (const auto & tensor : parsed_prefix.metadata.tensors) {
                tensors.push_back(tensor_to_json(tensor));
            }

            json body = {
                {"ok", true},
                {"source", "local"},
                {"path", path},
                {"metadata",
                    {
                        {"version", parsed_prefix.metadata.version},
                        {"kvCount", parsed_prefix.metadata.kv_count},
                        {"tensorCount", parsed_prefix.metadata.tensor_count},
                        {"bytesConsumed", parsed_prefix.metadata.bytes_consumed},
                        {"tensorListTruncated", parsed_prefix.metadata.tensor_list_truncated},
                        {"tensors", tensors},
                    }
                }
            };

            response = body.dump();
            return response.c_str();
        }

        if (parsed_prefix.status == vram::gguf_prefix_parse_status::need_more_data) {
            min_required = std::max(min_required, parsed_prefix.minimum_required_bytes);
            continue;
        }

        last_error = parsed_prefix.error.empty() ? "invalid_gguf_format" : parsed_prefix.error;
        break;
    }

    json error = {
        {"ok", false},
        {"error", last_error.empty() ? "insufficient_prefix_bytes" : last_error},
        {"path", path},
        {"minimumRequiredBytes", min_required}
    };

    response = error.dump();
    return response.c_str();
    } catch (const std::exception & error) {
        response = json({
            {"ok", false},
            {"error", "predict_exception"},
            {"message", error.what()}
        }).dump();
        return response.c_str();
    } catch (...) {
        response = json({
            {"ok", false},
            {"error", "predict_exception"},
            {"message", "unknown_exception"}
        }).dump();
        return response.c_str();
    }
}
