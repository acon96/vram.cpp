#include "vram/predictor_api.h"

#include "vram/fit_executor.h"
#include "vram/gguf_prefix_parser.h"
#include "vram/hf_range_fetch_helper.h"
#include "vram/hf_range_plan.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <exception>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using nlohmann::json;

constexpr uint64_t k_default_initial_prefix_bytes = 1024 * 1024;
constexpr uint64_t k_default_max_prefix_bytes = 8 * 1024 * 1024;
constexpr size_t k_max_tensor_preview = 32;

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

json parse_status_to_json(vram::gguf_prefix_parse_status status) {
    switch (status) {
        case vram::gguf_prefix_parse_status::complete:
            return "complete";
        case vram::gguf_prefix_parse_status::need_more_data:
            return "need_more_data";
        case vram::gguf_prefix_parse_status::invalid_format:
            return "invalid_format";
        default:
            return "unknown";
    }
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
        {"apiVersion", "0.1.0"},
        {"engine", "vram-cpp"},
        {"target", target},
        {"features",
            {
                {"metadataEstimator", false},
                {"llamaFitMode", vram::fit_execution_available()},
                {"hfRangeFetch", true},
                {"ggufPrefixParser", true},
                {"fitMemoryOverrideExecution", vram::fit_execution_available()}
            }
        }
    };

    response = body.dump();
    } catch (const std::exception & error) {
        response = json({
            {"ok", false},
            {"phase", "phase-0-stub"},
            {"error", "system_info_exception"},
            {"message", error.what()}
        }).dump();
    } catch (...) {
        response = json({
            {"ok", false},
            {"phase", "phase-0-stub"},
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
            {"phase", "phase-0-stub"},
            {"error", "null_request_json"}
        };
        response = error.dump();
        return response.c_str();
    }

    const json parsed = json::parse(request_json, nullptr, false);
    if (parsed.is_discarded()) {
        json error = {
            {"ok", false},
            {"phase", "phase-0-stub"},
            {"error", "invalid_json"}
        };
        response = error.dump();
        return response.c_str();
    }

    if (!parsed.is_object()) {
        json error = {
            {"ok", false},
            {"phase", "phase-2-prefix-parser"},
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
                {"engine", "vram-cpp"},
                {"apiVersion", "0.1.0"},
                {"phase", "phase-4-fit-parity"},
                {"error", "fit_mode_currently_requires_local_model_source"}
            };
            response = error.dump();
            return response.c_str();
        }

        if (!model.contains("path") || !model["path"].is_string()) {
            json error = {
                {"ok", false},
                {"phase", "phase-4-fit-parity"},
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
        if (device.contains("gpus")) {
            if (!device["gpus"].is_array()) {
                json error = {
                    {"ok", false},
                    {"phase", "phase-4-fit-parity"},
                    {"error", "device.gpus_must_be_array_when_present"}
                };
                response = error.dump();
                return response.c_str();
            }

            for (const auto & gpu : device["gpus"]) {
                if (!gpu.is_object() || !gpu.contains("free_bytes") || !gpu["free_bytes"].is_number_integer()) {
                    json error = {
                        {"ok", false},
                        {"phase", "phase-4-fit-parity"},
                        {"error", "device.gpus[].free_bytes_required_for_fit_override"}
                    };
                    response = error.dump();
                    return response.c_str();
                }

                const uint64_t free_bytes = gpu["free_bytes"].get<uint64_t>();
                override_device_free_mib.push_back(free_bytes / (1024 * 1024));

                if (gpu.contains("total_bytes") && gpu["total_bytes"].is_number_integer()) {
                    const uint64_t total_bytes = gpu["total_bytes"].get<uint64_t>();
                    override_device_total_mib.push_back(total_bytes / (1024 * 1024));
                }
            }
        }

        if (!override_device_total_mib.empty() && override_device_total_mib.size() != override_device_free_mib.size()) {
            json error = {
                {"ok", false},
                {"phase", "phase-4-fit-parity"},
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
            exec_request.override_device_free_mib = override_device_free_mib;
            exec_request.override_device_total_mib = override_device_total_mib;
            exec_request.has_override_host_free_mib = override_host_free_mib > 0;
            exec_request.has_override_host_total_mib = override_host_free_mib > 0;
            exec_request.override_host_free_mib = override_host_free_mib;
            exec_request.override_host_total_mib = override_host_free_mib;
            exec_request.show_fit_logs = show_fit_logs;
            exec_request.min_ctx = static_cast<uint32_t>(fit_ctx_min);
            exec_request.n_ctx = static_cast<uint32_t>(n_ctx > 0 ? n_ctx : 4096);
            exec_request.n_gpu_layers = runtime.contains("n_gpu_layers") && runtime["n_gpu_layers"].is_number_integer()
                ? static_cast<int32_t>(runtime["n_gpu_layers"].get<int64_t>())
                : -1;

            vram::fit_execution_result exec_result;
            std::string exec_error;
            if (!vram::execute_fit_request(exec_request, exec_result, exec_error)) {
                json error = {
                    {"ok", false},
                    {"engine", "vram-cpp"},
                    {"apiVersion", "0.1.0"},
                    {"phase", "phase-4-fit-parity"},
                    {"error", exec_error.empty() ? "fit_execution_failed" : exec_error}
                };
                response = error.dump();
                return response.c_str();
            }

            json body = {
                {"ok", exec_result.ok},
                {"engine", "vram-cpp"},
                {"apiVersion", "0.1.0"},
                {"phase", "phase-4-fit-parity"},
                {"mode", "fit"},
                {"memory",
                    {
                        {"weights_bytes", (exec_result.totals.model_mib * 1024ULL * 1024ULL)},
                        {"kv_cache_bytes", (exec_result.totals.context_mib * 1024ULL * 1024ULL)},
                        {"device_bytes", ((exec_result.totals.model_mib + exec_result.totals.context_mib + exec_result.totals.compute_mib) * 1024ULL * 1024ULL)},
                        {"host_bytes", ((exec_result.host.model_mib + exec_result.host.context_mib + exec_result.host.compute_mib) * 1024ULL * 1024ULL)}
                    }
                },
                {"fit",
                    {
                        {"executedInProcess", true},
                        {"binary", "in_process"},
                        {"fitTargetMiB", exec_result.fit_target_mib},
                        {"targetFreeMiB", target_free_mib},
                        {"overrideDeviceFreeMiB", exec_result.device_free_mib},
                        {"overrideDeviceTotalMiB", exec_result.device_total_mib},
                        {"overrideHostFreeMiB", exec_result.host_free_mib},
                        {"executeNative", false},
                        {"recommended_n_ctx", exec_result.n_ctx},
                        {"recommended_n_gpu_layers", exec_result.n_gpu_layers},
                        {"status", exec_result.status},
                        {"warnings", exec_result.warnings},
                        {"memoryBreakdown",
                            {
                                {"totals",
                                    {
                                        {"modelMiB", exec_result.totals.model_mib},
                                        {"contextMiB", exec_result.totals.context_mib},
                                        {"computeMiB", exec_result.totals.compute_mib}
                                    }
                                },
                                {"devices", [&]() {
                                    json devices = json::array();
                                    for (const auto & entry : exec_result.devices) {
                                        devices.push_back(fit_memory_entry_to_json(entry));
                                    }
                                    return devices;
                                }()},
                                {"host", fit_memory_entry_to_json(exec_result.host)}
                            }
                        },
                        {"limitations", {
                            "Fit request executed in-process through the vendored llama/common patch surface.",
                            "Reported breakdown reflects a fitted context instantiated in the current runtime after applying override-mode fit parameters."
                        }}
                    }
                }
            };
            response = body.dump();
            return response.c_str();
        }

        json body = {
            {"ok", true},
            {"engine", "vram-cpp"},
            {"apiVersion", "0.1.0"},
            {"phase", "phase-4-fit-parity"},
            {"mode", "fit"},
            {"fit",
                {
                    {"binary", harness_bin},
                    {"args", args},
                    {"executedInProcess", false},
                    {"fitTargetMiB", fit_target_mib},
                    {"targetFreeMiB", target_free_mib},
                    {"overrideDeviceFreeMiB", override_device_free_mib},
                    {"overrideDeviceTotalMiB", override_device_total_mib},
                    {"overrideHostFreeMiB", override_host_free_mib},
                    {"executeNative", false},
                    {"limitations", {
                        "Predictor API does not shell out to llama-fit executables.",
                        "Set fit.execute_in_process=true in a vendor-enabled build to run the fit request through the in-process override path.",
                        "Run in-process fit via vram_fit_harness (native) or embedded C++ fit bridge (wasm target integration step).",
                        "fit_target_mib maps to llama-fit margin-per-device semantics.",
                        "device.gpus[].free_bytes can override detected device memory for deterministic hardware simulation."
                    }}
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
                    {"phase", "phase-2-prefix-parser"},
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

            vram::hf_model_location loc;
            loc.repo = repo;
            loc.file = file;
            loc.revision = revision;

            const std::string url = vram::resolve_hf_file_url(loc);
            if (url.empty()) {
                json error = {
                    {"ok", false},
                    {"phase", "phase-2-prefix-parser"},
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

            const auto requests = vram::build_hf_prefix_range_requests(loc, initial_bytes, max_bytes, growth_factor, token);
            json planned = json::array();
            for (const auto & req : requests) {
                planned.push_back({
                    {"url", req.url},
                    {"start", req.start},
                    {"end", req.end},
                    {"headers", headers_to_json(req.headers)},
                });
            }

            if (!execute_remote) {
                json body = {
                    {"ok", true},
                    {"engine", "vram-cpp"},
                    {"apiVersion", "0.1.0"},
                    {"phase", "phase-2-prefix-parser"},
                    {"source", "huggingface"},
                    {"resolvedUrl", url},
                    {"plannedRequests", planned},
                    {"message", "HF range request planning is ready; set fetch.execute_remote=true to execute requests with platform backend (browser fetch in wasm, curl fallback in native)."}
                };
                response = body.dump();
                return response.c_str();
            }

            json attempts = json::array();
            uint64_t min_required = 0;
            std::string last_error;

            for (const auto & req : requests) {
                std::vector<uint8_t> fetched;
                std::string fetch_error;
                const bool ok_fetch = vram::fetch_hf_range_bytes(req, fetched, fetch_error);

                if (!ok_fetch) {
                    json error = {
                        {"ok", false},
                        {"engine", "vram-cpp"},
                        {"apiVersion", "0.1.0"},
                        {"phase", "phase-2-prefix-parser"},
                        {"source", "huggingface"},
                        {"error", "hf_range_fetch_failed"},
                        {"detail", fetch_error},
                        {"resolvedUrl", url},
                        {"plannedRequests", planned},
                        {"attempts", attempts}
                    };
                    response = error.dump();
                    return response.c_str();
                }

                const auto parsed_prefix = vram::parse_gguf_prefix(fetched.data(), fetched.size(), k_max_tensor_preview);
                attempts.push_back({
                    {"requestedBytes", req.end + 1},
                    {"fetchedBytes", fetched.size()},
                    {"status", parse_status_to_json(parsed_prefix.status)},
                    {"minimumRequiredBytes", parsed_prefix.minimum_required_bytes}
                });

                if (parsed_prefix.status == vram::gguf_prefix_parse_status::complete) {
                    json tensors = json::array();
                    for (const auto & tensor : parsed_prefix.metadata.tensors) {
                        tensors.push_back(tensor_to_json(tensor));
                    }

                    json body = {
                        {"ok", true},
                        {"engine", "vram-cpp"},
                        {"apiVersion", "0.1.0"},
                        {"phase", "phase-2-prefix-parser"},
                        {"mode", parsed.contains("mode") ? parsed["mode"] : json("metadata")},
                        {"source", "huggingface"},
                        {"resolvedUrl", url},
                        {"metadata",
                            {
                                {"version", parsed_prefix.metadata.version},
                                {"kvCount", parsed_prefix.metadata.kv_count},
                                {"tensorCount", parsed_prefix.metadata.tensor_count},
                                {"bytesConsumed", parsed_prefix.metadata.bytes_consumed},
                                {"tensorListTruncated", parsed_prefix.metadata.tensor_list_truncated},
                                {"tensors", tensors},
                            }
                        },
                        {"plannedRequests", planned},
                        {"attempts", attempts},
                        {"fetchExecuted", true}
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
                {"engine", "vram-cpp"},
                {"apiVersion", "0.1.0"},
                {"phase", "phase-2-prefix-parser"},
                {"source", "huggingface"},
                {"error", last_error.empty() ? "insufficient_prefix_bytes" : last_error},
                {"minimumRequiredBytes", min_required},
                {"resolvedUrl", url},
                {"plannedRequests", planned},
                {"attempts", attempts},
                {"fetchExecuted", true}
            };
            response = error.dump();
            return response.c_str();
        }

        json body = {
            {"ok", false},
            {"engine", "vram-cpp"},
            {"apiVersion", "0.1.0"},
            {"phase", "phase-2-prefix-parser"},
            {"error", "unsupported_model_source"},
            {"supportedSources", {"local", "huggingface"}}
        };
        response = body.dump();
        return response.c_str();
    }

    if (!model.contains("path") || !model["path"].is_string()) {
        json error = {
            {"ok", false},
            {"phase", "phase-2-prefix-parser"},
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
            {"phase", "phase-2-prefix-parser"},
            {"error", "file_open_failed"},
            {"path", path}
        };
        response = error.dump();
        return response.c_str();
    }

    if (file_size == 0) {
        json error = {
            {"ok", false},
            {"phase", "phase-2-prefix-parser"},
            {"error", "file_empty"},
            {"path", path}
        };
        response = error.dump();
        return response.c_str();
    }

    const std::vector<vram::byte_range> plan = vram::build_hf_prefix_range_plan(initial_bytes, max_bytes, growth_factor);
    if (plan.empty()) {
        json error = {
            {"ok", false},
            {"phase", "phase-2-prefix-parser"},
            {"error", "invalid_fetch_plan"}
        };
        response = error.dump();
        return response.c_str();
    }

    json attempts = json::array();
    uint64_t min_required = 0;
    std::string last_error;

    for (const vram::byte_range & range : plan) {
        const uint64_t capped_end = std::min(range.end, file_size - 1);
        const uint64_t requested_bytes_u64 = capped_end + 1;
        const size_t requested_bytes = static_cast<size_t>(requested_bytes_u64);

        std::vector<uint8_t> prefix;
        if (!read_file_prefix(path.c_str(), requested_bytes, prefix)) {
            json error = {
                {"ok", false},
                {"phase", "phase-2-prefix-parser"},
                {"error", "file_read_failed"},
                {"requestedBytes", requested_bytes_u64},
                {"path", path}
            };
            response = error.dump();
            return response.c_str();
        }

        const auto parsed_prefix = vram::parse_gguf_prefix(prefix.data(), prefix.size(), k_max_tensor_preview);
        attempts.push_back({
            {"requestedBytes", requested_bytes_u64},
            {"status", parse_status_to_json(parsed_prefix.status)},
            {"minimumRequiredBytes", parsed_prefix.minimum_required_bytes}
        });

        if (parsed_prefix.status == vram::gguf_prefix_parse_status::complete) {
            json tensors = json::array();
            for (const auto & tensor : parsed_prefix.metadata.tensors) {
                tensors.push_back(tensor_to_json(tensor));
            }

            json body = {
                {"ok", true},
                {"engine", "vram-cpp"},
                {"apiVersion", "0.1.0"},
                {"phase", "phase-2-prefix-parser"},
                {"mode", parsed.contains("mode") ? parsed["mode"] : json("metadata")},
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
                },
                {"attempts", attempts}
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
        {"engine", "vram-cpp"},
        {"apiVersion", "0.1.0"},
        {"phase", "phase-2-prefix-parser"},
        {"error", last_error.empty() ? "insufficient_prefix_bytes" : last_error},
        {"path", path},
        {"minimumRequiredBytes", min_required},
        {"attempts", attempts}
    };

    response = error.dump();
    return response.c_str();
    } catch (const std::exception & error) {
        response = json({
            {"ok", false},
            {"engine", "vram-cpp"},
            {"apiVersion", "0.1.0"},
            {"phase", "phase-0-stub"},
            {"error", "predict_exception"},
            {"message", error.what()}
        }).dump();
        return response.c_str();
    } catch (...) {
        response = json({
            {"ok", false},
            {"engine", "vram-cpp"},
            {"apiVersion", "0.1.0"},
            {"phase", "phase-0-stub"},
            {"error", "predict_exception"},
            {"message", "unknown_exception"}
        }).dump();
        return response.c_str();
    }
}
