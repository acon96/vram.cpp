#include "vram/predictor_api.h"

#include "vram/fit_executor.h"

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

const json json_obj_or_default( const json & obj, const char * key, const json & fallback) {
    if (!obj.is_object() || !obj.contains(key)) {
        return fallback;
    }

    const json & value = obj[key];
    if (!value.is_object()) {
        return fallback;
    }

    return value.get<json>();
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

        const std::string model = json_string_or_default(parsed, "model", "");
        const json runtime = json_obj_or_default(parsed, "runtime", json::object());
        const json device = json_obj_or_default(parsed, "device", json::object());
        const bool show_fit_logs = json_bool_or_default(parsed, "show_fit_logs", false);

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

                uint64_t total_bytes = free_bytes;

                if (gpu.contains("total_bytes") && gpu["total_bytes"].is_number_integer()) {
                    total_bytes = gpu["total_bytes"].get<uint64_t>();
                    override_device_total_mib.push_back(total_bytes / (1024 * 1024));
                }

                vram::sim_device_spec sim_device;
                sim_device.name = label;
                sim_device.description = vram::sim_backend_profile_name(backend_profile) + std::string(" device");
                sim_device.free_bytes = free_bytes;
                sim_device.total_bytes = std::max(total_bytes, free_bytes);
                sim_device.profile = backend_profile;
                simulated_devices.push_back(sim_device);
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

            std::vector<std::string> args;
            args.push_back("--model");
            args.push_back(model);
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

            const uint64_t fit_ctx_min = json_u64_or_default(runtime, "min_ctx", 0);
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

            vram::fit_execution_request::split_mode_type split_mode = vram::fit_execution_request::split_mode_type::layer;
            if (runtime.contains("split_mode")) {
                if (!runtime["split_mode"].is_string()) {
                    json error = {
                        {"ok", false},
                        {"error", "runtime.split_mode_must_be_string_when_present"}
                    };
                    response = error.dump();
                    return response.c_str();
                }

                const std::string split_mode_name = runtime["split_mode"].get<std::string>();
                if (split_mode_name == "layer") {
                    split_mode = vram::fit_execution_request::split_mode_type::layer;
                } else if (split_mode_name == "row") {
                    split_mode = vram::fit_execution_request::split_mode_type::row;
                } else if (split_mode_name == "tensor") {
                    split_mode = vram::fit_execution_request::split_mode_type::tensor;
                } else {
                    json error = {
                        {"ok", false},
                        {"error", "runtime.split_mode_invalid"},
                        {"value", split_mode_name}
                    };
                    response = error.dump();
                    return response.c_str();
                }

                args.push_back("--split-mode");
                args.push_back(split_mode_name);
            }

            if (show_fit_logs) {
                args.push_back("--show-fit-logs");
            }

            vram::fit_execution_request exec_request;
            exec_request.model_path = model;
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
            exec_request.split_mode = split_mode;

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
