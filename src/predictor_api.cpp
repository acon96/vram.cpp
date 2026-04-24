#include "vram/predictor_api.h"

#include "vram/gguf_prefix_parser.h"
#include "vram/hf_range_fetch_helper.h"
#include "vram/hf_range_plan.h"

#include <nlohmann/json.hpp>

#include <algorithm>
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

} // namespace

extern "C" const char * vram_predictor_get_system_info_json(void) {
    static std::string response;
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
                {"llamaFitMode", false},
                {"hfRangeFetch", true},
                {"ggufPrefixParser", true}
            }
        }
    };

    response = body.dump();
    return response.c_str();
}

extern "C" const char * vram_predictor_predict_json(const char * request_json) {
    static std::string response;
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
    const std::string source = model.contains("source") && model["source"].is_string()
        ? model["source"].get<std::string>()
        : "";

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
}
