#include "vram/predictor_api.h"

#include <nlohmann/json.hpp>
#include <string>

namespace {

using nlohmann::json;

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
                {"hfRangeFetch", false},
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

    json body = {
        {"ok", true},
        {"engine", "vram-cpp"},
        {"apiVersion", "0.1.0"},
        {"phase", "phase-0-stub"},
        {"message", "Predictor math is not implemented yet."},
        {"echoRequestPresent", true},
        {"requestKeys", parsed.is_object() ? parsed.size() : 0}
    };

    response = body.dump();
    return response.c_str();
}
