#include "vram/predictor_api.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <string>
#include <vector>

namespace {

bool contains(const std::string & haystack, const std::string & needle) {
    return haystack.find(needle) != std::string::npos;
}

std::string trim_copy(const std::string & value) {
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }

    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }

    return value.substr(begin, end - begin);
}

std::vector<std::string> collect_model_paths() {
    std::vector<std::string> out;

    const char * model_paths = std::getenv("VRAM_LLAMA_FIT_MODELS");
    if (model_paths != nullptr && *model_paths != '\0') {
        std::string joined(model_paths);
        size_t start = 0;

        while (start < joined.size()) {
            size_t end = start;
            while (end < joined.size() && joined[end] != ',' && joined[end] != ';') {
                ++end;
            }

            const std::string token = trim_copy(joined.substr(start, end - start));
            if (!token.empty()) {
                out.push_back(token);
            }

            start = end + 1;
        }
    }

    if (!out.empty()) {
        return out;
    }

    const char * single_model = std::getenv("VRAM_LLAMA_FIT_MODEL");
    if (single_model != nullptr && *single_model != '\0') {
        out.push_back(single_model);
    }

    return out;
}

void test_fit_mode_executes_in_process_with_overrides() {
    const std::vector<std::string> model_paths = collect_model_paths();
    if (model_paths.empty()) {
        std::puts("SKIP: set VRAM_LLAMA_FIT_MODEL or VRAM_LLAMA_FIT_MODELS");
        return;
    }

    for (const std::string & model_path : model_paths) {
        const std::string request =
            "{"
            "\"mode\":\"fit\","
            "\"model\":{\"source\":\"local\",\"path\":\"" + model_path + "\"},"
            "\"runtime\":{\"n_ctx\":4096,\"n_gpu_layers\":-1,\"cache_type_k\":\"f16\",\"cache_type_v\":\"f16\"},"
            "\"device\":{"
              "\"host_ram_bytes\":34359738368,"
              "\"fit_target_mib\":[512],"
              "\"target_free_mib\":[2048],"
                            "\"gpus\":[{\"id\":\"gpu0\",\"name\":\"A100\",\"index\":2,\"free_bytes\":4294967296,\"total_bytes\":8589934592}]"
            "},"
            "\"fit\":{\"min_ctx\":1024,\"execute_in_process\":true}"
            "}";

        const char * response = vram_predictor_predict_json(request.c_str());
        const std::string body(response == nullptr ? "" : response);

        assert(contains(body, "\"ok\":true"));
        assert(contains(body, "\"executedInProcess\":true"));
        assert(contains(body, "\"targets\":{\"fitMiB\":[2560],\"targetFreeMiB\":[2048]}"));
        assert(contains(body, "\"overrides\":{\"deviceFreeMiB\":[4096],\"deviceTotalMiB\":[8192],\"hostFreeMiB\":32768}"));
        assert(contains(body, "\"recommended\":{\"n_ctx\":4096,\"n_gpu_layers\":-1}"));
        assert(contains(body, "\"breakdown\""));
        assert(contains(body, "\"devices\":[{"));
        assert(contains(body, "\"name\":\"A100 [index 2]\""));
        assert(contains(body, "\"host\":{\"name\":\"Host\""));
        assert(contains(body, "\"totals\":{\"modelMiB\":"));
        assert(contains(body, "\"modelMiB\":"));
        assert(contains(body, "\"contextMiB\":"));
        assert(contains(body, "\"computeMiB\":"));
        assert(contains(body, "\"memoryBytes\""));
        assert(contains(body, "\"weights\":"));
        assert(contains(body, "\"kvCache\":"));
    }
}

} // namespace

int main() {
    test_fit_mode_executes_in_process_with_overrides();
    return 0;
}