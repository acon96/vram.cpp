#include "vram/predictor_api.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

bool contains(const std::string & haystack, const std::string & needle) {
    return haystack.find(needle) != std::string::npos;
}

void test_fit_mode_executes_in_process_with_overrides() {
    const char * model_path = std::getenv("VRAM_LLAMA_FIT_MODEL");
    if (model_path == nullptr) {
        std::puts("SKIP: set VRAM_LLAMA_FIT_MODEL");
        return;
    }

    const std::string request =
        "{"
        "\"mode\":\"fit\","
        "\"model\":{\"source\":\"local\",\"path\":\"" + std::string(model_path) + "\"},"
        "\"runtime\":{\"n_ctx\":4096,\"n_gpu_layers\":-1,\"cache_type_k\":\"f16\",\"cache_type_v\":\"f16\"},"
        "\"device\":{"
          "\"host_ram_bytes\":34359738368,"
          "\"fit_target_mib\":[512],"
          "\"target_free_mib\":[2048],"
          "\"gpus\":[{\"id\":\"gpu0\",\"free_bytes\":4294967296,\"total_bytes\":8589934592}]"
        "},"
        "\"fit\":{\"min_ctx\":1024,\"execute_in_process\":true}"
        "}";

    const char * response = vram_predictor_predict_json(request.c_str());
    const std::string body(response == nullptr ? "" : response);

    assert(contains(body, "\"ok\":true"));
    assert(contains(body, "\"executedInProcess\":true"));
    assert(contains(body, "\"fitTargetMiB\":[2560]"));
    assert(contains(body, "\"targetFreeMiB\":[2048]"));
    assert(contains(body, "\"overrideDeviceFreeMiB\":[4096]"));
    assert(contains(body, "\"overrideDeviceTotalMiB\":[8192]"));
    assert(contains(body, "\"overrideHostFreeMiB\":32768"));
    assert(contains(body, "\"recommended_n_ctx\":4096"));
    assert(contains(body, "\"recommended_n_gpu_layers\":-1"));
    assert(contains(body, "\"memoryBreakdown\""));
    assert(contains(body, "\"devices\":[{"));
    assert(contains(body, "\"host\":{\"name\":\"Host\""));
    assert(contains(body, "\"totals\":{\"modelMiB\":"));
    assert(contains(body, "\"modelMiB\":"));
    assert(contains(body, "\"contextMiB\":"));
    assert(contains(body, "\"computeMiB\":"));
    assert(contains(body, "\"weights_bytes\":"));
    assert(contains(body, "\"kv_cache_bytes\":"));
}

} // namespace

int main() {
    test_fit_mode_executes_in_process_with_overrides();
    return 0;
}