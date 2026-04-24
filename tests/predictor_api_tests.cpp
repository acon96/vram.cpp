#include "vram/predictor_api.h"

#include <cassert>
#include <cstdio>
#include <string>

namespace {

bool contains(const std::string & haystack, const std::string & needle) {
    return haystack.find(needle) != std::string::npos;
}

std::string resolve_fixture(const char * relative) {
    const char * prefixes[] = {
        "",
        "../",
        "../../",
    };

    for (const char * prefix : prefixes) {
        std::string path(prefix);
        path += relative;
        FILE * fp = std::fopen(path.c_str(), "rb");
        if (fp != nullptr) {
            std::fclose(fp);
            return path;
        }
    }

    return "";
}

void test_invalid_json() {
    const char * response = vram_predictor_predict_json("this-is-not-json");
    const std::string body(response == nullptr ? "" : response);

    assert(contains(body, "\"ok\":false"));
    assert(contains(body, "\"error\":\"invalid_json\""));
}

void test_local_fixture_parse() {
    const std::string path = resolve_fixture("vendor/llama-cpp/models/ggml-vocab-gpt-2.gguf");
    assert(!path.empty());

    const std::string request =
        "{"
        "\"mode\":\"metadata\","
        "\"model\":{\"source\":\"local\",\"path\":\"" + path + "\"},"
        "\"runtime\":{\"n_ctx\":1024,\"cache_type_k\":\"f16\",\"cache_type_v\":\"f16\"},"
        "\"device\":{\"host_ram_bytes\":34359738368},"
        "\"fetch\":{\"initial_bytes\":1024,\"max_bytes\":1048576,\"growth_factor\":2.0}"
        "}";

    const char * response = vram_predictor_predict_json(request.c_str());
    const std::string body(response == nullptr ? "" : response);

    assert(contains(body, "\"ok\":true"));
    assert(contains(body, "\"phase\":\"phase-2-prefix-parser\""));
    assert(contains(body, "\"version\":3"));
    assert(contains(body, "\"kvCount\":16"));
    assert(contains(body, "\"tensorCount\":0"));
}

} // namespace

int main() {
    test_invalid_json();
    test_local_fixture_parse();
    return 0;
}
