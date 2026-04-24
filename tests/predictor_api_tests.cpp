#include "vram/predictor_api.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

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
    assert(contains(body, "\"version\":3"));
    assert(contains(body, "\"kvCount\":16"));
    assert(contains(body, "\"tensorCount\":0"));
}

void test_metadata_only_fixture_matrix() {
    const std::vector<std::string> fixture_paths = {
        "vendor/llama-cpp/models/ggml-vocab-gpt-neox.gguf",
        "vendor/llama-cpp/models/ggml-vocab-llama-spm.gguf",
        "vendor/llama-cpp/models/ggml-vocab-starcoder.gguf",
    };

    for (const std::string & fixture_relative : fixture_paths) {
        const std::string path = resolve_fixture(fixture_relative.c_str());
        assert(!path.empty());

        const std::string request =
            "{"
            "\"mode\":\"metadata\","
            "\"model\":{\"source\":\"local\",\"path\":\"" + path + "\"},"
            "\"runtime\":{\"n_ctx\":1024,\"cache_type_k\":\"f16\",\"cache_type_v\":\"f16\"},"
            "\"device\":{\"host_ram_bytes\":34359738368},"
            "\"fetch\":{\"initial_bytes\":1024,\"max_bytes\":4194304,\"growth_factor\":2.0}"
            "}";

        const char * response = vram_predictor_predict_json(request.c_str());
        const std::string body(response == nullptr ? "" : response);

        assert(contains(body, "\"ok\":true"));
        assert(contains(body, "\"source\":\"local\""));
        assert(contains(body, "\"path\":\"" + path + "\""));
        assert(contains(body, "\"tensorCount\":0"));
    }
}

void test_hf_request_planning() {
    const std::string request =
        "{"
        "\"mode\":\"metadata\","
        "\"model\":{"
            "\"source\":\"huggingface\"," 
            "\"huggingFace\":{"
                "\"repo\":\"Qwen/Qwen2.5-0.5B-Instruct-GGUF\"," 
                "\"file\":\"qwen2.5-0.5b-instruct-q4_k_m.gguf\"," 
                "\"revision\":\"main\""
            "}"
        "},"
        "\"runtime\":{\"n_ctx\":1024,\"cache_type_k\":\"f16\",\"cache_type_v\":\"f16\"},"
        "\"device\":{\"host_ram_bytes\":34359738368},"
        "\"fetch\":{\"initial_bytes\":1024,\"max_bytes\":4096,\"growth_factor\":2.0}"
        "}";

    const char * response = vram_predictor_predict_json(request.c_str());
    const std::string body(response == nullptr ? "" : response);

    assert(contains(body, "\"ok\":true"));
    assert(contains(body, "\"source\":\"huggingface\""));
    assert(contains(body, "\"resolvedUrl\":\"https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf\""));
    assert(contains(body, "\"requests\""));
    assert(contains(body, "\"start\":0"));
    assert(contains(body, "\"end\":1023"));
}

void test_hf_split_gguf_request_planning() {
    const std::string request =
        "{"
        "\"mode\":\"metadata\","
        "\"model\":{"
            "\"source\":\"huggingface\","
            "\"huggingFace\":{"
                "\"repo\":\"bartowski/Llama-3.2-1B-Instruct-GGUF\","
                "\"file\":\"Llama-3.2-1B-Instruct-Q4_K_M-00001-of-00002.gguf\","
                "\"revision\":\"main\""
            "}"
        "},"
        "\"runtime\":{\"n_ctx\":1024,\"cache_type_k\":\"f16\",\"cache_type_v\":\"f16\"},"
        "\"device\":{\"host_ram_bytes\":34359738368},"
        "\"fetch\":{\"initial_bytes\":1024,\"max_bytes\":4096,\"growth_factor\":2.0}"
        "}";

    const char * response = vram_predictor_predict_json(request.c_str());
    const std::string body(response == nullptr ? "" : response);

    assert(contains(body, "\"ok\":true"));
    assert(contains(body, "\"source\":\"huggingface\""));
    assert(contains(body, "Llama-3.2-1B-Instruct-Q4_K_M-00001-of-00002.gguf"));
    assert(contains(body, "\"resolvedUrl\":\"https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M-00001-of-00002.gguf\""));
    assert(contains(body, "\"requests\""));
}

void test_fit_mode_command_planning() {
    const std::string request =
        "{"
        "\"mode\":\"fit\","
        "\"model\":{\"source\":\"local\",\"path\":\"/tmp/sample.gguf\"},"
                "\"runtime\":{\"n_ctx\":4096,\"n_gpu_layers\":-1,\"cache_type_k\":\"f16\",\"cache_type_v\":\"f16\"},"
                "\"device\":{"
                    "\"host_ram_bytes\":34359738368,"
                    "\"fit_target_mib\":[256,512],"
                    "\"target_free_mib\":[2048,1024],"
                    "\"gpus\":["
                        "{\"id\":\"gpu0\",\"free_bytes\":8589934592,\"total_bytes\":12884901888},"
                        "{\"id\":\"gpu1\",\"free_bytes\":6442450944,\"total_bytes\":8589934592}"
                    "]"
                "},"
                "\"fit\":{\"fit_harness_binary\":\"vram_fit_harness\",\"min_ctx\":1024,\"show_fit_logs\":true}"
        "}";

    const char * response = vram_predictor_predict_json(request.c_str());
    const std::string body(response == nullptr ? "" : response);

    assert(contains(body, "\"ok\":true"));
    assert(contains(body, "\"executedInProcess\":false"));
    assert(contains(body, "\"command\":{\"binary\":\"vram_fit_harness\""));
    assert(contains(body, "\"args\""));
    assert(contains(body, "--fit-target-mib"));
    assert(contains(body, "--target-free-mib"));
    assert(contains(body, "--override-device-free-mib"));
    assert(contains(body, "--override-device-total-mib"));
    assert(contains(body, "--override-host-free-mib"));
    assert(contains(body, "--n-gpu-layers"));
    assert(contains(body, "--show-fit-logs"));
    assert(contains(body, "256,512"));
    assert(contains(body, "2048,1024"));
    assert(contains(body, "8192,6144"));
    assert(contains(body, "12288,8192"));
    assert(contains(body, "\"targets\":{\"fitMiB\":[256,512],\"targetFreeMiB\":[2048,1024]}"));
    assert(contains(body, "\"overrides\":{\"deviceFreeMiB\":[8192,6144],\"deviceTotalMiB\":[12288,8192],\"hostFreeMiB\":32768}"));
}

void test_fit_mode_heterogeneous_gpu_planning() {
    const std::string request =
        "{"
        "\"mode\":\"fit\","
        "\"model\":{\"source\":\"local\",\"path\":\"/tmp/hetero.gguf\"},"
        "\"runtime\":{\"n_ctx\":4096,\"n_gpu_layers\":-1,\"cache_type_k\":\"f16\",\"cache_type_v\":\"f16\"},"
        "\"device\":{"
            "\"host_ram_bytes\":68719476736,"
            "\"fit_target_mib\":[512,768,1024],"
            "\"target_free_mib\":[2048],"
            "\"gpus\":["
                "{\"id\":\"gpu0\",\"free_bytes\":8589934592,\"total_bytes\":17179869184},"
                "{\"id\":\"gpu1\",\"free_bytes\":5368709120,\"total_bytes\":12884901888},"
                "{\"id\":\"gpu2\",\"free_bytes\":21474836480,\"total_bytes\":25769803776}"
            "]"
        "},"
        "\"fit\":{\"fit_harness_binary\":\"vram_fit_harness\",\"min_ctx\":1024}"
        "}";

    const char * response = vram_predictor_predict_json(request.c_str());
    const std::string body(response == nullptr ? "" : response);

    assert(contains(body, "\"ok\":true"));
    assert(contains(body, "\"targets\":{\"fitMiB\":[512,768,1024],\"targetFreeMiB\":[2048]}"));
    assert(contains(body, "\"overrides\":{\"deviceFreeMiB\":[8192,5120,20480],\"deviceTotalMiB\":[16384,12288,24576],\"hostFreeMiB\":65536}"));
}

} // namespace

int main() {
    test_invalid_json();
    test_local_fixture_parse();
    test_metadata_only_fixture_matrix();
    test_hf_request_planning();
    test_hf_split_gguf_request_planning();
    test_fit_mode_command_planning();
    test_fit_mode_heterogeneous_gpu_planning();
    return 0;
}
