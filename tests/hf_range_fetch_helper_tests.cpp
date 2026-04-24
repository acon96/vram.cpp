#include "vram/hf_range_fetch_helper.h"

#include <cassert>
#include <string>
#include <vector>

namespace {

void test_resolve_hf_file_url_defaults_revision() {
    const vram::hf_model_location loc = {
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        ""
    };

    const std::string url = vram::resolve_hf_file_url(loc);
    assert(url == "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf");
}

void test_resolve_hf_file_url_encodes_spaces() {
    const vram::hf_model_location loc = {
        "acme/my model",
        "weights/my model.gguf",
        "dev branch"
    };

    const std::string url = vram::resolve_hf_file_url(loc);
    assert(url == "https://huggingface.co/acme/my%20model/resolve/dev%20branch/weights/my%20model.gguf");
}

void test_build_range_requests() {
    const vram::hf_model_location loc = {
        "owner/repo",
        "model.gguf",
        "main"
    };

    const auto requests = vram::build_hf_prefix_range_requests(loc, 1024, 4096, 2.0, "hf_token");
    assert(requests.size() == 3);

    assert(requests[0].start == 0 && requests[0].end == 1023);
    assert(requests[1].start == 0 && requests[1].end == 2047);
    assert(requests[2].start == 0 && requests[2].end == 4095);

    assert(requests[0].headers.size() == 3);
    assert(requests[0].headers[0].first == "Range");
    assert(requests[0].headers[0].second == "bytes=0-1023");
    assert(requests[0].headers[2].first == "Authorization");
    assert(requests[0].headers[2].second == "Bearer hf_token");
}

void test_fetch_empty_url_fails() {
    vram::hf_range_request req;
    std::vector<uint8_t> bytes;
    std::string error;

    const bool ok = vram::fetch_hf_range_bytes(req, bytes, error);
    assert(!ok);
    assert(error == "empty_url");
}

} // namespace

int main() {
    test_resolve_hf_file_url_defaults_revision();
    test_resolve_hf_file_url_encodes_spaces();
    test_build_range_requests();
    test_fetch_empty_url_fails();
    return 0;
}
