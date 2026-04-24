#ifndef VRAM_HF_RANGE_FETCH_HELPER_H
#define VRAM_HF_RANGE_FETCH_HELPER_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vram {

struct byte_range {
    uint64_t start = 0;
    uint64_t end = 0;
};

struct hf_model_location {
    std::string repo;
    std::string file;
    std::string revision;
};

struct hf_range_request {
    std::string url;
    uint64_t start = 0;
    uint64_t end = 0;
    std::vector<std::pair<std::string, std::string>> headers;
};

std::vector<byte_range> build_hf_prefix_range_plan(
    uint64_t initial_bytes,
    uint64_t max_bytes,
    double growth_factor = 2.0);

std::string resolve_hf_file_url(const hf_model_location & location);

std::vector<hf_range_request> build_hf_prefix_range_requests(
    const hf_model_location & location,
    uint64_t initial_bytes,
    uint64_t max_bytes,
    double growth_factor,
    const std::string & bearer_token = "");

// Executes one range request using the active platform backend:
// - wasm/emscripten: browser fetch API
// - native: curl command fallback
bool fetch_hf_range_bytes(
    const hf_range_request & request,
    std::vector<uint8_t> & out_bytes,
    std::string & error);

} // namespace vram

#endif
