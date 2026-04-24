#ifndef VRAM_GGUF_PREFIX_PARSER_H
#define VRAM_GGUF_PREFIX_PARSER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vram {

enum class gguf_prefix_parse_status {
    complete,
    need_more_data,
    invalid_format,
};

struct gguf_tensor_info {
    std::string name;
    std::vector<uint64_t> dimensions;
    uint32_t ggml_type = 0;
    uint64_t data_offset = 0;
};

struct gguf_prefix_metadata {
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    uint64_t bytes_consumed = 0;
    bool tensor_list_truncated = false;
    std::vector<gguf_tensor_info> tensors;
};

struct gguf_prefix_parse_result {
    gguf_prefix_parse_status status = gguf_prefix_parse_status::invalid_format;
    std::string error;
    uint64_t minimum_required_bytes = 0;
    gguf_prefix_metadata metadata;
};

gguf_prefix_parse_result parse_gguf_prefix(
    const uint8_t * data,
    size_t size,
    size_t max_tensors_to_store = 256,
    size_t max_dims = 16);

} // namespace vram

#endif
