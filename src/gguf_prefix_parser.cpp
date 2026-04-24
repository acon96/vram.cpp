#include "vram/gguf_prefix_parser.h"

#include <cstring>
#include <limits>

namespace vram {
namespace {

constexpr uint32_t k_gguf_type_uint8 = 0;
constexpr uint32_t k_gguf_type_int8 = 1;
constexpr uint32_t k_gguf_type_uint16 = 2;
constexpr uint32_t k_gguf_type_int16 = 3;
constexpr uint32_t k_gguf_type_uint32 = 4;
constexpr uint32_t k_gguf_type_int32 = 5;
constexpr uint32_t k_gguf_type_float32 = 6;
constexpr uint32_t k_gguf_type_bool = 7;
constexpr uint32_t k_gguf_type_string = 8;
constexpr uint32_t k_gguf_type_array = 9;
constexpr uint32_t k_gguf_type_uint64 = 10;
constexpr uint32_t k_gguf_type_int64 = 11;
constexpr uint32_t k_gguf_type_float64 = 12;
constexpr uint32_t k_gguf_type_count = 13;

struct parser {
    const uint8_t * data = nullptr;
    size_t size = 0;
    size_t pos = 0;
    size_t need_at_least = 0;

    bool ensure(size_t n) {
        if (pos + n <= size) {
            return true;
        }
        need_at_least = pos + n;
        return false;
    }

    bool skip(size_t n) {
        if (!ensure(n)) {
            return false;
        }
        pos += n;
        return true;
    }

    bool read_u32(uint32_t & out) {
        if (!ensure(4)) {
            return false;
        }
        out =
            static_cast<uint32_t>(data[pos]) |
            (static_cast<uint32_t>(data[pos + 1]) << 8) |
            (static_cast<uint32_t>(data[pos + 2]) << 16) |
            (static_cast<uint32_t>(data[pos + 3]) << 24);
        pos += 4;
        return true;
    }

    bool read_u64(uint64_t & out) {
        if (!ensure(8)) {
            return false;
        }
        out =
            static_cast<uint64_t>(data[pos]) |
            (static_cast<uint64_t>(data[pos + 1]) << 8) |
            (static_cast<uint64_t>(data[pos + 2]) << 16) |
            (static_cast<uint64_t>(data[pos + 3]) << 24) |
            (static_cast<uint64_t>(data[pos + 4]) << 32) |
            (static_cast<uint64_t>(data[pos + 5]) << 40) |
            (static_cast<uint64_t>(data[pos + 6]) << 48) |
            (static_cast<uint64_t>(data[pos + 7]) << 56);
        pos += 8;
        return true;
    }

    bool read_string(std::string & out) {
        uint64_t len = 0;
        if (!read_u64(len)) {
            return false;
        }
        if (len > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return false;
        }
        const size_t n = static_cast<size_t>(len);
        if (!ensure(n)) {
            return false;
        }
        out.assign(reinterpret_cast<const char *>(data + pos), n);
        pos += n;
        return true;
    }

    bool skip_string() {
        uint64_t len = 0;
        if (!read_u64(len)) {
            return false;
        }
        if (len > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return false;
        }
        return skip(static_cast<size_t>(len));
    }
};

bool safe_mul_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if (a > std::numeric_limits<uint64_t>::max() / b) {
        return false;
    }
    out = a * b;
    return true;
}

bool gguf_type_size(uint32_t type, uint64_t & out_size) {
    switch (type) {
        case k_gguf_type_uint8:
        case k_gguf_type_int8:
        case k_gguf_type_bool:
            out_size = 1;
            return true;
        case k_gguf_type_uint16:
        case k_gguf_type_int16:
            out_size = 2;
            return true;
        case k_gguf_type_uint32:
        case k_gguf_type_int32:
        case k_gguf_type_float32:
            out_size = 4;
            return true;
        case k_gguf_type_uint64:
        case k_gguf_type_int64:
        case k_gguf_type_float64:
            out_size = 8;
            return true;
        default:
            return false;
    }
}

bool skip_gguf_value(parser & p, uint32_t value_type) {
    if (value_type >= k_gguf_type_count) {
        return false;
    }

    if (value_type == k_gguf_type_string) {
        return p.skip_string();
    }

    if (value_type == k_gguf_type_array) {
        uint32_t elem_type = 0;
        uint64_t n = 0;
        if (!p.read_u32(elem_type) || !p.read_u64(n)) {
            return false;
        }
        if (elem_type >= k_gguf_type_count || elem_type == k_gguf_type_array) {
            return false;
        }

        if (elem_type == k_gguf_type_string) {
            for (uint64_t i = 0; i < n; ++i) {
                if (!p.skip_string()) {
                    return false;
                }
            }
            return true;
        }

        uint64_t elem_size = 0;
        uint64_t bytes = 0;
        if (!gguf_type_size(elem_type, elem_size) || !safe_mul_u64(elem_size, n, bytes)) {
            return false;
        }
        if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return false;
        }
        return p.skip(static_cast<size_t>(bytes));
    }

    uint64_t scalar_size = 0;
    if (!gguf_type_size(value_type, scalar_size)) {
        return false;
    }
    return p.skip(static_cast<size_t>(scalar_size));
}

} // namespace

gguf_prefix_parse_result parse_gguf_prefix(
    const uint8_t * data,
    size_t size,
    size_t max_tensors_to_store,
    size_t max_dims) {
    gguf_prefix_parse_result result;
    result.status = gguf_prefix_parse_status::invalid_format;

    if (data == nullptr) {
        result.error = "null_input";
        return result;
    }

    parser p;
    p.data = data;
    p.size = size;

    if (!p.ensure(4)) {
        result.status = gguf_prefix_parse_status::need_more_data;
        result.minimum_required_bytes = p.need_at_least;
        return result;
    }

    if (std::memcmp(p.data, "GGUF", 4) != 0) {
        result.error = "invalid_magic";
        return result;
    }
    p.pos += 4;

    uint32_t version = 0;
    if (!p.read_u32(version)) {
        result.status = gguf_prefix_parse_status::need_more_data;
        result.minimum_required_bytes = p.need_at_least;
        return result;
    }
    if (version < 2 || version > 3) {
        result.error = "unsupported_version";
        return result;
    }

    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;
    if (!p.read_u64(n_tensors) || !p.read_u64(n_kv)) {
        result.status = gguf_prefix_parse_status::need_more_data;
        result.minimum_required_bytes = p.need_at_least;
        return result;
    }

    for (uint64_t i = 0; i < n_kv; ++i) {
        if (!p.skip_string()) {
            result.status = gguf_prefix_parse_status::need_more_data;
            result.minimum_required_bytes = p.need_at_least;
            return result;
        }

        uint32_t value_type = 0;
        if (!p.read_u32(value_type)) {
            result.status = gguf_prefix_parse_status::need_more_data;
            result.minimum_required_bytes = p.need_at_least;
            return result;
        }

        if (!skip_gguf_value(p, value_type)) {
            if (p.need_at_least > 0) {
                result.status = gguf_prefix_parse_status::need_more_data;
                result.minimum_required_bytes = p.need_at_least;
            } else {
                result.status = gguf_prefix_parse_status::invalid_format;
                result.error = "invalid_kv_value";
            }
            return result;
        }
    }

    for (uint64_t i = 0; i < n_tensors; ++i) {
        gguf_tensor_info tensor;
        if (!p.read_string(tensor.name)) {
            result.status = gguf_prefix_parse_status::need_more_data;
            result.minimum_required_bytes = p.need_at_least;
            return result;
        }

        uint32_t n_dims = 0;
        if (!p.read_u32(n_dims)) {
            result.status = gguf_prefix_parse_status::need_more_data;
            result.minimum_required_bytes = p.need_at_least;
            return result;
        }
        if (n_dims > max_dims) {
            result.status = gguf_prefix_parse_status::invalid_format;
            result.error = "too_many_dims";
            return result;
        }

        tensor.dimensions.reserve(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            uint64_t dim = 0;
            if (!p.read_u64(dim)) {
                result.status = gguf_prefix_parse_status::need_more_data;
                result.minimum_required_bytes = p.need_at_least;
                return result;
            }
            tensor.dimensions.push_back(dim);
        }

        if (!p.read_u32(tensor.ggml_type) || !p.read_u64(tensor.data_offset)) {
            result.status = gguf_prefix_parse_status::need_more_data;
            result.minimum_required_bytes = p.need_at_least;
            return result;
        }

        if (result.metadata.tensors.size() < max_tensors_to_store) {
            result.metadata.tensors.push_back(std::move(tensor));
        } else {
            result.metadata.tensor_list_truncated = true;
        }
    }

    result.status = gguf_prefix_parse_status::complete;
    result.metadata.version = version;
    result.metadata.tensor_count = n_tensors;
    result.metadata.kv_count = n_kv;
    result.metadata.bytes_consumed = p.pos;
    return result;
}

} // namespace vram
