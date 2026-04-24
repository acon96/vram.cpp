#include "vram/hf_range_plan.h"

#include <cmath>

namespace vram {

std::vector<byte_range> build_hf_prefix_range_plan(
    uint64_t initial_bytes,
    uint64_t max_bytes,
    double growth_factor) {
    std::vector<byte_range> ranges;
    if (max_bytes == 0) {
        return ranges;
    }

    if (initial_bytes == 0 || initial_bytes > max_bytes) {
        initial_bytes = max_bytes < 1024 * 1024 ? max_bytes : 1024 * 1024;
    }

    if (initial_bytes == 0) {
        initial_bytes = 1;
    }

    if (growth_factor <= 1.0) {
        growth_factor = 2.0;
    }

    uint64_t current = initial_bytes;
    while (true) {
        ranges.push_back({0, current - 1});
        if (current >= max_bytes) {
            break;
        }

        uint64_t next = static_cast<uint64_t>(std::ceil(static_cast<double>(current) * growth_factor));
        if (next <= current) {
            next = current + 1;
        }
        if (next > max_bytes) {
            next = max_bytes;
        }
        current = next;
    }

    return ranges;
}

} // namespace vram
