#ifndef VRAM_HF_RANGE_PLAN_H
#define VRAM_HF_RANGE_PLAN_H

#include <cstdint>
#include <vector>

namespace vram {

struct byte_range {
    uint64_t start = 0;
    uint64_t end = 0;
};

std::vector<byte_range> build_hf_prefix_range_plan(
    uint64_t initial_bytes,
    uint64_t max_bytes,
    double growth_factor = 2.0);

} // namespace vram

#endif
