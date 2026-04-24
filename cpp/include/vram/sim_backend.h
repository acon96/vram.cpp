#ifndef VRAM_SIM_BACKEND_H
#define VRAM_SIM_BACKEND_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if defined(VRAM_HAS_LLAMA_FIT_EXECUTION)
#include "ggml-backend.h"
#endif

namespace vram {

enum class sim_backend_profile {
    cuda,
    metal,
    vulkan,
    generic,
};

const char * sim_backend_profile_name(sim_backend_profile profile);
bool parse_sim_backend_profile(const std::string & value, sim_backend_profile & profile);

struct sim_device_spec {
    std::string name;
    std::string description;
    uint64_t total_bytes = 0;
    uint64_t free_bytes = 0;
    sim_backend_profile profile = sim_backend_profile::cuda;
};

class sim_backend {
public:
    explicit sim_backend(std::vector<sim_device_spec> specs);
    ~sim_backend();

    sim_backend(const sim_backend &) = delete;
    sim_backend & operator=(const sim_backend &) = delete;

    sim_backend(sim_backend && other) noexcept;
    sim_backend & operator=(sim_backend && other) noexcept;

    bool valid() const;
    size_t device_count() const;
    const sim_device_spec & spec(size_t index) const;
    const std::vector<sim_device_spec> & specs() const;

#if defined(VRAM_HAS_LLAMA_FIT_EXECUTION)
    ggml_backend_dev_t * devices();
#endif

private:
    struct impl;

    std::vector<sim_device_spec> specs_;
    std::unique_ptr<impl> impl_;
};

} // namespace vram

#endif
