#ifndef VRAM_FIT_EXECUTOR_H
#define VRAM_FIT_EXECUTOR_H

#include <cstdint>
#include <string>
#include <vector>

namespace vram {

struct fit_memory_breakdown_entry {
    std::string name;
    uint64_t total_mib = 0;
    uint64_t free_mib = 0;
    uint64_t model_mib = 0;
    uint64_t context_mib = 0;
    uint64_t compute_mib = 0;
    uint64_t unaccounted_mib = 0;
};

struct fit_memory_breakdown_totals {
    uint64_t model_mib = 0;
    uint64_t context_mib = 0;
    uint64_t compute_mib = 0;
};

struct fit_execution_request {
    std::string model_path;
    std::vector<uint64_t> fit_target_mib;
    std::vector<uint64_t> target_free_mib;
    std::vector<uint64_t> override_device_free_mib;
    std::vector<uint64_t> override_device_total_mib;
    std::vector<std::string> override_device_labels;
    bool has_override_host_free_mib = false;
    bool has_override_host_total_mib = false;
    uint64_t override_host_free_mib = 0;
    uint64_t override_host_total_mib = 0;
    bool show_fit_logs = false;
    uint32_t n_ctx = 4096;
    uint32_t n_batch = 0;
    uint32_t n_ubatch = 0;
    uint32_t min_ctx = 0;
    int32_t n_gpu_layers = -1;
};

struct fit_execution_result {
    bool ok = false;
    int status = 0;
    uint32_t n_ctx = 0;
    int32_t n_gpu_layers = 0;
    std::vector<uint64_t> fit_target_mib;
    std::vector<std::string> warnings;
    bool memory_override_enabled = false;
    std::vector<uint64_t> device_free_mib;
    std::vector<uint64_t> device_total_mib;
    bool host_override_enabled = false;
    uint64_t host_free_mib = 0;
    uint64_t host_total_mib = 0;
    fit_memory_breakdown_totals totals;
    std::vector<fit_memory_breakdown_entry> devices;
    fit_memory_breakdown_entry host;
};

bool fit_execution_available();

bool execute_fit_request(const fit_execution_request & request, fit_execution_result & result, std::string & error);

} // namespace vram

#endif