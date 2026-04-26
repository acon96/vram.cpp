#include "vram/sim_backend.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <utility>

#include "ggml.h"
#include "ggml-backend-impl.h"

#include <cstdlib>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace vram {
namespace {

std::string to_lower_ascii(const std::string & value) {
    std::string out = value;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

} // namespace

const char * sim_backend_profile_name(sim_backend_profile profile) {
    switch (profile) {
        case sim_backend_profile::cuda:
            return "CUDA";
        case sim_backend_profile::metal:
            return "Metal";
        case sim_backend_profile::vulkan:
            return "Vulkan";
        case sim_backend_profile::generic:
            return "Generic";
        default:
            return "Unknown";
    }
}

bool parse_sim_backend_profile(const std::string & value, sim_backend_profile & profile) {
    const std::string normalized = to_lower_ascii(value);
    if (normalized.empty() || normalized == "cuda") {
        profile = sim_backend_profile::cuda;
        return true;
    }
    if (normalized == "metal") {
        profile = sim_backend_profile::metal;
        return true;
    }
    if (normalized == "vulkan") {
        profile = sim_backend_profile::vulkan;
        return true;
    }
    if (normalized == "generic" || normalized == "default") {
        profile = sim_backend_profile::generic;
        return true;
    }
    return false;
}

namespace {

constexpr size_t k_tensor_alignment = 128;

struct sim_reg_context {
    std::vector<std::unique_ptr<ggml_backend_device>> * devices_owned = nullptr;
};

struct sim_backend_stream_context {
    ggml_backend_dev_t device = nullptr;
};

struct sim_device_context {
    sim_device_spec spec;
    ggml_backend_buffer_type buft = {};
    uint64_t allocated_bytes = 0;
};

struct sim_device_caps {
    bool async = true;
    bool host_buffer = false;
    bool buffer_from_host_ptr = false;
    bool events = true;
};

struct sim_buffer_context {
    sim_device_context * device = nullptr;
    void * data = nullptr;
};

void * sim_aligned_alloc(size_t size, size_t alignment) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    void * ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void sim_aligned_free(void * ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

static ggml_guid_t sim_backend_guid(void) {
    static ggml_guid guid = { 0x73, 0x69, 0x6d, 0x62, 0x61, 0x63, 0x6b, 0x65, 0x6e, 0x64, 0x76, 0x72, 0x61, 0x6d, 0x31, 0x00 };
    return &guid;
}

bool sim_supports_mul_mat_type(sim_backend_profile profile, ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_MXFP4:
            return true;
        case GGML_TYPE_Q8_K:
            return profile != sim_backend_profile::vulkan;
        case GGML_TYPE_NVFP4:
            return profile != sim_backend_profile::metal;
        default:
            return false;
    }
}

const char * sim_backend_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "SIM";
}

void sim_backend_free(ggml_backend_t backend) {
    if (backend == nullptr) {
        return;
    }

    delete static_cast<sim_backend_stream_context *>(backend->context);
    delete backend;
}

enum ggml_status sim_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);
    GGML_UNUSED(cgraph);
    return GGML_STATUS_SUCCESS;
}

const struct ggml_backend_i sim_backend_i = {
    /* .get_name                = */ sim_backend_get_name,
    /* .free                    = */ sim_backend_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ sim_backend_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

const char * sim_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "SIM_GPU";
}

void sim_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    if (buffer == nullptr) {
        return;
    }

    auto * ctx = static_cast<sim_buffer_context *>(buffer->context);
    if (ctx != nullptr) {
        if (ctx->device != nullptr && ctx->device->allocated_bytes >= buffer->size) {
            ctx->device->allocated_bytes -= static_cast<uint64_t>(buffer->size);
        } else if (ctx->device != nullptr) {
            ctx->device->allocated_bytes = 0;
        }

        if (ctx->data != nullptr) {
            sim_aligned_free(ctx->data);
        }

        delete ctx;
    }
}

void * sim_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<sim_buffer_context *>(buffer->context);
    return ctx != nullptr ? ctx->data : nullptr;
}

void sim_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    std::memset(static_cast<char *>(tensor->data) + offset, value, size);
}

void sim_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    std::memcpy(static_cast<char *>(tensor->data) + offset, data, size);
}

void sim_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

bool sim_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    if (src == nullptr || dst == nullptr || src->buffer == nullptr) {
        return false;
    }
    if (ggml_backend_buffer_is_host(src->buffer)) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
}

void sim_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    if (buffer == nullptr || buffer->context == nullptr) {
        return;
    }

    auto * ctx = static_cast<sim_buffer_context *>(buffer->context);
    if (ctx == nullptr || ctx->data == nullptr) {
        return;
    }

    std::memset(ctx->data, value, buffer->size);
}

const struct ggml_backend_buffer_i sim_buffer_i = {
    /* .free_buffer     = */ sim_buffer_free_buffer,
    /* .get_base        = */ sim_buffer_get_base,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ sim_buffer_memset_tensor,
    /* .set_tensor      = */ sim_buffer_set_tensor,
    /* .get_tensor      = */ sim_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ sim_buffer_cpy_tensor,
    /* .clear           = */ sim_buffer_clear,
    /* .reset           = */ nullptr,
};

ggml_backend_buffer_t sim_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * device_ctx = static_cast<sim_device_context *>(buft->context);

    if (device_ctx == nullptr) {
        return nullptr;
    }

    if (size > static_cast<size_t>(std::numeric_limits<uint64_t>::max())) {
        return nullptr;
    }

    const uint64_t request_size = static_cast<uint64_t>(size);
    if (request_size > device_ctx->spec.free_bytes ||
            device_ctx->allocated_bytes > device_ctx->spec.free_bytes - request_size) {
        return nullptr;
    }

    void * data = sim_aligned_alloc(size, k_tensor_alignment);
    if (data == nullptr) {
        return nullptr;
    }

    auto * buffer_ctx = new sim_buffer_context;
    if (buffer_ctx == nullptr) {
        sim_aligned_free(data);
        return nullptr;
    }

    buffer_ctx->device = device_ctx;
    buffer_ctx->data = data;

    if (device_ctx != nullptr) {
        device_ctx->allocated_bytes += static_cast<uint64_t>(size);
    }

    return ggml_backend_buffer_init(buft, sim_buffer_i, buffer_ctx, size);
}

size_t sim_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return k_tensor_alignment;
}

bool sim_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

const struct ggml_backend_buffer_type_i sim_buffer_type_i = {
    /* .get_name       = */ sim_buffer_type_get_name,
    /* .alloc_buffer   = */ sim_buffer_type_alloc_buffer,
    /* .get_alignment  = */ sim_buffer_type_get_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ sim_buffer_type_is_host,
};

const char * sim_device_get_name(ggml_backend_dev_t dev) {
    const auto * ctx = static_cast<const sim_device_context *>(dev->context);
    return ctx->spec.name.c_str();
}

const char * sim_device_get_description(ggml_backend_dev_t dev) {
    const auto * ctx = static_cast<const sim_device_context *>(dev->context);
    return ctx->spec.description.c_str();
}

void sim_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    const auto * ctx = static_cast<const sim_device_context *>(dev->context);
    const uint64_t allocated = std::min<uint64_t>(ctx->allocated_bytes, ctx->spec.free_bytes);
    const uint64_t adjusted_free = ctx->spec.free_bytes - allocated;

    if (free != nullptr) {
        *free = static_cast<size_t>(adjusted_free);
    }
    if (total != nullptr) {
        *total = static_cast<size_t>(ctx->spec.total_bytes);
    }
}

enum ggml_backend_dev_type sim_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

void sim_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    const auto * ctx = static_cast<const sim_device_context *>(dev->context);
    const uint64_t allocated = std::min<uint64_t>(ctx->allocated_bytes, ctx->spec.free_bytes);
    const uint64_t adjusted_free = ctx->spec.free_bytes - allocated;

    sim_device_caps caps = {};
    switch (ctx->spec.profile) {
        case sim_backend_profile::cuda:
            caps = {/* async = */ true, /* host_buffer = */ true, /* buffer_from_host_ptr = */ false, /* events = */ true};
            break;
        case sim_backend_profile::vulkan:
            caps = {/* async = */ true, /* host_buffer = */ true, /* buffer_from_host_ptr = */ false, /* events = */ true};
            break;
        case sim_backend_profile::metal:
            caps = {/* async = */ true, /* host_buffer = */ false, /* buffer_from_host_ptr = */ true, /* events = */ true};
            break;
        case sim_backend_profile::generic:
        default:
            caps = {/* async = */ true, /* host_buffer = */ false, /* buffer_from_host_ptr = */ false, /* events = */ true};
            break;
    }

    props->name = ctx->spec.name.c_str();
    props->description = ctx->spec.description.c_str();
    props->memory_free = static_cast<size_t>(adjusted_free);
    props->memory_total = static_cast<size_t>(ctx->spec.total_bytes);
    props->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->device_id = nullptr;
    props->caps = {
        /* .async                 = */ caps.async,
        /* .host_buffer           = */ caps.host_buffer,
        /* .buffer_from_host_ptr  = */ caps.buffer_from_host_ptr,
        /* .events                = */ caps.events,
    };
}

ggml_backend_buffer_type_t sim_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    const auto * ctx = static_cast<const sim_device_context *>(dev->context);
    if (ctx == nullptr) {
        return nullptr;
    }

    switch (ctx->spec.profile) {
        case sim_backend_profile::cuda:
        case sim_backend_profile::vulkan:
            return ggml_backend_cpu_buffer_type();
        case sim_backend_profile::metal:
        case sim_backend_profile::generic:
        default:
            return nullptr;
    }
}

ggml_backend_t sim_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    auto * ctx = new sim_backend_stream_context;
    if (ctx == nullptr) {
        return nullptr;
    }
    ctx->device = dev;

    auto * backend = new ggml_backend {
        /* .guid    = */ sim_backend_guid(),
        /* .iface   = */ sim_backend_i,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };

    if (backend == nullptr) {
        delete ctx;
        return nullptr;
    }

    return backend;
}

ggml_backend_buffer_type_t sim_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * ctx = static_cast<sim_device_context *>(dev->context);
    return &ctx->buft;
}

bool sim_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    if (dev == nullptr || op == nullptr) {
        return false;
    }

    const auto * ctx = static_cast<const sim_device_context *>(dev->context);

    if (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) {
        const struct ggml_tensor * src0 = op->src[0];
        const struct ggml_tensor * src1 = op->src[1];
        if (src0 == nullptr) {
            return false;
        }
        if (src1 != nullptr && src1->type == GGML_TYPE_F16 && src0->type != GGML_TYPE_F16) {
            return false;
        }
        return sim_supports_mul_mat_type(ctx->spec.profile, src0->type);
    }

    if (op->op == GGML_OP_GET_ROWS && op->src[0] != nullptr &&
            ctx->spec.profile == sim_backend_profile::metal &&
            op->src[0]->type == GGML_TYPE_NVFP4) {
        return false;
    }

    return true;
}

bool sim_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->device == dev;
}

bool sim_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    if (op == nullptr) {
        return false;
    }
    return op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID || op->op == GGML_OP_GET_ROWS;
}

const struct ggml_backend_device_i sim_device_i = {
    /* .get_name             = */ sim_device_get_name,
    /* .get_description      = */ sim_device_get_description,
    /* .get_memory           = */ sim_device_get_memory,
    /* .get_type             = */ sim_device_get_type,
    /* .get_props            = */ sim_device_get_props,
    /* .init_backend         = */ sim_device_init_backend,
    /* .get_buffer_type      = */ sim_device_get_buffer_type,
    /* .get_host_buffer_type = */ sim_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ sim_device_supports_op,
    /* .supports_buft        = */ sim_device_supports_buft,
    /* .offload_op           = */ sim_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

const char * sim_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "SIM";
}

size_t sim_reg_get_device_count(ggml_backend_reg_t reg) {
    const auto * reg_context = static_cast<const sim_reg_context *>(reg->context);
    if (reg_context == nullptr || reg_context->devices_owned == nullptr) {
        return 0;
    }
    return reg_context->devices_owned->size();
}

ggml_backend_dev_t sim_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    const auto * reg_context = static_cast<const sim_reg_context *>(reg->context);
    if (reg_context == nullptr || reg_context->devices_owned == nullptr) {
        return nullptr;
    }
    if (index >= reg_context->devices_owned->size()) {
        return nullptr;
    }
    return (*reg_context->devices_owned)[index].get();
}

void * sim_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

const struct ggml_backend_reg_i sim_reg_i = {
    /* .get_name         = */ sim_reg_get_name,
    /* .get_device_count = */ sim_reg_get_device_count,
    /* .get_device       = */ sim_reg_get_device,
    /* .get_proc_address = */ sim_reg_get_proc_address,
};

} // namespace

struct sim_backend::impl {
    ggml_backend_reg reg = {};
    sim_reg_context reg_context = {};
    std::vector<std::unique_ptr<sim_device_context>> device_contexts;
    std::vector<std::unique_ptr<ggml_backend_device>> devices_owned;
    std::vector<ggml_backend_dev_t> devices_terminated;
};

sim_backend::sim_backend(std::vector<sim_device_spec> specs)
    : specs_(std::move(specs)),
      impl_(nullptr) {
    if (specs_.empty()) {
        return;
    }

    impl_ = std::make_unique<impl>();
    if (!impl_) {
        return;
    }

    impl_->reg.api_version = GGML_BACKEND_API_VERSION;
    impl_->reg.iface = sim_reg_i;
    impl_->reg_context.devices_owned = &impl_->devices_owned;
    impl_->reg.context = &impl_->reg_context;

    impl_->device_contexts.reserve(specs_.size());
    impl_->devices_owned.reserve(specs_.size());
    impl_->devices_terminated.reserve(specs_.size() + 1);

    for (size_t i = 0; i < specs_.size(); ++i) {
        auto device_context = std::make_unique<sim_device_context>();
        device_context->spec = specs_[i];
        if (device_context->spec.name.empty()) {
            device_context->spec.name = "Sim GPU " + std::to_string(i);
        }
        if (device_context->spec.description.empty()) {
            device_context->spec.description = sim_backend_profile_name(device_context->spec.profile) + std::string(" backend");
        }
        if (device_context->spec.total_bytes < device_context->spec.free_bytes) {
            device_context->spec.total_bytes = device_context->spec.free_bytes;
        }

        device_context->buft.iface = sim_buffer_type_i;
        device_context->buft.device = nullptr;
        device_context->buft.context = device_context.get();

        auto device = std::make_unique<ggml_backend_device>();
        device->iface = sim_device_i;
        device->reg = &impl_->reg;
        device->context = device_context.get();

        device_context->buft.device = device.get();

        specs_[i] = device_context->spec;
        impl_->device_contexts.push_back(std::move(device_context));
        impl_->devices_owned.push_back(std::move(device));
    }

    for (const auto & device : impl_->devices_owned) {
        impl_->devices_terminated.push_back(device.get());
    }
    impl_->devices_terminated.push_back(nullptr);
}

sim_backend::~sim_backend() = default;

sim_backend::sim_backend(sim_backend && other) noexcept = default;

sim_backend & sim_backend::operator=(sim_backend && other) noexcept = default;

bool sim_backend::valid() const {
    return impl_ != nullptr && !impl_->devices_owned.empty();
}

size_t sim_backend::device_count() const {
    return specs_.size();
}

const sim_device_spec & sim_backend::spec(size_t index) const {
    return specs_.at(index);
}

const std::vector<sim_device_spec> & sim_backend::specs() const {
    return specs_;
}

ggml_backend_dev_t * sim_backend::devices() {
    if (!impl_ || impl_->devices_terminated.empty()) {
        return nullptr;
    }
    return impl_->devices_terminated.data();
}

} // namespace vram
