#include "vram/hf_range_fetch_helper.h"

#include "vram/hf_range_plan.h"

#include <cstdio>

namespace vram {
namespace {

std::string encode_component(const std::string & input, bool encode_slash) {
    static const char * k_hex = "0123456789ABCDEF";
    std::string out;
    out.reserve(input.size() * 3);

    for (const unsigned char c : input) {
        const bool unreserved =
            (c >= 'A' && c <= 'Z') ||
            (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') ||
            c == '-' || c == '_' || c == '.' || c == '~' ||
            (!encode_slash && c == '/');

        if (unreserved) {
            out.push_back(static_cast<char>(c));
            continue;
        }

        out.push_back('%');
        out.push_back(k_hex[(c >> 4) & 0x0F]);
        out.push_back(k_hex[c & 0x0F]);
    }

    return out;
}

} // namespace

std::string resolve_hf_file_url(const hf_model_location & location) {
    if (location.repo.empty() || location.file.empty()) {
        return "";
    }

    std::string revision = location.revision.empty() ? "main" : location.revision;
    std::string url = "https://huggingface.co/";
    url += encode_component(location.repo, false);
    url += "/resolve/";
    url += encode_component(revision, true);
    url += "/";
    url += encode_component(location.file, false);
    return url;
}

std::vector<hf_range_request> build_hf_prefix_range_requests(
    const hf_model_location & location,
    uint64_t initial_bytes,
    uint64_t max_bytes,
    double growth_factor,
    const std::string & bearer_token) {
    std::vector<hf_range_request> requests;
    const std::string url = resolve_hf_file_url(location);
    if (url.empty()) {
        return requests;
    }

    const std::vector<byte_range> plan = build_hf_prefix_range_plan(initial_bytes, max_bytes, growth_factor);
    requests.reserve(plan.size());

    for (const byte_range & range : plan) {
        hf_range_request req;
        req.url = url;
        req.start = range.start;
        req.end = range.end;

        char range_header[64];
        const int n = std::snprintf(range_header, sizeof(range_header), "bytes=%llu-%llu",
            static_cast<unsigned long long>(range.start),
            static_cast<unsigned long long>(range.end));
        if (n <= 0) {
            continue;
        }

        req.headers.push_back({"Range", range_header});
        req.headers.push_back({"Accept", "application/octet-stream"});
        if (!bearer_token.empty()) {
            req.headers.push_back({"Authorization", "Bearer " + bearer_token});
        }

        requests.push_back(std::move(req));
    }

    return requests;
}

} // namespace vram
