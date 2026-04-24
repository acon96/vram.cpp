#include "vram/hf_range_fetch_helper.h"

#include <array>
#include <cmath>
#include <cstring>
#include <cstdio>

#if defined(__EMSCRIPTEN__)
#include <emscripten/fetch.h>
#endif

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

std::string shell_escape_single_quoted(const std::string & input) {
    std::string out;
    out.reserve(input.size() + 2);
    out.push_back('\'');
    for (const char c : input) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

} // namespace

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

bool fetch_hf_range_bytes(
    const hf_range_request & request,
    std::vector<uint8_t> & out_bytes,
    std::string & error) {
    out_bytes.clear();
    error.clear();

    if (request.url.empty()) {
        error = "empty_url";
        return false;
    }

#if defined(__EMSCRIPTEN__)
    emscripten_fetch_attr_t attr;
    emscripten_fetch_attr_init(&attr);
    std::strcpy(attr.requestMethod, "GET");
    attr.attributes = EMSCRIPTEN_FETCH_LOAD_TO_MEMORY | EMSCRIPTEN_FETCH_SYNCHRONOUS;
    attr.timeoutMSecs = 30000;

    std::vector<const char *> headers;
    headers.reserve(request.headers.size() * 2 + 1);
    for (const auto & kv : request.headers) {
        headers.push_back(kv.first.c_str());
        headers.push_back(kv.second.c_str());
    }
    headers.push_back(nullptr);
    attr.requestHeaders = headers.data();

    emscripten_fetch_t * fetch = emscripten_fetch(&attr, request.url.c_str());
    if (fetch == nullptr) {
        error = "emscripten_fetch_failed";
        return false;
    }

    const int status = static_cast<int>(fetch->status);
    if (status < 200 || status >= 300) {
        error = "http_status_" + std::to_string(status);
        emscripten_fetch_close(fetch);
        return false;
    }

    if (fetch->numBytes <= 0 || fetch->data == nullptr) {
        error = "empty_response";
        emscripten_fetch_close(fetch);
        return false;
    }

    out_bytes.assign(
        reinterpret_cast<const uint8_t *>(fetch->data),
        reinterpret_cast<const uint8_t *>(fetch->data) + static_cast<size_t>(fetch->numBytes));
    emscripten_fetch_close(fetch);
    return true;
#else
    std::string command = "curl -L --silent --show-error --fail";
    command += " --url ";
    command += shell_escape_single_quoted(request.url);

    for (const auto & header : request.headers) {
        command += " -H ";
        command += shell_escape_single_quoted(header.first + ": " + header.second);
    }

    command += " 2>/dev/null";

    FILE * pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        error = "popen_failed";
        return false;
    }

    std::array<unsigned char, 16384> chunk{};
    while (true) {
        const size_t n = std::fread(chunk.data(), 1, chunk.size(), pipe);
        if (n > 0) {
            out_bytes.insert(out_bytes.end(), chunk.data(), chunk.data() + n);
        }

        if (n < chunk.size()) {
            if (std::feof(pipe)) {
                break;
            }
            if (std::ferror(pipe)) {
                break;
            }
        }
    }

    const int rc = pclose(pipe);
    if (rc != 0) {
        error = "curl_command_failed";
        out_bytes.clear();
        return false;
    }

    if (out_bytes.empty()) {
        error = "empty_response";
        return false;
    }

    return true;
#endif
}

} // namespace vram
