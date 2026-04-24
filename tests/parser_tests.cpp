#include "vram/gguf_prefix_parser.h"
#include "vram/hf_range_plan.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace {

void write_u32(std::vector<uint8_t> & out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 24) & 0xff));
}

void write_u64(std::vector<uint8_t> & out, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xff));
    }
}

void write_string(std::vector<uint8_t> & out, const std::string & value) {
    write_u64(out, static_cast<uint64_t>(value.size()));
    out.insert(out.end(), value.begin(), value.end());
}

std::vector<uint8_t> build_minimal_valid_prefix() {
    std::vector<uint8_t> bytes;

    bytes.push_back('G');
    bytes.push_back('G');
    bytes.push_back('U');
    bytes.push_back('F');

    write_u32(bytes, 3);
    write_u64(bytes, 1); // tensors
    write_u64(bytes, 1); // kv pairs

    write_string(bytes, "general.architecture");
    write_u32(bytes, 8); // GGUF_TYPE_STRING
    write_string(bytes, "llama");

    write_string(bytes, "tok_embeddings.weight");
    write_u32(bytes, 2); // n_dims
    write_u64(bytes, 64);
    write_u64(bytes, 32);
    write_u32(bytes, 0); // ggml_type
    write_u64(bytes, 0); // offset

    return bytes;
}

void test_parse_complete_prefix() {
    const std::vector<uint8_t> bytes = build_minimal_valid_prefix();
    const auto result = vram::parse_gguf_prefix(bytes.data(), bytes.size());

    assert(result.status == vram::gguf_prefix_parse_status::complete);
    assert(result.metadata.version == 3);
    assert(result.metadata.kv_count == 1);
    assert(result.metadata.tensor_count == 1);
    assert(result.metadata.tensors.size() == 1);
    assert(result.metadata.tensors[0].name == "tok_embeddings.weight");
    assert(result.metadata.tensors[0].dimensions.size() == 2);
    assert(result.metadata.tensors[0].dimensions[0] == 64);
    assert(result.metadata.tensors[0].dimensions[1] == 32);
}

void test_parse_need_more_data() {
    const std::vector<uint8_t> bytes = build_minimal_valid_prefix();
    const size_t cut = bytes.size() - 5;
    const auto result = vram::parse_gguf_prefix(bytes.data(), cut);

    assert(result.status == vram::gguf_prefix_parse_status::need_more_data);
    assert(result.minimum_required_bytes > cut);
}

void test_parse_invalid_magic() {
    const std::vector<uint8_t> bytes = {0, 1, 2, 3, 4, 5, 6, 7};
    const auto result = vram::parse_gguf_prefix(bytes.data(), bytes.size());

    assert(result.status == vram::gguf_prefix_parse_status::invalid_format);
}

void test_range_plan_growth() {
    const auto ranges = vram::build_hf_prefix_range_plan(1024, 8192, 2.0);
    assert(ranges.size() == 4);
    assert(ranges[0].start == 0 && ranges[0].end == 1023);
    assert(ranges[1].start == 0 && ranges[1].end == 2047);
    assert(ranges[2].start == 0 && ranges[2].end == 4095);
    assert(ranges[3].start == 0 && ranges[3].end == 8191);
}

void test_range_plan_clamp() {
    const auto ranges = vram::build_hf_prefix_range_plan(0, 128, 1.5);
    assert(!ranges.empty());
    assert(ranges.front().start == 0);
    assert(ranges.back().end == 127);
}

bool read_file_bytes(const char * path, std::vector<uint8_t> & out) {
    FILE * fp = std::fopen(path, "rb");
    if (fp == nullptr) {
        return false;
    }

    if (std::fseek(fp, 0, SEEK_END) != 0) {
        std::fclose(fp);
        return false;
    }

    const long n = std::ftell(fp);
    if (n < 0 || std::fseek(fp, 0, SEEK_SET) != 0) {
        std::fclose(fp);
        return false;
    }

    out.resize(static_cast<size_t>(n));
    if (!out.empty()) {
        const size_t read_n = std::fread(out.data(), 1, out.size(), fp);
        if (read_n != out.size()) {
            std::fclose(fp);
            return false;
        }
    }

    std::fclose(fp);
    return true;
}

std::string resolve_fixture(const char * relative) {
    const char * prefixes[] = {
        "",
        "../",
        "../../",
    };

    for (const char * prefix : prefixes) {
        std::string path(prefix);
        path += relative;
        FILE * fp = std::fopen(path.c_str(), "rb");
        if (fp != nullptr) {
            std::fclose(fp);
            return path;
        }
    }

    return "";
}

void test_golden_fixture(
    const char * relative_path,
    uint32_t expected_version,
    uint64_t expected_kv,
    uint64_t expected_tensors) {
    const std::string path = resolve_fixture(relative_path);
    assert(!path.empty());

    std::vector<uint8_t> bytes;
    const bool loaded = read_file_bytes(path.c_str(), bytes);
    assert(loaded);

    const auto result = vram::parse_gguf_prefix(bytes.data(), bytes.size());
    assert(result.status == vram::gguf_prefix_parse_status::complete);
    assert(result.metadata.version == expected_version);
    assert(result.metadata.kv_count == expected_kv);
    assert(result.metadata.tensor_count == expected_tensors);
}

void test_real_gguf_golden_fixtures() {
    test_golden_fixture("vendor/llama-cpp/models/ggml-vocab-llama-bpe.gguf", 3, 20, 0);
    test_golden_fixture("vendor/llama-cpp/models/ggml-vocab-gpt-2.gguf", 3, 16, 0);
    test_golden_fixture("vendor/llama-cpp/models/ggml-vocab-qwen2.gguf", 3, 20, 0);
}

} // namespace

int main() {
    test_parse_complete_prefix();
    test_parse_need_more_data();
    test_parse_invalid_magic();
    test_range_plan_growth();
    test_range_plan_clamp();
    test_real_gguf_golden_fixtures();
    return 0;
}
