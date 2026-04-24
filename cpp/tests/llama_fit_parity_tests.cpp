#include "vram/predictor_api.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

bool contains(const std::string & haystack, const std::string & needle) {
    return haystack.find(needle) != std::string::npos;
}

std::string read_command_output(const std::string & command, int & exit_code) {
    exit_code = -1;
    std::string out;

    FILE * pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return out;
    }

    char buf[4096];
    while (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
        out += buf;
    }

    exit_code = pclose(pipe);
    return out;
}

std::string find_fit_cli_line(const std::string & output) {
    size_t pos = 0;
    while (pos < output.size()) {
        size_t end = output.find('\n', pos);
        if (end == std::string::npos) {
            end = output.size();
        }
        const std::string line = output.substr(pos, end - pos);
        if (line.find("-c ") != std::string::npos && line.find(" -ngl ") != std::string::npos) {
            return line;
        }
        pos = end + 1;
    }
    return "";
}

void test_optional_native_fit_parity() {
    const char * model_path = std::getenv("VRAM_LLAMA_FIT_MODEL");
    const char * fit_bin = std::getenv("VRAM_LLAMA_FIT_BINARY");
    const char * harness_bin = std::getenv("VRAM_FIT_HARNESS_BINARY");

    if (model_path == nullptr || fit_bin == nullptr || harness_bin == nullptr) {
        std::puts("SKIP: set VRAM_LLAMA_FIT_MODEL, VRAM_LLAMA_FIT_BINARY, VRAM_FIT_HARNESS_BINARY");
        return;
    }

    std::string cmd_fit = std::string(fit_bin) + " --model '" + model_path + "' --fit on --fit-target 512 --fit-ctx 1024 -c 4096 2>&1";
    int ec_fit = -1;
    const std::string out_fit = read_command_output(cmd_fit, ec_fit);
    assert(ec_fit == 0);

    std::string cmd_h = std::string(harness_bin) + " --model '" + model_path + "' --fit-target-mib 512 --fit-ctx 1024 -c 4096 2>&1";
    int ec_h = -1;
    const std::string out_h = read_command_output(cmd_h, ec_h);
    assert(ec_h == 0);

    const std::string line_fit = find_fit_cli_line(out_fit);
    assert(!line_fit.empty());

    assert(contains(out_h, "\"ok\":true"));
    assert(contains(out_h, "\"n_ctx\":"));
    assert(contains(out_h, "\"n_gpu_layers\":"));
    assert(contains(line_fit, "-c "));
    assert(contains(line_fit, "-ngl "));

    const std::string request =
        "{"
        "\"mode\":\"fit\","
        "\"model\":{\"source\":\"local\",\"path\":\"" + std::string(model_path) + "\"},"
        "\"runtime\":{\"n_ctx\":4096,\"cache_type_k\":\"f16\",\"cache_type_v\":\"f16\"},"
        "\"device\":{\"host_ram_bytes\":34359738368,\"fit_target_mib\":[512]},"
        "\"fit\":{\"fit_harness_binary\":\"vram_fit_harness\",\"min_ctx\":1024}"
        "}";

    const char * response = vram_predictor_predict_json(request.c_str());
    const std::string body(response == nullptr ? "" : response);

    assert(contains(body, "\"ok\":true"));
    assert(contains(body, "\"executedInProcess\":false"));
    assert(contains(body, "\"command\":{\"binary\":\"vram_fit_harness\""));
}

} // namespace

int main() {
    test_optional_native_fit_parity();
    return 0;
}
