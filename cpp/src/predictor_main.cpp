#include "vram/predictor_api.h"

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#define VRAM_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define VRAM_EXPORT
#endif

extern "C" {

VRAM_EXPORT const char * vram_predictor_predict_json(const char * request_json);

}

#ifndef __EMSCRIPTEN__

#include <cstring>
#include <cstdio>
#include <string>

namespace {

std::string read_stdin() {
    std::string input;
    char buffer[4096];
    while (std::fgets(buffer, static_cast<int>(sizeof(buffer)), stdin) != nullptr) {
        input += buffer;
    }
    return input;
}

bool response_is_ok(const char * response) {
    if (response == nullptr) {
        return false;
    }

    // Keep native CLI behavior simple: return non-zero when API reports ok:false.
    return std::strstr(response, "\"ok\":false") == nullptr;
}

} // namespace

int main(int argc, char * argv[]) {
    // Native testing path: accept JSON from argv[1], or stdin when no args are provided.
    std::string request_json;
    if (argc > 1) {
        request_json = argv[1];
    } else {
        request_json = read_stdin();
    }

    if (request_json.empty()) {
        request_json = "{}";
    }

    const char * request_str = request_json.c_str();
    const char* response = vram_predictor_predict_json(request_str);
    std::puts(response);
    return response_is_ok(response) ? 0 : 1;
}

#endif
