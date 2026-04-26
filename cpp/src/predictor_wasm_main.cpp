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

#include <cstdio>

int main(int argc, char * argv[]) {
    // interpret the first argument as a JSON request and run a prediction
    // then print it to stdout (for testing purposes)
    const char* request_str = argc > 1 ? argv[1] : "{}";
    const char* response = vram_predictor_predict_json(request_str);
    std::puts(response);
    return 0;
}

#endif