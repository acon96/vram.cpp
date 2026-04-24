#include "vram/predictor_api.h"

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#define VRAM_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define VRAM_EXPORT
#endif

extern "C" {

VRAM_EXPORT const char * vram_predictor_get_system_info_json(void);
VRAM_EXPORT const char * vram_predictor_predict_json(const char * request_json);

}

#ifndef __EMSCRIPTEN__

#include <cstdio>

int main() {
    std::puts(vram_predictor_get_system_info_json());
    return 0;
}

#endif
