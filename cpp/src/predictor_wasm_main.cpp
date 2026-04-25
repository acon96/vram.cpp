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
