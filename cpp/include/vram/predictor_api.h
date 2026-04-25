#ifndef VRAM_PREDICTOR_API_H
#define VRAM_PREDICTOR_API_H

#ifdef __cplusplus
extern "C" {
#endif

// accepts request JSON and returns the fitted prediction result
const char * vram_predictor_predict_json(const char * request_json);

#ifdef __cplusplus
}
#endif

#endif
