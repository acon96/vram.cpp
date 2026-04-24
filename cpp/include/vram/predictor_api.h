#ifndef VRAM_PREDICTOR_API_H
#define VRAM_PREDICTOR_API_H

#ifdef __cplusplus
extern "C" {
#endif

// Returns static JSON with build/system information.
const char * vram_predictor_get_system_info_json(void);

// Phase-0 stub: accepts request JSON and returns a structured placeholder response.
const char * vram_predictor_predict_json(const char * request_json);

#ifdef __cplusplus
}
#endif

#endif
