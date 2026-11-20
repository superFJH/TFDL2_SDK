#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// ABI 1. Callbacks receive mono FP32 at the model sample rate. Return zero to
// continue or nonzero to cancel; data is valid only during the callback.
// Dialogue mode invokes callbacks on a worker thread. Calls remain ordered and
// finish before generate returns. Do not reenter the same handle from a callback.
typedef int (*tfllm_breeze_audio_cb)(const float*,size_t,void*);
int tfllm_breeze_abi_version(void);
int tfllm_breeze_has_npu(void);
const char* tfllm_breeze_last_error(void);
void* tfllm_breeze_create(const char* model,const char* options_json);
void tfllm_breeze_destroy(void*);
// Returned strings are thread-local and valid until the next call on that
// thread. NULL means failure; inspect last_error. Handles must outlive calls.
const char* tfllm_breeze_info(void*);
int tfllm_breeze_validate_request(void*,const char* request_json);
const char* tfllm_breeze_generate(void*,const char* request_json,tfllm_breeze_audio_cb,void*);
#ifdef __cplusplus
}
#endif
