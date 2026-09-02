#pragma once
#include <cstdint>
#ifdef _WIN32
#define AS_API __declspec(dllexport)
#else
#define AS_API
#endif
extern "C" {
struct ASNgxConfig { uint32_t width; uint32_t height; uint32_t device_index; uint32_t backend; };
struct ASNgxStatus { uint32_t abi_version; uint32_t adapter_ready; uint64_t context_vram_bytes; char message[512]; };
AS_API int as_ngx_query(ASNgxStatus* out_status);
AS_API int as_ngx_create(const ASNgxConfig* config, void** out_handle);
AS_API int as_ngx_destroy(void* handle);
}
