// Minimal process-local CUPTI Activity tracer. Inspired by NVIDIA's CUPTI
// 13.0 cupti_trace_injection/activity_trace_async samples; no sample code copied.
// Enable only for the target child via CUDA_INJECTION64_PATH and set
// SAGE_CUPTI_TRACE_PATH to a new local TSV file. No hardware counters are used.
#include <windows.h>
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

static std::mutex output_mutex;
static FILE* output = nullptr;
static constexpr size_t kBufferBytes = 8 * 1024 * 1024;

static void status(const char* operation, CUptiResult result) {
    const char* message = nullptr;
    cuptiGetResultString(result, &message);
    std::lock_guard<std::mutex> guard(output_mutex);
    if (output) { std::fprintf(output, "status\t%s\t%d\t%s\n", operation, int(result), message ? message : "unknown"); std::fflush(output); }
}

static void CUPTIAPI request_buffer(uint8_t** buffer, size_t* size, size_t* max_records) {
    *buffer = static_cast<uint8_t*>(_aligned_malloc(kBufferBytes, 8));
    *size = *buffer ? kBufferBytes : 0;
    *max_records = 0;
}

static void CUPTIAPI complete_buffer(CUcontext context, uint32_t stream, uint8_t* buffer, size_t, size_t valid) {
    if (!buffer) return;
    std::lock_guard<std::mutex> guard(output_mutex);
    CUpti_Activity* record = nullptr;
    CUptiResult result = CUPTI_SUCCESS;
    while (valid && (result = cuptiActivityGetNextRecord(buffer, valid, &record)) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            auto* k = reinterpret_cast<CUpti_ActivityKernel10*>(record);
            std::fprintf(output, "kernel\t%llu\t%llu\t%u\t%u\t%u\t%s\n",
                static_cast<unsigned long long>(k->start), static_cast<unsigned long long>(k->end),
                k->correlationId, k->graphId, k->streamId, k->name ? k->name : "unknown");
        } else if (record->kind == CUPTI_ACTIVITY_KIND_DRIVER || record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
            auto* api = reinterpret_cast<CUpti_ActivityAPI*>(record);
            const char* name = nullptr;
            const auto domain = record->kind == CUPTI_ACTIVITY_KIND_DRIVER ? CUPTI_CB_DOMAIN_DRIVER_API : CUPTI_CB_DOMAIN_RUNTIME_API;
            cuptiGetCallbackName(domain, api->cbid, &name);
            if (name && std::strstr(name, "Launch"))
                std::fprintf(output, "api\t%llu\t%llu\t%u\t%u\t%s\n",
                    static_cast<unsigned long long>(api->start), static_cast<unsigned long long>(api->end),
                    api->correlationId, api->threadId, name);
        }
    }
    size_t dropped = 0;
    CUptiResult drops_result = cuptiActivityGetNumDroppedRecords(context, stream, &dropped);
    if (drops_result != CUPTI_SUCCESS || dropped) std::fprintf(output, "dropped\t%zu\t%d\n", dropped, int(drops_result));
    if (valid && result != CUPTI_ERROR_MAX_LIMIT_REACHED && result != CUPTI_SUCCESS) std::fprintf(output, "record_error\t%d\n", int(result));
    std::fflush(output);
    _aligned_free(buffer);
}

static DWORD WINAPI flush_worker(void*) {
    for (;;) {
        Sleep(1000);
        uint64_t timestamp = 0;
        FILETIME wall;
        GetSystemTimePreciseAsFileTime(&wall);
        CUptiResult clock_result = cuptiGetTimestamp(&timestamp);
        {
            std::lock_guard<std::mutex> guard(output_mutex);
            ULARGE_INTEGER ticks;
            ticks.LowPart = wall.dwLowDateTime; ticks.HighPart = wall.dwHighDateTime;
            std::fprintf(output, "clock\t%llu\t%llu\t%d\n", static_cast<unsigned long long>(timestamp),
                static_cast<unsigned long long>(ticks.QuadPart), int(clock_result));
            std::fflush(output);
        }
        CUptiResult result = cuptiActivityFlushAll(0);
        if (result != CUPTI_SUCCESS) { status("flush", result); return 1; }
    }
}

extern "C" __declspec(dllexport) int InitializeInjection() {
    char path[32768];
    DWORD length = GetEnvironmentVariableA("SAGE_CUPTI_TRACE_PATH", path, sizeof(path));
    if (!length || length >= sizeof(path)) return 0;
    if (fopen_s(&output, path, "wx") != 0 || !output) return 0;
    std::fprintf(output, "schema\t1\n"); std::fflush(output);
    CUptiResult result = cuptiActivityRegisterCallbacks(request_buffer, complete_buffer);
    status("register", result); if (result != CUPTI_SUCCESS) return 0;
    for (auto kind : {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_DRIVER, CUPTI_ACTIVITY_KIND_RUNTIME}) {
        result = cuptiActivityEnable(kind); status("enable", result); if (result != CUPTI_SUCCESS) return 0;
    }
    HANDLE worker = CreateThread(nullptr, 0, flush_worker, nullptr, 0, nullptr);
    if (!worker) { status("create_worker", CUPTI_ERROR_UNKNOWN); return 0; }
    CloseHandle(worker);
    status("initialized", CUPTI_SUCCESS);
    return 1;
}
