#include "sage/strata_expert_iq4xs.hpp"

#include <cstdlib>
#include <stdexcept>

namespace {
int calls = 0, enqueued = 0, id_calls = 0;
sage::cuda::IQ4XSIds seen_ids{};
bool accept = true, fail = false;
sage::cuda::IQ4XSSlots seen{};
int seen_count = 0, seen_in = 0, seen_out = 0;
void* seen_stream = nullptr;

void set_flag(const char* value) {
#ifdef _WIN32
    _putenv_s("STRATA_SAGE_EXPERT_IQ4XS", value ? value : "");
#else
    if (value) setenv("STRATA_SAGE_EXPERT_IQ4XS", value, 1);
    else unsetenv("STRATA_SAGE_EXPERT_IQ4XS");
#endif
}
} // namespace

// Mock only the enqueue boundary: this host test exercises the production adapter.
bool sage::cuda::try_iq4xs_slots_q8_1(const IQ4XSSlots& slots, int count, int n_in,
                                   int n_out, void* stream) {
    ++calls;
    if (fail) throw std::runtime_error("launch failure");
    if (!accept) return false;
    seen = slots;
    seen_count = count;
    seen_in = n_in;
    seen_out = n_out;
    seen_stream = stream;
    ++enqueued;
    return true;
}

bool sage::cuda::try_iq4xs_ids_q8_1(const IQ4XSIds& ids,int,int,int,void*) {
    ++id_calls; seen_ids=ids; return true;
}

int main() {
    struct RuntimeSlots {
        const void* w[8];
        const void* x[8];
        float* y[8];
    } slots{};
    float weights[8]{}, inputs[8]{}, outputs[8]{};
    int stream_token = 0;
    void* stream = &stream_token;
    for (int i = 0; i < 8; ++i) {
        slots.w[i] = weights + i;
        slots.x[i] = inputs + i;
        slots.y[i] = outputs + i;
    }
    const auto dispatch = [&](int type = 23, int count = 8, int n_in = 2048, int n_out = 1536) {
        return sage::strata::try_expert_iq4xs_slots(type, slots, count, n_in, n_out, stream);
    };
    const char* disabled[] = {nullptr, "0", "01", "true", "1 "};
    for (const char* value : disabled) {
        set_flag(value);
        if (dispatch() || calls != 0 || enqueued != 0) return 1;
    }
    set_flag("1");
    // Unsupported Q4_K and invalid metadata must reach the original dispatcher untouched.
    if (dispatch(12) || dispatch(23, 0) || dispatch(23, 9) || dispatch(23, 1, 255) ||
        dispatch(23, 1, 0) || dispatch(23, 1, 256, 0) || calls != 0 || enqueued != 0)
        return 2;
    if (!dispatch() || calls != 1 || enqueued != 1 || seen_count != 8 ||
        seen_in != 2048 || seen_out != 1536 || seen_stream != stream)
        return 3;
    for (int i = 0; i < 8; ++i)
        if (seen.w[i] != slots.w[i] || seen.x[i] != slots.x[i] || seen.y[i] != slots.y[i]) return 4;
    // One cached expert and a default stream are supported too.
    if (!sage::strata::try_expert_iq4xs_slots(23, slots, 1, 256, 3, nullptr) || seen_stream != nullptr)
        return 5;
    struct RuntimeIds { const void* const* table; const int* ids; unsigned long long offset;
        const void* x; unsigned long long x_stride; int x_every; float* y; unsigned long long y_stride; };
    const void* table[1]={weights}; int indices[1]={0};
    RuntimeIds ids{table,indices,16,inputs,144,2,outputs,2048};
    set_flag("0");
    if(sage::strata::try_expert_iq4xs_ids(23,ids,1,2048,1024,stream,7) || id_calls) return 8;
    set_flag("1");
    if(sage::strata::try_expert_iq4xs_ids(14,ids,1,2048,1024,stream,7) || id_calls) return 9;
    if(!sage::strata::try_expert_iq4xs_ids(23,ids,1,2048,1024,stream,7) || id_calls!=1 ||
       seen_ids.table!=table || seen_ids.ids!=indices || seen_ids.offset!=16 || seen_ids.x!=inputs ||
       seen_ids.x_stride!=144 || seen_ids.x_every!=2 || seen_ids.y!=outputs || seen_ids.y_stride!=2048 || seen_ids.layer!=7) return 10;
    accept = false;
    if (dispatch() || enqueued != 2) return 6;
    fail = true;
    try { dispatch(); }
    catch (const std::runtime_error&) { set_flag(nullptr); return 0; }
    return 7;
}

