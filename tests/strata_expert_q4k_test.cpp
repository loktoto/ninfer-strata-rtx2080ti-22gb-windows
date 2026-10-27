#include "sage/strata_expert_q4k.hpp"

#include <cstdlib>
#include <stdexcept>

namespace {
int calls = 0, enqueued = 0;
bool accept = true, fail = false;
sage::cuda::Q4KSlots seen{};
int seen_count = 0, seen_in = 0, seen_out = 0;
void* seen_stream = nullptr;

void set_flag(const char* value) {
#ifdef _WIN32
    _putenv_s("STRATA_SAGE_EXPERT_Q4K", value ? value : "");
#else
    if (value) setenv("STRATA_SAGE_EXPERT_Q4K", value, 1);
    else unsetenv("STRATA_SAGE_EXPERT_Q4K");
#endif
}
} // namespace

// Mock only the enqueue boundary: this host test exercises the production adapter.
bool sage::cuda::try_q4k_slots_q8_1(const Q4KSlots& slots, int count, int n_in,
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
    const auto dispatch = [&](int type = 12, int count = 8, int n_in = 2048, int n_out = 1536) {
        return sage::strata::try_expert_q4k_slots(type, slots, count, n_in, n_out, stream);
    };
    const char* disabled[] = {nullptr, "0", "01", "true", "1 "};
    for (const char* value : disabled) {
        set_flag(value);
        if (dispatch() || calls != 0 || enqueued != 0) return 1;
    }
    set_flag("1");
    // Unsupported Q6_K and invalid metadata must reach the original dispatcher untouched.
    if (dispatch(14) || dispatch(12, 0) || dispatch(12, 9) || dispatch(12, 1, 255) ||
        dispatch(12, 1, 0) || dispatch(12, 1, 256, 0) || calls != 0 || enqueued != 0)
        return 2;
    if (!dispatch() || calls != 1 || enqueued != 1 || seen_count != 8 ||
        seen_in != 2048 || seen_out != 1536 || seen_stream != stream)
        return 3;
    for (int i = 0; i < 8; ++i)
        if (seen.w[i] != slots.w[i] || seen.x[i] != slots.x[i] || seen.y[i] != slots.y[i]) return 4;
    // One cached expert and a default stream are supported too.
    if (!sage::strata::try_expert_q4k_slots(12, slots, 1, 256, 3, nullptr) || seen_stream != nullptr)
        return 5;
    accept = false;
    if (dispatch() || enqueued != 2) return 6;
    fail = true;
    try { dispatch(); }
    catch (const std::runtime_error&) { set_flag(nullptr); return 0; }
    return 7;
}
