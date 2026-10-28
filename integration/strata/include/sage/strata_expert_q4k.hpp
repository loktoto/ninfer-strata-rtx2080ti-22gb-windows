#pragma once

#include "sage/cuda/q4k_slots.hpp"
#include "sage/dispatch_trace.hpp"

#include <cstdlib>
#include <cstring>

namespace sage::strata {
// Borrow the runtime's exact GGUF Q4_K slots and Q8_1 scratch on its stream.
// Returning false never enqueues work, so the original dispatcher can run.
template <class Slots>
inline bool try_expert_q4k_slots(int ggml_type, const Slots& slots, int count,
                                 int n_in, int n_out, void* stream) {
    const char* enabled = std::getenv("STRATA_SAGE_EXPERT_Q4K");
    if (!enabled || std::strcmp(enabled, "1") != 0 || ggml_type != 12 ||
        count < 1 || count > 8 || n_in <= 0 || n_in % 256 != 0 || n_out <= 0)
        return false;
    sage::cuda::Q4KSlots borrowed{};
    for (int i = 0; i < count; ++i) {
        borrowed.w[i] = slots.w[i];
        borrowed.x[i] = slots.x[i];
        borrowed.y[i] = slots.y[i];
    }
    if (!sage::cuda::try_q4k_slots_q8_1(borrowed, count, n_in, n_out, stream)) return false;
    trace_dispatch(DispatchTrace::expert_q4k);
    return true;
}
} // namespace sage::strata
