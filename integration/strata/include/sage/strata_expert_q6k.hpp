#pragma once
#include "sage/cuda/q6k_slots.hpp"
#include "sage/dispatch_trace.hpp"
#include <cstdlib>
#include <cstring>

namespace sage::strata {
template <class Slots>
inline bool try_expert_q6k_slots(int type, const Slots& slots, int count,
                                int n_in, int n_out, void* stream) {
    const char* enabled = std::getenv("STRATA_SAGE_EXPERT_Q6K");
    if (!enabled || std::strcmp(enabled, "1") != 0 || type != 14 ||
        count < 1 || count > 8 || n_in <= 0 || n_in % 256 || n_out <= 0) return false;
    sage::cuda::Q6KSlots borrowed{};
    for (int j = 0; j < count; ++j) {
        borrowed.w[j] = slots.w[j]; borrowed.x[j] = slots.x[j]; borrowed.y[j] = slots.y[j];
    }
    if (!sage::cuda::try_q6k_slots_q8_1(borrowed, count, n_in, n_out, stream)) return false;
    trace_dispatch(DispatchTrace::expert_q6k);
    return true;
}
} // namespace sage::strata
