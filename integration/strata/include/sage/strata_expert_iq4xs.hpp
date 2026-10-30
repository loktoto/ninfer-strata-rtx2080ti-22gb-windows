#pragma once
#include "sage/cuda/iq4xs_slots.hpp"
#include "sage/dispatch_trace.hpp"
#include <cstdlib>
#include <cstring>

namespace sage::strata {
template <class Slots>
inline bool try_expert_iq4xs_slots(int type, const Slots& slots, int count,
                                int n_in, int n_out, void* stream) {
    const char* enabled = std::getenv("STRATA_SAGE_EXPERT_IQ4XS");
    if (!enabled || std::strcmp(enabled, "1") != 0 || type != 23 ||
        count < 1 || count > 8 || n_in <= 0 || n_in % 256 || n_out <= 0) return false;
    sage::cuda::IQ4XSSlots borrowed{};
    for (int j = 0; j < count; ++j) {
        borrowed.w[j] = slots.w[j]; borrowed.x[j] = slots.x[j]; borrowed.y[j] = slots.y[j];
    }
    if (!sage::cuda::try_iq4xs_slots_q8_1(borrowed, count, n_in, n_out, stream)) return false;
    trace_dispatch(DispatchTrace::expert_iq4xs);
    return true;
}
template<class Ids>
inline bool try_expert_iq4xs_ids(int type, const Ids& ids, int count, int n_in, int n_out, void* stream, int layer) {
    const char* enabled=std::getenv("STRATA_SAGE_EXPERT_IQ4XS");
    if (!enabled || std::strcmp(enabled,"1") != 0 || type != 23) return false;
    sage::cuda::IQ4XSIds borrowed{ids.table,ids.ids,ids.offset,ids.x,ids.x_stride,ids.x_every,ids.y,ids.y_stride,layer};
    return sage::cuda::try_iq4xs_ids_q8_1(borrowed,count,n_in,n_out,stream);
}
} // namespace sage::strata

