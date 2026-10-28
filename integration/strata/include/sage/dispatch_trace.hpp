#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace sage::strata {
// Reports a host dispatch enqueue, including enqueue during graph capture. It
// does not report device execution, completion, or numerical correctness.
enum class DispatchTrace { rmsnorm, expert_q4k };

inline void trace_dispatch(DispatchTrace kind) {
    const char* enabled = std::getenv("STRATA_SAGE_TRACE");
    if (!enabled || std::strcmp(enabled, "1") != 0) return;
    static std::atomic_flag rmsnorm = ATOMIC_FLAG_INIT;
    static std::atomic_flag expert_q4k = ATOMIC_FLAG_INIT;
    std::atomic_flag& once = kind == DispatchTrace::rmsnorm ? rmsnorm : expert_q4k;
    if (!once.test_and_set(std::memory_order_relaxed))
        std::fprintf(stderr, "sage-dispatch: %s\n", kind == DispatchTrace::rmsnorm ? "rmsnorm" : "expert-q4k");
}
} // namespace sage::strata
