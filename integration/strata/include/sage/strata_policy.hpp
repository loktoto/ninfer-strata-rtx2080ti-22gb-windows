#pragma once
#include <stdexcept>

namespace sage::strata {
// Original top-k routing remains authoritative: every selected expert executes.
inline void require_complete_routing(int tail_skip, float skip_weight) {
    if (tail_skip != 0 || skip_weight != 0.0f)
        throw std::invalid_argument("Sage requires --tail-skip 0 and --skip-weight 0; "
                                    "also set STRATA_ROUTE_TAIL_SKIP=0");
}
} // namespace sage::strata
