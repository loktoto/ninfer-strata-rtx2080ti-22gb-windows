#include "sage/strata_policy.hpp"
#include <limits>
#include <stdexcept>

int main() {
    sage::strata::require_complete_routing(0, 0.0f);
    const auto rejected = [](int rank, float weight) {
        try { sage::strata::require_complete_routing(rank, weight); }
        catch (const std::invalid_argument&) { return true; }
        return false;
    };
    if (!rejected(1, 0) || !rejected(-1, 0) || !rejected(0, 0.01f) ||
        !rejected(0, -0.01f) || !rejected(0, std::numeric_limits<float>::quiet_NaN()))
        return 1;
    return 0;
}
