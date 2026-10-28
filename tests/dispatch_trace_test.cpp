#include "sage/strata_expert_q4k.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <io.h>
#define sage_dup _dup
#define sage_dup2 _dup2
#define sage_close _close
#define sage_fileno _fileno
#else
#include <unistd.h>
#define sage_dup dup
#define sage_dup2 dup2
#define sage_close close
#define sage_fileno fileno
#endif

namespace {
bool accept = false;
void env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}
}

bool sage::cuda::try_q4k_slots_q8_1(const Q4KSlots&, int, int, int, void*) { return accept; }

int main() {
    FILE* output = std::tmpfile();
    if (!output) return 1;
    const int saved = sage_dup(sage_fileno(stderr));
    if (saved < 0 || sage_dup2(sage_fileno(output), sage_fileno(stderr)) < 0) return 2;
    struct Slots { const void* w[8]{}; const void* x[8]{}; float* y[8]{}; } slots;
    env("STRATA_SAGE_TRACE", "0");
    sage::strata::trace_dispatch(sage::strata::DispatchTrace::rmsnorm);
    env("STRATA_SAGE_TRACE", "1");
    env("STRATA_SAGE_EXPERT_Q4K", "1");
    if (sage::strata::try_expert_q4k_slots(14, slots, 1, 256, 1, nullptr) ||
        sage::strata::try_expert_q4k_slots(12, slots, 1, 256, 1, nullptr)) return 3;
    std::fflush(stderr);
    if (std::ftell(output) != 0) return 4; // rejected adapter calls did not trace
    accept = true;
    if (!sage::strata::try_expert_q4k_slots(12, slots, 1, 256, 1, nullptr)) return 5;
    std::vector<std::thread> threads;
    for (int i = 0; i < 16; ++i) threads.emplace_back([] {
        sage::strata::trace_dispatch(sage::strata::DispatchTrace::rmsnorm);
        sage::strata::trace_dispatch(sage::strata::DispatchTrace::expert_q4k);
    });
    for (auto& t : threads) t.join();
    std::fflush(stderr);
    sage_dup2(saved, sage_fileno(stderr));
    sage_close(saved);
    std::rewind(output);
    char buffer[256]{};
    const size_t n = std::fread(buffer, 1, sizeof(buffer), output);
    std::fclose(output);
    env("STRATA_SAGE_TRACE", nullptr);
    env("STRATA_SAGE_EXPERT_Q4K", nullptr);
    return std::string(buffer, n) == "sage-dispatch: expert-q4k\nsage-dispatch: rmsnorm\n" ? 0 : 6;
}
