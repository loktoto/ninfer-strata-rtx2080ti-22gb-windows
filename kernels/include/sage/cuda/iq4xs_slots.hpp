#pragma once

namespace sage::cuda {

// Borrowed row-major GGUF IQ4_XS matrices (136 bytes per 256 weights), Q8_1
// inputs (36 bytes per 32 values), and FP32 outputs for up to eight experts.
struct IQ4XSSlots {
    const void* w[8];
    const void* x[8];
    float* y[8];
};

// count 1..8, n_in positive and divisible by 256, n_out positive. Active
// pointers must be non-null, 4-byte aligned, device accessible, and output
// ranges must not overlap inputs or each other. Inputs may alias. The caller
// owns the memory until stream completion. A null stream denotes the default.
// Returns false before enqueue on invalid shape/pointer/address overflow.
// Launch failures throw; success means enqueue only. No allocation or copies.
bool try_iq4xs_slots_q8_1(const IQ4XSSlots& slots, int count, int n_in, int n_out,
                        void* stream);
void iq4xs_slots_q8_1(const IQ4XSSlots& slots, int count, int n_in, int n_out,
                    void* stream);
} // namespace sage::cuda

