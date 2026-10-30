#pragma once

namespace sage::cuda {

// Borrowed row-major GGUF IQ4_XS matrices (136 bytes per 256 weights), Q8_1
// inputs (36 bytes per 32 values), and FP32 outputs for up to eight experts.
struct IQ4XSSlots {
    const void* w[8];
    const void* x[8];
    float* y[8];
};

// Device-selected experts: table[ids[slot]] + offset, strided activations and
// output (y_stride is in floats, x_stride and offset in bytes). Caller owns
// device table/ids and ensures IDs address valid entries;
// K is positive, divisible by 256, and at most 2048. False enqueues nothing.
struct IQ4XSIds {
    const void* const* table;
    const int* ids;
    unsigned long long offset;
    const void* x;
    unsigned long long x_stride;
    int x_every;
    float* y;
    unsigned long long y_stride;
    int layer; // diagnostic counter index; -1 disables per-layer accounting
};
void report_iq4xs_counts();
bool try_iq4xs_ids_q8_1(const IQ4XSIds& ids, int count, int n_in, int n_out, void* stream);

// count 1..8, n_in positive, divisible by 256, and at most 2048; n_out positive. Active
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

