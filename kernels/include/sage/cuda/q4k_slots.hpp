#pragma once

namespace sage::cuda {

// Borrowed device pointers, identical field order to Strata's MmvqSlots.
// Each matrix is row-major GGUF Q4_K: 144 bytes per 256 weights.
// Each x is one contiguous CUDA Q8_1 column: 36 bytes per 32 values
// (FP16 d, FP16 s, then 32 signed bytes). Q4_K uses d, not s, exactly as
// Strata's Q4_K MMVQ does: the minimum correction uses d*sum(qs).
struct Q4KSlots {
    const void* w[8];
    const void* x[8];
    float* y[8];
};

// Enqueue y[j] = w[j] * x[j] for count independently selected experts.
// count is 1..8, n_in > 0 is divisible by 256, n_out > 0. All active
// pointers must be non-null, 4-byte aligned, device accessible and nonoverlapping
// with outputs; outputs must also be mutually nonoverlapping. Inputs may alias.
// Caller retains all allocations until stream completion. No allocation, copy,
// repacking, second expert bank, or synchronization. Null stream is the default.
// False rejects invalid shapes, null/misaligned pointers or address overflow
// before enqueue; allocation size, accessibility and overlap are caller contracts.
// CUDA launch errors throw std::runtime_error. Success means enqueue only.
bool try_q4k_slots_q8_1(const Q4KSlots& slots, int count, int n_in, int n_out,
                       void* stream);

// Same operation; invalid arguments throw std::invalid_argument.
void q4k_slots_q8_1(const Q4KSlots& slots, int count, int n_in, int n_out,
                   void* stream);

} // namespace sage::cuda
