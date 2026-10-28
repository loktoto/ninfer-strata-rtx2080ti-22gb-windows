// SPDX-License-Identifier: Apache-2.0
// Q6_K bit decoding adapted from NInfer-all ggml_bridge_vec.cuh, commit
// 71d3f8190772ef7089f8eda4e22371114fd5bbf9, Decoder<GGML_TYPE_Q6_K>.
#include "sage/cuda/q6k_slots.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sage::cuda {
namespace {
__device__ float half_at(const std::uint8_t* p) {
    return __half2float(__ushort_as_half(std::uint16_t(p[0]) | (std::uint16_t(p[1]) << 8)));
}

// Scalar form of the donor Decoder's ql/qh primitive. q is the 32-value
// slice within a 128-value half; the original decoder packs four bytes at a
// time and subtracts 32 in each byte. This operates on one byte at a time.
__device__ int donor_q6(const std::uint8_t* b, int u, int i) {
    const int h = u / 4, q = u % 4;
    const auto* ql = b + 64 * h + 32 * (q & 1);
    const auto* qh = b + 128 + 32 * h;
    const unsigned v = ((ql[i] >> (4 * (q >> 1))) & 15u) |
                       (((qh[i] >> (2 * q)) << 4) & 48u);
    return int(v) - 32;
}
__global__ void kernel(Q6KSlots slots, int n_in, int n_out) {
    const int slot = int(blockIdx.y), warp = int(threadIdx.x) / 32;
    const int lane = int(threadIdx.x) % 32, group = lane / 4, quarter = lane % 4;
    const std::size_t row = std::size_t(blockIdx.x) * 4 + warp;
    if (row >= std::size_t(n_out)) return;
    const int blocks = n_in / 256;
    const auto* w = static_cast<const std::uint8_t*>(slots.w[slot]) + row * std::size_t(blocks) * 210;
    const auto* x = static_cast<const std::uint8_t*>(slots.x[slot]);
    float sum = 0;
    for (int b = 0; b < blocks; ++b) {
        const auto* wb = w + std::size_t(b) * 210;
        const auto* xb = x + (std::size_t(b) * 8 + group) * 36;
        int dot = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int i = quarter * 8 + j;
            const int activation = int(static_cast<const std::int8_t*>(static_cast<const void*>(xb + 4))[i]);
            dot += donor_q6(wb, group, i) * activation;
        }
        dot += __shfl_down_sync(0xffffffffu, dot, 1, 4);
        const int second = __shfl_down_sync(0xffffffffu, dot, 2, 4);
        if (quarter == 0) {
            const auto* scales = reinterpret_cast<const std::int8_t*>(wb + 192);
            const float d = half_at(wb + 208);
            // Match donor's FP32 coefficient rounding, then its Q8_1 dot.
            const float a = __fmul_rn(d, float(scales[group * 2]));
            const float c = __fmul_rn(d, float(scales[group * 2 + 1]));
            const float value = __fmul_rn(half_at(xb), __fadd_rn(__fmul_rn(a, float(dot)),
                                                                   __fmul_rn(c, float(second))));
            sum += value;
        }
    }
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) slots.y[slot][row] = sum;
}
bool valid_range(const void* p, std::uint64_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}
} // namespace
bool try_q6k_slots_q8_1(const Q6KSlots& slots, int count, int n_in, int n_out, void* stream) {
    if (count < 1 || count > 8 || n_in <= 0 || n_in % 256 || n_out <= 0) return false;
    const std::uint64_t wb = std::uint64_t(n_in / 256) * std::uint64_t(n_out) * 210;
    const std::uint64_t xb = std::uint64_t(n_in / 32) * 36;
    const std::uint64_t yb = std::uint64_t(n_out) * sizeof(float);
    for (int j = 0; j < count; ++j)
        if (!valid_range(slots.w[j], wb) || !valid_range(slots.x[j], xb) || !valid_range(slots.y[j], yb)) return false;
    const dim3 grid(unsigned((std::uint64_t(n_out) + 3) / 4), unsigned(count));
    kernel<<<grid, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(slots, n_in, n_out);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(std::string("sage Q6_K slots launch: ") + cudaGetErrorString(status));
    return true;
}
void q6k_slots_q8_1(const Q6KSlots& slots, int count, int n_in, int n_out, void* stream) {
    if (!try_q6k_slots_q8_1(slots, count, n_in, n_out, stream))
        throw std::invalid_argument("sage Q6_K slots: invalid shape or pointer");
}
} // namespace sage::cuda
