// SPDX-License-Identifier: Apache-2.0
// Original correctness-first GGUF Q4_K implementation. Uses NInfer's borrowed
// pointer / caller stream integration pattern, not NInfer's signed-Q4 format or
// kernel. Layout and mathematical operation checked against PolyStrata's
// native_mmvq.cu (pinned llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d).
#include "sage/cuda/q4k_slots.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sage::cuda {
namespace {

struct alignas(4) Q4KBlock {
    std::uint16_t d, dmin;
    std::uint8_t scales[12];
    std::uint8_t qs[128];
};
struct alignas(4) Q81Block {
    std::uint16_t d, s;
    std::int8_t qs[32];
};
static_assert(sizeof(Q4KBlock) == 144 && offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q81Block) == 36 && offsetof(Q81Block, qs) == 4);

__device__ float half_value(std::uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

// Four lanes cooperate on a group of 32 weights. Eight groups cover one GGUF
// block. Unsigned nibbles and the separately packed six-bit minima are essential:
// these bytes cannot be reinterpreted as NInfer signed Q4.
__global__ void q4k_slots_kernel(Q4KSlots slots, int n_in, int n_out) {
    const int slot = static_cast<int>(blockIdx.y);
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const std::size_t row = std::size_t(blockIdx.x) * 4 + warp;
    if (row >= static_cast<std::size_t>(n_out)) { return; }
    const int group = lane / 4;
    const int quarter = lane % 4;
    const int blocks = n_in / 256;
    const auto* w = static_cast<const Q4KBlock*>(slots.w[slot]) + row * blocks;
    const auto* x = static_cast<const Q81Block*>(slots.x[slot]);
    float sum = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const Q4KBlock& weight = w[b];
        const Q81Block& input = x[std::size_t(b) * 8 + group];
        const int scale = group < 4 ? (weight.scales[group] & 63) :
            ((weight.scales[group + 4] & 15) | ((weight.scales[group - 4] >> 6) << 4));
        const int minimum = group < 4 ? (weight.scales[group + 4] & 63) :
            ((weight.scales[group + 4] >> 4) | ((weight.scales[group] >> 6) << 4));
        int dot = 0, input_sum = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int element = quarter * 8 + j;
            const unsigned packed = weight.qs[(group / 2) * 32 + element];
            const int q = (packed >> (4 * (group % 2))) & 15;
            const int activation = static_cast<int>(input.qs[element]);
            dot += q * activation;
            input_sum += activation;
        }
        dot += __shfl_down_sync(0xffffffffu, dot, 2, 4);
        input_sum += __shfl_down_sync(0xffffffffu, input_sum, 2, 4);
        dot += __shfl_down_sync(0xffffffffu, dot, 1, 4);
        input_sum += __shfl_down_sync(0xffffffffu, input_sum, 1, 4);
        if (quarter == 0) {
            const float dx = half_value(input.d);
            sum += half_value(weight.d) * (dx * static_cast<float>(scale * dot)) -
                   half_value(weight.dmin) * (dx * static_cast<float>(minimum * input_sum));
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) { slots.y[slot][row] = sum; }
}

bool valid_range(const void* pointer, std::uint64_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return pointer && address % 4 == 0 &&
           bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}

} // namespace

bool try_q4k_slots_q8_1(const Q4KSlots& slots, int count, int n_in, int n_out,
                       void* stream) {
    if (count < 1 || count > 8 || n_in <= 0 || n_in % 256 != 0 || n_out <= 0) {
        return false;
    }
    const std::uint64_t weight_bytes = std::uint64_t(n_in / 256) * n_out * 144;
    const std::uint64_t input_bytes = std::uint64_t(n_in / 32) * 36;
    const std::uint64_t output_bytes = std::uint64_t(n_out) * sizeof(float);
    for (int j = 0; j < count; ++j) {
        if (!valid_range(slots.w[j], weight_bytes) || !valid_range(slots.x[j], input_bytes) ||
            !valid_range(slots.y[j], output_bytes)) { return false; }
    }
    const dim3 grid(static_cast<unsigned>((std::uint64_t(n_out) + 3) / 4),
                    static_cast<unsigned>(count));
    q4k_slots_kernel<<<grid, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(slots, n_in, n_out);
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("sage Q4_K slots launch: ") + cudaGetErrorString(result));
    }
    return true;
}

void q4k_slots_q8_1(const Q4KSlots& slots, int count, int n_in, int n_out, void* stream) {
    if (!try_q4k_slots_q8_1(slots, count, n_in, n_out, stream)) {
        throw std::invalid_argument("sage Q4_K slots: invalid shape or pointer");
    }
}

} // namespace sage::cuda
