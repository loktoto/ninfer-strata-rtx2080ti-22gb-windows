// SPDX-License-Identifier: Apache-2.0
// Derived from NInfer by Neroued and its loktoto SM75 fork; see ../README.md.
// MODIFIED: FP32 storage, optional identity gain, strided output, external-stream
// API, and host validation replace NInfer's BF16 tensor/epilogue interface.
#include "sage/cuda/rms_norm.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sage::cuda {
namespace {

__launch_bounds__(256) __global__ void rms_norm_f32_kernel(
    const float* x, const float* w, float* y, int n, int rows, float eps, int y_stride) {
    const std::int64_t row = blockIdx.x;
    if (row >= rows) { return; }
    const std::int64_t base = row * n;
    float sum = 0.0f;
    for (std::int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[base + i];
        sum += v * v;
    }
    __shared__ float scratch[256];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { scratch[threadIdx.x] += scratch[threadIdx.x + stride]; }
        __syncthreads();
    }
    const float inv = rsqrtf(scratch[0] / static_cast<float>(n) + eps);
    for (std::int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        y[row * y_stride + i] = x[base + i] * inv * (w ? w[i] : 1.0f);
    }
}

bool valid_range(const void* ptr, std::uint64_t count, std::uintptr_t& end) {
    const auto begin = reinterpret_cast<std::uintptr_t>(ptr);
    if (!ptr || begin % alignof(float) != 0 ||
        count > (std::numeric_limits<std::uintptr_t>::max() - begin) / sizeof(float)) {
        return false;
    }
    end = begin + static_cast<std::uintptr_t>(count) * sizeof(float);
    return true;
}

bool overlaps(const void* a, std::uintptr_t a_end, const void* b, std::uintptr_t b_end) {
    return reinterpret_cast<std::uintptr_t>(a) < b_end &&
           reinterpret_cast<std::uintptr_t>(b) < a_end;
}

} // namespace

bool try_rms_norm_f32(const float* x, const float* w, float* y, int n, int rows,
                      float eps, void* stream, int y_stride) {
    if (n <= 0 || rows <= 0 || y_stride < n || !(eps > 0.0f) || !std::isfinite(eps)) {
        return false;
    }
    const auto input_count = static_cast<std::uint64_t>(rows) * n;
    const auto output_count = static_cast<std::uint64_t>(rows - 1) * y_stride + n;
    std::uintptr_t x_end = 0, w_end = 0, y_end = 0;
    if (!valid_range(x, input_count, x_end) || !valid_range(y, output_count, y_end) ||
        overlaps(x, x_end, y, y_end) ||
        (w && (!valid_range(w, n, w_end) || overlaps(w, w_end, y, y_end)))) {
        return false;
    }
    rms_norm_f32_kernel<<<static_cast<unsigned>(rows), 256, 0,
                          reinterpret_cast<cudaStream_t>(stream)>>>(
        x, w, y, n, rows, eps, y_stride);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("sage RMSNorm launch: ") + cudaGetErrorString(status));
    }
    return true;
}

} // namespace sage::cuda
