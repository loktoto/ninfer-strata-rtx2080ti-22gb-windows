#pragma once

namespace sage::cuda {

// Enqueue FP32 RMSNorm: y[r*y_stride+i] = x[r*n+i] *
// rsqrt(sum_j(x[r*n+j]^2)/n + eps) * (w ? w[i] : 1).
// x is compact; y_stride is in floats and must be >= n. Null w means identity.
// x/y must be non-null, float-aligned device-accessible pointers; y must not
// overlap x or w. The caller owns all storage until stream completion.
// A null stream denotes CUDA's default stream. No allocation or synchronization.
// False means rejected before any kernel was enqueued; CUDA launch errors throw
// std::runtime_error. Successful enqueue does not establish asynchronous success.
bool try_rms_norm_f32(const float* x, const float* w, float* y, int n, int rows,
                      float eps, void* stream, int y_stride);

} // namespace sage::cuda
