#include "sage/cuda/rms_norm.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(e)); }
}
struct Allocation {
    float* data = nullptr;
    explicit Allocation(std::size_t floats) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), floats * sizeof(float)));
    }
    ~Allocation() { if (data) { cudaFree(data); } }
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};

double run_case(int n, int rows, float eps, int pattern, bool identity, bool default_stream) {
    const int stride = n + 7;
    constexpr float sentinel = 123456.0f;
    std::vector<float> x(static_cast<std::size_t>(n) * rows), w(n);
    std::vector<float> y(static_cast<std::size_t>(stride) * rows, sentinel);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float base = static_cast<float>(static_cast<int>((i * 17) % 101) - 50) / 13.0f;
        x[i] = pattern == 0 ? 0.0f : pattern == 1 ? base * 1e-8f :
               pattern == 2 ? base * 1e10f : base;
    }
    for (int i = 0; i < n; ++i) { w[i] = static_cast<float>((i % 23) - 11) / 7.0f; }
    Allocation dx(x.size()), dw(w.size()), dy(y.size());
    Stream owned;
    cudaStream_t stream = default_stream ? nullptr : owned.value;
    check(cudaMemcpyAsync(dx.data, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(dw.data, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(dy.data, y.data(), y.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    if (!sage::cuda::try_rms_norm_f32(dx.data, identity ? nullptr : dw.data, dy.data,
                                     n, rows, eps, stream, stride)) {
        throw std::runtime_error("valid RMSNorm call rejected");
    }
    check(cudaMemcpyAsync(y.data(), dy.data, y.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    double max_error = 0;
    for (int row = 0; row < rows; ++row) {
        // Independent mathematical oracle: complete dot sum and square root in
        // long double from the represented FP32 inputs, with no GPU reduction tree.
        long double sum = 0;
        for (int i = 0; i < n; ++i) {
            const long double value = x[static_cast<std::size_t>(row) * n + i];
            sum += value * value;
        }
        const long double inv = 1 / std::sqrt(sum / n + static_cast<long double>(eps));
        for (int i = 0; i < stride; ++i) {
            const float actual = y[static_cast<std::size_t>(row) * stride + i];
            if (i >= n) {
                if (actual != sentinel) { throw std::runtime_error("output padding overwritten"); }
                continue;
            }
            const long double expected = x[static_cast<std::size_t>(row) * n + i] * inv *
                                         (identity ? 1.0L : static_cast<long double>(w[i]));
            const double error = static_cast<double>(std::abs(actual - expected));
            max_error = std::max(max_error, error);
            if (!std::isfinite(actual) || error > 3e-5 + 3e-5 * std::abs(expected)) {
                throw std::runtime_error("oracle mismatch n=" + std::to_string(n) +
                                         " row=" + std::to_string(row));
            }
        }
    }
    // Rejected calls must preserve destination and perform no launch.
    const auto reject = [&](const float* xp, const float* wp, float* yp,
                            int width, int count, float e, int ys) {
        if (sage::cuda::try_rms_norm_f32(xp, wp, yp, width, count, e, stream, ys)) {
            throw std::runtime_error("invalid RMSNorm call accepted");
        }
    };
    reject(nullptr, dw.data, dy.data, n, rows, eps, stride);
    reject(dx.data, dw.data, dy.data, 0, rows, eps, stride);
    reject(dx.data, dw.data, dy.data, n, 0, eps, stride);
    reject(dx.data, dw.data, dy.data, n, rows, 0, stride);
    reject(dx.data, dw.data, dy.data, n, rows, std::numeric_limits<float>::quiet_NaN(), stride);
    reject(dx.data, dw.data, dy.data, n, rows, std::numeric_limits<float>::infinity(), stride);
    reject(dx.data, dw.data, dy.data, n, rows, eps, n - 1);
    reject(dx.data, dw.data, dx.data, n, rows, eps, n);
    reject(dx.data, dy.data, dy.data, n, rows, eps, stride);
    std::vector<float> after(y.size());
    check(cudaMemcpyAsync(after.data(), dy.data, after.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    if (after != y) { throw std::runtime_error("rejected call changed output"); }
    return max_error;
}
} // namespace

int main() {
    try {
        int count = 0;
        check(cudaGetDeviceCount(&count));
        if (count == 0) { throw std::runtime_error("no CUDA GPU: qualification cannot run"); }
        int device = 0, driver = 0, runtime = 0;
        check(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device));
        check(cudaDriverGetVersion(&driver));
        check(cudaRuntimeGetVersion(&runtime));
        std::cout << "device=" << properties.name << " sm=" << properties.major << properties.minor
                  << " total_vram_bytes=" << properties.totalGlobalMem
                  << " driver=" << driver << " runtime=" << runtime << '\n';
        if (properties.major != 7 || properties.minor != 5) {
            throw std::runtime_error("SM75 hardware required for this qualification executable");
        }
        int cases = 0;
        double max_error = 0;
        for (int n : {1, 7, 31, 128, 255, 256, 257, 2048, 4097}) {
            for (int rows : {1, 3, 17}) {
                for (float eps : {1e-6f, 1e-5f, 1e-2f}) {
                    for (int pattern = 0; pattern != 4; ++pattern) {
                        max_error = std::max(max_error, run_case(n, rows, eps, pattern,
                                                                (cases % 2) == 0, (cases % 7) == 0));
                        ++cases;
                    }
                }
            }
        }
        std::cout << "PASS rmsnorm_cases=" << cases << " max_abs_error=" << max_error
                  << " criterion=3e-5+3e-5*abs(oracle)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
