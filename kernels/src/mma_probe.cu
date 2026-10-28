// SPDX-License-Identifier: Apache-2.0
// The mma_f16_m16n8k8 primitive below is copied from ninfer,
// src/ops/common/mma.cuh, commit de849e (Apache-2.0).
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
// Turing-native FP16 MMA: 16x8x8 (ninfer, commit de849e).
__device__ __forceinline__ void mma_f16_m16n8k8(float& c0, float& c1, float& c2, float& c3,
                                                unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}
#endif

constexpr int kThreads = 128;
constexpr int kIterations = 512;
constexpr int kAccumulators = 4;
constexpr int kWarmups = 2;
constexpr int kRepetitions = 7;

__global__ void mma_probe(float* output) {
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
    // Two packed FP16 ones per register; every logical A and B entry is 1.
    constexpr unsigned one = 0x3c003c00u;
    float c[4][4];
#pragma unroll
    for (int j = 0; j < kAccumulators; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k) c[j][k] = static_cast<float>(j);
    // Distinct initial C values prevent identical accumulator chains from being
    // common-subexpression candidates in the device assembler.
    for (int i = 0; i < kIterations; ++i) {
#pragma unroll
        for (int j = 0; j < kAccumulators; ++j)
            mma_f16_m16n8k8(c[j][0], c[j][1], c[j][2], c[j][3], one, one, one);
    }
#pragma unroll
    for (int j = 0; j < kAccumulators; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k)
            output[thread * 16 + j * 4 + k] = c[j][k];
#else
    // Prevent a different compiled architecture from silently passing validation.
    for (int i = 0; i < 16; ++i) output[thread * 16 + i] = -1.0f;
#endif
}

static std::string json_string(const std::string& s) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : s) {
        switch (c) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20 || c >= 0x80) {
                out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<unsigned>(c) << std::dec;
            } else out << static_cast<char>(c);
        }
    }
    out << '"';
    return out.str();
}

static std::string uuid_string(const cudaUUID_t& uuid) {
    std::ostringstream out;
    out << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out << '-';
        out << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return out.str();
}

int main(int argc, char** argv) {
    try {
        std::string label = "unclassified";
        if (argc == 3 && std::strcmp(argv[1], "--label") == 0 &&
            (std::strcmp(argv[2], "locked") == 0 || std::strcmp(argv[2], "unlocked") == 0))
            label = argv[2];
        else if (argc != 1)
            throw std::runtime_error("usage: sage_mma_probe [--label locked|unlocked]");

        int count = 0;
        check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (!count) throw std::runtime_error("no CUDA device available");
        int device = 0;
        check(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
        if (prop.major != 7 || prop.minor != 5)
            throw std::runtime_error("this probe requires a compute capability 7.5 device");
        int driver = 0, runtime = 0, clock_rate = 0;
        check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
        check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
        check(cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device), "cudaDeviceGetAttribute clock rate");

        const int blocks = std::min(256, std::max(1, prop.multiProcessorCount * 4));
        const std::size_t elements = static_cast<std::size_t>(blocks) * kThreads * 16;
        std::vector<float> host(elements);
        float* device_output = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&device_output), elements * sizeof(float)), "cudaMalloc");
        cudaEvent_t start{}, stop{};
        check(cudaEventCreate(&start), "cudaEventCreate start");
        check(cudaEventCreate(&stop), "cudaEventCreate stop");
        std::vector<float> times;
        for (int run = -kWarmups; run < kRepetitions; ++run) {
            check(cudaEventRecord(start), "cudaEventRecord start");
            mma_probe<<<blocks, kThreads>>>(device_output);
            check(cudaGetLastError(), "mma_probe launch");
            check(cudaEventRecord(stop), "cudaEventRecord stop");
            check(cudaEventSynchronize(stop), "cudaEventSynchronize");
            float ms = 0;
            check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
            if (ms <= 0) throw std::runtime_error("invalid event timing");
            check(cudaMemcpy(host.data(), device_output, elements * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
            for (std::size_t i = 0; i < elements; ++i) {
                const int expected = 8 * kIterations + static_cast<int>((i % 16) / 4);
                if (host[i] != static_cast<float>(expected)) {
                    std::ostringstream error;
                    error << "validation failed at output " << i << ": got " << host[i]
                          << ", expected " << expected;
                    throw std::runtime_error(error.str());
                }
            }
            if (run >= 0) times.push_back(ms);
        }
        check(cudaEventDestroy(start), "cudaEventDestroy start");
        check(cudaEventDestroy(stop), "cudaEventDestroy stop");
        check(cudaFree(device_output), "cudaFree");
        std::sort(times.begin(), times.end());
        const std::uint64_t warps = static_cast<std::uint64_t>(blocks) * kThreads / 32;
        const std::uint64_t flops = warps * kIterations * kAccumulators * 2048ULL;
        const double median = times[times.size() / 2];
        std::cout << std::setprecision(9)
                  << "{\"probe\":\"sm75_fp16_m16n8k8\",\"label\":" << json_string(label)
                  << ",\"label_is_user_supplied\":" << (argc == 3 ? "true" : "false")
                  << ",\"device_name\":" << json_string(prop.name)
                  << ",\"device_uuid\":" << json_string(uuid_string(prop.uuid))
                  << ",\"total_memory_bytes\":" << prop.totalGlobalMem
                  << ",\"compute_capability\":\"7.5\",\"driver_version\":" << driver
                  << ",\"runtime_version\":" << runtime
                  << ",\"nominal_clock_rate_khz\":" << clock_rate
                  << ",\"blocks\":" << blocks << ",\"threads_per_block\":" << kThreads
                  << ",\"warps\":" << warps << ",\"iterations\":" << kIterations
                  << ",\"independent_accumulators\":" << kAccumulators
                  << ",\"warmups\":" << kWarmups << ",\"repetitions\":" << kRepetitions
                  << ",\"validated_outputs_per_run\":" << elements
                  << ",\"expected_output_base\":" << 8 * kIterations
                  << ",\"accumulator_initial_values\":[0,1,2,3]"
                  << ",\"flops_per_run\":" << flops
                  << ",\"time_ms_min\":" << times.front() << ",\"time_ms_median\":" << median
                  << ",\"time_ms_max\":" << times.back()
                  << ",\"median_tflops\":" << (static_cast<double>(flops) / (median * 1.0e9))
                  << ",\"validation\":\"pass\"}" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sage_mma_probe: " << error.what() << '\n';
        return 1;
    }
}
