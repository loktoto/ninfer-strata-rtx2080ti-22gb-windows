#include "sage/cuda/q4k_slots.hpp"
#ifndef SAGE_Q4K_CPU_ORACLE_ONLY
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Bytes = std::vector<std::uint8_t>;

void require(bool ok, const char* message) {
    if (!ok) { throw std::runtime_error(message); }
}

// IEEE half decoding in long double, independently of CUDA's half conversion.
long double half(const std::uint8_t* p) {
    const unsigned bits = unsigned(p[0]) | (unsigned(p[1]) << 8);
    const unsigned exponent = (bits >> 10) & 31;
    const unsigned mantissa = bits & 1023;
    require(exponent != 31, "nonfinite test half");
    const long double magnitude = exponent == 0 ? std::ldexp(static_cast<long double>(mantissa), -24) :
        std::ldexp(static_cast<long double>(1024 + mantissa), static_cast<int>(exponent) - 25);
    return bits & 32768 ? -magnitude : magnitude;
}

void put_half(std::uint8_t* p, unsigned bits) {
    p[0] = static_cast<std::uint8_t>(bits);
    p[1] = static_cast<std::uint8_t>(bits >> 8);
}

// Encode the test fixture from logical scale/min values. The independent oracle
// below reads individual bits through a bit-location table, not GPU expressions.
void pack_scales(std::uint8_t* p, const std::array<int, 8>& scales,
                 const std::array<int, 8>& minima) {
    std::fill(p, p + 12, 0);
    for (int g = 0; g < 4; ++g) {
        p[g] = static_cast<std::uint8_t>(scales[g] | ((scales[g + 4] >> 4) << 6));
        p[g + 4] = static_cast<std::uint8_t>(minima[g] | ((minima[g + 4] >> 4) << 6));
        p[g + 8] = static_cast<std::uint8_t>((scales[g + 4] & 15) | ((minima[g + 4] & 15) << 4));
    }
}

unsigned unpack_six(const std::uint8_t* p, int group, bool minimum) {
    static constexpr int scale_bits[8][6] = {
        {0,1,2,3,4,5}, {8,9,10,11,12,13}, {16,17,18,19,20,21}, {24,25,26,27,28,29},
        {64,65,66,67,6,7}, {72,73,74,75,14,15}, {80,81,82,83,22,23}, {88,89,90,91,30,31}
    };
    static constexpr int min_bits[8][6] = {
        {32,33,34,35,36,37}, {40,41,42,43,44,45}, {48,49,50,51,52,53}, {56,57,58,59,60,61},
        {68,69,70,71,38,39}, {76,77,78,79,46,47}, {84,85,86,87,54,55}, {92,93,94,95,62,63}
    };
    unsigned value = 0;
    for (int bit = 0; bit < 6; ++bit) {
        const int location = minimum ? min_bits[group][bit] : scale_bits[group][bit];
        value |= ((p[location / 8] >> (location % 8)) & 1u) << bit;
    }
    return value;
}

void oracle_self_test() {
    for (int base = 0; base < 64; ++base) {
        std::array<int, 8> scales{}, minima{};
        std::uint8_t packed[12];
        for (int g = 0; g < 8; ++g) {
            scales[g] = (base + 9 * g) % 64;
            minima[g] = (63 - base + 13 * g) % 64;
        }
        pack_scales(packed, scales, minima);
        for (int g = 0; g < 8; ++g) {
            require(unpack_six(packed, g, false) == unsigned(scales[g]), "scale fixture round trip");
            require(unpack_six(packed, g, true) == unsigned(minima[g]), "minimum fixture round trip");
        }
    }
    const std::uint8_t one[2] = {0, 60}, subnormal[2] = {1, 0}, negative[2] = {0, 188};
    require(half(one) == 1 && half(negative) == -1 && half(subnormal) == std::ldexp(1.0L, -24),
            "independent half decoder");
}

struct Fixture {
    Bytes w, x;
    Fixture(int n_in, int n_out, int slot, int pattern)
        : w(std::size_t(n_in / 256) * n_out * 144), x(std::size_t(n_in / 32) * 36) {
        const int blocks = n_in / 256;
        static constexpr int boundary[8] = {0, 1, 15, 16, 31, 32, 62, 63};
        for (int row = 0; row < n_out; ++row) {
            for (int b = 0; b < blocks; ++b) {
                auto* p = w.data() + (std::size_t(row) * blocks + b) * 144;
                put_half(p, 0x2800 + ((slot + b + row) % 7) * 0x100);
                put_half(p + 2, pattern == 0 ? 0 : 0x2c00 + ((slot + row) % 5) * 0x100);
                std::array<int, 8> scales{}, minima{};
                for (int g = 0; g < 8; ++g) {
                    scales[g] = boundary[(g + slot + b) % 8];
                    minima[g] = boundary[(7 - g + row + slot) % 8];
                }
                pack_scales(p + 4, scales, minima);
                for (int i = 0; i < 128; ++i) {
                    const int lo = (i * 3 + row + slot * 5 + b) % 16;
                    const int hi = (i * 7 + row * 3 + slot + b) % 16;
                    p[16 + i] = pattern == 1 ? 0 : static_cast<std::uint8_t>(lo | (hi << 4));
                }
            }
        }
        for (int b = 0; b < n_in / 32; ++b) {
            auto* p = x.data() + std::size_t(b) * 36;
            put_half(p, 0x2400 + ((b + slot) % 5) * 0x100);
            // Deliberately unrelated to sum(qs)*d: Q4_K must ignore this field.
            put_half(p + 2, 0x7000 + slot);
            for (int i = 0; i < 32; ++i) {
                const int value = pattern == 1 ? 23 + slot : ((i * 17 + b * 29 + slot * 11) % 256) - 128;
                p[4 + i] = static_cast<std::uint8_t>(value & 255);
            }
        }
    }
};

struct Reference { long double value = 0, magnitude = 0; };
Reference oracle(const Fixture& f, int n_in, int row) {
    Reference result;
    for (int column = 0; column < n_in; ++column) {
        const auto* w = f.w.data() + (std::size_t(row) * (n_in / 256) + column / 256) * 144;
        const int within = column % 256;
        const int group = within / 32;
        const int packed_index = (within / 64) * 32 + within % 32;
        const unsigned nibble = (w[16 + packed_index] >> ((within % 64 >= 32) ? 4 : 0)) & 15;
        const long double weight = half(w) * unpack_six(w + 4, group, false) * nibble -
                                   half(w + 2) * unpack_six(w + 4, group, true);
        const auto* x = f.x.data() + std::size_t(column / 32) * 36;
        const int encoded = x[4 + column % 32];
        const long double activation = half(x) * (encoded < 128 ? encoded : encoded - 256);
        const long double product = weight * activation;
        result.value += product;
        result.magnitude += std::abs(product);
    }
    return result;
}

#ifndef SAGE_Q4K_CPU_ORACLE_ONLY
void check(cudaError_t status) {
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}
struct Allocation {
    void* pointer = nullptr;
    explicit Allocation(std::size_t bytes) { check(cudaMalloc(&pointer, bytes)); }
    ~Allocation() { cudaFree(pointer); }
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
};
struct Stream {
    cudaStream_t stream = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(stream); }
};

double run_case(int n_in, int n_out, int count, int pattern, bool capture, bool default_stream) {
    constexpr float sentinel = 123456.0f;
    Stream owned;
    const cudaStream_t stream = default_stream ? nullptr : owned.stream;
    sage::cuda::Q4KSlots slots{};
    std::array<std::unique_ptr<Allocation>, 8> dw, dx, dy;
    std::vector<Fixture> fixtures;
    std::array<std::vector<float>, 8> outputs;
    for (int j = 0; j < count; ++j) {
        fixtures.emplace_back(n_in, n_out, j, pattern);
        const Fixture& f = fixtures.back();
        dw[j] = std::make_unique<Allocation>(f.w.size());
        dx[j] = std::make_unique<Allocation>(f.x.size());
        dy[j] = std::make_unique<Allocation>(std::size_t(n_out + 2) * sizeof(float));
        outputs[j].assign(std::size_t(n_out) + 2, sentinel);
        slots.w[j] = dw[j]->pointer;
        slots.x[j] = dx[j]->pointer;
        slots.y[j] = static_cast<float*>(dy[j]->pointer) + 1;
        check(cudaMemcpyAsync(dw[j]->pointer, f.w.data(), f.w.size(), cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(dx[j]->pointer, f.x.data(), f.x.size(), cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(dy[j]->pointer, outputs[j].data(), outputs[j].size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
    }
    // Graph capture verifies that the caller's nondefault stream receives exactly
    // the one kernel and that the API performs no hidden allocation/copy/wait.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (capture) {
        check(cudaStreamSynchronize(stream));
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    }
    require(sage::cuda::try_q4k_slots_q8_1(slots, count, n_in, n_out, stream), "valid slots rejected");
    if (capture) {
        check(cudaStreamEndCapture(stream, &graph));
        std::size_t nodes = 0;
        check(cudaGraphGetNodes(graph, nullptr, &nodes));
        require(nodes == 1, "caller stream capture must contain one kernel");
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        check(cudaGraphLaunch(executable, stream));
    }
    for (int j = 0; j < count; ++j) {
        check(cudaMemcpyAsync(outputs[j].data(), dy[j]->pointer, outputs[j].size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
    }
    check(cudaStreamSynchronize(stream));
    if (executable) { check(cudaGraphExecDestroy(executable)); }
    if (graph) { check(cudaGraphDestroy(graph)); }
    double maximum_error = 0;
    for (int j = 0; j < count; ++j) {
        require(outputs[j].front() == sentinel && outputs[j].back() == sentinel, "output guard changed");
        for (int row = 0; row < n_out; ++row) {
            const Reference expected = oracle(fixtures[j], n_in, row);
            const float actual = outputs[j][std::size_t(row) + 1];
            const long double error = std::abs(static_cast<long double>(actual) - expected.value);
            maximum_error = std::max(maximum_error, static_cast<double>(error));
            if (!std::isfinite(actual) || error > 2e-5L + 3e-6L * expected.magnitude) {
                throw std::runtime_error("Q4_K oracle mismatch slot=" + std::to_string(j) +
                    " n_in=" + std::to_string(n_in) + " n_out=" + std::to_string(n_out) +
                    " row=" + std::to_string(row));
            }
        }
    }
    // All these calls reject synchronously and must enqueue no kernel.
    check(cudaStreamBeginCapture(owned.stream, cudaStreamCaptureModeGlobal));
    const auto reject = [&](const sage::cuda::Q4KSlots& s, int c, int ni, int no) {
        require(!sage::cuda::try_q4k_slots_q8_1(s, c, ni, no, owned.stream), "invalid slots accepted");
    };
    reject(slots, 0, n_in, n_out); reject(slots, 9, n_in, n_out);
    reject(slots, count, 0, n_out); reject(slots, count, n_in - 1, n_out);
    reject(slots, count, n_in, 0);
    auto bad = slots; bad.w[count - 1] = nullptr; reject(bad, count, n_in, n_out);
    bad = slots; bad.x[0] = nullptr; reject(bad, count, n_in, n_out);
    bad = slots; bad.y[0] = nullptr; reject(bad, count, n_in, n_out);
    bad = slots; bad.w[0] = static_cast<const std::uint8_t*>(slots.w[0]) + 1;
    reject(bad, count, n_in, n_out);
    cudaGraph_t empty = nullptr;
    check(cudaStreamEndCapture(owned.stream, &empty));
    std::size_t nodes = 0;
    check(cudaGraphGetNodes(empty, nullptr, &nodes));
    require(nodes == 0, "rejected call enqueued work");
    check(cudaGraphDestroy(empty));
    return maximum_error;
}
#endif
} // namespace

int main() {
    try {
        oracle_self_test();
#ifdef SAGE_Q4K_CPU_ORACLE_ONLY
        // This mode checks test data/oracle construction only; it does not claim
        // CUDA compilation, execution, parity or SM75 qualification.
        for (int pattern = 0; pattern < 3; ++pattern) {
            const Fixture f(768, 7, 7, pattern);
            for (int row = 0; row < 7; ++row) {
                const Reference r = oracle(f, 768, row);
                require(std::isfinite(r.value) && r.magnitude > 0, "invalid CPU fixture");
                if (pattern == 1) { require(r.value < 0, "minimum-only fixture must be negative"); }
            }
        }
        std::cout << "PASS CPU fixture/oracle self-check only; GPU parity untested\n";
#else
        int devices = 0;
        check(cudaGetDeviceCount(&devices));
        require(devices > 0, "no CUDA GPU: qualification cannot run");
        int device = 0;
        check(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, device));
        std::cout << "device=" << prop.name << " sm=" << prop.major << prop.minor << '\n';
        require(prop.major == 7 && prop.minor == 5, "SM75 hardware required for qualification");
        int cases = 0;
        double maximum_error = 0;
        for (int ni : {256, 768, 2048, 6144}) {
            for (int no : {1, 7, 33}) {
                for (int count : {1, 3, 8}) {
                    for (int pattern = 0; pattern < 3; ++pattern) {
                        maximum_error = std::max(maximum_error, run_case(ni, no, count, pattern,
                            ni == 768 && no == 7 && count == 8, cases % 17 == 0));
                        ++cases;
                    }
                }
            }
        }
        std::cout << "PASS q4k_slots_cases=" << cases << " max_abs_error=" << maximum_error
                  << " criterion=2e-5+3e-6*sum_abs_products\n";
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
