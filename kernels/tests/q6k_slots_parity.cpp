#include "sage/cuda/q6k_slots.hpp"
#ifndef SAGE_Q6K_CPU_ORACLE_ONLY
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
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
long double half(const std::uint8_t* p) {
    unsigned bits = unsigned(p[0]) | (unsigned(p[1]) << 8);
    unsigned exp = (bits >> 10) & 31, mantissa = bits & 1023;
    require(exp != 31, "nonfinite test half");
    long double m = exp == 0 ? std::ldexp(static_cast<long double>(mantissa), -24) :
        std::ldexp(static_cast<long double>(1024 + mantissa), int(exp) - 25);
    return bits & 32768 ? -m : m;
}
void put_half(std::uint8_t* p, unsigned bits) { p[0] = bits & 255; p[1] = bits >> 8; }

// Encode each logical value by bit placement. The oracle reads every bit by
// its independent absolute bit address; neither uses the device decoder.
void pack_code(std::uint8_t* p, int index, int code) {
    const int half = index / 128, chunk = (index % 128) / 32, pos = index % 32;
    const int low_byte = half * 64 + (chunk % 2) * 32 + pos;
    const int high_byte = 128 + half * 32 + pos;
    p[low_byte] |= std::uint8_t((code & 15) << (chunk >= 2 ? 4 : 0));
    p[high_byte] |= std::uint8_t((code >> 4) << (chunk * 2));
}
int read_code(const std::uint8_t* p, int index) {
    const int half = index / 128, chunk = (index / 32) % 4, pos = index % 32;
    int code = 0;
    for (int bit = 0; bit < 6; ++bit) {
        int address = bit < 4 ? (half * 64 + (chunk % 2) * 32 + pos) * 8 + (chunk / 2) * 4 + bit :
            (128 + half * 32 + pos) * 8 + chunk * 2 + bit - 4;
        code |= ((p[address / 8] >> (address % 8)) & 1) << bit;
    }
    return code - 32;
}
int signed_byte(std::uint8_t b) { return b < 128 ? int(b) : int(b) - 256; }
void oracle_self_test() {
    for (int code = 0; code < 64; ++code) {
        std::array<std::uint8_t, 210> p{};
        for (int i = 0; i < 256; ++i) pack_code(p.data(), i, (code + i * 7) % 64);
        for (int i = 0; i < 256; ++i) require(read_code(p.data(), i) == (code + i * 7) % 64 - 32, "code round trip");
    }
    const std::uint8_t one[] = {0, 60}, neg[] = {0, 188}, sub[] = {1, 0};
    require(half(one) == 1 && half(neg) == -1 && half(sub) == std::ldexp(1.0L, -24), "half decode");
}
struct Fixture {
    Bytes w, x;
    Fixture(int ni, int no, int slot, int pattern): w(std::size_t(ni / 256) * no * 210), x(std::size_t(ni / 32) * 36) {
        constexpr int scales[] = {-128, -127, -33, -1, 0, 1, 31, 127};
        for (int row = 0; row < no; ++row) for (int b = 0; b < ni / 256; ++b) {
            auto* p = w.data() + (std::size_t(row) * (ni / 256) + b) * 210;
            for (int i = 0; i < 256; ++i) {
                int code = pattern == 1 ? (i % 2 ? 63 : 0) : (i * 17 + row * 3 + slot * 11 + b) % 64;
                pack_code(p, i, code);
            }
            for (int i = 0; i < 16; ++i) p[192 + i] = std::uint8_t(scales[(i + row + slot + b) % 8] & 255);
            put_half(p + 208, 0x2801 + ((row + b + slot) % 7) * 0x100);
        }
        for (int b = 0; b < ni / 32; ++b) {
            auto* p = x.data() + std::size_t(b) * 36;
            put_half(p, 0x2401 + ((b + slot) % 5) * 0x100);
            put_half(p + 2, 0x7000 + slot); // deliberately unrelated to sum(qs)
            for (int i = 0; i < 32; ++i) p[4 + i] = std::uint8_t((i * 17 + b * 29 + slot * 11 + pattern * 7) % 256);
        }
    }
};
struct Reference { long double value = 0, magnitude = 0; };
Reference oracle(const Fixture& f, int ni, int row) {
    Reference r;
    for (int col = 0; col < ni; ++col) {
        const auto* w = f.w.data() + (std::size_t(row) * (ni / 256) + col / 256) * 210;
        const auto* x = f.x.data() + std::size_t(col / 32) * 36;
        const int index = col % 256;
        long double product = half(w + 208) * signed_byte(w[192 + index / 16]) * read_code(w, index) *
                              half(x) * signed_byte(x[4 + col % 32]);
        r.value += product; r.magnitude += std::abs(product);
    }
    return r;
}
#ifndef SAGE_Q6K_CPU_ORACLE_ONLY
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
    sage::cuda::Q6KSlots slots{};
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
    require(sage::cuda::try_q6k_slots_q8_1(slots, count, n_in, n_out, stream), "valid slots rejected");
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
                throw std::runtime_error("Q6_K oracle mismatch slot=" + std::to_string(j) +
                    " n_in=" + std::to_string(n_in) + " n_out=" + std::to_string(n_out) +
                    " row=" + std::to_string(row));
            }
        }
    }
    // All these calls reject synchronously and must enqueue no kernel.
    check(cudaStreamBeginCapture(owned.stream, cudaStreamCaptureModeGlobal));
    const auto reject = [&](const sage::cuda::Q6KSlots& s, int c, int ni, int no) {
        require(!sage::cuda::try_q6k_slots_q8_1(s, c, ni, no, owned.stream), "invalid slots accepted");
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
#ifdef SAGE_Q6K_CPU_ORACLE_ONLY
        // This mode checks test data/oracle construction only; it does not claim
        // CUDA compilation, execution, parity or SM75 qualification.
        for (int pattern = 0; pattern < 3; ++pattern) {
            const Fixture f(768, 7, 7, pattern);
            for (int row = 0; row < 7; ++row) {
                const Reference r = oracle(f, 768, row);
                require(std::isfinite(r.value) && r.magnitude > 0, "invalid CPU fixture");

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
        for (int ni : {256, 512, 768, 2048}) {
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
        maximum_error = std::max(maximum_error, run_case(512, 2048, 8, 0, true, false));
        maximum_error = std::max(maximum_error, run_case(2048, 1024, 8, 0, false, false));
        cases += 2;
        std::cout << "PASS q6k_slots_cases=" << cases << " max_abs_error=" << maximum_error
                  << " criterion=2e-5+3e-6*sum_abs_products\n";
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
