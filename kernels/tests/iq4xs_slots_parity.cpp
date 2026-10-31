#include "sage/cuda/iq4xs_slots.hpp"
#ifndef SAGE_IQ4XS_CPU_ORACLE_ONLY
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

// Independent fixture encoder and scalar code lookup, indexed by logical value.
void pack_code(std::uint8_t* p,int index,int code) {
    int u=index/32, i=index%32;
    p[8+16*u+(i%16)] |= std::uint8_t(code << (i<16?0:4));
}
int read_code(const std::uint8_t* p,int index) {
    constexpr int table[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    int u=index/32,i=index%32; return table[(p[8+16*u+i%16] >> (i<16?0:4)) & 15];
}
int signed_byte(std::uint8_t b) { return b < 128 ? int(b) : int(b) - 256; }
void oracle_self_test() {
    constexpr int table[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    for (int c=0;c<16;++c) {
        std::array<std::uint8_t,136> p{};
        for(int i=0;i<256;++i) pack_code(p.data(),i,(c+i*7)%16);
        for(int i=0;i<256;++i) require(read_code(p.data(),i)==table[(c+i*7)%16],"IQ4_XS code round trip");
    }
    const std::uint8_t one[] = {0, 60}, neg[] = {0, 188}, sub[] = {1, 0};
    require(half(one) == 1 && half(neg) == -1 && half(sub) == std::ldexp(1.0L, -24), "half decode");
}
struct Fixture {
    Bytes w, x;
    Fixture(int ni, int no, int slot, int pattern): w(std::size_t(ni / 256) * no * 136), x(std::size_t(ni / 32) * 36) {
        
        for (int row = 0; row < no; ++row) for (int b = 0; b < ni / 256; ++b) {
            auto* p = w.data() + (std::size_t(row) * (ni / 256) + b) * 136;
            for(int i=0;i<256;++i) {
                unsigned z=unsigned(i + 257*row + 65537*slot + 17*b + 12345);
                z ^= z >> 16; z *= 0x7feb352du; z ^= z >> 15; z *= 0x846ca68bu; z ^= z >> 16;
                pack_code(p,i,pattern==2 ? int(z&15) : (i*7+row*3+slot*11+b+pattern)%16);
            }
            unsigned sh=0;
            for(int u=0;u<8;++u) { int scale=(u*9+row+slot+b)%64; p[4+u/2] |= (scale&15) << (4*(u&1)); sh |= unsigned(scale>>4) << (2*u); }
            p[2]=sh&255; p[3]=sh>>8; put_half(p,0x2801+((row+b+slot)%7)*0x100);
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
        const auto* w = f.w.data() + (std::size_t(row) * (ni / 256) + col / 256) * 136;
        const auto* x = f.x.data() + std::size_t(col / 32) * 36;
        const int index = col % 256;
        int u=index/32; unsigned sh=unsigned(w[2]) | (unsigned(w[3])<<8);
        int scale=((w[4+u/2]>>(4*(u&1)))&15) | (((sh>>(2*u))&3)<<4);
        long double product = half(w) * (scale-32) * read_code(w, index) *
                              half(x) * signed_byte(x[4 + col % 32]);
        r.value += product; r.magnitude += std::abs(product);
    }
    return r;
}
#ifndef SAGE_IQ4XS_CPU_ORACLE_ONLY
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
    sage::cuda::IQ4XSSlots slots{};
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
    require(sage::cuda::try_iq4xs_slots_q8_1(slots, count, n_in, n_out, stream), "valid slots rejected");
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
                throw std::runtime_error("IQ4_XS oracle mismatch slot=" + std::to_string(j) +
                    " n_in=" + std::to_string(n_in) + " n_out=" + std::to_string(n_out) +
                    " row=" + std::to_string(row));
            }
        }
    }
    // All these calls reject synchronously and must enqueue no kernel.
    check(cudaStreamBeginCapture(owned.stream, cudaStreamCaptureModeGlobal));
    const auto reject = [&](const sage::cuda::IQ4XSSlots& s, int c, int ni, int no) {
        require(!sage::cuda::try_iq4xs_slots_q8_1(s, c, ni, no, owned.stream), "invalid slots accepted");
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
// Exercise device-side routing, nonzero weight offsets, permuted experts,
// activation/output strides and graph replay against the same scalar oracle.
double run_ids_case(int ni,int no,int count,int x_every) {
    Stream stream; auto st=stream.stream; const size_t wb=size_t(ni/256)*no*136, xb=size_t(ni/32)*36;
    const size_t x_stride=xb+16, y_stride=size_t(no)+4;
    std::vector<Fixture> f; f.reserve(count);
    std::vector<std::unique_ptr<Allocation>> dw(count); std::vector<const void*> host_table(count); std::vector<int> host_ids(count);
    for(int j=0;j<count;++j) {
        f.emplace_back(ni,no,j,2); const int idx=count-1-j; host_ids[j]=idx;
        dw[idx]=std::make_unique<Allocation>(wb+16);host_table[idx]=dw[idx]->pointer;
        check(cudaMemcpyAsync(static_cast<std::uint8_t*>(dw[idx]->pointer)+16,f[j].w.data(),wb,cudaMemcpyHostToDevice,st));
    }
    Allocation table(count*sizeof(void*)), ids(count*sizeof(int)), x(count*x_stride), y(count*y_stride*sizeof(float));
    check(cudaMemcpyAsync(table.pointer,host_table.data(),count*sizeof(void*),cudaMemcpyHostToDevice,st));
    check(cudaMemcpyAsync(ids.pointer,host_ids.data(),count*sizeof(int),cudaMemcpyHostToDevice,st));
    for(int j=0;j<count/x_every+(count%x_every!=0);++j)
        check(cudaMemcpyAsync(static_cast<std::uint8_t*>(x.pointer)+j*x_stride,f[j].x.data(),xb,cudaMemcpyHostToDevice,st));
    std::vector<float> out(count*y_stride,123456.f);
    check(cudaMemcpyAsync(y.pointer,out.data(),out.size()*sizeof(float),cudaMemcpyHostToDevice,st));
    sage::cuda::IQ4XSIds args{static_cast<const void* const*>(table.pointer),static_cast<const int*>(ids.pointer),16,
        x.pointer,x_stride,x_every,static_cast<float*>(y.pointer),y_stride,-1};
    check(cudaStreamSynchronize(st));check(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));
    require(sage::cuda::try_iq4xs_ids_q8_1(args,count,ni,no,st),"ids rejected");
    cudaGraph_t graph;cudaGraphExec_t exec;check(cudaStreamEndCapture(st,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    check(cudaGraphLaunch(exec,st));check(cudaGraphLaunch(exec,st));
    check(cudaMemcpyAsync(out.data(),y.pointer,out.size()*sizeof(float),cudaMemcpyDeviceToHost,st));check(cudaStreamSynchronize(st));
    check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
    double maxerr=0;
    for(int j=0;j<count;++j) { Fixture expected=f[j]; expected.x=f[j/x_every].x;
        for(int row=0;row<no;++row) { auto ref=oracle(expected,ni,row);float actual=out[size_t(j)*y_stride+row];
            long double err=std::abs((long double)actual-ref.value);maxerr=std::max(maxerr,(double)err);
            require(std::isfinite(actual) && err<=2e-5L+3e-6L*ref.magnitude,"ids oracle mismatch"); }
        for(size_t k=no;k<y_stride;++k)require(out[size_t(j)*y_stride+k]==123456.f,"ids output guard");
    }
    auto invalid=args; invalid.x_every=0; require(!sage::cuda::try_iq4xs_ids_q8_1(invalid,count,ni,no,st),"invalid x_every accepted");
    invalid=args; invalid.y_stride=no-1; require(!sage::cuda::try_iq4xs_ids_q8_1(invalid,count,ni,no,st),"invalid y_stride accepted");
    return maxerr;
}

#endif
} // namespace

int main() {
    try {
        oracle_self_test();
#ifdef SAGE_IQ4XS_CPU_ORACLE_ONLY
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
        for(int ni : {512,2048}) for(int count : {1,3,8,32}) for(int every : {1,2,8}) {
            maximum_error=std::max(maximum_error,run_ids_case(ni,ni==512?2048:1024,count,every)); ++cases;
        }
        std::cout << "PASS iq4xs_slots_cases=" << cases << " max_abs_error=" << maximum_error
                  << " criterion=2e-5+3e-6*sum_abs_products\n";
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

