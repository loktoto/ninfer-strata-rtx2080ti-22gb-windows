// SPDX-License-Identifier: Apache-2.0
// IQ4_XS bit decoding adapted from NInfer-all ggml_bridge_vec.cuh, commit
// 71d3f8190772ef7089f8eda4e22371114fd5bbf9, Decoder<GGML_TYPE_IQ4_XS>.
#include "sage/cuda/iq4xs_slots.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <cstdlib>

namespace sage::cuda {
namespace {
// Adapted from NInfer-all ggml_bridge_vec.cuh Decoder<IQ4_XS> and its
// two-row warp vector schedule, commit 71d3f8190772ef7089f8eda4e22371114fd5bbf9.
// Strata supplies interleaved Q8_1 instead of NInfer's planar activations.
__device__ float half_at(const std::uint8_t* p) {
    return __half2float(__ushort_as_half(std::uint16_t(p[0]) | (std::uint16_t(p[1]) << 8)));
}
__device__ int code(int x) {
    constexpr signed char v[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    return v[x];
}
__device__ int iq4xs_trace_once = 0;
__global__ void kernel(IQ4XSSlots slots, int n_in, int n_out, bool trace) {
    if (trace && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && atomicCAS(&iq4xs_trace_once, 0, 1) == 0)
        printf("sage-device: expert-iq4xs executed n_in=%d n_out=%d\n", n_in, n_out);
    const int slot=int(blockIdx.y), warp=int(threadIdx.x)/32, lane=int(threadIdx.x)%32;
    const int row0=(int(blockIdx.x)*4+warp)*2;
    const auto* x=static_cast<const std::uint8_t*>(slots.x[slot]);
    float sum[2]={0,0};
    for(int slice=lane;slice<n_in/32;slice+=32) {
        const int u=slice%8, block=slice/8;
        const auto* xb=x+std::size_t(slice)*36;
        for(int r=0;r<2;++r) if(row0+r<n_out) {
            const auto* b=static_cast<const std::uint8_t*>(slots.w[slot])+(std::size_t(row0+r)*(n_in/256)+block)*136;
            const unsigned sh=unsigned(b[2]) | (unsigned(b[3])<<8);
            const int ls=((b[4+u/2]>>(4*(u&1)))&15) | (((sh>>(2*u))&3)<<4);
            int dot=0;
            for(int j=0;j<16;++j) {
                const unsigned q=b[8+16*u+j];
                dot+=code(q&15)*int(reinterpret_cast<const std::int8_t*>(xb+4)[j]);
                dot+=code(q>>4)*int(reinterpret_cast<const std::int8_t*>(xb+4)[j+16]);
            }
            const float factor=__fmul_rn(half_at(b),float(ls-32));
            sum[r]+=__fmul_rn(__fmul_rn(factor,half_at(xb)),float(dot));
        }
    }
    for(int o=16;o;o>>=1) for(int r=0;r<2;++r) sum[r]+=__shfl_down_sync(0xffffffffu,sum[r],o);
    if(lane==0) for(int r=0;r<2;++r) if(row0+r<n_out) slots.y[slot][row0+r]=sum[r];
}
bool valid_range(const void* p, std::uint64_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}
} // namespace
bool try_iq4xs_slots_q8_1(const IQ4XSSlots& slots, int count, int n_in, int n_out, void* stream) {
    if (count < 1 || count > 8 || n_in <= 0 || n_in % 256 || n_out <= 0) return false;
    const std::uint64_t wb = std::uint64_t(n_in / 256) * std::uint64_t(n_out) * 136;
    const std::uint64_t xb = std::uint64_t(n_in / 32) * 36;
    const std::uint64_t yb = std::uint64_t(n_out) * sizeof(float);
    for (int j = 0; j < count; ++j)
        if (!valid_range(slots.w[j], wb) || !valid_range(slots.x[j], xb) || !valid_range(slots.y[j], yb)) return false;
    const dim3 grid(unsigned((std::uint64_t(n_out) + 7) / 8), unsigned(count));
    kernel<<<grid, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(slots, n_in, n_out, std::getenv("STRATA_SAGE_TRACE") && std::string(std::getenv("STRATA_SAGE_TRACE")) == "1");
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(std::string("sage IQ4_XS slots launch: ") + cudaGetErrorString(status));
    return true;
}
void iq4xs_slots_q8_1(const IQ4XSSlots& slots, int count, int n_in, int n_out, void* stream) {
    if (!try_iq4xs_slots_q8_1(slots, count, n_in, n_out, stream))
        throw std::invalid_argument("sage IQ4_XS slots: invalid shape or pointer");
}
} // namespace sage::cuda

