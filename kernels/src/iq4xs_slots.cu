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
// NInfer-all's IQ4_XS decoder expands four bytes of packed nibbles into
// two signed byte vectors; __dp4a consumes each against four Q8_1 values.
// The byte-permutation lookup is the pinned ggml/NInfer table operation.
__device__ __forceinline__ int2 decode4(unsigned q) {
    constexpr unsigned t0=0xbfad9881u,t1=0xf6eaddcfu,t2=0x26190d01u,t3=0x71594535u;
    const unsigned sel=0x32103210u | ((q & 0x88888888u)>>1);
    unsigned t[2];
#pragma unroll
    for(int i=0;i<2;++i) {
        const int shift=16*i;
        t[i]=__byte_perm(__byte_perm(t0,t1,q>>shift),__byte_perm(t2,t3,q>>shift),sel>>shift);
    }
    return make_int2(__byte_perm(t[0],t[1],0x6420),__byte_perm(t[0],t[1],0x7531));
}
__device__ int iq4xs_trace_once = 0;
__device__ int iq4xs_ids_trace_once = 0;
__device__ unsigned long long iq4xs_ids_counts[256][2];
template<bool DeviceIds>
__global__ void kernel(IQ4XSSlots slots, IQ4XSIds ids, int n_in, int n_out, bool trace) {
    if (trace && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 &&
        atomicCAS(DeviceIds ? &iq4xs_ids_trace_once : &iq4xs_trace_once, 0, 1) == 0)
        printf("sage-device: expert-iq4xs-%s executed n_in=%d n_out=%d\n", DeviceIds ? "ids" : "slots", n_in, n_out);
    if constexpr (DeviceIds) {
        if(trace && blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && ids.layer>=0 && ids.layer<256)
            atomicAdd(&iq4xs_ids_counts[ids.layer][n_in==2048 ? 0 : 1],1ULL);
    }
    const int slot=int(blockIdx.y), warp=int(threadIdx.x)/32, lane=int(threadIdx.x)%32;
    const auto* w=DeviceIds ? static_cast<const std::uint8_t*>(ids.table[ids.ids[slot]])+ids.offset : static_cast<const std::uint8_t*>(slots.w[slot]);
    const auto* x=DeviceIds ? static_cast<const std::uint8_t*>(ids.x)+std::size_t(slot/ids.x_every)*ids.x_stride : static_cast<const std::uint8_t*>(slots.x[slot]);
    float* out=DeviceIds ? ids.y+std::size_t(slot)*ids.y_stride : slots.y[slot];
    // Each lane accumulates the same virtual-warp slices in the same order
    // as native's lane-wise partial-warp addition, then uses its reduction.
    const int row0=(int(blockIdx.x)*4+warp)*2;
    float sum[2]={0,0};
    for(int slice=lane;slice<n_in/32;slice+=32) {
        const int u=slice%8, block=slice/8;
        const auto* xb=x+std::size_t(slice)*36;
        const int* xa=reinterpret_cast<const int*>(xb+4);
        for(int r=0;r<2;++r) if(row0+r<n_out) {
            const auto* b=w+(std::size_t(row0+r)*(n_in/256)+block)*136;
            const unsigned sh=unsigned(b[2]) | (unsigned(b[3])<<8);
            const int ls=((b[4+u/2]>>(4*(u&1)))&15) | (((sh>>(2*u))&3)<<4);
            const int* q=reinterpret_cast<const int*>(b+8+16*u);
            int dot=0;
#pragma unroll
            for(int j=0;j<4;++j) { const int2 v=decode4(unsigned(q[j])); dot=__dp4a(v.x,xa[j],dot); dot=__dp4a(v.y,xa[j+4],dot); }
            dot*=ls-32;
            const float d=__fmul_rn(half_at(b),half_at(xb));
            sum[r]=__fadd_rn(sum[r],__fmul_rn(d,float(dot)));
        }
    }
    for(int o=16;o;o>>=1) for(int r=0;r<2;++r) sum[r]+=__shfl_down_sync(0xffffffffu,sum[r],o);
    if(lane==0) for(int r=0;r<2;++r) if(row0+r<n_out) out[row0+r]=sum[r];
}
bool valid_range(const void* p, std::uint64_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= std::numeric_limits<std::uintptr_t>::max() - address;
}
} // namespace
bool try_iq4xs_slots_q8_1(const IQ4XSSlots& slots, int count, int n_in, int n_out, void* stream) {
    if (count < 1 || count > 8 || n_in <= 0 || n_in > 2048 || n_in % 256 || n_out <= 0) return false;
    const std::uint64_t wb = std::uint64_t(n_in / 256) * std::uint64_t(n_out) * 136;
    const std::uint64_t xb = std::uint64_t(n_in / 32) * 36;
    const std::uint64_t yb = std::uint64_t(n_out) * sizeof(float);
    for (int j = 0; j < count; ++j)
        if (!valid_range(slots.w[j], wb) || !valid_range(slots.x[j], xb) || !valid_range(slots.y[j], yb)) return false;
    const dim3 grid(unsigned((std::uint64_t(n_out)+7)/8), unsigned(count));
    kernel<false><<<grid, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(slots, IQ4XSIds{}, n_in, n_out, std::getenv("STRATA_SAGE_TRACE") && std::string(std::getenv("STRATA_SAGE_TRACE")) == "1");
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(std::string("sage IQ4_XS slots launch: ") + cudaGetErrorString(status));
    return true;
}
void report_iq4xs_counts() {
    const char* t=std::getenv("STRATA_SAGE_TRACE");
    if(!t || std::string(t)!="1") return;
    unsigned long long c[256][2]{};
    const auto status=cudaMemcpyFromSymbol(c,iq4xs_ids_counts,sizeof(c));
    if(status!=cudaSuccess) { std::fprintf(stderr,"sage-device: count read failed: %s\n",cudaGetErrorString(status)); return; }
    for(int i=0;i<256;++i) if(c[i][0] || c[i][1])
        std::fprintf(stderr,"sage-device-count: layer=%d in2048=%llu other=%llu\n",i,c[i][0],c[i][1]);
}
bool try_iq4xs_ids_q8_1(const IQ4XSIds& ids, int count, int n_in, int n_out, void* stream) {
    if (count < 1 || count > 65535 || n_in <= 0 || n_in > 2048 || n_in % 256 || n_out <= 0 || ids.x_every < 1 ||
        ids.x_stride % 4 || ids.offset % 4 || ids.y_stride < static_cast<unsigned long long>(n_out)) return false;
    const std::uint64_t xrow=std::uint64_t(n_in/32)*36, xlast=(count-1)/ids.x_every;
    const std::uint64_t ymax=std::numeric_limits<std::uint64_t>::max()/sizeof(float);
    if(ids.x_stride < xrow || (xlast && ids.x_stride > (std::numeric_limits<std::uint64_t>::max()-xrow)/xlast) ||
       (count>1 && ids.y_stride > (ymax-std::uint64_t(n_out))/(count-1))) return false;
    const std::uint64_t xbytes=xlast*ids.x_stride+xrow;
    const std::uint64_t ybytes=((count-1)*ids.y_stride+std::uint64_t(n_out))*sizeof(float);
    if(!valid_range(ids.table,sizeof(void*)) || !valid_range(ids.ids,std::uint64_t(count)*sizeof(int)) ||
       !valid_range(ids.x,xbytes) || !valid_range(ids.y,ybytes)) return false;
    const dim3 grid(unsigned((std::uint64_t(n_out)+7)/8), unsigned(count));
    const char* t=std::getenv("STRATA_SAGE_TRACE");
    kernel<true><<<grid,128,0,reinterpret_cast<cudaStream_t>(stream)>>>(IQ4XSSlots{},ids,n_in,n_out,t && std::string(t)=="1");
    const auto status=cudaGetLastError();
    if(status != cudaSuccess) throw std::runtime_error(std::string("sage IQ4_XS ids launch: ")+cudaGetErrorString(status));
    return true;
}
void iq4xs_slots_q8_1(const IQ4XSSlots& slots, int count, int n_in, int n_out, void* stream) {
    if (!try_iq4xs_slots_q8_1(slots, count, n_in, n_out, stream))
        throw std::invalid_argument("sage IQ4_XS slots: invalid shape or pointer");
}
} // namespace sage::cuda

