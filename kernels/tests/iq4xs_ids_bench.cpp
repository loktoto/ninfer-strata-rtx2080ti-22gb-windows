#include "sage/cuda/iq4xs_slots.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
int main(){ try {
  cudaStream_t st; CK(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
  for(int ni : {512,2048}) { int no=ni==512?2048:1024, count=8; size_t wb=size_t(ni/256)*no*136, xb=size_t(ni/32)*36;
    std::vector<unsigned char> w(wb), x(xb*count); for(size_t i=0;i<w.size();++i)w[i]=(i*37+i/29)%256;
    for(size_t i=0;i<w.size();i+=136){unsigned h=0x2001u+unsigned((i/136*37)%0x0bffu);w[i]=h&255;w[i+1]=h>>8;}
    for(size_t i=0;i<x.size();++i)x[i]=(i*23+i/13)%256;
    for(size_t i=0;i<x.size();i+=36){unsigned h=0x1c01u+unsigned((i/36*53)%0x0bffu);x[i]=h&255;x[i+1]=h>>8;}
    void* ws[8]{}; void* dx; float *dy,*dn; const void** table; int* did; int ids[8]={7,6,5,4,3,2,1,0};
    for(int i=0;i<count;++i){CK(cudaMalloc(&ws[i],wb));CK(cudaMemcpyAsync(ws[i],w.data(),wb,cudaMemcpyHostToDevice,st));}
    CK(cudaMalloc(&dx,x.size()));CK(cudaMemcpyAsync(dx,x.data(),x.size(),cudaMemcpyHostToDevice,st));
    CK(cudaMalloc(&dy,size_t(no)*count*sizeof(float)));CK(cudaMalloc(&dn,size_t(no)*count*sizeof(float)));
    CK(cudaMalloc(&table,8*sizeof(void*)));CK(cudaMemcpyAsync(table,ws,8*sizeof(void*),cudaMemcpyHostToDevice,st));
    CK(cudaMalloc(&did,8*sizeof(int)));CK(cudaMemcpyAsync(did,ids,8*sizeof(int),cudaMemcpyHostToDevice,st));
    sage::cuda::IQ4XSIds candidate{table,did,0,dx,xb,1,dy,(unsigned long long)no,-1};
    strata::kernels::MmvqIds native{table,did,0,dx,xb,1,dn,(size_t)no};
    if(!sage::cuda::try_iq4xs_ids_q8_1(candidate,count,ni,no,st) || !strata::kernels::native_mmvq_ids(23,native,count,ni,no,st))throw std::runtime_error("dispatch");
    std::vector<float> a(size_t(no)*count), b(a.size()); CK(cudaMemcpyAsync(a.data(),dy,a.size()*4,cudaMemcpyDeviceToHost,st)); CK(cudaMemcpyAsync(b.data(),dn,b.size()*4,cudaMemcpyDeviceToHost,st)); CK(cudaStreamSynchronize(st));
    double maxerr=0;for(size_t i=0;i<a.size();++i) {if(!std::isfinite(a[i])||!std::isfinite(b[i]))throw std::runtime_error("nonfinite"); maxerr=std::max(maxerr,double(std::abs(a[i]-b[i])));}
    printf("shape %d->%d max_native_abs=%g\n",ni,no,maxerr);
    if(maxerr!=0) throw std::runtime_error("native exact mismatch");
    for(int pass=0;pass<6;++pass) for(int cand : (pass%2==0?std::vector<int>{0,1}:std::vector<int>{1,0})) {
      CK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));
      for(int i=0;i<50;++i) {if(cand) sage::cuda::try_iq4xs_ids_q8_1(candidate,count,ni,no,st); else strata::kernels::native_mmvq_ids(23,native,count,ni,no,st);}
      cudaGraph_t g;cudaGraphExec_t exec; CK(cudaStreamEndCapture(st,&g)); CK(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0));
      for(int i=0;i<5;++i)CK(cudaGraphLaunch(exec,st)); CK(cudaStreamSynchronize(st));cudaEvent_t a0,a1; CK(cudaEventCreate(&a0));CK(cudaEventCreate(&a1));CK(cudaEventRecord(a0,st));
      for(int i=0;i<20;++i)CK(cudaGraphLaunch(exec,st)); CK(cudaEventRecord(a1,st));CK(cudaEventSynchronize(a1)); float ms;CK(cudaEventElapsedTime(&ms,a0,a1));
      printf("%s pass=%d us_per_call=%.3f\n",cand?"candidate":"native",pass,ms); CK(cudaEventDestroy(a0));CK(cudaEventDestroy(a1));CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(g));
    }
    for(auto p:ws)CK(cudaFree(p)); CK(cudaFree(dx)); CK(cudaFree(dy)); CK(cudaFree(dn)); CK(cudaFree(table)); CK(cudaFree(did));
  } CK(cudaStreamDestroy(st));return 0;
} catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
