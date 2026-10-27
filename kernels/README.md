# NInfer-derived SM75 kernels

`sage_sm75_kernels` is a small CUDA static library linked into the unified
runtime. It currently provides FP32 RMSNorm. Strata owns every allocation and
the execution stream. This kernel performs no allocation, copies, or stream
synchronization, and preserves Strata's FP32 activation/weight/output storage.
It is a SIMT reduction, not a Tensor Core matrix multiplication.

## Source and license

Derived from [loktoto/ninfer-2080ti-22g](https://github.com/loktoto/ninfer-2080ti-22g)
commit `de849e1f131a8c615ae2943d2922d217c488fb0e`, itself derived from
[Neroued/ninfer](https://github.com/Neroued/ninfer). The exact source is
`src/ops/kernel/rmsnorm.cuh`, the `rmsnorm_generic_kernel` implementation,
and its plain `rmsnorm_epilogue` formula. See the included full [Apache 2.0
license](LICENSE).

`src/rms_norm.cu` is a modified derivative: BF16 storage and conversions were
replaced with FP32; only plain RMSNorm is retained; null gain means identity;
output rows can have a stride; the tensor wrapper is replaced by a caller-owned
pointer/stream API with prelaunch validation. The original complete reduction
with 256 threads, FP32 partial sums, shared-memory tree and `rsqrtf` remains.
No upstream file is modified, and no native BF16 Tensor Core capability is
claimed for Turing.

## API contract

Include `sage/cuda/rms_norm.hpp` and call `sage::cuda::try_rms_norm_f32`.
Inputs are compact FP32 `[rows,n]`, optional gain `[n]`, and output FP32 with
`y_stride >= n` in floats. Gain can be null. Epsilon must be positive and finite;
dimensions must be positive. The default/null CUDA stream is accepted.

False is returned before any kernel launch for invalid dimensions, epsilon,
pointer alignment, overflowing pointer ranges, or output overlap with either
input or gain. The output overlap check conservatively includes row padding.
Pointers must address sufficient device-accessible storage on the current
device; the caller is responsible for this and for finite values whose FP32
sum of squares does not overflow. Invalid stream/device state or launch failure
raises `std::runtime_error`; successful enqueue does not imply later device
execution success. The caller retains storage until its stream completes.

## Build and numerical qualification

The parent project must enable CUDA before adding this subdirectory. Targets:

```text
cmake --build <build-dir> --target sage_sm75_kernels sage_rmsnorm_parity -j
<build-dir>/kernels/sage_rmsnorm_parity
```

The library compiles for SM75 (with CUDA's forward PTX path). The parity
executable requires an actual compute capability 7.5 GPU and exits nonzero if
there is no CUDA device or another architecture is selected. It prints the
device name, reported VRAM bytes, driver and runtime versions. Numerical success
does not establish modified 22GB-card qualification, end-to-end model support,
memory savings, or throughput improvement.

The independent CPU oracle evaluates the complete mathematical formula in
`long double` from represented FP32 inputs. (MSVC gives `long double` the same
precision as double, still independent of the FP32 GPU reduction.) Cases cover
widths 1, 7, 31, 128, 255, 256, 257, Sage's 2048, and 4097; 1/3/17 rows;
epsilon 1e-6/1e-5/1e-2; zero, tiny, large and mixed-sign inputs; optional gain;
default and nonblocking streams; output padding; and rejected arguments.
The output criterion is `abs(error) <= 3e-5 + 3e-5*abs(oracle)` per element.
GPU compilation and execution remain pending until a CUDA toolkit and physical
SM75 GPU are available; the implementation must not be called hardware-qualified
based on host checks.
