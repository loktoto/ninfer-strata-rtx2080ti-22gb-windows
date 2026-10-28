# SM75 FP16 MMA probe

`sage_mma_probe [--label locked|unlocked]` runs a bounded, synthetic CUDA probe on the current CUDA device, which must report compute capability 7.5. It emits one JSON object to standard output on success; errors go to standard error and result in a nonzero exit. The optional label is supplied by the user, defaults to `unclassified`, and does not verify the device's lock or unlock state. Run with the same device and conditions to compare measurements.

The kernel issues `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`: actual FP16 inputs with FP32 accumulation, with no BF16 emulation. The packed inputs are all ones. Each warp computes four independent 16×8 output tiles, each accumulated for 512 iterations. The host checks every output register of every lane, warp, and tile after every warmup and measured run against exactly 4096 + accumulator index (0 through 3). A failure aborts without a success JSON object. This uniform input is a basic arithmetic check, not a comprehensive matrix-layout or numerical accuracy test.

CUDA events measure each kernel launch, with two warmups and seven measured runs. JSON reports minimum, median, and maximum milliseconds, plus median TFLOP/s based on `warps × 512 iterations × 4 accumulators × (2 × 16 × 8 × 8)` operations. Results include launch and store overhead, repeated dependent accumulations, and scheduling effects; they are a synthetic workload measurement, not a claim of peak Tensor Core throughput or proof of unlock. Device `nominal_clock_rate_khz` is the CUDA device clock-rate attribute, not a live clock measurement. Driver and runtime versions use CUDA's integer encoding. The probe neither changes system state nor writes result files. Redirect stdout if a saved result is desired. Work is capped at 256 blocks per launch, and does not alter watchdog settings.

The inline MMA primitive is copied from `src/ops/common/mma.cuh` in [ninfer-2080ti-22g-windows](https://github.com/loktoto/ninfer-2080ti-22g-windows), commit `de849e`, under Apache-2.0; see that project's license. The probe requires an SM75 binary, configured by the build target.

The four accumulator chains start at distinct values 0, 1, 2 and 3 to prevent
identical-chain common-subexpression elimination. Each output is checked against
its own initial value plus 8 times the iteration count.
