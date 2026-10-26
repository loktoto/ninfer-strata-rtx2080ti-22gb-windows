# Windows RTX 2080 Ti 22GB Inference Stack — Proposed Architecture

> **Research design, not an implementation report.** No integrated inference binary or physical-device benchmark is published in this repository yet.

## Separation of concerns

```text
          Local user / OpenAI-compatible API client
                            |
                   [NInfer serving runtime]
                            |
               [NInfer CUDA SM75 execution]
                            |
            [Explicit memory-tier interface]  <-- only after compatibility proof
                 /                   \
         22GB GPU VRAM          Windows host RAM
                 \                   /
                 [Measured paging / copies]
                            |
             NVIDIA Windows driver + CUDA
                            |
              Physical TU102 / RTX 2080 Ti

Optional and independent research track:
  UEFI / pre-boot Turing unlock prototype
                 |
  Driver and device qualification + recovery testing
                 |
  Must independently pass G3 before comparing with baseline
```

## Interface contracts to validate

1. **Hardware identity:** confirm actual TU102, compute capability 7.5, physical VRAM reported by CUDA, relevant BIOS/board information, driver and firmware baseline. A modified 22 GB card cannot be assumed to behave like a stock 11 GB card.
2. **Model artifact contract:** record model name, quantization, file checksum, runtime/container version and tokenizer compatibility before performance comparisons.
3. **Memory ownership:** determine whether the inference runtime allows controlled CPU/GPU placement for weights or KV cache without modifying its kernel correctness assumptions.
4. **Transfer discipline:** define explicit, bounded host RAM ↔ VRAM staging, synchronization, pinned memory policy and transfer/compute overlap; measure PCIe transfer costs.
5. **Kernel correctness:** the memory manager must not silently change tensor layouts, quantization interpretation, attention semantics, execution ordering or SM75 support.
6. **Windows correctness:** compile and measure the native Windows stack independently of WSL2/Linux; do not extrapolate from CI or Linux.
7. **Unlock boundary:** a UEFI or driver feature-unlock is not part of the CUDA kernel. Test it in a separate hardware recovery and qualification lane and add it to the matrix only after successful gated verification.

## Optimization hypotheses (all unproven)

| Hypothesis | Potential upside | Principal failure mode | Evidence needed |
|---|---|---|---|
| Selective host RAM tiering for inactive data | Larger feasible working set/context | PCIe stalls exceed saved VRAM benefit | A/B VRAM/RAM/TTFT/decode traces |
| More careful KV-cache placement | More usable context under fixed VRAM | Cache misses, attention overhead and quality regressions | Repeatable retrieval and latency tests |
| Concurrent staging / CUDA compute | Hide part of transfer latency | Synchronization bugs or no true overlap | Timeline/profiler evidence |
| Optional qualified GPU unlock | Potential performance delta if features become usable | Driver mismatch, instability or zero gain | Separate driver/CUDA and kernel A/B evidence |
| Combined configuration | Potential additive benefits | Interaction effects cause *negative* scaling | Full factorial / ablation benchmarks |

## Experimental matrix

Keep model and build identical within each controlled comparison:

| Profile | Memory tier | GPU unlock | Status |
|---|---|---|---|
| A: native Windows baseline | Off | Off | Planned evidence gate |
| B: memory-tier-only | On | Off | Planned; depends on validated adapter |
| C: unlock-only | Off | On | Not eligible until the separate hardware gate passes |
| D: combined | On | On | Not eligible until B and C independently pass |

All profiles should record the same exact driver/build/model conditions, and the team must explicitly report when the comparisons cannot be held constant.

## Links

- [Parent README](../README.md)
- [Verification gates](ROADMAP.md)
- [NInfer Windows runtime](https://github.com/loktoto/ninfer-2080ti-22g-windows)
- [Turing Windows research](https://github.com/loktoto/Turing-unlock-windows)
