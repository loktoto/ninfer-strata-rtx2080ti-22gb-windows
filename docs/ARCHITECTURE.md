# NInfer × Strata × Turing Unlock — Windows Architecture Research

**Design stage only.** The engines are separate implementations with different model and kernel paths. This document does not establish that Strata's memory algorithms can directly run inside NInfer.

## System boundary: a unified platform is not automatically one fused CUDA runtime

```text
      OpenAI / Anthropic-compatible app or coding agent
                          |
             Windows orchestration / router (PROPOSED)
                + benchmarking + provenance
                   /                   \
          Backend A: NInfer       Backend B: Strata
          RTX 2080Ti/SM75        supported MoE model
          Qwen3.8-27B path       GPU/RAM/SSD placement
                   \                   /
           NVIDIA Windows driver and CUDA
                          |
             Modified TU102 / RTX 2080 Ti 22GB

      OPTIONAL SEPARATE EXPERIMENTAL PLATFORM LANE
               Windows Turing unlock research
        (not yet validated, not required by either backend)
```

**Do not infer simultaneous co-execution.** Initial architecture should start **one backend at a time** and use distinct API ports when testing multiple servers. Unified API and benchmarks do not imply shared CUDA context, allocator, KV cache, runtime kernels or model format.

## Workstream A — NInfer engine

Source: [NInfer upstream](https://github.com/Neroued/ninfer) and [local SM75 Windows derivative](https://github.com/loktoto/ninfer-2080ti-22g-windows).

- Identify exact supported model container/version, quantization, ops, model architecture and kernel capabilities of the **local fork** (do not assume current upstream RTX 5090 requirements apply unchanged to the fork).
- Test Windows builds on an actual modified 22 GB / SM75 card. Do not equate hosted CI success with working GPU inference.
- Keep the NInfer baseline independent of Strata until explicit boundaries are proven.

## Workstream B — Strata engine and techniques

Source: [Strata](https://github.com/kossum/strata).

- Strata is a **full local inference product**; memory tiering, CPU/GPU expert placement and SSD usage are internally coupled to its runtime and models.
- Start with independently supported Windows + CUDA build/model testing; document any compatibility restrictions for modified Turing hardware.
- Only then study **one of two** integration options: (1) route clients to Strata as its own engine; (2) port a narrowly defined, license-compatible **idea** or explicitly portable subsystem to a compatible NInfer model/ops path.
- MoE *expert* placement does not directly solve dense model weight capacity or KV-cache growth. Map each memory object and bottleneck explicitly; do not claim universal context extension.

## Workstream C — optional Turing GPU unlock

Sources: [Linux Turing-unlock](https://github.com/nlzy/turing-unlock) and [Windows port research](https://github.com/loktoto/Turing-unlock-windows).

- Linux upstream describes a TU102 FP16 Tensor Core throughput unlock on particular tested hardware and driver configurations. This is **not** equivalent to enabling extra CUDA cores, VRAM or doubling all operations.
- Our Windows project currently offers a portable host-validation toolkit; **no qualified Windows firmware/UEFI GPU unlock** has been demonstrated in that repository.
- Gate hardware changes behind recovery procedures and independent testing. Test only relevant FP16 input / FP32 accumulate GEMM paths plus correctness and device health.
- Never require the unlock to run the normal Windows platform/backend baseline.

## Possible integration levels

| Level | What is shared | Main unknown | Initial priority |
|---|---|---|---|
| L0 — Coordinated serving | Common launcher, API selection, test and telemetry format; independent runtimes | Driver, ports, installation and model footprints | High |
| L1 — Unified capacity experiments | Comparable model- and memory-specific policies between runtime experiments | Different model architectures; PCIe, RAM, SSD latency | Medium |
| L2 — Code-level memory integration | Explicit NInfer runtime API / allocator / model support, if engineered | Ownership, CUDA streams, tensor formats, paging and numerical parity | Experimental |
| L3 — Unlock-assisted serving | Pre-qualified platform unlock used below either backend | Windows pre-boot/driver transition and real kernel uplift | Blocked |
| L4 — Fully integrated kernels | Single runtime supporting desired model/placement/unlock combination | Large engineering scope; no established interface | Research only |

## Experimental methodology

- A/B **within the same engine and same model** for offload controls. Do not compare a 27B dense model to a ~125B MoE model and call the difference a Strata kernel gain.
- Hardware unlock A/B must use the same engine, model, quantization, clock/power envelope and GEMM kernel path, as closely as possible.
- Report baseline and experimental VRAM, host RAM, SSD reads, PCIe traffic, time-to-first-token, prefill, decode, p50/p95 latency, output quality and error rates.
- Note conditions where larger working set is possible but per-token decode speed decreases.

## Upstream and licensing

NInfer (Apache-2.0), Strata (MIT) and turing-unlock (GPL-2.0-only) are independent projects. The unified launcher concept alone does not grant permission to redistribute other projects' binaries or proprietary firmware. Assess distribution and source obligations before packaging.

See [Roadmap](ROADMAP.md) and [Project README](../README.md).
