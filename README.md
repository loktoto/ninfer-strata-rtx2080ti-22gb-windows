# NInfer × Strata × Turing GPU Unlock — RTX 2080 Ti 22GB on Windows

**Experimental Windows local LLM inference research: combine the NInfer CUDA SM75 performance path, Strata's large-model MoE execution and GPU/RAM/SSD expert-tiering design, and independent Turing (TU102) FP16 Tensor Core unlock experiments on a modified NVIDIA RTX 2080 Ti 22GB.**

> **Project status: integration research only.** No three-way integrated inference kernel, Windows GPU unlock, performance gain, installable release or extended-context guarantee has been validated in this repository.

**Proposed GitHub repository name:** `ninfer-strata-gpu-unlock-rtx2080ti-22gb-windows`. **Correct spelling:** `NInfer` (capital letter **I**) / `ninfer` (lowercase); `Nlnfer` with a lowercase **l** is a typo. The repository has **not** been renamed yet.

繁體中文：研究點樣喺 **Windows + 改裝 RTX 2080 Ti 22GB**，發揮 **NInfer CUDA 推理、Strata 的 MoE／GPU-RAM-SSD 分層技術，以及 Turing FP16 Tensor Core Unlock**。唔係聲稱三套系統已合併成功；以兼容性、實卡驗證、推理準確性同效能 A/B 測試作為必要門檻。

## The three technologies (and their distinct boundaries)

| Technology | Real role | Status relevant to this project |
| --- | --- | --- |
| **[NInfer](https://github.com/Neroued/ninfer)** / [SM75 Windows fork](https://github.com/loktoto/ninfer-2080ti-22g-windows) | A standalone C++/CUDA inference engine; our fork investigates Qwen3.8-27B inference on modified RTX 2080 Ti 22GB / SM75 under Windows | The Windows fork is a development project, **not yet a physically qualified Windows release** |
| **[Strata](https://github.com/kossum/strata)** | A separate end-to-end local inference engine, with its own model execution, MoE expert placement / GPU-RAM-SSD memory strategy, and Windows support for specified hardware | **Not** a drop-in NInfer memory plugin; compatibility and integration must be proven |
| **[Turing Unlock for Windows](https://github.com/loktoto/Turing-unlock-windows)** / [Linux upstream](https://github.com/nlzy/turing-unlock) | Independent TU102 feature-unlock research aimed at FP16 Tensor Core throughput; it operates below inference kernels, in the platform/driver boot path | Linux upstream reports device-specific tests; our Windows port has **host-level tests only**, no qualified Windows unlock |

**Important model distinction:** NInfer's targeted **Qwen3.8-27B** path and Strata's **Qwen3.8-Flash-Next MoE** path are not identical workloads. They must not be benchmarked against each other as if both were the same model or as if Strata's scheduler could simply be plugged into NInfer.

## What "integration" could mean

1. **Platform-first (lowest-risk):** a Windows launcher / serving layer selects NInfer or Strata as distinct backends, sharing configuration, benchmarking, model provenance and monitoring. They need not run the same kernel.
2. **Selective technology integration (research):** identify whether specific memory-tiering / expert placement ideas can be adapted to a *compatible* NInfer runtime and model architecture. Measure real PCIe/host RAM costs, cache behavior, synchronization and correctness. This may be infeasible for some model families.
3. **GPU unlock A/B (independent and optional):** only after a separately qualified Windows TU102 procedure exists, compare **the same model, kernels and settings** with and without the unlock on an actual device. Do not promise blanket GPU or LLM speedups.
4. **Combined experiments (last):** try any mutually compatible combinations after all isolated gates pass, report negative outcomes and preserve a no-unlock fallback.

This project targets **22 GB VRAM**, not a reduced-VRAM configuration. Host RAM/SSD is a potential capacity tier for larger models, inactive MoE experts or context state, **not automatically a speed improvement**.

## What this repository contains

- [Architecture and compatibility boundaries](docs/ARCHITECTURE.md)
- [Roadmap, evidence gates and benchmarks](docs/ROADMAP.md)
- References to the two separate, upstream-derived development repos

**No integrated source tree, kernel patch, Windows installer, reliable benchmark, or real-device GPU unlock is currently published here.**

## What must be demonstrated before making performance claims

- Actual modified TU102 / RTX 2080 Ti 22GB board, detected VRAM, Windows build, driver and CUDA compute capability.
- Separate runnable NInfer and Strata baselines on supported model/driver combinations, with checksums and exact build revisions.
- Strict model/quantization/input parity for all within-engine A/B tests; independent correctness and 8K/32K/64K semantic-retrieval tests where supported.
- PCIe copy bytes/time, GPU residency, RAM/SSD I/O, TTFT, prefill tokens/s, decode tokens/s and stability.
- GPU-unlock and no-unlock results on the **same** model and compute paths, independently of RAM offload; measurements must identify FP16/FP32 accumulate kernel use.
- Public failure results, recovery gates, license/attribution compliance and rollback steps.

## Accuracy, safety and licensing

- An upstream Linux TU102 unlock is **not proof of Windows unlock support**. GPU feature unlock experiments may involve hardware/power risk; never infer a general-purpose performance improvement.
- No confidential firmware dumps, private keys or licensed model artifacts are hosted here.
- NInfer upstream: **Apache-2.0**; Strata upstream: **MIT**; Turing-unlock upstream: **GPL-2.0-only** according to their respective source repositories. A future distribution must assess each component's license and provenance independently.
- All proposed optimizations are **hypotheses until independently replicated on physical hardware**.

Maintained by [loktoto](https://github.com/loktoto). Contributions should label each claim **planned / source-built / host-tested / real-hardware-tested**.

## Discoverability

Suggested GitHub Topics: `ninfer`, `strata`, `turing-unlock`, `gpu-unlock`, `rtx-2080-ti`, `windows`, `cuda`, `sm75`, `llm-inference`, `local-llm`, `moe`, `gpu-offloading`.

Repository Description (manual GitHub About field): *Windows RTX 2080 Ti 22GB research integrating NInfer CUDA SM75, Strata MoE GPU/RAM/SSD offloading and experimental Turing FP16 Tensor Core unlock. Not yet integrated.*
