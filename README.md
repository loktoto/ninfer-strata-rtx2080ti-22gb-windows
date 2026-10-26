# RTX 2080 Ti 22GB Local LLM Inference Stack — Windows Research

**Experimental integration roadmap for NInfer + Strata memory offloading + Turing GPU-unlock research on a modified NVIDIA RTX 2080 Ti 22GB (CUDA SM75).**

> **Status: architecture and feasibility research only.** This repository is **not** a working unified inference engine, a one-click installer, a validated GPU unlock, or a benchmarked performance claim. The planned components have different responsibilities and cannot be assumed to work together merely because each exists individually.

繁體中文：呢個專案研究在 **Windows** 上，如何令 **RTX 2080 Ti 22GB** 同時善用 **NInfer 推理運算、Strata 式 GPU/RAM 記憶體分層，以及獨立驗證的 Turing GPU Unlock**。目前只係技術整合規劃，未有可供正式使用嘅整合版本。

## Why this project exists

A modified 22 GB RTX 2080 Ti can be a useful low-cost local AI device, but an inference backend, memory-tiering system and pre-boot GPU feature-unlock experiment operate at **different layers**. The goal is to establish compatible interfaces and measurements before attempting integration.

- **Target workload:** local LLM / Qwen inference with reproducible context-window and latency benchmarks.
- **Target platform:** native Windows first; **NVIDIA Turing / SM75**, focusing on a modified RTX 2080 Ti 22GB.
- **VRAM policy:** keep the 22 GB target. Use additional system RAM **only if** it provides a measured net benefit for relevant workloads, not as an excuse to reduce the VRAM budget.
- **Quality policy:** do not publish unverified CUDA support, unlock capability, large-context reliability, or tokens/second.

## Components and ownership

| Component | Intended responsibility | Verified status in *this* repository |
| --- | --- | --- |
| [NInfer — RTX 2080 Ti 22GB Windows fork](https://github.com/loktoto/ninfer-2080ti-22g-windows) | CUDA SM75 model execution, local serving and measured GPU inference | External development repository; **not integrated here** |
| Strata / memory-tier research | Profile GPU VRAM, host RAM and transfer costs; experiment with model/KV memory placement only through explicit, compatible interfaces | **Integration not implemented or benchmarked** |
| [Turing Unlock for Windows](https://github.com/loktoto/Turing-unlock-windows) | Separately test the TU102 / Windows pre-boot unlock research path | Host tests only; **no verified bootable unlock** |
| This integration repo | Compatibility contract, architecture, reproducible benchmark methodology and integration gates | **Documentation / planning stage** |

**Critical distinction:** A pre-boot GPU unlock is not an inference CUDA kernel, and a memory-offload mechanism is not automatically compatible with an existing model runtime. Combining them requires driver/firmware qualification, compatible memory ownership and synchronization, and end-to-end Windows testing.

## Planned execution order

1. **G0 — Establish a reproducible 22 GB baseline.** Capture exact Windows build, driver, VBIOS/device characteristics and detected VRAM; obtain known-good NInfer 8K inference measurements on an actual physical card.
2. **G1 — Validate native Windows inference independently.** Reproduce a specific model artifact, build/packaging SHA, model quality checks and stable serving before any memory-tier or unlock change.
3. **G2 — Prototype optional memory tiering.** Identify the exact API boundary; measure host RAM, PCIe transfer, prefill, TTFT and decode against G1 with no unverifiable throughput claims.
4. **G3 — Qualify GPU-unlock research *separately*.** Follow read-only diagnostics and recovery gates first; do not require firmware or unlock changes for the baseline path.
5. **G4 — Compare controlled combinations.** Benchmark baseline, tiering-only, any independently qualified unlock-only mode and finally combined mode **only if** all prerequisite gates pass.
6. **G5 — Release only after hardware validation.** Ship reproducible scripts, limitations, rollback instructions, licenses/attribution and an evidence-backed compatibility matrix.

See [Architecture](docs/ARCHITECTURE.md) and [Validation Roadmap](docs/ROADMAP.md).

## Measurements we will publish when evidence exists

- Exact GPU/driver/Windows/model revision and integrity SHA-256
- Physical VRAM use, system RAM use, host-device transfer volume and PCIe bottlenecks
- Time-to-first-token (TTFT), prefill throughput and decode tokens/s; run-to-run variation
- Repeatable 8K / 32K / 64K context tests **as attempted gates, not promises**
- Model correctness and any regressions introduced by offload or unlock experiments
- Separate performance deltas for each component, including negative results

## Safety, provenance and licensing

This repo **does not contain a functioning GPU unlocker** and must not be used as a firmware flashing guide. Avoid collecting or committing proprietary firmware dumps, keys, passwords or licensed artifacts. Any future inclusion of upstream code must respect the corresponding original licenses and notices. The NInfer derivative and the Turing-unlock port have distinct licensing obligations; there is **no assumed combined binary license**.

## Related repositories

- [NInfer for RTX 2080 Ti 22GB on Windows](https://github.com/loktoto/ninfer-2080ti-22g-windows)
- [Turing Unlock for Windows — TU102](https://github.com/loktoto/Turing-unlock-windows)

Maintained by [loktoto](https://github.com/loktoto). Contributions should distinguish **implemented**, **tested on CI**, **tested on real hardware**, and **planned**.
