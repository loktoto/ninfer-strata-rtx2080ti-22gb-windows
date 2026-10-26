# Verification Roadmap — NInfer × Strata × Turing GPU Unlock (Windows)

**Status:** planned research. Project-specific gates are not passed by README publication, by green CI, or by upstream performance claims from a different machine.

| Gate | What must be established | Status |
| --- | --- | --- |
| G0 Target & safety | Identify actual RTX 2080 Ti 22GB TU102, board mod, VRAM, OS, driver, CUDA, baseline health and rollback plan | Not evidenced here |
| G1 NInfer independent baseline | Windows NInfer fork on physical card; pinned supported 27B model, correct outputs, reproducible 8K/prefill/decode tests | Not evidenced here |
| G2 Strata independent baseline | Independently supported Windows Strata execution on *this exact device*; pinned MoE model and actual GPU/RAM/SSD traces | Not evidenced here |
| G3 Coordinated-platform prototype | Safe backend selection, nonconflicting ports, pinned artifacts, shared quality/telemetry schema, distinct model benchmark reports | Not implemented here |
| G4 Narrow memory integration | Explicitly identify adaptable Strata technique and corresponding *compatible* NInfer model/runtime interfaces; retain numerical correctness | Not demonstrated |
| G5 GPU unlock experiment | Independent Windows TU102 boot/driver/recovery qualification and relevant FP16 kernel A/B on real card | Not validated |
| G6 Controlled combined measurements | Matched A/B against appropriate baselines; measured component interactions; record negative and inconclusive results | Blocked |
| G7 Public release | Reproducible Windows install, compatible models/boards, benchmark evidence, attribution/license review and secure defaults | Not started |

## Reproducibility record

Record exact Git commit hashes, driver/Windows version, physical board, detected VRAM, toolkit and clock/power settings. For each engine use a pinned model revision/quantization plus integrity hash.

### Separate measurements per engine

- NInfer: target dense-model behavior and SM75 kernels of the **local** fork.
- Strata: supported MoE model behavior and memory residency; separately profile GPU RAM, system RAM, SSD and expert-fetch traffic.
- Do not cross-compare unlike model families to claim a kernel gain.
- Both: TTFT, prefill and decode, context semantic correctness, multi-run variance, memory usage, failures.

### Controlled experiments

1. Baseline backend unchanged, unlock off.
2. Within-backend memory-policy A/B where the exact backend supports that control.
3. GPU unlock off/on with the **same model and relevant GEMM ops**, only after G5 hardware safety validation.
4. Both features only when independent tests show a compatible implementation path.

Run prefill / TTFT / decode and precision/semantic tests. 8K, 32K, 64K and higher contexts are *test candidates*, not demonstrated capabilities.

## Stop criteria

- Cannot run either chosen backend on tested hardware without instability.
- Any unaccounted model-output/precision regression.
- Unbounded RAM/SSD traffic, repeatable worse TTFT/decode without a clear capacity tradeoff.
- No safe Windows recovery or driver's compatibility proof for optional unlock.
- Unclear redistribution right for included code/binaries.

## GitHub discovery settings (manual update)

**Recommended new repository name:** `ninfer-strata-gpu-unlock-rtx2080ti-22gb-windows`

**Repository Description:** `Windows RTX 2080 Ti 22GB research integrating NInfer CUDA SM75, Strata MoE GPU/RAM/SSD offloading and experimental Turing FP16 Tensor Core unlock. Not yet integrated.`

**Suggested Topics:** `ninfer`, `strata`, `gpu-unlock`, `turing-unlock`, `rtx-2080-ti`, `windows`, `cuda`, `sm75`, `llm-inference`, `local-llm`, `moe`, `gpu-offloading`.

**Reminder:** Text in this file is not the GitHub About Description or Topics metadata; the owner must apply those in GitHub repository settings.
