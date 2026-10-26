# Validation Roadmap and Evidence Gates

**Repository maturity:** architecture / research kickoff. These gates are proposed; none should be treated as completed merely because a README exists.

| Gate | Exit criteria | Status |
|---|---|---|
| G0 Hardware baseline | 22 GB modified GPU identified with exact board, driver, CUDA-visible memory, OS build, rollback and integrity checks | Not evidenced here |
| G1 Native Windows inference | Specific NInfer build + pinned model on real GPU; correctness checks and reproducible baseline latency / throughput | Not evidenced here |
| G2 Offload prototype | Confirmed compatible memory interface, bounded RAM staging and repeatable prefill/TTFT/decode A/B versus G1 | Not implemented here |
| G3 Unlock qualification | Independent recovery path, read-only diagnosis, GPU/driver validation and actual-hardware acceptance | Not qualified |
| G4 Combined A/B | Reproduce A/B/C/D experimental matrix; measure any negative interactions and correctness regressions | Blocked by G2 and G3 |
| G5 Release | Working Windows package, checksums, supported hardware matrix, attribution/license audit and reproducible evidence | Not started |

## Evidence standards

For each test record:

- Date/time, Windows build, GPU board/mod details, detected VRAM, driver/CUDA runtime and power/clock settings.
- Exact repository commit, binary checksum, NInfer model checksum and quantization.
- Input length, generated tokens, context-window setting, request concurrency and warm/cold start.
- Median, spread and tail latency for TTFT; prefill and decode throughput; physical VRAM and host RAM peak use.
- Instrumented host ↔ device copy duration and bytes; any pinned memory or IO parallelism.
- Explicit **PASS / FAIL / NOT RUN** for semantic correctness, stability and expected behavior.
- Comparison against the exact G1 baseline; preserve negative or inconclusive results.

A hosted CI green badge is **not** real-device evidence. A successful Windows compile is **not** proof that a modified GPU works, and a device showing 22 GB is **not** proof that a UEFI unlock has succeeded.

## Documentation update policy

Whenever a gate advances:

1. Link the qualifying reproducible logs and source revision, without credentials or sensitive device data.
2. Update the relevant status row and date.
3. Keep unsupported workloads documented.
4. Never replace an unverified claim with a performance headline.

## Public search / discovery metadata (manual GitHub repository settings)

The GitHub repository owner may set:

- Suggested repository name: `rtx-2080ti-22gb-inference-stack`
- Suggested Description: `Research roadmap for RTX 2080 Ti 22GB local LLM inference combining NInfer, Strata memory offloading and experimental GPU unlock on Windows.`
- Suggested Topics: `local-llm`, `llm-inference`, `gpu-offloading`, `nvidia`, `cuda`, `windows`, `ninfer`, `strata`, `gpu-unlock`, `rtx-2080-ti`

Topic and description edits are **suggestions only** until committed via GitHub's repository settings. Unlike README content, these metadata fields cannot be set through a file commit.
