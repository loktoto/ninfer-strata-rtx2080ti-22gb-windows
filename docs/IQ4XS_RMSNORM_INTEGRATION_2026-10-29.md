# NInfer-derived RMSNorm integration: Windows model diagnostics

Tested 2026-10-29 on the RTX 2080 Ti 22 GB, continuing the [initial IQ4_XS smoke test](IQ4XS_WINDOWS_RESULTS_2026-10-29.md). This document reports one short diagnostic fixture, not qualification or a performance benchmark.

## Setup

- The same pinned and patched Strata executable, model, 32,768-token allocation and strict settings recorded in the initial report were used. The full model SHA-256 was rechecked against a separately created **local identity record**, not the official Q4_K_M lock.
- The fixture used the model's exported tokenizer and chat template for `Say hello.` (13 prompt IDs). The first 16 IDs encoded from the previously observed baseline response were teacher-forced as a continuation. Forced IDs test the same input path, not the accuracy of the model's choices.
- Both baseline and candidate ran in separate processes with FP16 KV, zero expert skipping, frozen expert residency, no graphs or draft, and tracing enabled. The candidate alone selected `STRATA_SAGE_RMSNORM=1`; the baseline set it to `0`.
- The test compared all FP32 logits at each of 16 continuation steps, using the preselected illustrative tolerances `atol=0.0001` and `rtol=0.001`. These tolerances are not established quality acceptance thresholds.

## Observations

Both runs completed successfully, consumed all 13 prompt IDs, emitted the 16 forced IDs and produced full logit dumps. The candidate stderr included `sage-dispatch: rmsnorm`, demonstrating that Strata enqueued the NInfer-derived RMSNorm path in this workload. The baseline did not report this dispatch. This trace records host enqueue, not GPU completion or numerical correctness by itself. The earlier independent GPU RMSNorm oracle passed; the complete candidate inference here also finished without a reported CUDA error.

The full-logit comparison **failed** the chosen tolerances:

| Measurement | Result |
| --- | --- |
| Values compared | 3,973,120 (16 steps × 248,320 vocabulary entries) |
| Outside tolerance | 3,851,231 |
| Maximum absolute difference | 1.602045 |
| RMS difference | 0.212061 |
| Top-1 agreement | 16/16 steps |
| Mean KL divergence, baseline to candidate | 0.00128309 |

The same baseline and candidate runs were each independently repeated with the same fixture and settings. Baseline versus baseline and candidate versus candidate were both bit-identical across all 3,973,120 logits. The repeated baseline-versus-candidate comparison reproduced the same metrics above. Thus the difference is repeatable under this fixture and is associated with enabling the RMSNorm candidate; its underlying cause and practical quality impact have not been established.

The first difference in the rounded layer-summary logs was observed at position 0, block 1 `attn_in`; these summaries do not expose complete tensors and cannot identify the cause. Identical top-1 choices and prior matching API text do not override the failed full-logit comparison.

The GGUF contains 120 IQ4_XS routed expert tensors in blocks 0–39 and three additional block-40 expert tensors (two Q4_K, one Q6_K). The observed no-draft diagnostic reported 10,240/10,240 routed experts resident in VRAM. No Q4_K or Q6_K candidate dispatch was requested or demonstrated by these runs, and no NInfer IQ4_XS expert kernel is integrated.

## Conclusion and next work

Strata and an NInfer-derived RMSNorm kernel are connected in the same inference process, and the candidate was enqueued on this GPU. **The candidate has not passed end-to-end numerical qualification:** leave it opt-in and retain the original Strata RMSNorm as the default. Do not claim equivalent logits, a speedup, MTP4 performance or 32K actual prompt coverage from these tests.

Investigate the first divergence with full intermediate tensor comparisons and routing decisions, characterize accumulation/rounding and expert-selection sensitivity, and compare against an independent reference before changing any tolerances. Extend testing to varied prompts and free-generation quality. Any expert-kernel dispatch requires a separate workload that actually selects a compatible Q4_K or Q6_K path.
