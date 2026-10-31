# Methodology, implementation history, and benchmark evidence

This is the index for the October 29–30, 2026 Windows Sage IQ4_XS investigation. It records successful results, failed experiments, corrected claims, and outstanding comparisons. The model stays in GGUF for Strata; the integrated component is an adapted NInfer-all CUDA kernel, not a second NInfer engine.

## A/B/C/D definitions

| Variant | Definition | Qualification |
| --- | --- | --- |
| A | The same Sage 1.5 IQ4_XS model through the installed llama.cpp server | Measured engine baseline. Sage is a model, so the runtime must be named. |
| B | Native Strata, candidate flags off | Measured native control. |
| C | Standalone NInfer running the same model | Not measured; a qualified Sage artifact/import and standalone runtime are still missing. No throughput is assigned. |
| D | Strata with the NInfer-all-derived IQ4_XS resident expert kernel | Measured integration; RMSNorm off. |

The [A/B/C/D report](ABCD_COMPARISON_2026-10-30.md) records the fresh engine comparison. C is not interchangeable with D. The dense [Turing NInfer branch](https://github.com/mr-september/ninfer-2080ti-22g/blob/8dfa5cc96620b646be4608ffb1246cb89df9a444/README.md) explicitly omits working Turing MoE kernels. [NInfer-all](https://github.com/iamwavecut/ninfer-all/blob/71d3f8190772ef7089f8eda4e22371114fd5bbf9/README.md) supports SM75 and broader MoE/GGUF-block operations, but that does not establish import compatibility for this exact Sage model. Its pinned [GGUF documentation](https://github.com/iamwavecut/ninfer-all/blob/71d3f8190772ef7089f8eda4e22371114fd5bbf9/docs/gguf.md) describes importing encoded blocks into its artifact container. Its [dense GGUF adapter](https://github.com/iamwavecut/ninfer-all/blob/71d3f8190772ef7089f8eda4e22371114fd5bbf9/tools/convert/gguf_blocks.py) validates `qwen35` geometry, whereas this model is `qwen35moe`. This is a concrete import gap, not a claim that standalone NInfer can never support Sage. Converting a different dense model would not complete C.

## What changed and why

1. Established native Strata and llama.cpp baselines using the same local model. Inventoried actual tensor encodings rather than trusting the GGUF summary label. The routed experts are IQ4_XS; Q4_K/Q6_K-only adaptations did not cover that work.
2. Tested the FP32 RMSNorm candidate with an independent numerical oracle and full teacher-forced logits. Its small oracle passed, but full-model distributions differed. It remains disabled and unqualified; top-1 agreement was not accepted as sufficient correctness.
3. Traced startup failures. Native executables stalled inside the managed sandbox, including `--help`, but ran outside it. The earlier startup failures did not establish a graph-mode defect.
4. Ran an off/on/on/off graph-option comparison. Decode improved strongly, but that was a native Strata configuration effect. It must not be presented as an NInfer kernel gain.
5. Adapted [NInfer-all's IQ4_XS decoder and two-row warp schedule](https://github.com/iamwavecut/ninfer-all/blob/71d3f8190772ef7089f8eda4e22371114fd5bbf9/src/ops/linear/gguf/ggml_bridge_vec.cuh), with Apache-2.0 attribution, to Strata's interleaved Q8_1 activations and existing expert buffers.
6. Found and fixed a coverage bug: the first adapter intercepted host-routed slots, while all-resident decode could use `moe_card()` and bypass it. The first exact-logit report therefore did not establish candidate execution throughout decode. The [original report was corrected](IQ4XS_EXPERT_INTEGRATION_2026-10-30.md).
7. Added the device-selected expert-ID path for gate/up and down projections. Device counters verify execution per layer/projection, including graph replay, and are read only after synchronization. A one-time host or GPU marker alone is not treated as full decode coverage.
8. Replaced scalar decoding with packed byte permutations and integer `dp4a`, then corrected floating-point factor/reduction ordering. The first resident variant failed logits; the final tested variant matched native. A warp-local layout removed synchronization overhead while retaining the tested arithmetic order.
9. Built in a separate clean Release/SM75 directory and repeated correctness and same-session performance checks. The original build was already Release/SM75; the useful fixes were coverage and kernel implementation, not merely renaming the build type.
10. Tested the model's actual MTP block in a separate [draft-window study](IQ4XS_MTP_RESULTS_2026-10-30.md), including native/candidate/candidate/native and a fresh llama.cpp/candidate/candidate/llama.cpp comparison. Decode mode is labeled explicitly; this is separate from the no-draft kernel ablation.
11. Fixed the API launcher's saved configuration: it previously omitted the IQ4_XS selection despite setting it for the initial process. Restarts now preserve that selection and disable inherited tracing. Added explicit graph/draft options with 17 passing launcher checks, including restart-environment regression coverage. The benchmark harness's explicit configs did not have this omission.

## Correctness and build procedure

Pinned PolyStrata and ggml source revisions are in `config/sources.lock.json`; the integration applies an ordered patch series through `tools/bootstrap.py`. A CUDA-capable Windows x64 developer environment builds with CMake/Ninja, Release, architecture 75, optimized MSVC flags, and no CUDA device debug flag. See the [reproducible optimized build](IQ4XS_OPTIMIZED_SUMMARY_2026-10-30.md).

The evidence has several layers:

- Independent long-double oracle: the original 122 GPU cases were extended to 134, adding 32 device-selected expert IDs with eight experts per activation for four-token verification. Coverage includes packed-code patterns, real matrix shapes, permuted expert IDs, nonzero weight offsets, shared/strided activations, guarded outputs, and graph replay. Tolerance is `2e-5 + 3e-6 * sum(abs(products))`; passing does not imply bitwise equality to the oracle. The production executable was unchanged by this test extension.
- A separate native-versus-candidate GPU check uses varied FP16 scales and requires exact equality for both model shapes.
- A fixed prompt and 16 teacher-forced continuation positions compare the entire vocabulary, with RMSNorm and other candidates disabled. All 3,973,120 logits matched exactly on the clean build. Device counters showed 16 executions of each projection in every layer 0–39. Teacher forcing fixes inputs; the full logits, rather than the forced token IDs, are the correctness evidence.
- All 88 Python tests passed, including the repaired Windows CRLF patch fixtures. The host adapter and patch-series checks passed. The older 85/88 result is retained as historical evidence rather than overwritten.

The [numeric correctness export](../evidence/published-benchmarks/2026-10-30-correctness.json) includes the full-logit summary and per-layer counts. `tools/validate_iq4xs_resident.py` reproduces the model check using user-supplied local paths. A separate real-model GPU launch timeline was not collected; graph-option-enabled execution and the kernel oracle's replay checks should not be overstated as that timeline.

## Performance procedure

Use the same model, tokenizer/template, prompt set, GPU and server timing definitions where the runtimes expose them. Fix 32K configured capacity, FP16 KV, 16 CPU threads, greedy sampling (`temperature=0`, `top_p=1`, seed 1), no speculative decoding, no expert/layer skipping, one request at a time and all available experts resident. Disable candidate RMSNorm in B and D. Allow each engine's graph-capable path for the main comparison. Engine batching, attention implementation and graph internals still differ; A versus B/D is an engine comparison, while B versus D uses the same binary and changes the expert flag.

Each process start runs arithmetic, Python generation, and retrieval in three interleaved repetitions. Exclude the first repetition of each workload/start from the primary warmed median, but publish it in the numeric sample export. Check completion, token counts, prompt-cache reports, exact output agreement and sampled GPU conditions. Prompt caching must be disabled; contaminated runs are kept and explicitly excluded. Report prefill, decode and first-content latency separately. Retrieval has only nine output tokens, so its decode rate is not a robust throughput benchmark.

For the optimized kernel, use native/candidate/candidate/native starts. Four warmed samples per variant across two starts give arithmetic **107.85 → 115.00 tok/s (+6.63%)** and Python **108.60 → 115.95 (+6.77%)**. Retrieval prefill is **2233.65 → 2213.65 tok/s (-0.90%)**. All response text and token counts matched. The [sample export](../evidence/published-benchmarks/2026-10-30-optimization-samples.json) and [summary](../evidence/published-benchmarks/2026-10-30-optimization-summary.json) preserve this trial independently of the newer A/B/C/D run.

The no-draft trial does not demonstrate a 40% whole-model uplift. The native graph-option experiment and isolated expert-kernel timings must not be substituted for this B-versus-D result. Two starts per variant and a small prompt set do not establish a confidence interval, general quality or occupied-32K performance. The later [MTP study](IQ4XS_MTP_RESULTS_2026-10-30.md) records drafting separately. Session-to-session rates changed materially, so comparing yesterday's A with today's D cannot establish a controlled uplift.

The [NInfer research](NINFER_UPLIFT_REPORTS_2026-10-30.md) preserves the hardware,
model and decoding conditions behind published large gains. The [Sage Strata
fork review](STRATA_SAGE_RESEARCH_2026-10-30.md) distinguishes the separate
`src/qwen35` implementation from our pinned PolyStrata `src/poly` engine and
records the concrete Windows/SM75 build and model-loader investigation.

## October 29 evidence recovered and indexed

The October 29 reports were already on GitHub, but several underlying numeric datasets existed only locally. The [audit](EVIDENCE_AUDIT_2026-10-30.md) records which families were missing and what has been exported. The folder date is an experiment-group label; later revisions saved in that folder do not all represent October 29 measurements.

| Historical study | Report | Restored evidence |
| --- | --- | --- |
| Windows build, oracle checks, initial Python failures | [Windows results](IQ4XS_WINDOWS_RESULTS_2026-10-29.md) | Report retained; later test repair is identified above |
| llama.cpp / Strata / RMSNorm / MTP4 comparison | [Engine comparison](IQ4XS_ENGINE_COMPARISON_2026-10-29.md) | [Per-request numeric samples](../evidence/published-benchmarks/2026-10-29-samples.json) |
| 31,996-token retrieval and six strict-answer tasks | [Engine comparison](IQ4XS_ENGINE_COMPARISON_2026-10-29.md) | [Long-context and quality results](../evidence/published-benchmarks/2026-10-29-additional-tests.json) |
| RMSNorm numerical failure | [RMSNorm report](IQ4XS_RMSNORM_INTEGRATION_2026-10-29.md) | Full-logit metrics in the additional-tests export |
| Graph-option experiment and format research | [Format/startup report](IQ4XS_FORMAT_STARTUP_2026-10-29.md) | Per-request samples and previously published paired summary |
| Why the initial uplift was small | [Diagnosis](IQ4XS_UPLIFT_DIAGNOSIS_2026-10-29.md) | Historical interpretation; resident-path correction supersedes the initial coverage claim |

Exports retain numeric timings, token/cache counts, completion status and case labels. They exclude raw prompts, generated text, personal absolute paths, model metadata, binaries, tokenizers and full logit tensors. The original evidence remains local. Missing from GitHub does not mean lost; the audit distinguishes unpublished originals from recovered numeric exports. Failed and superseded experiments remain labeled instead of being removed from the history.
