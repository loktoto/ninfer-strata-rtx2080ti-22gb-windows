# IQ4_XS format and startup investigation — 2026-10-29

## Container, metadata and weight encoding

The tested file `Sage-A1.5-IQ4_XS-exps.gguf` is GGUF version 3. Its optional `general.file_type` metadata is 15 (`MOSTLY_Q4_K_M`), a coarse descriptive label, not a constraint on each tensor. The inventory identifies 120 routed-expert tensors actually stored as IQ4_XS in blocks 0–39. The [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) explicitly describes the metadata as a majority-type enumeration inferable from tensor types. The local identity is 19,322,710,880 bytes, SHA-256 `be92beb4e0fda7b40fc4dd2705831095c27792d342833e9d06d9366850bf7069`; see `evidence/iq4xs-20261029/model-inventory.json`.

The [pinned NInfer donor](https://github.com/mr-september/ninfer-2080ti-22g/blob/8dfa5cc96620b646be4608ffb1246cb89df9a444/README.md) uses its own `.ninfer` v3 container and Q4/Q5/Q6 signed groupwise integer blocks with one FP16 scale per 64 values, with Q8 or BF16 elsewhere. Its converter reads source safetensors or FP8, not GGUF; its Turing kernels are registered for dense Qwen 27B-class shapes and its MoE experts are unsupported. IQ4_XS packed data cannot be passed directly to that signed-Q4 kernel; re-quantization would change weights and require separate fidelity tests.

A different container or requantization is not required in principle to optimize this GGUF model. [llama.cpp's CUDA code](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cu) dispatches an IQ4_XS quantized matrix kernel. Pinned PolyStrata `52c31cfcb55d7e42506a57757886af73f9547314` already implements CUDA IQ4_XS paths, including grouped expert gate/up kernels and MoE prefill. These examples establish feasibility, not a measured speedup for this model. An improved kernel could consume original GGUF packed blocks or load/repack them in memory; memory use, load time, layout and numerical behavior must be measured.

## Startup obstruction

A 15-second direct `strata-poly.exe --help` diagnostic in the managed sandbox stalled at zero CPU and approximately 3.9 MB working set with waiting threads. The same executable and CUDA DLL path outside that sandbox returned usage and exit code 2 in under one second. An unsandboxed server using the same GGUF then reached readiness in 68.2 seconds and completed nine no-graphs requests. This establishes a sandbox-associated launch obstruction for these trials; it does not identify the specific Windows mechanism. No system security settings or unrelated applications were changed.

## Correctness and profiling status

The previously observed full-logit discrepancy is still unresolved; matching top-1 and small quality checks do not establish equivalence. The candidate remains opt-in. Native and candidate kernels use the same mathematical form but different reduction and reciprocal-square-root implementations; attributing the accumulated difference requires captured identical inputs and weights, route/state comparisons, and independent reference evaluation. Nsight Systems, Nsight Compute, and nvprof were unavailable in the environment. Without a device timeline and kernel duration attribution, do not assert IQ4_XS is the bottleneck or claim a custom kernel will deliver a particular speedup. The prior native implementation already has specialized IQ4_XS expert paths, so a new replacement requires evidence and matched correctness and end-to-end tests.

A new analysis of the saved 16 teacher-forced rows applies softmax at temperature 1. The maximum total variation distance is 0.07745, at row 12, where the shared top token's probability changes from 0.6401 to 0.7158; maximum KL(baseline || candidate) is 0.01601 and median total variation is 0.0000243. This quantifies a distribution difference in this fixture, not its cause or general quality impact; the existing full-logit failure remains decisive for the stated diagnostic threshold. The calculation and full per-row results are in `analyze_rmsnorm_probabilities.py` and `rmsnorm-probability-analysis.json`.

## Controlled native graph-option comparison

Four fresh server starts ran in ABBA order: graphs off, on, on, off. All used the same GGUF, FP16 KV, 32,768 context, no draft, temperature zero, identical prompts and three repetitions each. The first repetition of each start is excluded from the results below. All 36 outputs matched byte-for-byte for their respective prompts across the four runs, with 52/240, 42/110 and 1133/9 prompt/completion tokens respectively. GPU memory and process working set were sampled; clocks were not collected. Samples are individual server-reported decode tokens per second, not confidence intervals.

| Workload | Graph option off | Graph option on | Medians |
|---|---|---|---|
| Arithmetic decode | 61.2, 55.2, 56.8, 56.2 | 98.0, 97.9, 96.9, 96.6 | 56.5 → 97.4 tok/s (+72%) |
| Python decode | 62.0, 60.7, 59.4, 57.1 | 95.7, 97.4, 98.4, 98.9 | 60.05 → 97.9 tok/s (+63%) |
| Retrieval decode (9 tokens) | 65.2, 64.7, 69.1, 66.6 | 107.1, 106.4, 107.3, 105.0 | 65.9 → 106.75 tok/s; short sample |

The graph option alone was changed apart from server port; no candidate kernel was enabled. This exceeds 1.4× for these two short decode workloads but is not a universal model-speed claim. The source includes graph capture and launch under the option, but a profiler trace of execution was not available; the measured result is an option-level comparison, not per-kernel attribution. Exact text agreement on three prompts is not a full-logit equivalence or broad quality test. The complete run records, settings and script are local under `evidence/iq4xs-20261029/unsandboxed-*` and `run_unsandboxed_pairs.py`; `summarize_unsandboxed_pairs.py` produces the compact summary.

For the 1133-token retrieval prompt, warmed prefill medians were 2018.5 tok/s off and 2005.5 tok/s on; this experiment does not show a prefill gain. The retrieval decode is only nine tokens and should not be used as a stable long-decode estimate.

## Graph-on RMSNorm candidate

After ABBA, a fifth fresh server start used the same graph-enabled native configuration except `STRATA_SAGE_RMSNORM=1` and the one-time trace enabled. Its engine log reported `sage-dispatch: rmsnorm`; that marker establishes a host enqueue, including possible graph capture, rather than proving device execution or replay. All nine requests completed. Excluding the first repetition, arithmetic decode was 95.4 and 95.8 tok/s versus native graph-on samples 98.0, 97.9, 96.9, 96.6. Python was 97.1 and 95.6 versus 95.7, 97.4, 98.4, 98.9. This one-start candidate sample provides no evidence of an incremental speedup; it is not an interleaved candidate study or confidence interval.

The Python and retrieval outputs matched the native graph-on outputs. Arithmetic answered 217 in both cases, but one sentence differed and candidate completion length was 241 tokens versus 240. Such a greedy-output difference and the existing full-logit failure make this candidate unsuitable for default enablement without further investigation. These are different samples, so the small rate differences should not be assigned solely to RMSNorm. See `unsandboxed-5-rms-graphs-comparison.json` and its engine log for the raw results.

The pinned [PolyStrata source](https://github.com/VecSzn/PolyStrata/blob/52c31cfcb55d7e42506a57757886af73f9547314/src/kernels/cuda/iq_kernels.cu) contains grouped native IQ4_XS expert kernels; its [MoE prefill source](https://github.com/VecSzn/PolyStrata/blob/52c31cfcb55d7e42506a57757886af73f9547314/src/prefill/moe_fused_iq.cu) includes the same format. Their existence does not prove which individual GPU kernels ran during this benchmark. A full expert replacement is gated on measuring the actual operation shares and verifying the precise dispatch; no dedicated GPU timeline or device profile was available here. If warranted, first validate the GGUF IQ4_XS block decoder and a candidate on real expert shapes and routing, then compare logits and quality and perform matched end-to-end trials. Graphs alone provide the demonstrated workload-specific throughput improvement above.


