# IQ4_XS uplift diagnosis — 2026-10-29

## What was tested

The tested integration is a partial port: only the opt-in NInfer-derived FP32 RMSNorm has demonstrated a host enqueue on this model. Eligibility is restricted to 2048-wide, non-overlapping calls. The trace is a one-time enqueue marker, not a call count or proof of device execution under graph replay. The model inventory contains 120 IQ4_XS routed expert tensors in blocks 0–39; no IQ4_XS expert candidate is integrated. These remain on Strata's native path. The separately implemented Q4_K path is original GGUF SIMT code, not NInfer's signed-Q4 kernel; Q4_K and Q6_K have standalone oracle results but no demonstrated matching expert dispatch on the measured workload.

The earlier three-run medians with 32K configured context, FP16 KV, no MTP and no graphs were 56.6 vs 57.0 tok/s on arithmetic (+0.7%), 57.1 vs 58.3 on Python (+2.1%), and 1967 vs 1997 prompt tok/s on the 1133-token retrieval (+1.5%) for native vs RMSNorm candidate. These small differences have no paired confidence interval and may include session variation. The target of 1.4x throughput requires eliminating at least 28.6% of baseline latency; without measured operation shares this does not establish an achievable limit or identify a bottleneck.

## Correctness is unresolved

Full-logit comparison failed the diagnostic atol=1e-4, rtol=1e-3 criterion: 3,851,231 of 3,973,120 values exceeded it; max absolute difference 1.602 and overall RMS 0.212. Top-1 agreed for all 16 positions; repeat runs were bit-identical within each configuration. Six small quality tasks per variant passed, which does not establish broad equivalence. An independent 324-case synthetic RMSNorm oracle passed (maximum absolute error 3.9e-7). Native and candidate implement the same formula and epsilon, but use different reduction trees, thread counts, and reciprocal-square-root operations; rounding is a plausible contributor, not an established root cause. Rounded layer summaries first differed at block 1, not a full-tensor root-cause localization. Keep the candidate opt-in and investigate before qualification.

## Graphs and upstream comparisons

A requested CUDA-graphs-disabled llama.cpp run had lower decode rates than its default run, but this is an association, not a quantified Strata graph effect. The llama log's “graphs reused” counter persists in the disabled run and cannot be treated as proof of CUDA graph replay. Strata source has explicit CUDA graph capture and launch paths, yet strict validation disabled them.

A bounded native Strata graph-enabled rerun was attempted with the existing model, harness and settings, changing only the graph option. It did not reach readiness after approximately 130 seconds and was stopped. A no-graphs startup recheck also timed out at 45 seconds, and a separate executable help invocation stalled. The CUDA DLL directory was prepended to PATH; during the first attempt the Strata process had approximately 3.4 MiB working set and zero CPU, and GPU memory use was approximately 872 MiB by desktop processes. Cause is unknown. These attempts provide no graph performance result. Local scripts, configurations and server logs are in `evidence/iq4xs-20261029/`; profiler tools nsys, ncu and nvprof were not found.

The pinned [Turing donor README](https://github.com/mr-september/ninfer-2080ti-22g/blob/8dfa5cc96620b646be4608ffb1246cb89df9a444/README.md) reports a different dense Qwen3.8-27B, groupwise-int format, INT8 KV and CUDA graphs. Its MTP3 code example is 47.5 vs 26.3 tok/s without speculation, while prose is 28.8 vs 26.0. It explicitly says its MoE expert operations do not run on this card. These figures do not promise a 40% uplift for this GGUF MoE workload, and the donor warns of several-percent session variation.

## Next measurements and work

1. Resolve the shared executable startup stall without changing unrelated applications; repeat the matched native graphs-on/off comparison in alternating order, after warmup, collecting per-run values, actual token counts, clocks, VRAM and graph capture/replay evidence. Record a bounded failure if startup remains blocked.
2. Obtain GPU timeline evidence on representative decode and prefill workloads: GPU kernel durations, CPU launch gaps, graph replay, expert matvecs, attention and memory traffic. Check trace overhead separately and distinguish enqueue from executed kernels. Do not infer an operation's latency share from enqueue count.
3. Isolate the first full-tensor divergence using identical captured inputs and weights for native and candidate norms, independent high-precision oracles and layer-by-layer comparisons; check router scores/selected experts and recurrent state. Establish agreed precision and quality criteria before enabling the candidate.
4. If profiling identifies IQ4_XS expert products as a major cost, design a format-correct candidate for actual GGUF IQ4_XS and target SM75. Prove byte decoding and numerics on adversarial shapes, actual runtime dispatch and routing, graph lifetime, full logits and workload quality before measuring end-to-end gain. Evaluate fusion and graph overhead only against evidence; no uplift is guaranteed.
5. Benchmark speculation/MTP and KV changes separately with verified acceptance, output equivalence and memory capacity. They are different configurations and must not be attributed to RMSNorm.
