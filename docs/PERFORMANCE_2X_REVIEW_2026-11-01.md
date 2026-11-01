# Review of the 2x Sage target — 2026-11-01

**100% throughput uplift versus the normal baseline has not been demonstrated.** This review checks existing evidence and pinned source; it is not a new GPU benchmark.

A is the recorded llama.cpp no-draft baseline using the same Sage IQ4_XS GGUF. B is native Strata. C is standalone NInfer, still incomplete for the full model. D is Strata with adapted NInfer expert kernels and MTP. Sage itself is a model, not a runtime.

| Workload | Recorded A tok/s | Recorded D tok/s | Uplift | 2x-A target | Further uplift needed from D |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arithmetic | 88.37 | 146.70 | 66.0% | 176.74 | 20.48% |
| Python | 88.40 | 140.40 | 58.8% | 176.80 | 25.93% |

These reference targets come from the [same-session bracketed experiment](IQ4XS_MTP_RESULTS_2026-10-30.md). A fresh campaign must remeasure A. Comparing later 162.40 fusion throughput with the earlier 88.37 baseline mixes sessions. The fusion table measures an incremental change within D. A 100% throughput uplift corresponds to 50% less decode time for the same output work, not necessarily 50% less total response time.

## Findings

- The matched CUPTI diagnostic showed 27.2% lower IQ4_XS expert duration, while whole-model no-draft throughput improved only 6.6–6.8%. Other operations remain; summed kernel durations are not wall-time shares.
- Native normalization fusion regressed prefill and some request latency. It does not establish the missing 20–26%.
- Packed Q6 output-head testing had no consistent benefit and cost another 397.9 MiB VRAM.
- Recorded MTP acceptance was around 82–88%. Draft and verification costs still matter. Acceptance alone is not throughput.
- Pinned src/poly/main.cpp caps block guesses at three. With lookup=0, increasing raw window beyond four does not automatically produce more MTP guesses.
- Changing GGUF to a .ninfer container cannot supply missing architecture support or make existing kernels twice as fast.

## Optimization priorities and acceptance gates

1. Mark prefill and decode separately in device traces and profile MTP on as well as off. Current traces mix phases.
2. Benchmark actual Q4_K/shared-expert and Q6_K projection shapes at real verification batch sizes. Existing kernels already use DP4A; alternate packing must beat them without unacceptable VRAM cost.
3. Measure MTP draft/verification overhead with windows 2/3/4. Do not weaken acceptance or skip experts to reach a target.
4. Evaluate expert residency and occupied-context pressure. Lower context or quantized KV cannot be silently applied only to D.
5. Require operator parity, applicable full-logit checks, graph replay and output/quality validation before a fresh bracketed A/D campaign. Keep identical model, precision, prompts, sampling and cache settings; retain warmups, multiple starts, all samples and telemetry. Expand workloads and actual occupied contexts before general claims.

C still needs converter/frontend, routing, model-owned bindings and GDN controls before complete A/B/C/D testing. The target is an optimization objective, not a promised hardware capability.

## Setup changes and checks

The launcher now accepts -MaxContext, -Threads and -DraftWindow, preserving defaults 32768/16/4. -NoDraft overrides the window. Six actual Windows PowerShell launch-plan checks passed: default, 8192/8/window2, window3, no-draft, native, fusion. These are plan checks, not new context or throughput qualification.

See the [Traditional Chinese setup walkthrough](IQ4XS_SETUP_ZH_HK.md).

