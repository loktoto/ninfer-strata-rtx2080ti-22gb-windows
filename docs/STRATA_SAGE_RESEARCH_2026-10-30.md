# Existing Strata work for Sage-A1.5 — research 2026-10-30

Yes. A separate contributor has made and tested substantial Sage-A1.5-specific Strata work: [xyehya/Strata q36-r2](https://github.com/xyehya/Strata/tree/379f095027097937f28c5c009cacbbb66504d957), with a [Sage artifact and model card](https://huggingface.co/ykarout/Strata-Sage-A1.5-GGUF). This branch is built on **Niko1221/Strata's `src/qwen35` engine**, distinct from the **VecSzn/PolyStrata `src/poly` engine** in our tests. They implement the same model family but do not share an identical driver, buffers, quantization or scheduling contract. We have not built or run this separate fork locally.

## What was actually tested

- [PR 3699](https://github.com/Niko1221/Strata/pull/3699), open and unmerged at research time, adds Q4_K expert down support. The tested Sage Q4_K_M model is modified: its quantized `ssm_alpha` and `ssm_beta` were replaced with original BF16 gates because this engine could not load those quantized tensors. On RTX 5080 16 GB, Fedora, CUDA 13.4, 32K configured context, 112 of 256 experts per layer resident, MTP2 and three code prompts, the author reports 133.9 / 153.2 / 154.7 tok/s; this is not our unchanged IQ4_XS file or Windows/SM75 system. Teacher-forced KL against llama.cpp was not run on that PR.
- [PR 3641](https://github.com/Niko1221/Strata/pull/3641), open, is independent F16 Sage support. It reports 36.58 versus 29.95 tok/s llama.cpp on RTX 5090, F16 weights and KV, 261,888 actual input tokens, 256 greedy outputs and MTP off; one final run each. It does not measure IQ4_XS.
- [PR 3903](https://github.com/Niko1221/Strata/pull/3903), open, reports 104.9 to 167.6 tok/s (+60%) after 16K occupied tokens on RTX 5080, 65K configured, forced text, MTP off, two alternating runs per arm. This combines a change from FP32 to INT8 KV, grouped-query attention and increasing resident experts from 90 to 120, so it is not an isolated kernel gain. The INT8 cache changes precision. Its `--attn-mma` and optimized shared-memory INT8 path require sm_80+; on sm_75 the former is ignored and the latter uses a fallback. SM75 was compiled, not executed.
- [PR 3904](https://github.com/Niko1221/Strata/pull/3904), open, reports +54% prompt throughput at 90K and +159% at 200K for combined `--fused-prompt-attn --chunked-gdn` on RTX 5080. The former requires sm_80+, both change reduction order, and the PR did not run full flag-on KL on that revision. These are long-prefill results, not short decode results.
- [PR 3905](https://github.com/Niko1221/Strata/pull/3905), open, reports +4.7% forced decode at 4K on RTX 5080 from reduction/quantization work, with selected identical-logit tests. Its additional `--side-stream` and `--split-stage-copy` combined raise the reported gain to +9.1%, but the timing did not separate those flags. The side stream is not used for captured decode graphs, and the copy optimization targets hybrid placement. Neither gain is established on the fully resident 2080 Ti.
- [PR 3906](https://github.com/Niko1221/Strata/pull/3906) reports +17.3% at 4K using a group of hybrid expert-placement flags, with non-additive and sometimes negative individual effects; timing-dependent flags can alter reproducibility. [PR 3908](https://github.com/Niko1221/Strata/pull/3908) reports +23% greedy MTP2 versus plain decode at 4K on the same 5080 setup, and +21.8% sampled MTP/GPU-sampling on its separate workload. MTP acceptance, precision, hardware and residency must be measured here.
- The [PR 3910](https://github.com/Niko1221/Strata/pull/3910) chain explicitly lists #3699, #3903–#3910 as stacked open work. Its report builds `75;120` but runs on the 5080, not SM75; Windows and 24/32 GB cards were not tested. The branch's [CMake configuration](https://github.com/xyehya/Strata/blob/379f095027097937f28c5c009cacbbb66504d957/CMakeLists.txt#L180-L219) permits SM75 and includes MSVC conditions, so Windows/SM75 is a candidate, not a proven blocker or a verified working build. The default CUDA architecture is 120: a local build must explicitly select 75.

## Our pinned engine and newer PolyStrata

The measured executable uses PolyStrata [52c31cf](https://github.com/VecSzn/PolyStrata/tree/52c31cfcb55d7e42506a57757886af73f9547314), 2026-10-26, plus our integration patches. Its qwen35moe support is why the actual Sage IQ4_XS model already runs here. The [local A/B/C/D report](ABCD_COMPARISON_2026-10-30.md) measures native Strata and the adapted NInfer-all expert path; standalone NInfer remains unmeasured.

At this research snapshot, [PolyStrata main 20d839d](https://github.com/VecSzn/PolyStrata/commit/20d839dbbaf34f28413e84d6e99301725f4f624b) is 63 commits ahead of the pin. The engine release merge supports mixed gate/up quant types, relevant to some Sage APEX NanoPlus files but not necessary for the IQ4_XS file already running; enables FATTN/FDEC by default for Ampere and later, not SM75; and adds opt-in cache-biased routing. [PR 104](https://github.com/VecSzn/PolyStrata/pull/104) explicitly leaves card-routed all-resident calls untouched and changes selected experts in hybrid calls, so it is not a strict same-routing acceleration for our fully resident benchmark. Its [documented test](https://github.com/VecSzn/PolyStrata/blob/20d839dbbaf34f28413e84d6e99301725f4f624b/docs/BACKENDS.md) shows 121–126 of 128 top tokens matching at one bias setting; do not silently enable it. A source diff alone does not establish any gain from updating the pin on this system.

## Specific transfer candidates and tests

1. **Residual/norm/Q8_1 fusion:** xyehya's [`k_add_rms_q` and `k_moe_out_s8`](https://github.com/xyehya/Strata/blob/379f095027097937f28c5c009cacbbb66504d957/src/qwen35/q36_kernels.cuh#L194) use registers and a one-barrier reduction with the original reduction pairing. Our `src/poly/qwen35.cpp` `Qwen35::layer_head` currently launches `rms_norm`, `native_quantize_q8_1`, residual `axpy`, then another norm and quantization. Inspect actual call and Q8_1 contracts before adapting: this is a different engine, not drop-in code. Add graph-compatible opt-in, compare device outputs to independent oracle and exact native full logits over representative token positions, then measure paired short and occupied-context runs.
2. **MoE combine:** our `Engine::moe_card` calls `moe_sum` after expert products; compare its dataflow and summation order against `k_moe_out_s8`. Avoid assuming their residual, activation, quantized buffer or routing layouts match. Measure kernel and end-to-end time before investing; no SM75 gain is known.
3. **Attention/GDN:** check shape- and architecture-guarded PR paths against `Qwen35::gdn`, `gdn_step`, and native graph capture; the fork's sm_80-only prompt attention/MMA cannot run as advertised on SM75. Grouped attention and INT8 KV are separate quality/configuration experiments, not equal-FP16 comparisons.
4. **Update study:** compare fixed revisions and integration patch applicability in an isolated checkout, identify changed SM75-capable runtime paths, run exact-model baseline and candidate under matched flags, and retain the original pin until proven. Profile before prioritizing.
5. **Standalone Strata fork:** verify its loader's supported expert formats before attempting unchanged IQ4_XS. The Q4_K Sage graft is a separate artifact and must never be substituted in a same-weight comparison. Establish CMake/MSVC/CUDA 75 build, tokenization, logits, memory fit and graph execution before benchmarking; use a separate clone and no default routing bias.

The initial sandboxed clone failed at network name resolution (`getaddrinfo() thread failed to start`); a retry outside that sandbox succeeded in a separate `strata-sage-research` checkout. Its HEAD is `379f095027097937f28c5c009cacbbb66504d957`. Pinned source files and PR metadata were also read through the GitHub connector.

## Local build investigation — October 31 update

An isolated Release build configured successfully with CUDA 13.0.88, MSVC 19.44,
architecture 75, native experts and Q6_K enabled, and the same pinned ggml source
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. At this publication, compilation is
still running; no successful full build, model execution or speed result is claimed.

The exact local model header establishes a loader blocker: all 60 `ssm_alpha`
and `ssm_beta` tensors are Q4_K with shape `[2048,32]`. This fork's `upload_f32`
accepts only F32, BF16, F16 and Q8_0. A local loader patch uses the pinned ggml
Q4_K decoder to widen the values already represented by those blocks to FP32.
It does not replace them with another checkpoint's BF16 values or rewrite the
GGUF. A focused CPU fixture covers packed nibble order, signed scale, nonzero
minima, high scale/minimum bits, guarded outputs and invalid sizes. Those code
changes are not qualified until the build and tests finish. The existing tested
PolyStrata binary and model remain the benchmark reference.

The fork's [expert format lists](https://github.com/xyehya/Strata/blob/379f095027097937f28c5c009cacbbb66504d957/src/kernels/cuda/iq_kernels.cu#L762-L773) include ggml tensor type 23 (IQ4_XS) in both gate/up and down lists. This is stronger than a generic architecture claim, but the exact Sage file's loader, tensor geometry, remaining weights, Q8_1 layout, build options and SM75 runtime still need validation; no exact-file inference is reported here. Q6_K expert support is build-option dependent (`STRATA_Q6K_EXPERTS`), relevant to the model's MTP tensors if that path reaches these experts.

