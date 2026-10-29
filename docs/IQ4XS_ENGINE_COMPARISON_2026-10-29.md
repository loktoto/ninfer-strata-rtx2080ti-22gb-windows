# Sage-A1.5 IQ4_XS: Windows engine comparison

Tested on 2026-10-29 (Hong Kong time). This compares the same local GGUF on the owner's RTX 2080 Ti 22 GB; it is not a general model-quality benchmark or a hardware acceptance decision.

## Setup

- Model: `Sage-A1.5-IQ4_XS-exps.gguf`, 19,322,710,880 bytes, SHA-256 `be92beb4e0fda7b40fc4dd2705831095c27792d342833e9d06d9366850bf7069`.
- GPU: RTX 2080 Ti SM75, 22,528 MiB; driver 616.56. The same computer and model were used sequentially, with only one inference engine resident at a time.
- Project: repository revision `973fcd043d8906b6ac3be72cd9e1c13a3ed5c568`, pinned PolyStrata `52c31cfcb55d7e42506a57757886af73f9547314`, CUDA 13.0.88, executable SHA-256 `8cfebf4f3625d86f2edb772dc7834af5cd37c7b2c9bd87d7f3e0321e957d5ec6`. Both project configurations used the strict launcher, `--no-draft --no-graphs --kv fp16 --max-context 32768 --threads 16`, no expert skipping and frozen expert residency. The candidate additionally set `STRATA_SAGE_RMSNORM=1`.
- Comparison: existing llama.cpp Windows build 11222, commit `a97cce86a`; server executable SHA-256 `11b1a4aea38359202647a40bb04a8c2300d856a1d5a4521c7806bd0c39003bde`, server implementation DLL `6f0760074d9fee915ea718906dfa86e917357e49412cf16e54f38c1f91866a9c`, CUDA backend DLL `90ac71801d4266269ab75d93faaf742fee583badc14619afc749531d77c4ac6e`.
- llama.cpp used `--ctx-size 32768 --parallel 1 --n-gpu-layers all --fit off --cache-type-k f16 --cache-type-v f16 --threads 16 --threads-batch 16 --flash-attn on --no-cache-prompt`. The default comparison used `--spec-type none`. A second run set `GGML_CUDA_DISABLE_GRAPHS=1` in that server's process environment. A third used `--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75`; this is a separate MTP reference, not the user's previously reported exact configuration. The engines retain different internal batching, kernels, caches and memory layouts; settings cannot make them identical. The no-CUDA-graphs environment was requested but an independent CUDA profiler trace was not collected.

## Repeated short and medium requests

Each configuration ran three repetitions of the same arithmetic request, Python prime-function request, and 1,133-token retrieval request, interleaved by case. Requests used temperature 0, top-p 1, seed 1, `enable_thinking=false`, maximum 256 output tokens, and streaming. Table values are medians of three reported output-generation speeds in tokens/s. The retrieval answer is only nine tokens; its generation speed is sensitive to fixed overhead and should not be treated as a sustained decode benchmark.

| Configuration | Arithmetic: 52 prompt / 240–241 output tokens | Python: 42 / 110 | Retrieval: 1,133 / 9 |
| --- | ---: | ---: | ---: |
| llama.cpp, default graph setting, no MTP | 85.2 | 84.6 | 73.7 |
| llama.cpp, CUDA graphs disabled via environment, no MTP | 46.3 | 42.3 | 41.3 |
| Project: Strata native RMSNorm, no graphs or MTP | 56.6 | 57.1 | 62.7 |
| Project: NInfer-derived RMSNorm candidate, no graphs or MTP | 57.0 | 58.3 | 65.5 |
| llama.cpp, MTP4, default graph setting | 73.9 | 62.7 | 77.4 |

For the 1,133-token retrieval case, median prompt-processing rates were 1,547.5 tokens/s for default llama.cpp, 1,558.7 for llama.cpp with CUDA graphs disabled, 1,967.0 for Strata native and 1,997.2 for the RMSNorm candidate. Median first-content latency for this case was 0.802 s, 0.810 s, 0.579 s and 0.586 s respectively. Each reported zero cached prompt tokens. Prompt rates for very short prompts are strongly affected by fixed overhead.

MTP draft acceptance in the arithmetic case was 143/166, 142/167 and 142/167 across the three runs. Despite successful drafting, MTP4 was slower than the default no-MTP llama.cpp configuration in the arithmetic and Python cases under these settings. Neither result validates or contradicts a previous 100 tokens/s result produced with different settings.

The three repetitions produced identical text within each configuration. Strata native, default llama.cpp, no-graphs llama.cpp and MTP4 returned identical text on all three prompts. The RMSNorm candidate returned the same Python and retrieval text; its arithmetic answer was still 217 but one sentence differed and it used 241 instead of 240 output tokens.

The sampled maximum total GPU memory used during these requests was 19,679 MiB for default llama.cpp, 19,666 MiB for no-graphs llama.cpp, 20,428 MiB for Strata native, 20,280 MiB for the RMSNorm candidate and 20,573 MiB for MTP4. Samples were taken roughly once per second, include desktop usage, and are not guaranteed peak allocations.

## Near-32K context

A synthetic retrieval fixture contained 1,278 records. Both model tokenizers reported exactly 31,996 prompt tokens, and every configuration found the correct early-record code `K0027-002`. Each row is **one** request, no cached prompt tokens, and 10 completion tokens. Input time is the engine-reported prompt processing time; wall time includes API overhead and completion.

| Configuration | Prompt time | Wall time | Prompt tokens/s |
| --- | ---: | ---: | ---: |
| llama.cpp default, no MTP | 21.04 s | 21.33 s | 1,520.6 |
| llama.cpp CUDA graphs disabled | 20.53 s | 20.90 s | 1,558.3 |
| Strata native | 16.60 s | 16.86 s | 1,927.6 |
| Strata + RMSNorm candidate | 16.48 s | 16.75 s | 1,941.0 |
| llama.cpp MTP4 | 22.25 s | 22.56 s | 1,438.2 |

This establishes that each tested configuration processed this particular near-32K prompt and returned its expected answer; it does not establish general long-context quality, sustained throughput, or peak VRAM.

## Additional functional and numerical checks

Six strict-answer cases covered integer arithmetic, ordering, leap-day arithmetic, Traditional Chinese translation, factorial and a JSON object. llama.cpp without CUDA graphs, Strata native and the RMSNorm candidate each passed 6/6. The three repeated benchmark cases also returned the correct arithmetic result, a plausible prime function and the expected retrieval code. These small fixtures do not measure broad quality.

The earlier [full-logit comparison](IQ4XS_RMSNORM_INTEGRATION_2026-10-29.md) remains failed: 3,851,231 of 3,973,120 values were outside `atol=0.0001, rtol=0.001`, with maximum absolute difference 1.602045; each configuration reproduced its own logits exactly. The highest-scoring token agreed in all 16 teacher-forced steps, while top-five sets differed in 3 steps. Removing each step's mean logit offset left RMS differences between 0.0688 and 0.2366, so the discrepancy is not solely a uniform logit shift. Neither root cause nor impact across general prompts is established.

## Interpretation

On this PC and these small repeated fixtures, the project generated faster than llama.cpp with CUDA graphs disabled, while the existing llama.cpp default graph configuration generated faster. Project prompt processing was faster on this single near-32K retrieval fixture. The candidate RMSNorm's small measured speed differences from Strata native do not establish a speedup, and its failed logit test means it remains unqualified and opt-in. IQ4_XS experts still use Strata's native implementation; no integrated NInfer IQ4_XS expert acceleration or matched MTP4 project benchmark is claimed.

Raw request/response JSON, per-second memory samples, fixtures, scripts, full logits and logs were preserved locally under `D:\\AI\\ninfer-strata-test\\evidence\\iq4xs-20261029`. More prompts, repeated long-context trials, CUDA profiling, exact-token independent reference comparisons, and investigation of the RMSNorm divergence are needed before a performance or quality claim beyond these observations.


See [uplift diagnosis](IQ4XS_UPLIFT_DIAGNOSIS_2026-10-29.md) for dispatch coverage, correctness, graph-test limitations and prioritized next measurements.
