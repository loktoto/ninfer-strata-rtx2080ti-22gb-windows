# A/B/C/D comparison — October 30, 2026

This page preserves the original **no-draft** comparison. A later, separately
controlled [MTP study](IQ4XS_MTP_RESULTS_2026-10-30.md) measured D at 146.70 versus
A at 88.37 tokens/s on arithmetic (+66.0%), and 140.40 versus 88.40 on Python
(+58.8%). Those are whole-configuration gains with drafting in D and no drafting
in A, not NInfer-only gains. The same study includes a matched B/D MTP comparison
and a slower llama.cpp MTP control. C remains unmeasured; see the updated
[standalone port status](NINFER_STANDALONE_PROGRESS_2026-10-31.md). Do not combine
samples or percentages across these experiments.

## Definitions and completion status

| Variant | Engine | Same Sage IQ4_XS model | Status |
| --- | --- | --- | --- |
| A | Installed llama.cpp server, build 11222 / a97cce86a | Yes | Two accepted process starts |
| B | Native Strata from the clean optimized build, expert flag off | Yes | Two process starts |
| C | Standalone NInfer | Not yet imported/qualified | Not measured; no throughput value |
| D | Same Strata binary with NInfer-all-derived IQ4_XS expert flag on | Yes | Two process starts |

This is a completed three-engine-configuration measurement with an explicit C preflight gap, not a completed four-runtime benchmark. A means Sage through llama.cpp: Sage itself is the model, not an engine. D embeds the adapted NInfer-all kernel in Strata and does not start the standalone NInfer engine.

## Fresh measured results

Medians use four warmed samples per variant/workload, across two independent starts. Units are tokens/s.

| Variant | Arithmetic decode | Python decode | Retrieval prefill |
| --- | ---: | ---: | ---: |
| A: llama.cpp | 95.32 | 94.53 | 934.14 |
| B: native Strata | 107.95 | 108.50 | 2202.40 |
| C: standalone NInfer | Not measured | Not measured | Not measured |
| D: Strata + adapted NInfer-all expert path | 115.10 | 115.90 | 2238.50 |

D versus B is **+6.62% arithmetic decode and +6.82% Python decode**. D versus A is **+20.75% and +22.60%**, respectively, under these engine settings. B versus A is +13.25% and +14.77%. The requested 40% whole-model uplift is not established. Retrieval prefill differs strongly between engines, but the nine output tokens are too few for a strong decode comparison; its decode samples are retained in the export without a headline claim.

All content and reasoning strings and token counts matched between A, B and D on all three workloads, including first repetitions. All accepted requests completed successfully and reported zero cached prompt tokens. Exact response text is checked locally; the published dataset contains the boolean comparison and numeric samples, not the text.

The [machine-readable summary](../evidence/published-benchmarks/abcd-summary.json) contains every warmed sample, medians, pairwise percent changes, token/cache counts, TTFT and exact-text checks. The [per-request numeric export](../evidence/published-benchmarks/2026-10-30-abcd-samples.json) includes all 63 requests, including nine excluded cached-control requests and first repetitions. Fifty-four requests belong to the accepted A/B/D starts; 36 warmed requests feed the primary medians.

## Controls and run order

All accepted runs used the same local model and prompt set, 32K configured capacity, FP16 KV, 16 CPU threads, one request at a time, greedy generation, seed 1, maximum 256 output tokens, thinking disabled, no speculative decoding, and no expert/layer skipping. The arithmetic fixture has 52 prompt/240 output tokens, Python 42/110, and retrieval 1133/9. Each start made three repetitions with the three cases interleaved. First repetitions are retained but excluded from warmed medians. The context was not filled to 32K in these short workloads.

A used all GPU layers, one parallel slot, flash attention on, fit disabled, batch 512 and microbatch 128. B and D used the same clean Release/SM75 binary, existing Strata attention/batch implementation, fixed expert residency, and graph option enabled. In B all candidate flags were off; in D only `STRATA_SAGE_EXPERT_IQ4XS=1` was enabled. RMSNorm, Q4_K and Q6_K candidates remained off, and tracing was disabled for timing. Different engine internals mean A versus B/D is a configuration comparison, not isolation of a single kernel.

The intended A/B/D/D/B/A ordering exposed a prompt-cache issue in the first A start: repeated Python/retrieval prompts reused prefixes. That run (`1-A`) is retained as excluded. `cache_prompt=false` was added to the local request harness, and two cache-disabled A starts were accepted (`6-A`, `7-A`). B/D starts reported zero cached tokens throughout. The accepted order is B/D/D/B/A/A. B-versus-D retains its reversed-pair control; A's two accepted starts are later, so the overall engine comparison is not fully balanced against session drift. No confidence interval or broad superiority claim follows from this small study.

The [earlier optimized B/D study](IQ4XS_OPTIMIZED_SUMMARY_2026-10-30.md) remains separately published. Its 107.85→115.00 and 108.60→115.95 results are not overwritten by the fresh trial. Neither study should be combined with yesterday's engine baseline to manufacture a larger percentage.

## Why C remains open

The pinned dense [NInfer Turing branch](https://github.com/mr-september/ninfer-2080ti-22g/blob/8dfa5cc96620b646be4608ffb1246cb89df9a444/README.md) is a dense-model product whose README says its retained MoE code lacks working Turing kernels. It is not an eligible same-model C runtime.

The broader pinned [NInfer-all branch](https://github.com/iamwavecut/ninfer-all/blob/71d3f8190772ef7089f8eda4e22371114fd5bbf9/README.md) supports SM75 and broader MoE operations. Its [GGUF-block path](https://github.com/iamwavecut/ninfer-all/blob/71d3f8190772ef7089f8eda4e22371114fd5bbf9/docs/gguf.md) imports encoded tensors into a `.ninfer` artifact. The reviewed [dense GGUF adapter](https://github.com/iamwavecut/ninfer-all/blob/71d3f8190772ef7089f8eda4e22371114fd5bbf9/tools/convert/gguf_blocks.py) validates a fixed `qwen35` tensor layout. Its [Qwen MoE recipe](https://github.com/iamwavecut/ninfer-all/blob/71d3f8190772ef7089f8eda4e22371114fd5bbf9/tools/convert/official_recipes.py) assigns native Q4/Q5/Q6 groupwise formats rather than establishing a byte-preserving import for this Sage `qwen35moe` IQ4_XS file. No validated exact-model artifact or standalone binary is installed here, so this study did not execute C.

This is a source/installation preflight finding, not an observed C runtime failure or proof of impossibility. Completing C requires mapping and validating the Sage MoE artifact import, building or installing that NInfer runtime, checking model quality/numerical behavior, then rerunning all accepted variants with a balanced order and comparable settings. Different source weights, quantization, KV precision or MTP settings would make it a separate experiment. Upstream performance figures on other models cannot fill this row.

See [methodology](METHODOLOGY_AND_RESULTS.md) for kernel provenance, numerical validation, known bugs and limits, and the [evidence recovery audit](EVIDENCE_AUDIT_2026-10-30.md) for the October 29 data restored to GitHub.
