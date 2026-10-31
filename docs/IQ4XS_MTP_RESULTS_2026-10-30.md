# IQ4_XS expert path and speculative decoding results

These measurements were collected October 30–31, 2026 on a 22 GB RTX 2080 Ti. The tested optimized Release binary targets CUDA SM 75 and has SHA-256 `b1aa241711c97a708c61209a2baf4809e9fe175d2c2a7c449e201f788faf4ae6`. The binary was unchanged during these tests.

## Bracketed configuration comparison

The sequence was A, D, D, A, then one A-MTP start. A is the local llama.cpp baseline without drafting; D is the optimized Strata engine with the opt-in NInfer-derived IQ4_XS expert path and its built-in MTP draft block enabled. Both use the same model, 32,768-token configured context, FP16 KV cache, 16 CPU threads, GPU offload, greedy sampling, and disabled prompt caching. Their implementations and batching settings differ, so this compares complete configurations, not the isolated contribution of the expert kernel. D uses `--lookup 0 --window 4`: one known token and up to three guesses. No model layers are skipped. A-MTP is a separate bounded test of llama.cpp's draft-MTP mode with draft maximum 4 and minimum probability 0.75.

Each A and D server start produced one warmup and two measured requests per case; the table gives the median of four measured decode rates across two starts per configuration. A-MTP has only one start and two measured requests. Values are tokens per second.

| Case | Output tokens | A no draft | D, window 4 | D vs A | A-MTP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arithmetic | 240 | 88.37 | 146.70 | +66.0% | 79.07 |
| Python | 110 | 88.40 | 140.40 | +58.8% | 77.37 |

The arithmetic and Python D runs accepted respectively 570 of 651 offered draft tokens (87.6%) and 240 of 291 (82.5%). The bounded A-MTP run accepted 286/332 and 124/142 but was slower than A in these samples. A third retrieval case produced only nine tokens; its timing is too short to support a throughput conclusion. All measured content and reasoning strings for all three cases matched the first A response exactly. This is a small greedy-output check, not proof of general model equivalence or a full-logit test of speculation.

Median prompt processing was 438.6 vs 355.3 tokens/s for arithmetic and 400.5 vs 294.9 for Python (A vs D). Corresponding median reported prompt times were 118.654 vs 146.400 ms and 104.950 vs 142.450 ms. Median time to first token was 0.224933 vs 0.166515 seconds for arithmetic and 0.149142 vs 0.167818 seconds for Python; median total response time was 2.943156 vs 1.793220 seconds and 1.383188 vs 0.949996 seconds, respectively. All measured A/D samples reported zero cached tokens. Different token timing implementations and short prompts limit cross-engine interpretation. The baseline drifted upward between the two A starts, and these are few samples on a shared desktop, so these figures are workload-specific observations, not a guaranteed general uplift. Sampled median SM clocks were 1710–1725 MHz, memory clock 7000 MHz, temperatures 55–58 °C, and utilization 87–89% across the four A/D starts; samples also include idle transitions.

## Isolating the expert path

An earlier matched, same-binary, graph-enabled no-draft ABBA comparison measured native versus candidate medians of 107.85 vs 115.00 tokens/s for arithmetic (+6.63%) and 108.60 vs 115.95 for Python (+6.77%). In a separate window-4 matched ABBA experiment, native versus candidate medians were 130.95 vs 145.60 (+11.19%) and 125.10 vs 136.20 (+8.87%). These are separate sessions and cannot be added to obtain the combined gain above. In the window-4 experiment candidate draft acceptance was 581/655 for arithmetic and 240/291 for Python. Rates drifted upward across starts. Window 2 and window 3 were also checked in one candidate start each; window 4 was selected for the bracketed study, not established as universally optimal.

The standalone NInfer ≥40% target has not been demonstrated: this work integrates a specific NInfer-derived packed IQ4_XS decoding technique into Strata and measures the combined system; it does not provide a working standalone NInfer Sage baseline. The +66.0% arithmetic and +58.8% Python combined-configuration observations do not prove the requested 60–100% range on other workloads, and Python is below 60% in this sample.

## Correctness, coverage, and provenance

With drafting disabled and CUDA graphs eligible, 16 teacher-forced positions produced 3,973,120 bitwise-identical logits between native and the optimized expert path. Debug device counters recorded both expert projections in all 40 layers at each of the 16 positions. This demonstrates execution in that test; graph eligibility alone does not prove a particular replay trace. The expanded isolated GPU oracle passed 134 cases, including 32 device-selected expert IDs, `x_every=8`, varied strides and offsets, permuted IDs, and graph replay. Its maximum absolute error was 0.00255803 under the scale-aware threshold `2e-5 + 3e-6 * sum_abs_products`. The updated test is `kernels/tests/iq4xs_slots_parity.cpp`.

The packed lookup idea derives from NInfer-all; IQ4_XS layout handling, the device-selected expert adapter, and the integration into PolyStrata are adaptations for this model and runtime. The experimental RMSNorm path remains disabled because its earlier full-logit discrepancy is unresolved. The second D start used the actual launcher with CUDA graphs, window 4, and the IQ4_XS opt-in; its generated configuration recorded the expert path enabled and RMSNorm disabled.

Sanitized [per-sample numeric data, acceptance counts and telemetry](../evidence/published-benchmarks/2026-10-30-mtp-samples.json), the [matched summary](../evidence/published-benchmarks/2026-10-30-mtp-abba-summary.json), and the [bracketed summary](../evidence/published-benchmarks/2026-10-30-mtp-best-abba-summary.json) exclude request and response text and local paths. `python tools/export_mtp_numeric.py` regenerates these exports from the retained local originals. The October 30 folder name identifies the experiment group; the final bracketed runs crossed midnight into October 31 in Hong Kong.
