# Evidence recovery audit — October 30, 2026

Audited the repository at commit `12418313e25f462bc76f06f05225abac462fc64e` against the two local experiment groups. The recursive GitHub tree was complete, not truncated. This audit concerns saved test evidence, not a new benchmark run.

The October 29 Windows, engine-comparison, RMSNorm, diagnosis and format/startup reports were already published. Some were missing from the local snapshot even though they remained on GitHub. The README did not provide a useful central index and still described the first IQ4_XS attempt as having no speedup; the new methodology index and current-result links correct that discoverability problem.

## What was missing

| Local experiment group | Selected source files | Already published | Local-only before audit |
| --- | ---: | ---: | ---: |
| `iq4xs-20261029` | 19 | 6 | 13 |
| `iq4xs-optimization-20261030` | 4 | 0 | 4 |

The selected files are per-engine request comparisons, near-32K tests and strict-answer quality results. This is a scoped count, not a count of all logs, build products or tokenizer files. The October 29 group includes later revisions saved under the same folder label.

The 13 local-only sources in the first group were five original engine comparison files (llama.cpp default, llama.cpp requested graphs-off, llama.cpp MTP4, Strata native and Strata RMSNorm), five near-32K result files, and three quality-test files. The second group lacked all four native/candidate/candidate/native per-start comparison sources. The original source files are retained locally; the restored GitHub exports contain their numeric evidence without raw text.

## Recovery mapping

- [October 29 request samples](../evidence/published-benchmarks/2026-10-29-samples.json): all 11 saved comparison runs, including already published graph experiments and the five original engine comparisons. First repetitions are retained; each study's report identifies which samples its medians use.
- [October 29 additional tests](../evidence/published-benchmarks/2026-10-29-additional-tests.json): five near-32K timing/pass results, three sets of six quality checks, and the failed RMSNorm full-logit comparison.
- [October 30 optimized request samples](../evidence/published-benchmarks/2026-10-30-optimization-samples.json): four starts and all 36 requests, including warmups. This recovers the data behind the previously published aggregate gain.
- [Optimization summary](../evidence/published-benchmarks/2026-10-30-optimization-summary.json): all warmed workload samples, medians, token/cache counts and response-equality checks.
- [Correctness summary](../evidence/published-benchmarks/2026-10-30-correctness.json): exact full-logit metrics and device-execution counters for all 40 layers.
- [Fresh A/B/D numeric samples](../evidence/published-benchmarks/2026-10-30-abcd-samples.json): includes the initial cache-contaminated A control, labeled and excluded by the [A/B/C/D report](ABCD_COMPARISON_2026-10-30.md). C has no measured runtime data.
- [Machine-readable recovery audit](../evidence/published-benchmarks/recovery-audit.json): lists the selected local source filenames and their pre-audit publication status.
- [Pre-audit GitHub inventory](../evidence/published-benchmarks/pre-audit-github-files.json): records the docs/evidence filenames at the audited commit.

`tools/export_benchmark_evidence.py` reproduces the numeric exports from retained local originals. It uses explicit output fields; it does not copy arbitrary request/response dictionaries. Timings, token counts, finish status, case labels and pass/fail results are preserved. Raw prompts, generated responses, personal absolute paths, model metadata, binaries, tokenizers and full logit arrays are excluded. The previously blocked broad upload was not retried with those contents.

## Corrections preserved

The first expert adapter's exact logits did not prove sustained resident-decode candidate coverage. The original report carries a correction, and the later device-ID implementation supplies per-layer execution counts. The resident prototype that failed logits was corrected before performance claims. RMSNorm's full-model numerical difference remains unresolved. Native graphs-on gains, isolated kernel speedups and the measured B-versus-D model gain are separate results. The requested 40% whole-model uplift has not been achieved.

See [methodology and results](METHODOLOGY_AND_RESULTS.md) for implementation history and experimental controls. These recovered exports add missing evidence; they do not turn failed or unrun tests into passing results.
