# Qwen35 residual/norm/Q8 fusion: results and limits

This opt-in experiment adds a small, workload-dependent improvement to the
combined Strata/NInfer IQ4_XS configuration. It also regressed prefill and some
request timings, so it remains disabled by default. It does not establish
another 40% improvement or a new comparison with llama.cpp.

## Implementation and build

Patch `0010-qwen35-fused-norm-q8.patch` wires PolyStrata's existing
`native_add_rms_norm_q8_1` operation into Qwen35's two layer-head normalization
sites. The first combines norm and Q8_1 quantization; the second also includes
the residual addition. It preserves the original reduction order, residual
values, Q8_1 block representation, stream and debug-output placement. Other
architectures in the pinned source already use this operation. This is native
Strata fusion, not another NInfer-derived RMSNorm kernel.

`STRATA_SAGE_FUSED_NORM_Q8=1` selects it; default is off. Simultaneous selection
of the separate, numerically unqualified `STRATA_SAGE_RMSNORM` is rejected.
Both launchers save the explicit setting and clear an inherited opt-in for the
default control. The API offers `--fused-norm-q8`.

The isolated `build-fused` Release/SM75 build used CUDA 13.0.88, MSVC 19.44,
PolyStrata `52c31cfcb55d7e42506a57757886af73f9547314`, the project integration
patches through 0010, and ggml `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
It reused the pinned ggml checkout instead of fetching another copy. Relevant
build options matched the earlier optimized build; the ggml source path differed.
The build completed all 134 steps, including the parity target.

- Measured fusion executable SHA-256: `fadc9398c14e10abe398b20df0f7cdd01f1e97de23a28f740797d229e885c86f`.
- Earlier optimized executable SHA-256: `b1aa241711c97a708c61209a2baf4809e9fe175d2c2a7c449e201f788faf4ae6`; preserved unchanged.
- Complete logit-dump SHA-256 for the old candidate, new fusion-off and new fusion-on runs: `d2f5e7d1a28f37b22bb3e84173f2eaaa1c7f0b1186a3f65f1b9c0999ec751e15`.

## Correctness and device execution

The existing GPU parity test passed 50 cases: widths 512–5120, one and multiple
rows, with/without residual addition, a 1024-row prompt case and large
activations. Residual, normalized output and Q8_1 blocks were byte-identical.
The real-model comparison matched all 3,973,120 logits across 16 teacher-forced
positions exactly, with IQ4_XS expert device counters covering all 40 layers.
That bounded check does not establish all-context or broad model quality.

A later accepted CUPTI capture records 19,360 fused-kernel executions. Total
request-window kernel records fell from 317,259 in the earlier no-draft candidate
trace to 288,219, with 283,680 nonzero-graph-ID records across 122 graph IDs.
No dropped records or Activity/parse errors were reported. These counts include
prefill and decode; the instrumented timings are not benchmark results.
The [fused trace summary](../evidence/published-benchmarks/2026-11-01-cupti-fused.json)
preserves the observed symbols and counts.

The first attempt completed the API request but produced no trace: a relative
CUPTI DLL path was interpreted from the child engine's different working
directory. That attempt is retained locally and excluded. The runner now
resolves the DLL directory absolutely and fails if no trace file is produced.
Only the corrected capture and successfully parsed summary support the device
claim above. The full Python suite passed 95 tests, including explicit selection,
saved-setting restart and incompatible-backend checks.

## Same-binary MTP comparison

The order was old-binary control, new-binary fusion-off sanity, then
**off/on/on/off** on the same new executable. Each process made three interleaved
requests for each of three workloads; first repetitions were retained but
excluded from warmed medians. The four main starts yield four warmed samples
per condition/workload. All 54 requests completed, reported zero cached prompt
tokens, and matched both content and reasoning exactly.

Both main conditions used the same unchanged IQ4_XS GGUF, FP16 KV, 32,768
configured context capacity, 16 threads, graph eligibility, MTP window 4,
IQ4_XS expert opt-in, greedy sampling, seed 1, max 256 tokens, no skipping and
cache-disabled requests. The prompts occupied 52, 42 and 1133 tokens; this is
not an occupied-32K test. GPU telemetry is included in the numeric export.

| Workload | Fusion off decode tok/s | Fusion on decode tok/s | Change |
| --- | ---: | ---: | ---: |
| Arithmetic, 240 output tokens | 155.60 | 162.40 | +4.37% |
| Python, 110 output tokens | 150.70 | 151.95 | +0.83% |
| Retrieval, 9 output tokens | 178.30 | 161.30 | -9.53% |

The retrieval continuation is too short for a reliable decode conclusion; its
unfavorable result is still retained. The arithmetic off samples were
155.4/154.2/155.8/161.6, versus 164.9/162.3/162.5/161.8 on. Python off samples
were 150.5/147.5/151.5/150.9, versus 152.2/151.7/152.4/151.6 on. Two starts and
four warmed samples per condition do not establish a confidence interval or a
general speed guarantee.

| Workload | Prefill tok/s, off → on | Wall seconds, off → on | TTFT seconds, off → on |
| --- | ---: | ---: | ---: |
| Arithmetic | 376.30 → 346.30 | 1.70195 → 1.64955 | 0.15836 → 0.16871 |
| Python | 329.15 → 304.55 | 0.87158 → 0.87640 | 0.14329 → 0.15401 |
| Retrieval | 2268.50 → 2231.90 | 0.57121 → 0.58343 | 0.52063 → 0.51972 |

Prefill regressed in all three cases. Arithmetic wall time improved, while
Python and retrieval wall time regressed. Draft acceptance also varied despite
identical final text: arithmetic off 569/656 versus on 578/648; Python off
240/288 versus on 241/302. This is a configuration-level comparison, not fixed
amounts of kernel work. The extra build controls matched outputs but are only
one start per binary and are not an independent build-speed claim.

The [complete numeric export](../evidence/published-benchmarks/2026-10-31-fusion.json)
contains all six starts, all 54 requests, telemetry, first repetitions, correctness
and paired summaries. Raw prompts, responses, configurations and logs remain
local. Do not multiply this experiment's gains by the earlier +66.0%/+58.8%
[llama.cpp comparison](IQ4XS_MTP_RESULTS_2026-10-30.md): session rates differ,
and llama.cpp was not retested in this fusion trial.

## Reproduce

Build the pinned patched source in an x64 developer shell using the
[setup guide](IQ4XS_SETUP.md), selecting `-BuildDirectory build-fused`.
Build and run `poly_norm_q8_test` before the model checks. The supplied model,
tokenizer and CUDA runtime paths must resolve on the test PC.

```powershell
python tools/validate_iq4xs_resident.py --engine <build-fused>/strata/strata-poly.exe --model <model.gguf> --prompt-file tests/fixtures/iq4xs-prompt.ids --follow-file tests/fixtures/iq4xs-follow.ids --candidate fused-norm-q8 --output <new-correctness-directory>
python tools/profiling/run_fusion_abba.py --config <saved-window-4-config.json> --engine <build-fused>/strata/strata-poly.exe --reference-engine <old-build>/strata-poly.exe --fusion-env STRATA_SAGE_FUSED_NORM_Q8 --output-dir <new-benchmark-directory>
python tools/export_optimization_numeric.py --folder <benchmark-directory> --correctness <correctness-directory>/results.json --stem <numeric-export-name>
```

The benchmark harness requires `psutil` in addition to the server's Python
dependencies. It binds localhost and stops only its own process tree. New
output directories are required. Rebuilt binaries and other models need their
own validation; the old measured hashes are identities, not promised rebuild hashes.
