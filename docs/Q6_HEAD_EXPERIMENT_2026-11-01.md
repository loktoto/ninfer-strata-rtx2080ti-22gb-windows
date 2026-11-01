# Packed Q6_K output head: tested, not promoted

The default and tuned packed-head configurations preserved the checked outputs,
but did not provide a consistent speed benefit on this RTX 2080 Ti 22 GB. The
normal launcher keeps this experiment disabled. No end-to-end gain is claimed.

The GPU profile identified substantial native Q6_K matrix work. Source inspection
showed the kernels already use packed integer dot products; the earlier scalar
expert adapter would not be a suitable dense replacement. An existing native
Strata optimization repacks a Q6_K matrix into separate planes. PolyStrata's
output-head path had not registered that representation, although its other
engine already did. We tested it rather than assuming it would transfer well.

Patch `0011-poly-q6-packed-head.patch` registers the real 248320 × 2048 Q6_K
output matrix after upload, before expert-arena sizing and graph capture, when
`STRATA_Q6_PACKED=1`. The original GGUF layout remains available. Teardown joins
the streams, destroys captured graphs, then unregisters/frees the packed copy.
Preparation failure is reported rather than silently claiming the candidate ran.
With the flag off, the existing path and teardown order remain selected.

The isolated Release/SM75 `build-head` build completed 131 steps with CUDA
13.0.88, MSVC 19.44 and the same pinned PolyStrata/ggml sources as the fusion
study. Executable SHA-256:
`1a8a823e933f5c5491dab9da84b12ea0f6d9ba3f647c969cdefdf17d73421de0`.
The measured `build-optimized` and `build-fused` executables were preserved.

## Correctness and resource cost

Both grid 640 and grid 20480 matched all 3,973,120 real-model logits at the same
16 teacher-forced positions exactly, with IQ4_XS enabled and norm fusion off.
Their dumps also match the earlier native/candidate dump hash. The load-time
self-test compared every output row at one through four columns byte for byte.
These are bounded checks, not full-context or broad quality qualification.

The packed copy requires 417,177,600 additional device bytes (397.9 MiB), while
packing temporarily holds two similarly sized CPU buffers, approximately
796 MiB combined, plus self-test scratch. The model file is unchanged. This
memory cost matters even when a microbenchmark is slightly faster.

## Measurements

The native self-test takes 11 alternating paired samples, each timing five
launches with CUDA events. Its medians below are microseconds per head call,
not tokens/s or whole-model latency. Random Q8_1 inputs and the real model's
output-head weights are used; this is a loaded-head benchmark with broad sample
ranges, not a steady-state model workload.

| Run / grid | Columns | Native median µs | Packed median µs |
| --- | ---: | ---: | ---: |
| Initial / 640 | 1 | 1073.2 | 1085.5 |
| Initial / 640 | 2 | 1162.4 | 1430.0 |
| Initial / 640 | 4 | 1711.7 | 1804.0 |
| Sweep process / 640 | 1 | 879.9 | 1095.3 |
| Sweep process / 640 | 2 | 1033.0 | 1423.8 |
| Sweep process / 640 | 4 | 1597.1 | 1827.5 |
| Follow-up / 20480 | 1 | 876.0 | 1045.7 |
| Follow-up / 20480 | 2 | 1074.3 | 1090.8 |
| Follow-up / 20480 | 4 | 1571.2 | 1524.5 |

An exploratory sweep tried grids 160, 320, 640, 1280, 2560, 5120 and 20480,
with seven samples per grid/column count. Grid 20480 looked promising for one
column (832.3 µs), but that sweep did not interleave a fresh native measurement
at every grid. The independent paired follow-up above did not reproduce a
one-column advantage: packed duration increased 19.4% at one column and 1.5%
at two, and decreased 3.0% at four. We therefore did not promote it or start an
end-to-end benchmark campaign based on the best exploratory sample.

The [numeric export](../evidence/published-benchmarks/2026-11-01-q6-head.json)
retains all 21 sweep entries, all nine paired summaries and their min/max ranges,
and both full-model correctness summaries. Raw logs remain local. There is no
throughput value for this candidate in the A/B/C/D table.

## Reproduce the bounded checks

Use a separate `build-head` directory with the source patches through 0011,
following the [Windows build guide](IQ4XS_SETUP.md). Run the validator with
the actual local model, first with `--q6-grid 640`, then a new output directory
with `--q6-grid 20480`:

```powershell
python tools/validate_iq4xs_resident.py --engine <build-head>/strata/strata-poly.exe --model <model.gguf> --prompt-file tests/fixtures/iq4xs-prompt.ids --follow-file tests/fixtures/iq4xs-follow.ids --candidate q6-head --q6-grid 20480 --output <new-directory>
```

It explicitly enables the head self-test in the candidate process and keeps
norm fusion off. The existing upstream `STRATA_Q6P_SELFTEST=2` enables the grid
sweep; its temporary grid overrides are reset afterward. Normal API/start
wrappers disable inherited head/self-test flags. The experiment is preserved
for research, not recommended as a faster setting.
