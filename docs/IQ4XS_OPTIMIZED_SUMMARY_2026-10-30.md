# Resident IQ4_XS optimization: verified build and results

The opt-in NInfer-all-derived IQ4_XS implementation now intercepts both host-routed expert slots and resident GPU-ID decode. The previous adapter missed the resident path; its earlier exact-logit result did not establish candidate coverage throughout decode. The original report contains a coverage correction.

The scalar inner loop was replaced with packed byte-permutation decoding and integer dp4a. Factor ordering and warp reduction preserve native results for the tested model shapes. The NInfer-all provenance and Apache-2.0 attribution remain in the source and NOTICE. This embeds an adapted kernel in Strata; it does not link a second inference engine.

A separate clean Ninja build used Release, CUDA architecture 75, optimized MSVC flags and no CUDA device debug flag. It passed 122 GPU oracle cases, the adapter check, exact native kernel comparisons with varied scales, and a full-model comparison of 3,973,120 logits. Device counters recorded 16 executions for each projection in every one of 40 layers. The full-model comparison had zero logit differences. All 88 Python tests passed. These are bounded checks, not general model-quality qualification or an independent graph-launch timeline.

The same clean executable ran native/candidate/candidate/native with graphs allowed, tracing disabled, FP16 KV, fixed expert residency, no speculation and 32K configured capacity. Each start ran three repetitions. Excluding its first repetition leaves four samples per variant across two starts:

| Workload | Native | Candidate | Change |
| --- | ---: | ---: | ---: |
| Arithmetic decode | 107.85 tok/s | 115.00 tok/s | +6.63% |
| Python decode | 108.60 tok/s | 115.95 tok/s | +6.77% |
| Retrieval prefill | 2233.65 tok/s | 2213.65 tok/s | -0.90% |

All response text and token counts matched in this trial. The small workload set and two starts per variant do not establish a confidence interval. Retrieval generated too few tokens for a meaningful decode comparison. A 40% whole-model uplift was not reached. Further optimization needs a representative GPU timeline; isolated expert-kernel timings do not measure the remaining operations' runtime share.

Set `STRATA_SAGE_EXPERT_IQ4XS=1` in an otherwise graph-enabled Strata launch. Leave `STRATA_SAGE_RMSNORM=0`; the older RMSNorm candidate has unresolved full-model numerical differences. Both candidates remain off by default. CPU experts, other quantizations and batched prefill use native fallbacks.

Reproduce a clean build from the pinned dependencies in an x64 Visual Studio developer shell with CUDA available:

```powershell
python tools/bootstrap.py
cmake -S . -B build-optimized -G Ninja -DCMAKE_BUILD_TYPE=Release -DSAGE_ENABLE_CUDA=ON -DSAGE_BUILD_STRATA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-optimized --target strata-poly sage_iq4xs_slots_parity strata_expert_iq4xs_test sage_iq4xs_ids_bench --parallel 4
.\build-optimized\kernels\sage_iq4xs_slots_parity.exe
.\build-optimized\strata_expert_iq4xs_test.exe
```

`tools/validate_iq4xs_resident.py` accepts local engine/model/token paths and writes the logits and execution counters locally. Raw prompts, model metadata, machine paths and raw benchmark dumps are excluded from this publication.
