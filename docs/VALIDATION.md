# Validation — 2026-10-27

This is an initial source implementation, not a passed P1 or release milestone.

| Check | Observed result |
| --- | --- |
| Linux GCC 14.2 / CMake 4.4.4 Release host build | Passed |
| Host CTest suites including logit writer and dispatch traces | 7/7 passed |
| Python inspector, launch, API, full-logit, suite and evidence contracts | 72/72 passed |
| Slot policy strict compiler warnings | Passed |
| Slot policy ASan + UBSan | Passed |
| Patch against locked PolyStrata commit | Applied successfully |
| Repeated bootstrap | Recognized existing patch; no second application |
| Official Sage GGUF metadata | Parsed bounded 16 MiB prefix; provenance recorded |
| Selected SM75 CUDA kernels and both GPU oracle executables | Compile/link passed, NVIDIA CUDA 13.0.88, GNU 14.2 |
| Complete patched strata-poly runtime, Linux development build | All 127 build/link steps passed at 9307e76; both candidate symbols linked |
| Runtime help and missing token-file error | Passed with official driver stub for loader only; no inference |
| Windows MSVC / CUDA compile | Not run |
| RMSNorm numerical GPU oracle | Implemented, not run |
| Sage logits, 32K inference, API, performance | Not run |
| 64K / 128K / 256K quality | Not run |
| Windows unlock / post-driver MMA | Not implemented or verified |

Host tests validate policy behavior, not actual GPU completion or race freedom.
The scheduler policy library is not connected to Strata's live event/eviction path.
The second patch adds direct cached Q4_K slot kernels at gate/up and down
projections; GPU numerical execution remains untested. New kernels remain opt-in, and baseline skips remain forbidden.

The official GGUF payload was not downloaded in full. The model lock's SHA256
comes from the publisher's LFS metadata; only the downloaded header-prefix hash
was verified here. Header agreement does not establish numerical compatibility.

First pushed host CI passed Windows C++ build/tests and Linux checks; one Windows
Python expectation compared a short temporary path to its resolved long form.
The expectation now uses canonical paths, matching the existing launcher behavior.

Commit `4fbf077` contains the direct-slot milestone. Its GitHub jobs could not
start: the check annotation reports account payments/spending-limit restrictions.
This is a CI availability block, not a compiler/test result. The new compiler-only
workflow remains unexecuted on hosted Windows until that account issue is resolved.

The pinned CUDA 13.0.2 redistributable installer was exercised on Linux. Both
kernel objects and both parity executables compile/link for SM75 with nvcc
13.0.88 and GCC 14.2. Running the Q4_K GPU oracle here fails explicitly because
the CUDA driver/runtime is unavailable; no GPU numerical pass is recorded.

The pinned upstream tokenizer extractor also successfully read the verified
16 MiB Sage header prefix. It exported vocabulary 248320, merges 247587 and
the 9025-byte GGUF chat template (SHA256
`0080393e13df7322bed792a78e75daf8a0374a80928996e5cf3cfd2ab4c1e1be`).
Five encode/decode round trips passed for English, Cantonese, emoji, combining
marks, source code and whitespace using regex 2026.9.29. This is extraction and
round-trip evidence, not tokenizer parity with an independent reference or a
test of the model's reasoning quality.

See FULL_RUNTIME_BUILD.md for the complete runtime build command, artifact
fingerprint and exact scope of the driver-stub CLI checks.

On 2026-10-28 the runtime rebuilt and linked after all six integration patches,
including full-logit capture, live ownership auditing and dispatch tracing. Its
help smoke printed `--logits-file`; the official driver stub was used solely for
loader resolution. The native FP16 MMA probe also compiled/linked for SM75 and
exited nonzero at device discovery because no working GPU driver is present.
No new hardware gate is established by these checks.

Next hardware step: native Windows build, kernel oracle, short-prompt same-GGUF
reference comparison, then a real 32768-token prompt with output allowance. Do
not interpret compilation or successful model loading as passing that sequence.
