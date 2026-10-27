# Validation — 2026-10-27

This is an initial source implementation, not a passed P1 or release milestone.

| Check | Observed result |
| --- | --- |
| Linux GCC 14.2 / CMake 4.4.4 Release host build | Passed |
| CTest slot-pool and strict routing policy suites | 4/4 passed |
| Python GGUF inspector and strict launch contracts | 19/19 passed |
| Slot policy strict compiler warnings | Passed |
| Slot policy ASan + UBSan | Passed |
| Patch against locked PolyStrata commit | Applied successfully |
| Repeated bootstrap | Recognized existing patch; no second application |
| Official Sage GGUF metadata | Parsed bounded 16 MiB prefix; provenance recorded |
| Selected SM75 CUDA kernels and both GPU oracle executables | Compile/link passed, NVIDIA CUDA 13.0.88, GNU 14.2 |
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

Next hardware step: native Windows build, kernel oracle, short-prompt same-GGUF
reference comparison, then a real 32768-token prompt with output allowance. Do
not interpret compilation or successful model loading as passing that sequence.
