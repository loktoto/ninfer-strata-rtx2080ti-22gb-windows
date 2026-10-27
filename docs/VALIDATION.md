# Validation — 2026-10-27

This is an initial source implementation, not a passed P1 or release milestone.

| Check | Observed result |
| --- | --- |
| Linux GCC 14.2 / CMake 4.4.4 Release host build | Passed |
| CTest slot-pool and strict routing policy suites | 2/2 passed |
| Python GGUF inspector and strict launch contracts | 16/16 passed |
| Slot policy strict compiler warnings | Passed |
| Slot policy ASan + UBSan | Passed |
| Patch against locked PolyStrata commit | Applied successfully |
| Repeated bootstrap | Recognized existing patch; no second application |
| Official Sage GGUF metadata | Parsed bounded 16 MiB prefix; provenance recorded |
| CUDA configuration in current workspace | Blocked: `Failed to find nvcc` |
| Windows MSVC / CUDA compile | Not run |
| RMSNorm numerical GPU oracle | Implemented, not run |
| Sage logits, 32K inference, API, performance | Not run |
| 64K / 128K / 256K quality | Not run |
| Windows unlock / post-driver MMA | Not implemented or verified |

Host tests validate policy behavior, not actual GPU completion or race freedom.
The scheduler policy library is not connected to Strata's live event/eviction path.
The RMSNorm patch links a candidate operation; it does not implement direct expert
slot kernels. New kernels remain opt-in, and baseline skips remain forbidden.

The official GGUF payload was not downloaded in full. The model lock's SHA256
comes from the publisher's LFS metadata; only the downloaded header-prefix hash
was verified here. Header agreement does not establish numerical compatibility.

Next hardware step: native Windows build, kernel oracle, short-prompt same-GGUF
reference comparison, then a real 32768-token prompt with output allowance. Do
not interpret compilation or successful model loading as passing that sequence.
