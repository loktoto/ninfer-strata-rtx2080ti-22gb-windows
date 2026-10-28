# Testing readiness and remaining blockers — 2026-10-28

This is a development system prepared for target testing. It is not a completed
Strata + NInfer release, and no hardware gate is marked passed. The owner has
deferred Turing Unlock: active gates are G1–G4/G7; G5/G6 are outside this release.

The owner now reports IQ4_XS/MTP4 at a 32K configured limit. This differs from
the locked Q4_K_M/non-MTP validation path. See [IQ4XS_TESTING.md](IQ4XS_TESTING.md)
for the supported smoke test and remaining model/format/speculation gaps.

Implemented and locally checked:

- One compiled Strata-owned runtime, with opt-in NInfer-derived FP32 RMSNorm
  and direct cached Q4_K/Q6_K slot operations. No second expert weight bank.
- Exact long token-file inputs; full FP32 teacher-forced logit capture and
  comparison; optional evidence that selected dispatch paths were enqueued.
- Opt-in live cache ownership auditing at allocation/publication/resize
  boundaries, reusing Strata's existing stream and event ordering.
- A staged short/32K/64K/128K/256K validation plan, separate non-diagnostic
  performance runs, local API checks and evidence review tooling.
- A true native SM75 FP16 MMA probe for matched post-driver experiments.
  Labels never certify lock state. Output correctness/timing are untested here.
- Linux CUDA 13 SM75 compile/link checks for the complete patched runtime and
  probe and Q6_K oracle; nine host C++ suites and 88 Python tests passed
  after the acceptance-scope and IQ4_XS inventory updates.
- A Windows development ZIP packager with recursive DLL import checks,
  source fingerprints and per-file hashes. Native packaging remains untested.

Read [DONOR_REVIEW.md](DONOR_REVIEW.md) for the current mr-september and
NInfer-all comparison, RESEARCH_FINDINGS.md for unlock research and references,
SCHEDULER_VALIDATION.md for the ordering analysis, and TESTING.md for the fixture
manifest and test procedure. The hardware workflow prepares Python/MSVC build
tools and retains raw kernel outputs plus tested binary hashes as an artifact.
Its MMA diagnostic is optional and disabled by default; it is not an unlock test.

The following still prevent full completion:

| Requirement | Remaining work |
| --- | --- |
| Native Windows target | An available Windows 11 x64 runner with the physical TU102 22GB SM75, full CUDA toolkit and working NVIDIA driver |
| Exact Sage model | Full locked GGUF and checksum, representative exact-token fixtures, independent same-GGUF reference outputs |
| G1 / G2 | Actual 32K input consumption, GPU operator oracles and complete-logit comparisons on that target; source/build success is insufficient |
| G3 | GPU delayed-copy and failure injection, adaptive eviction/KV relocation and graph replay validation; ownership audit alone is insufficient |
| G4 | Real long-document/retrieval quality assertions and reviewed free-generation outputs at each stage; the runner records expectations but does not score semantics |
| G7 | Target MTP correctness, streaming/concurrency, endurance, Windows package/DLL and rollback qualification remain pending |

GitHub hosted jobs previously could not start because of the account
payments/spending-limit restriction. The previously dispatched Windows workflow
was still queued at the last observation; a queued job supplies no GPU evidence.
The latest source must be tested by its exact commit, not an older queued revision.

Do not flash firmware or change boot/watchdog settings as part of these tests.
The supplied runtime/probe/workflows perform none of those operations. G5/G6
are deferred; no Windows unlock work or qualification is required for the current
Strata + NInfer release. A future unlock phase needs separate hardware validation.
