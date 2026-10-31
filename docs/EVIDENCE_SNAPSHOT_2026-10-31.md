# Evidence preservation — 31 October 2026

The [evidence index](../evidence/published-benchmarks/2026-10-31-local-evidence-index.json)
records 365 files in a local snapshot: raw benchmark requests/responses, launch
configs, server logs, numerical evidence, source patches, tests and documentation.
Each entry has a repository-relative path, byte count and SHA-256. The archive
itself has a SHA-256 and passed ZIP CRC verification. It is approximately 349 MB.
It preserves the files at the recorded capture time; ongoing build work is a
separate, later result. The full model, dependency checkouts and build trees are
not duplicated in this snapshot. Their revisions and the measured executable
identity are recorded in the relevant reports.

The raw archive is saved locally under `evidence-archives`; its exact filename
and digest are in the index. GitHub contains the numerical exports, findings,
methodology, source changes and the digest index. Raw prompt/response text and
machine-specific launch paths remain in the local originals and archive.

The [earlier evidence audit](EVIDENCE_AUDIT_2026-10-30.md) accounts for the data
from October 29 that had not previously been uploaded. The [MTP study](IQ4XS_MTP_RESULTS_2026-10-30.md)
adds all 12 later runs, including draft-window exploration, the slower llama.cpp
MTP control and the fresh no-draft control. Its group name is October 30, while
the bracketed comparison crossed midnight into October 31 in Hong Kong.

`python tools/save_evidence_snapshot.py` creates another timestamped local
archive and updates the digest index. It never replaces an existing archive.
`python tools/export_benchmark_evidence.py` regenerates the earlier numeric
exports; `python tools/export_mtp_numeric.py` regenerates the draft study's
samples, telemetry and summaries from the retained originals.

At this snapshot's capture time, the separate Sage Strata fork was still being
built and the standalone NInfer port was incomplete. The later
[fork build and smoke report](STRATA_SAGE_FORK_BUILD_2026-10-31.md) and
[standalone progress](NINFER_STANDALONE_PROGRESS_2026-10-31.md) distinguish
subsequent observed results from source-only work; neither is assigned an
unmeasured speed.
