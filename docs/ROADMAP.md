# Sage-A1.5 unified runtime roadmap

The current product direction is one Strata-owned inference runtime on native
Windows 11 / TU102 SM75 / RTX 2080 Ti 22GB, with NInfer-derived operations and
independently qualified Windows boot unlock capability. This supersedes the
older coordinated standalone-backend proposal in repository history.

Implementation milestones are pushed individually. A source milestone does not
pass a hardware gate. Keep original routing, all selected experts and GGUF
quantization; compare numerical behavior before accepting performance changes.

| Gate | Required evidence | Current status |
| --- | --- | --- |
| G1 Sage baseline | Exact Q4_K_M and Windows binary, 32K real prompt plus output on target GPU, reference logits, misses/residency and latency | Pending Windows machine |
| G2 Kernel integration | Kernels read Strata expert slots without second bank; independent numerical and model parity on GPU | Direct-slot implementation in progress; not qualified |
| G3 Scheduler integrity | Actual CUDA/CPU completion, eviction, dynamic KV and graph pointer lifetime tests | Host policy tests pass; live event integration pending |
| G4 Long context | 64K, 128K, 256K actual input consumption and document/retrieval quality within memory budget | Not run |
| G5 Windows unlock | Boot implementation/recovery, post-driver matched MMA correctness and throughput | Host research only; hardware adapter pending |
| G6 Unified tuning | Same-model locked/unlocked matched end-to-end tests including transfer/CPU costs | Blocked on G2 and G5 |
| G7 Release | Native API, MTP validation, endurance, reproducible package and rollback | Pending |

Model identity and source revisions are pinned in `config/`. Tests and observed
limitations are recorded in `docs/VALIDATION.md`. Linux build checks are development
checks and do not qualify the Windows product.
