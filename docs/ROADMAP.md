# Sage-A1.5 unified runtime roadmap

The current product direction is one Strata-owned inference runtime on native
Windows 11 / TU102 SM75 / RTX 2080 Ti 22GB, with NInfer-derived operations on the normal SM75 path.

Scope decision, 2026-10-28: the owner deferred Turing Unlock. Current release
acceptance covers G1–G4 and G7. Preserve G5/G6 identifiers for the deferred
unlock work; they do not block this release. Kernel selection and performance
comparisons on normal SM75 remain part of G2/G7. This supersedes the
older coordinated standalone-backend proposal in repository history.

Implementation milestones are pushed individually. A source milestone does not
pass a hardware gate. Keep original routing, all selected experts and GGUF
quantization; compare numerical behavior before accepting performance changes.

| Gate | Required evidence | Current status |
| --- | --- | --- |
| G1 Sage baseline | Exact Q4_K_M and Windows binary, 32K real prompt plus output on target GPU, reference logits, misses/residency and latency | Pending Windows machine |
| G2 Kernel integration | Kernels read Strata expert slots without second bank; independent numerical and model parity on GPU | Direct-slot kernels, full-logit comparison and dispatch evidence implemented; GPU qualification pending |
| G3 Scheduler integrity | Actual CUDA/CPU completion, eviction, dynamic KV and graph pointer lifetime tests | Existing barriers reviewed; live ownership audit implemented; GPU stress validation pending |
| G4 Long context | 64K, 128K, 256K actual input consumption and document/retrieval quality within memory budget | Staged runner prepared; quality/hardware tests not run |
| G5 Windows unlock | Future boot implementation/recovery and matched post-driver MMA evidence | Deferred by owner; excluded from current release |
| G6 Unlock-aware tuning | Future same-model locked/unlocked comparisons including transfer/CPU costs | Deferred with G5; excluded from current release |
| G7 Release | Native API, MTP validation, endurance, reproducible package, rollback and matched normal-SM75 performance | API and test tooling implemented; release qualification pending |

Model identity and source revisions are pinned in `config/`. Tests and observed
limitations are recorded in `docs/VALIDATION.md`. Linux build checks are development
checks and do not qualify the Windows product.

Current implementation order:

1. Establish native Windows short-input correctness and the exact-model 32K baseline.
2. Qualify RMSNorm and direct Q4_K/Q6_K expert-slot candidates independently,
   then together; retain the original kernels unless correctness and repeated
   end-to-end measurements justify selection.
3. Validate CPU/GPU completion, KV growth, eviction and graph lifetimes on target.
4. Scale context one stage at a time, then qualify API, MTP, endurance and the
   packaged Windows executable. Context capacity is a test target, not a guarantee.

New acceptance manifests use `schema_version: 2`, `profile: "strata_ninfer"`,
all seven gate keys, and `status: "deferred"` for G5/G6. Active gates still need
real native Windows evidence. Legacy schema-version-1 manifests retain the
original three-component contract and its matched unlock evidence requirements.
Neither schema grants automatic acceptance.

A starting evidence manifest is provided in
[`config/acceptance-strata-ninfer.json`](../config/acceptance-strata-ninfer.json).
It declares every active gate `not_run`; add actual evidence before submitting
any pass to `python tools/check_acceptance.py <manifest>`. Inventory and staged
suite reports also identify the current profile, but are separate report formats
and are not acceptance manifests.
