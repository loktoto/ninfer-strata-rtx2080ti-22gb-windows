# Pushed implementation milestones

Source milestones are distinct from hardware acceptance gates in ROADMAP.md.

| Commit | Delivered | Evidence |
| --- | --- | --- |
| `0a631e0` | Pinned Sage/Strata baseline, linked RMSNorm candidate, host slot policies, inventory/launch/build tools | Local 2 C++ suites / 16 Python tests; hosted Windows C++ and Linux passed; Windows path expectation subsequently corrected |
| `4fbf077` | Direct cached Q4_K expert-slot kernel and dispatcher hook; unchanged mixed-format fallback | Local 4 C++ suites / 19 Python tests; selected CUDA SM75 objects and both oracle executables compile/link |
| `90c5235` | Pinned Windows/Linux CUDA compiler setup and hosted compile workflow | Linux installer exercised; Windows archive layout inspected; hosted CI blocked by account billing |
| `aaeab2e` | Exact token-file input for long Windows prompts and continuations | Host parser tests include 262144 IDs; all three patches apply and reapply safely |
| `9307e76` | Strict local API launcher and baseline/candidate diagnostic runner | 51 Python tests; actual GGUF header tokenizer extraction and five round trips; Windows inference pending |

The complete runtime at `9307e76` subsequently passed all 127 Linux SM75
compile/link steps and non-inference CLI smoke checks. See FULL_RUNTIME_BUILD.md
for the artifact and toolchain evidence. The Windows GPU workflow for that commit
was dispatched and remains queued at the last observation.

GPU execution remains required to pass G2. Live event-backed scheduler integration and Windows boot
unlock remain unfinished; no source milestone promotes those gates.

2026-10-28 Terra research and testing milestones:

| Commit | Delivered | Evidence |
| --- | --- | --- |
| `72c2c5d` | Repeatable overlapping patch series | Partial/complete reruns, CRLF and divergent edit tests |
| `0c59b84` | Complete teacher-forced FP32 logit capture/comparison | Writer and comparator tests; full runtime rebuilt |
| `fff3aa5` | Opt-in live cache ownership audit | Source ordering review; full runtime compiled; GPU stress pending |
| `2dd2a8a` | Researched stage runner and acceptance evidence review | Host tests; no synthetic hardware approval |
| `ea87bd9` | Optional candidate enqueue evidence | Fallback/no-enqueue and once-only trace tests; runtime rebuilt |
| `8299f89` | Native SM75 FP16 MMA probe | Compiles/links; no GPU execution |

Concurrent privacy-guard commit `3a09615` was preserved by merge `ed06faa`.
See TEST_READINESS.md for unimplemented or hardware-blocked acceptance work.
