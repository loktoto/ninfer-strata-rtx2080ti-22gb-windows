# Pushed implementation milestones

Source milestones are distinct from hardware acceptance gates in ROADMAP.md.

| Commit | Delivered | Evidence |
| --- | --- | --- |
| `0a631e0` | Pinned Sage/Strata baseline, linked RMSNorm candidate, host slot policies, inventory/launch/build tools | Local 2 C++ suites / 16 Python tests; hosted Windows C++ and Linux passed; Windows path expectation subsequently corrected |
| `4fbf077` | Direct cached Q4_K expert-slot kernel and dispatcher hook; unchanged mixed-format fallback | Local 4 C++ suites / 19 Python tests; selected CUDA SM75 objects and both oracle executables compile/link |

Compiler automation is a subsequent source milestone: pinned cross-platform
CUDA components and a hosted compiler-only workflow. GPU execution remains
required to pass G2. Live event-backed scheduler integration and Windows boot
unlock remain unfinished; no source milestone promotes those gates.
