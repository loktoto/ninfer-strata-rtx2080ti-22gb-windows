# Sage-A1.5 Windows Unified Engine

One Strata-owned runtime for Sage-A1.5 Q4_K_M on Windows 11 x64 and the
RTX 2080 Ti 22GB (SM75), with selected NInfer-derived CUDA operations.

**Development implementation; hardware acceptance is pending.** This repository
does not yet provide a qualified Windows executable. No throughput or 256K-context success is claimed.

**Current scope: Strata + NInfer on normal SM75. Turing Unlock is deferred**
by the project owner; it is not a dependency of this release. Preserve its
research for a later phase. See the [active roadmap](docs/ROADMAP.md).

The owner's current test target is **Sage-A1.5 IQ4_XS, MTP4, 32K configured
capacity**. Read [IQ4_XS testing readiness](docs/IQ4XS_TESTING.md) before testing:
current strict launchers disable MTP and the original Q4_K_M model lock cannot
qualify a different IQ4_XS file. Developer smoke testing is prepared; a matched
IQ4_XS/MTP4 performance comparison is not yet prepared.

## What is implemented

- Pinned PolyStrata integration patch and reproducible source preparation.
- An opt-in NInfer-derived FP32 RMSNorm kernel using Strata-owned buffers/stream,
  plus independent numerical GPU parity tests. Default execution keeps Strata's
  original kernel until target validation permits selecting the new path.
- A strict baseline launcher that disables expert skipping and speculative decode.
- An opt-in Q4_K expert-slot GEMV path using Strata-owned Q8_1 scratch and its
  existing stream, with independent GPU oracle tests; numerical qualification pending.
- An opt-in Q6_K expert path adapted from NInfer-all, borrowing the same slots
  without converting weights.
- Host slot-generation/reader-lease and KV-budget contracts, tested independently;
  these are not yet connected to Strata's CUDA events or live eviction logic.
- An official Sage model lock and bounded GGUF tensor inventory tool.
- Windows hardware inventory capture and host test CI for Windows/Linux.

The unlock repository is retained in a separate deferred research lock only. Its current host parser tests
are not hardware unlock evidence. See [architecture and gates](docs/ARCHITECTURE.md).

## Host development checks

Requires Python 3.10+, CMake 3.28+, and a C++20 compiler. These tests need no GPU
or model download. Linux is used for development checks, not as the product target.

```text
python -m pip install cmake ninja
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
python -m unittest discover -s tests -p "test_*.py" -v
```

## Build the Windows inference runtime

Use an x64 Visual Studio developer PowerShell with CUDA's compiler on PATH.
The CUDA/MSVC combination still requires target-machine qualification. The build
script installs no drivers and changes no boot or watchdog settings.

```powershell
python -m pip install cmake ninja
.\tools\build_windows.ps1
```

This fetches the pinned Strata source into `.deps/PolyStrata`, applies the local
patch, configures `SAGE_ENABLE_CUDA=ON`, `SAGE_BUILD_STRATA=ON`, and architecture
`75`, then builds `strata-poly`. Strata fetches its own pinned ggml dependency.
The executable is normally `build-windows/strata/strata-poly.exe` with Ninja.
The build links one inference engine; it does not launch a second NInfer engine.

Build and run the separate numerical kernel check on the SM75 machine:

```powershell
cmake --build build-windows --target sage_rmsnorm_parity sage_q4k_slots_parity sage_q6k_slots_parity --parallel 2
.\build-windows\kernels\sage_rmsnorm_parity.exe
.\build-windows\kernels\sage_q4k_slots_parity.exe
.\build-windows\kernels\sage_q6k_slots_parity.exe
```

Then follow the [strict launch instructions](integration/strata/README.md).
For the local OpenAI-compatible API, follow [native API startup](docs/API.md).
For full prompt/continuation diagnostic capture, use the
[Windows model-validation procedure](docs/MODEL_VALIDATION.md).
Keep the original kernel selected for the baseline; opt into the candidate only
for matched correctness/performance comparisons. A manual self-hosted Windows
workflow is included for a runner labelled `sm75`; it expects the same developer
toolchain on PATH and does not publish a release.

For a compiler-only setup without installing a GPU driver, Python 3.12 can fetch
the pinned CUDA 13.0.2 components (SHA256 checked) with:

```text
python tools/setup_cuda.py --directory .cuda
```

The destination must be absent or empty. The hosted CUDA workflow uses this
setup to compile the selected kernel library and numerical executables on Windows
and Linux. It does not execute GPU tests or build/qualify the complete Strata
runtime. Hosted execution currently requires resolving GitHub's account
payments/spending-limit block reported for commit `4fbf077`.

For a manually prepared source tree, use `-DSAGE_STRATA_SOURCE_DIR=...`. The tree
must match the source lock and have the integration patch applied.

## Model identity

The official artifact is
[InternScience/Sage-A1.5-Q4_K_M-GGUF](https://huggingface.co/InternScience/Sage-A1.5-Q4_K_M-GGUF/tree/10b179b52f2c841030db327627e707d076b4c8cf).
Use the revision and expected SHA256 in [model.lock.json](config/model.lock.json).
The 21.7 GB model is not bundled or automatically downloaded by setup.
The [inspected official header](config/sage-gguf-inventory.json) confirms
`qwen35moe` with mixed Q4_K/Q6_K/F32 tensors. It is a header inventory, not a
completed model correctness test.

Inspect a local file without loading its tensors:

```powershell
python tools/inspect_gguf.py D:\models\Sage-A1.5-Q4_K_M.gguf
```

Inventory is not a checksum verification or a numerical inference test. Capture
full file hashes, binary hash, GPU/driver and compiler information separately:

```powershell
python tools/capture_baseline.py --model D:\models\Sage-A1.5-Q4_K_M.gguf --engine build-windows\strata\strata-poly.exe --source-lock config\sources.lock.json --output evidence\p0-001.json
```

That collector only runs on native Windows and never marks an inference gate as
passed. It hashes full files, so allow time for the model read.

## Project scope

Retain original routing and all selected experts; retain the GGUF's mixed tensor
formats. “No expert skipping” does not mean Q4 quantization is lossless. Adapt
NInfer operations at borrowed-buffer boundaries, not through its separately owning
Engine or incompatible packed Q4 loader. Long-context performance requires target evidence. GPU unlock research and
unlocked autotuning are deferred; current integration and performance testing
use the normal SM75 path.

Source attribution and license boundaries are documented in [NOTICE](NOTICE).
The local checks and unverified gates are recorded in [validation](docs/VALIDATION.md).
The [complete Linux development build](docs/FULL_RUNTIME_BUILD.md) also passed
for the patched SM75 runtime, including both linked candidate operations.

Current [testing readiness and blockers](docs/TEST_READINESS.md),
[Terra research](docs/RESEARCH_FINDINGS.md), [full-logit validation](docs/LOGIT_VALIDATION.md)
and [staged testing procedure](docs/TESTING.md) describe the next target-machine
work. Hardware acceptance and a qualified Strata + NInfer release remain pending.
Windows unlock is outside the current release scope.

The [Q6_K donor integration](docs/Q6K_DONOR.md) is selected with
`--ninfer-expert-q6k` in strict/API launchers. Validation accepts
`--candidate expert-q6k` or `--candidate all`; `both` retains RMSNorm plus Q4_K.
All candidates remain off by default. The [Windows testing packager](docs/WINDOWS_PACKAGE.md)
resolves transitive DLL imports for an unqualified development ZIP.
