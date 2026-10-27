# Sage-A1.5 Windows Unified Engine

One Strata-owned runtime for Sage-A1.5 Q4_K_M on Windows 11 x64 and the
RTX 2080 Ti 22GB (SM75), with selected NInfer-derived CUDA operations.

**Development implementation; hardware acceptance is pending.** This repository
does not yet provide a qualified Windows executable or a verified Windows GPU unlock. No throughput or 256K-context success is claimed.

## What is implemented

- Pinned PolyStrata integration patch and reproducible source preparation.
- An opt-in NInfer-derived FP32 RMSNorm kernel using Strata-owned buffers/stream,
  plus independent numerical GPU parity tests. Default execution keeps Strata's
  original kernel until target validation permits selecting the new path.
- A strict baseline launcher that disables expert skipping and speculative decode.
- An opt-in Q4_K expert-slot GEMV path using Strata-owned Q8_1 scratch and its
  existing stream, with independent GPU oracle tests; numerical qualification pending.
- Host slot-generation/reader-lease and KV-budget contracts, tested independently;
  these are not yet connected to Strata's CUDA events or live eviction logic.
- An official Sage model lock and bounded GGUF tensor inventory tool.
- Windows hardware inventory capture and host test CI for Windows/Linux.

The unlock repository is tracked as research only. Its current host parser tests
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
cmake --build build-windows --target sage_rmsnorm_parity sage_q4k_slots_parity --parallel 2
.\build-windows\kernels\sage_rmsnorm_parity.exe
.\build-windows\kernels\sage_q4k_slots_parity.exe
```

Then follow the [strict launch instructions](integration/strata/README.md).
Keep the original kernel selected for the baseline; opt into the candidate only
for matched correctness/performance comparisons. A manual self-hosted Windows
workflow is included for a runner labelled `sm75`; it expects the same developer
toolchain on PATH and does not publish a release.

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
Engine or incompatible packed Q4 loader. GPU unlock and long-context performance
remain evidence-dependent workstreams, not launch-time assumptions.

Source attribution and license boundaries are documented in [NOTICE](NOTICE).
The local checks and unverified gates are recorded in [validation](docs/VALIDATION.md).
