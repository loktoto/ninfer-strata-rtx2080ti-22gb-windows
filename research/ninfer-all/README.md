# Experimental NInfer-all Sage port

Both patches apply to a fresh checkout of NInfer-all commit `71d3f8190772ef7089f8eda4e22371114fd5bbf9`. They are partial research work, not a complete converter or standalone Sage inference runtime. The CPU patch replaces the earlier nine-test snapshot with the current 17-test adapter. Use a fresh pinned checkout rather than applying it over the old patch.

```powershell
git apply --check <path-to-sage-iq4xs-cpu-experiment.patch>
git apply <path-to-sage-iq4xs-cpu-experiment.patch>
git apply --check <path-to-sage-gguf-operator-experiment.patch>
git apply <path-to-sage-gguf-operator-experiment.patch>
```

Use Python 3.11 with NInfer's converter dependencies (CPU PyTorch, NumPy, safetensors and gguf) in an isolated environment. Set `NINFER_TEST_SAGE_GGUF` to the explicitly selected local Sage IQ4_XS GGUF to run real-weight checks; without it those checks skip. The artifact test needs hard-link support.

```powershell
python -m unittest tests.convert.test_sage_gguf_experts tests.convert.test_sage_iq4xs_oracle -v
```

The CPU patch preserves sampled expert block bytes and checks other candidate tensor mappings. It neither produces a complete artifact nor qualifies a model. The separate operator patch adds a Sage-shaped staged GGUF operation and a minimal numerical test executable. It also extracts common code used by the original Flash-Next operator; that original translation unit is compiled as a regression check, but its full runtime has not been retested.

In an x64 Visual Studio developer shell with CUDA 13.0.88 and CMake/Ninja:

```powershell
cmake -S experiments/sage-op-compile -B build-sage-op -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 `
  -DCMAKE_CUDA_COMPILER=<cuda-root>/bin/nvcc.exe `
  -DCUDAToolkit_ROOT=<cuda-root> -DSAGE_LINK_MINIMAL=ON
cmake --build build-sage-op --target sage_moe_minimal flash_moe_regression_probe --parallel 1
.\build-sage-op\sage_moe_minimal.exe
```

Make the corresponding CUDA DLLs available to this process. The test does not load a model or require FFmpeg, but compiles the real GGUF dispatch dependencies and therefore takes substantially longer than its GPU execution. A single build job limits memory pressure on the 16 GB host. See the [progress report](../../docs/NINFER_STANDALONE_PROGRESS_2026-10-31.md) for the numerical criteria, results, build warnings and remaining integration.
