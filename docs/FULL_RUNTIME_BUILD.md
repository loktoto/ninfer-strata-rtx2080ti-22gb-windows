# Complete runtime build verification — 2026-10-27

The complete patched `strata-poly` executable compiled and linked successfully
at project commit `9307e7626c4f8691a7849949d5bdb0cd4c952862`. All 127 build
steps passed. This was a Linux development check for SM75; native Windows
compilation and model execution remain separate acceptance requirements.

Environment: GCC 14.2, NVIDIA CUDA compiler 13.0.88 from CUDA 13.0.2,
CMake 4.4.4 and Ninja. PolyStrata and ggml revisions match `config/sources.lock.json`.
The official CUDA 13.0.2 cuBLAS redistributable was additionally installed in the
development toolkit after checking its published SHA256. No source changes were
needed to make the complete build pass.

Equivalent configuration with a complete CUDA toolkit:

```text
python tools/bootstrap.py
cmake -S . -B build-full -G Ninja -DCMAKE_BUILD_TYPE=Release -DSAGE_ENABLE_CUDA=ON -DSAGE_BUILD_STRATA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc -DCUDAToolkit_ROOT=/path/to/cuda
cmake --build build-full --target strata-poly --parallel 2
```

The compiler-only installer covers selected kernel builds. Building all of Strata
also requires the complete toolkit's cuBLAS headers/libraries. On Windows use
`tools/build_windows.ps1` with the real installed driver/toolkit and MSVC.

The resulting local executable was approximately 73 MiB. Its recorded SHA256 is
`ca9286cf164fcdb6459f3eb20869deb0a93b67e0c9e06dfa6e480b49b257e0ae`.
This identifies this build artifact, not a promise of byte-identical output from
other toolchains or source paths. It is not a Windows release binary.

Checks after linking:

- Symbol inspection confirmed both `sage::cuda::try_rms_norm_f32` and
  `sage::cuda::try_q4k_slots_q8_1` are defined in the executable.
- `--help` printed both token-file flags. Upstream deliberately returns exit 2
  for its help/usage branch; that expected status was retained.
- A missing `--tokens-file` returned exit 2 and its explicit file-open error
  before model loading.

Startup checks used CUDA runtime/cuBLAS libraries plus a temporary loader path
containing the official toolkit's driver stub under its `libcuda.so.1` SONAME.
There is no working NVIDIA driver/GPU in this workspace. The stub was used only
to load the executable for non-inference CLI checks; it provides no hardware or
numerical evidence. No model weights were loaded and no inference was performed.

The Windows SM75 workflow was dispatched for the same source commit as run
[40566043784](https://github.com/loktoto/ninfer-strata-gpu-unlock-rtx2080ti-22gb-windows/actions/runs/40566043784)
and remained queued at the last observation. Hosted CI separately encountered
the account billing/spending-limit restriction already recorded in VALIDATION.md.
