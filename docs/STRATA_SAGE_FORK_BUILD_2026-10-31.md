# Sage-focused Strata fork: Windows SM75 build and smoke — 2026-10-31

## Scope

This is a bounded compatibility test of [xyehya/Strata q36-r2](https://github.com/xyehya/Strata/tree/379f095027097937f28c5c009cacbbb66504d957), a different engine from the tested PolyStrata integration. It does not establish performance or full-logit equivalence. The existing local Sage IQ4_XS GGUF was read unchanged; no BF16 graft or model conversion was used.

## Build and loader change

- Upstream revision `379f095027097937f28c5c009cacbbb66504d957`; ggml source revision `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
- Native Windows, Ninja, MSVC 19.44.35229, CUDA compiler 13.0.88, Release, `CMAKE_CUDA_ARCHITECTURES=75`, `STRATA_NATIVE_EXPERTS=ON`, `STRATA_Q6K_EXPERTS=ON`. Build targets `strata-q36` and `q36_q4k_upload_test` completed successfully with two workers. CUDA header code-page warnings were emitted. An optional expert arena test was not registered because its fixture was absent.
- Executable SHA-256: `8711ce375708511b0ffb75fec240d3eae252cf40a94ae67d755a03a7df943f53`.
- Header inspection found 60 `ssm_alpha` / `ssm_beta` tensors in the model, all Q4_K with dimensions 2048 × 32. The unmodified fork's float-upload helper supports F32, BF16, F16 and Q8_0, and would reject these tensors. The local change adds Q4_K decoding through the pinned ggml `to_float` trait. It widens the values represented by the stored Q4_K blocks to FP32, without changing the GGUF or requantizing it. It is gated on native-expert support. The [portable patch](../research/strata-sage/q36-q4k-upload.patch) is against the fork revision above; reverse-apply checking against the modified checkout passed.
- The CPU decoder test passed: two-block packing and nibble order, a signed scale, nonzero minimum, upper packed scale/min bits, destination guards, null inputs, zero, unaligned, and oversized element counts. This checks the decoder interface against explicit expected values, not end-to-end inference.

## Unchanged-model smoke

One CLI process ran on the modified RTX 2080 Ti 22 GB with the original IQ4_XS model, 32,768 configured context, FP16 KV, 16 CPU pool workers, MTP drafting set to zero, a 13-token prompt and 16 freely generated greedy tokens. The loader reported 753 tensors and successful layout checks; all routed experts were resident in VRAM. The first 16 generated IDs matched the saved PolyStrata API continuation fixture exactly (16/16). The process exited successfully and GPU memory returned to desktop baseline.

The log reports a single 16-token decode rate, but this short startup smoke is not a throughput benchmark. The fork reported chunked GDN enabled for prompt chunks, grouped-query attention enabled and SM80-only attention MMA disabled; an MTP block was loaded even though drafting was disabled. These engine choices may differ from PolyStrata. The previous fixture came from a separate session, and no full-logit, teacher-forced KL, extended quality, occupied-32K context, MTP, graph replay or repeated benchmark was run here. Matching 16 IDs does not establish numerical equivalence.

Configuration, build and smoke logs remain in the local isolated checkout and in the local evidence snapshot; they are not included in the portable patch. Raw logs contain local paths and token IDs and require sanitization before publication. No model file, raw prompt or generated response is distributed in this report. The next step is a balanced, same-model multi-engine comparison only after agreement on corresponding cache, graph, MTP and correctness settings; preserve these smoke results separately from performance claims.


## Reproduction and local evidence

In an x64 Visual Studio developer environment with CUDA 13.0.88 on PATH, use the pinned fork and ggml checkout, then apply the patch before configuration. The following uses placeholders for local paths rather than distributing the model or machine-specific directories:

```powershell
git checkout 379f095027097937f28c5c009cacbbb66504d957
git apply <path-to-q36-q4k-upload.patch>
cmake -S . -B <build-dir> -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 `
  -DCMAKE_CUDA_COMPILER=<nvcc.exe> -DCUDAToolkit_ROOT=<cuda-dir> `
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON `
  -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON `
  -DSTRATA_GGML_DIR=<pinned-llama-cpp-checkout> `
  -DIQ_FIXTURE_PYTHON=<python-with-numpy>
cmake --build <build-dir> --target strata-q36 q36_q4k_upload_test --parallel 2
<build-dir>/q36_q4k_upload_test.exe
```

The existing ggml source was reused; its revision was checked before configuration. CMake reported ggml `3cf03257f`, `arch 75`, and CPU AVX2/FMA/F16C. The patch's test executable was run directly and returned exit code 0. A subsequent Ninja dry run reported no work for both requested targets. The raw CLI smoke used `--max-context 32768 --kv fp16 --pool-workers 16 --mtp 0 --tokens-file <13-token-fixture> --n 16` against the local original file, and exited with code 0 after 93.84 seconds including model loading. The previous continuation fixture was derived from a separate PolyStrata API run; comparison here is token IDs only.

Local artifacts are retained without uploading their raw contents. SHA-256 values identify the files:

| Local artifact | SHA-256 |
| --- | --- |
| Final configure log | `3416aa43a0f64cd6305d4e9a8fac4ca46f3e87c483d84e364b22f9d9571a3c2a` |
| Build log | `726887628f0c95be6c203c5661bf33fb483e6666c3f2077b237b5e5f1f61d9a5` |
| Smoke stdout | `0571abd4b08225d03caf38c6e624d9f7ab411dfdc083b6042adafb75afb62e5a` |
| Smoke stderr | `59c68fd72011a7d6635d74423fbe50e25d5d6a641ce8313e3ed647390523788d` |
| Portable patch | `fde7b974159fd9aa97281159fbccc101e388267be45729590a87220e3a3df4b5` |

The executable and logs are local evidence, not published runtime binaries. No TensorRT/NInfer integration was part of this test.

