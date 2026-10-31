# Run the IQ4_XS Strata integration on Windows

This guide starts the experimental, measured profile on Windows x64 with a compatible NVIDIA GPU and a locally supplied Sage IQ4_XS GGUF. The server runs on this computer at `127.0.0.1:8080`; the model is not downloaded for you. The combined profile uses Strata's MTP draft block and an opt-in NInfer-derived expert kernel inside the same Strata process. The separate experimental RMSNorm kernel stays off.

## Start on an already prepared PC

Open PowerShell in the project directory. Replace the example model path with your complete local GGUF; keep the quotes if the path contains spaces. Check the plan first:

```powershell
.\tools\start_iq4xs.ps1 -Model 'D:\models\Sage-IQ4_XS.gguf' -DryRun
```

The dry run checks the model, engine and pinned source paths, and prints the engine arguments and environment. It does not start the GPU engine, extract the tokenizer, validate the chat template, or test inference. If the plan succeeds, start the server:

```powershell
.\tools\start_iq4xs.ps1 -Model 'D:\models\Sage-IQ4_XS.gguf'
```

The default engine is `build-optimized\strata\strata-poly.exe`; use `-Engine 'D:\path\strata-poly.exe'` if built elsewhere. The script uses `.venv\Scripts\python.exe` and temporarily adds `.cuda\bin\x64` to `PATH` if that directory exists. Each start creates a new `sage-api-<timestamp>-<id>` directory in the project directory; it does not overwrite a prior run. Wait until the server reports it is ready. Leave that PowerShell window open and press **Ctrl+C** to stop it.

The default profile enables CUDA graphs, window 4 (one known token and up to three MTP guesses), IQ4_XS expert opt-in, 32,768 configured context tokens and 16 CPU threads. The script temporarily clears an inherited `GGML_CUDA_DISABLE_GRAPHS` while the server runs and restores it afterward. It does not skip layers. The model must contain the compatible draft block; other GGUF variants have not been qualified by these tests. The configured context capacity is not proof that a full 32K prompt or any larger context has been qualified.

## Verify and connect a client

In another PowerShell window, while the first remains open:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/v1/models
$request = @{
    model = 'sage-a1.5'
    messages = @(@{ role = 'user'; content = 'Say hello.' })
    temperature = 0
    max_tokens = 128
    stream = $false
} | ConvertTo-Json -Depth 8
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8080/v1/chat/completions' -ContentType 'application/json' -Body $request
```

For an OpenAI-compatible desktop application, set its base URL to `http://127.0.0.1:8080/v1` and model name to `sage-a1.5`. The listener is local only. If `STRATA_API_KEY` is configured, the client must send the same value as a Bearer token; otherwise use the client's usual no-key or placeholder-key option if required. The upstream server also supports `/health` for a readiness check.

## Compare configurations

Stop the running server before starting another on port 8080. Keep the same model, prompt, context, sampling and machine conditions for comparisons. These switches can be combined:

```powershell
# Native Strata expert kernels, still using MTP and graphs
.\tools\start_iq4xs.ps1 -Model 'D:\models\Sage-IQ4_XS.gguf' -Native
# Optimized expert kernel with drafting off, graphs still on
.\tools\start_iq4xs.ps1 -Model 'D:\models\Sage-IQ4_XS.gguf' -NoDraft
# Native kernels and no draft, graphs still on
.\tools\start_iq4xs.ps1 -Model 'D:\models\Sage-IQ4_XS.gguf' -Native -NoDraft
```

`-Native` is native **Strata**, not standalone NInfer or the llama.cpp baseline. Bounded runs on an RTX 2080 Ti 22 GB measured a 6.6–6.8% no-draft expert-path decode improvement in two tasks, and a separate combined MTP configuration measured +66.0% and +58.8% versus llama.cpp without drafting. These are small, workload-specific experiments, not expected speeds on every prompt or a standalone NInfer speed claim. Prefill was slower in that combined comparison. See [the measured settings and full limitations](IQ4XS_MTP_RESULTS_2026-10-30.md).

## Restart a saved configuration

The launcher prints the newly saved `strata-sage.json` path. To reuse its extracted tokenizer and settings, stop the first instance and substitute that path below. The config records absolute model, executable and tokenizer paths, so keep those files in place or make a fresh run after moving them.

```powershell
$config = (Resolve-Path '.\sage-api-YOUR-RUN\strata-sage.json').Path
$python = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$oldPath = $env:PATH
$oldGraphSetting = $env:GGML_CUDA_DISABLE_GRAPHS
try {
    if (Test-Path '.\.cuda\bin\x64') { $env:PATH = "$(Resolve-Path '.\.cuda\bin\x64');$oldPath" }
    Remove-Item Env:GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue
    Push-Location '.\.deps\PolyStrata'
    try { & $python -m serve.server --engine strata --config $config --host 127.0.0.1 --port 8080 }
    finally { Pop-Location }
} finally {
    $env:PATH = $oldPath
    if ($null -ne $oldGraphSetting) { $env:GGML_CUDA_DISABLE_GRAPHS = $oldGraphSetting }
}
```

## Fresh source setup and build

You need Git, Python 3.10+, CMake, Ninja, an x64 Visual Studio C++ developer environment, a compatible NVIDIA driver and CUDA toolkit/compiler. The measured build used CUDA 13.0.88 and MSVC 19.44 on SM75. Other toolchains need validation. For a new installation, clone into a new directory; repository access is required:

```powershell
git clone https://github.com/loktoto/ninfer-strata-gpu-unlock-rtx2080ti-22gb-windows.git
Set-Location ninfer-strata-gpu-unlock-rtx2080ti-22gb-windows
```

On the already prepared PC, use the existing project directory instead of cloning over it. Run the following in an **x64 Visual Studio developer PowerShell** with `cl`, `nvcc` and Git on `PATH`. It fetches the pinned PolyStrata and its build dependencies and installs Python packages in the project virtual environment; it does not install or change a GPU driver. Network access is required for this setup.

```powershell
python -m venv .venv
$env:PATH = "$(Resolve-Path '.\.venv\Scripts');$env:PATH"
python -m pip install cmake ninja
python tools\bootstrap.py
python -m pip install -r .deps\PolyStrata\requirements.txt
.\tools\build_windows.ps1 -BuildDirectory build-optimized -Jobs 2
```

The build script rechecks/prepares the pinned source and patch series, configures Release with CUDA, `SAGE_BUILD_STRATA=ON` and CUDA architecture 75, and builds `strata-poly`. It refuses to overwrite divergent patched source. A newly compiled binary is not automatically the previously benchmarked binary; rebuilds and different toolchains need their own validation. The local build may require CUDA runtime DLLs on `PATH`; see below. The server script performs no installation or driver changes.

The original repository model lock refers to a **different Q4_K_M file**, not the tested IQ4_XS GGUF. Its hash must not be used to identify or claim validation of an IQ4_XS model. The launcher checks that a model file exists and that its tokenizer can be extracted at startup; it does not establish the model's identity or correctness. Obtain the intended complete IQ4_XS file independently, check its provenance and hash against its own publisher, and retain enough disk space for the source, build and extracted tokenizer.

## Troubleshooting

- **Missing engine or Python:** build the engine and create `.venv` as above, or pass the actual executable with `-Engine`. Use the filename `strata-poly.exe` because the server selects its protocol from it.
- **Missing DLL or CUDA runtime:** verify the driver and runtime required by your build, and make the corresponding DLL directory available on `PATH`. The launcher automatically adds the project-local `.cuda\bin\x64` only when present. It does not install DLLs. Reopen a PowerShell after changing system settings.
- **Source pin or patch mismatch:** inspect `config\sources.lock.json` and run `python tools\bootstrap.py` from the project virtual environment. Preserve and review any local source edits; bootstrap intentionally refuses an unexpected source revision or divergent patches.
- **Port 8080 already in use:** stop your own previous server with Ctrl+C and check the port with `Get-NetTCPConnection -LocalPort 8080 -State Listen`. Do not terminate another application's process without identifying it. The simple launcher binds only `127.0.0.1:8080`.
- **GPU memory allocation or load failure:** close workloads you control, check free memory with `nvidia-smi`, and confirm that the model, KV cache and GPU match the tested configuration. A 32K configured context consumes memory; this simple launcher fixes that setting. Use the lower-level `tools\serve_sage.py --help` to choose a smaller `--max-context` for diagnosis, and revalidate performance and correctness for that configuration.
- **Tokenizer or template error:** use a complete compatible GGUF with its own tokenizer and chat template. A failed run directory is retained; a new launch creates a different one. Do not invent a replacement template to bypass the check.
- **PowerShell blocks scripts:** follow your organization's PowerShell execution policy; you can also invoke `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\start_iq4xs.ps1 -Model 'D:\models\Sage-IQ4_XS.gguf'` for this process if permitted. This does not change the persistent policy.
