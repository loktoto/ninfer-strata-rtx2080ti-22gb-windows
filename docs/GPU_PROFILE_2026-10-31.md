# Process-local GPU activity trace, October 31

Two bounded captures traced one 240-token arithmetic response each with CUDA graphs enabled, drafting disabled and the same model and context settings. One used native Strata IQ4_XS expert slots; the other enabled the NInfer-derived candidate. Both returned HTTP 200, produced the same content hash, and reported zero cached prompt tokens. This is a single traced request per variant, not a throughput benchmark or broad correctness test.

NVIDIA CUPTI Activity tracing initialized successfully without administrator elevation. Each request interval contained 317,259 GPU kernel records, including 312,480 records with nonzero graph IDs across 122 IDs, and 599 `cudaGraphLaunch` API records. No dropped records, failed Activity calls or record parsing errors were reported. At each one-second clock anchor the tracer records a CUPTI timestamp and Windows wall-clock time; the observed offset spread was 24.3 microseconds native and 14.6 microseconds candidate. Request boundaries are based on the client's wall clock. Their small mapping uncertainty can affect records near the boundaries.

| GPU kernel group | Calls | Native cumulative ms | Candidate cumulative ms |
| --- | ---: | ---: | ---: |
| IQ4_XS expert slots | 19,200 | 508.7 | 370.2 |
| Native Q4_K matrix-vector | 60,000 | 469.0 | 470.0 |
| Native Q6_K matrix-vector | 5,040 | 298.2 | 300.0 |
| Float GEMV | 19,280 | 130.9 | 122.1 |
| GDN state | 7,260 | 75.3 | 86.3 |
| RMSNorm | 19,600 | 83.6 | 83.8 |

The 508.7 to 370.2 ms change is a 27.2% reduction in aggregate measured IQ4_XS kernel duration, or a 37.4% increase in reciprocal aggregate kernel rate for the same number of calls. Neither figure is an end-to-end improvement or wall-time share. The table sums device kernel durations in the entire request interval, including prefill and decode; the capture did not mark the first generated token, so it cannot isolate decode. Kernels can overlap, records near the boundary may be included differently, and instrumentation changes execution. The same graph and kernel counts and returned content support a useful matched diagnostic, but individual times can vary. In particular, this does not establish an RMSNorm speed opportunity or justify adding candidate and MTP gains. The existing Q4_K and Q6_K matrix-vector work is a concrete next area for profiling and optimization; its larger cumulative time does not by itself establish an achievable end-to-end improvement.

CUDA launch API duration sums also include CPU-side time and may overlap device work. Activity tracing has substantial overhead: use the separate uninstrumented ABBA results for performance claims. Kernel symbols, counts and times are preserved in the sanitized [native](../evidence/published-benchmarks/2026-10-31-cupti-native.json) and [candidate](../evidence/published-benchmarks/2026-10-31-cupti-candidate.json) summaries. The raw trace and local configuration remain local.

Local raw traces, configuration, logs and a build manifest are archived under `evidence/cupti-20261031`. The tracer source SHA-256 is `91819134507979af63d1a3ae8e1e931e95cae87c327b7b652c7ef823de769489`. The archived DLL was rebuilt from that same source after both captures to test the build script; the exact capturing DLL hash was not recorded. A host-only summarizer test verifies window selection and rejection of dropped records, parse errors and non-success status.

## Reproduce the trace

This experimental tracer uses NVIDIA's Activity API, not hardware performance counters; it does not alter driver profiling permissions. It uses `CUDA_INJECTION64_PATH` for the target engine child only and is inactive otherwise. The approach follows the documented interface in NVIDIA's CUPTI `cupti_trace_injection` sample. The implementation is original and does not use the Windows sample's exit detour: it flushes completed Activity buffers once per second while the process is alive. The bounded runner waits four seconds after the request before terminating its own server tree. A crash or forced early termination can still lose the final records. A failed CUPTI initialization, missing records or reported drops is a failed trace, not performance evidence.

For a prepared source/build environment, obtain the Windows CUPTI 13.0.85 archive from NVIDIA's [CUDA 13.0.2 manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.2.json), check SHA-256 `e739e96415e1fb9a650eaac826f7cd24701e18ae2f7c26b78f3fabaf64c8e39b`, and extract it beneath `.profiling`. The [CUPTI Activity API](https://docs.nvidia.com/cupti/13.0.1/main/main.html) describes the GPU and CPU activity records. Do not install the package globally or modify driver permissions.

From an x64 Visual Studio developer PowerShell, in the project root, with an existing local CUDA toolkit under `.cuda`:

```powershell
$cupti = 'D:\path\to\extracted-cuda-cupti-archive'
.\tools\profiling\build_cupti_tracer.ps1 -CuptiRoot $cupti
$config = 'D:\path\to\your\strata-sage.json'
.\.venv\Scripts\python.exe tools\profiling\run_bounded_trace.py --config $config --cupti-root $cupti --output-dir .profiling\candidate-trial --no-draft
.\.venv\Scripts\python.exe tools\profiling\run_bounded_trace.py --config $config --cupti-root $cupti --output-dir .profiling\native-trial --no-draft --native
.\.venv\Scripts\python.exe tools\profiling\summarize_activity.py .profiling\candidate-trial --output .profiling\candidate-summary.json
.\.venv\Scripts\python.exe tools\profiling\summarize_activity.py .profiling\native-trial --output .profiling\native-summary.json
```

Use a new output directory for each run and ensure port 18360 is free. The input config can be made with the normal launcher from [the setup guide](IQ4XS_SETUP.md), then the server must be stopped before tracing. The config must point to the local engine, model and tokenizer. The runner copies it locally, selects the native or candidate setting, and starts only its own server and engine tree. It writes request timing, output hash, local server log and raw GPU activities; its config and log can contain machine paths, so review them before sharing. A profile may affect GPU memory usage and inference behavior; do not run it during another workload or treat the traced rate as normal throughput.
