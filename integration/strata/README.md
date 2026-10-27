# Strata integration

`patches/0001-sage-external-rmsnorm-and-strict-policy.patch` applies to PolyStrata
commit `52c31cfcb55d7e42506a57757886af73f9547314`. Bootstrap applies it; the root
CMake build includes this module only when `SAGE_BUILD_STRATA=ON`.

The module links `sage_sm75_kernels` directly into `strata-poly`. It preserves
Strata's model loader, GGUF tensors, routing, expert cache, CPU workers, CUDA
stream, scratch and output ownership. `Engine::open` rejects any nonzero expert
skip policy, including values supplied by `STRATA_ROUTE_TAIL_SKIP` through the
CLI. Every original top-k selection must execute on CPU or GPU.

The RMSnorm wrapper uses the upstream kernel by default. Set the launcher's
`--ninfer-rmsnorm` flag to try the linked NInfer-derived FP32 kernel for
2048-wide out-of-place norms. Other widths and unsupported calls retain the
original path. This operation requires no expert conversion, scratch allocation
or second weight bank. It does not accelerate Q4_K expert products or prove an
end-to-end speedup. Numerical reduction differs and needs CUDA parity plus
same-GGUF end-to-end validation before acceptance.

```powershell
python integration/strata/launch_strict.py --engine build-windows/strata/strata-poly.exe --model D:/models/Sage-A1.5-Q4_K_M.gguf -- --tokens 1,2,3 --max-new 16
python integration/strata/launch_strict.py --engine build-windows/strata/strata-poly.exe --model D:/models/Sage-A1.5-Q4_K_M.gguf --ninfer-rmsnorm -- --tokens 1,2,3 --max-new 16
```

Token IDs above are examples; use the model's verified tokenizer and template.
The launcher fixes skip policies to zero, disables draft/lookup/graphs, uses
FP16 KV and a one-token window, and freezes expert residency for comparisons.
It rejects conflicting forwarded options. `--dry-run` prints exact arguments
and controlled environment values without running the engine. Engine and model
files must exist. Paths are passed as an argument vector with no shell.

The second patch adds `--ninfer-expert-q4k`: selected cached Q4_K gate/up and
down matrices are borrowed directly from existing expert slots, using the
existing Q8_1 scratch/output pointers and stream. Q6_K, unsupported shapes and
all-cached device-ID routing retain native dispatch. No expert bank is created.
This candidate requires SM75 numerical and same-GGUF validation before use in
a performance profile; CPU and GPU expert partitioning remain Strata's.
