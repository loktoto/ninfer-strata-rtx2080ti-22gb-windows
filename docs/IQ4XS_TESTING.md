# Sage-A1.5 IQ4_XS testing readiness

Current scope is **Strata + NInfer only, normal SM75**. Bootstrap fetches only
PolyStrata; CMake links the borrowed-buffer kernel library into `strata-poly`.
No second NInfer runtime is launched. Deferred unlock source metadata is stored
separately in `config/deferred-research.lock.json`, outside the active source
lock. No unlock component is fetched, linked or executed by these entry points.

## Current user configuration

The owner reports Sage-A1.5 IQ4_XS, MTP4, a configured context limit of 32768,
and approximately 100 tokens/s. The 32K figure is capacity, not measured prompt
occupancy. The exact model filename/hash, producing engine/command, KV format,
actual prompt/output lengths and timing definition have not yet been supplied.
This is a user-reported result, not a benchmark of this integration.

| Test | Readiness |
| --- | --- |
| Build and host contracts | Full Linux SM75 runtime compiled previously; current host suites pass. Native Windows build/GPU execution pending. |
| IQ4_XS model smoke | Strata has native IQ4_XS dispatch. Supply the exact complete GGUF and inspect its header/hash. Actual Sage IQ4_XS inference is unverified here. |
| NInfer RMSNorm comparison | Candidate exists for eligible 2048-wide out-of-place norms, independently of the expert quantization. Enqueue and numerical/GPU evidence still required. |
| IQ4_XS expert acceleration | No NInfer IQ4_XS expert candidate is integrated. IQ4_XS matrices stay on Strata's native path. A mixed recipe may separately contain Q4_K/Q6_K matrices; inspect actual tensors. |
| MTP4 performance comparison | Not ready as a matched test: strict CLI/API launchers disable draft, graphs and adaptive residency. Pinned `strata-poly` also caps its block-based speculation at one known token plus three guesses; this must not be called four MTP drafts. |
| Official gate qualification | Existing model lock and acceptance checker bind the original Q4_K_M artifact. They must not certify the user's IQ4_XS file using that identity. |

## Initial native Windows smoke procedure

Build from the repository using the README's native Windows instructions and
run the independent kernel oracles first. Set `$SageModel` to the **exact local
IQ4_XS filename**; the filename below is illustrative, not a pinned artifact.
Run from the repository root:

```powershell
$SageModel = 'D:\models\Sage-A1.5-IQ4_XS.gguf'
python tools/inspect_gguf.py $SageModel
Get-FileHash $SageModel -Algorithm SHA256
python tools/serve_sage.py --engine .\build-windows\strata\strata-poly.exe --model $SageModel --max-context 32768 --output-dir .\iq4xs-baseline
```

Follow [API.md](API.md) to check health and send a short deterministic request.
Stop that server, then use a new directory and the same model/request for the
RMSNorm candidate:

```powershell
python tools/serve_sage.py --engine .\build-windows\strata\strata-poly.exe --model $SageModel --max-context 32768 --ninfer-rmsnorm --output-dir .\iq4xs-rmsnorm
```

These are **non-MTP diagnostic smoke tests** with FP16 KV, no graphs, no expert
skipping and frozen residency. They are not representative of the reported
100 tokens/s MTP4 configuration. A successful API response is partial smoke
evidence, not numerical parity or a speed improvement. A 32768 allocation also
does not prove full-context quality or consumption.

Before a performance claim, capture the original launch settings and exact
model identity; check free-generation numerical/quality behavior; then compare
same settings and prompt/output lengths with repeated non-diagnostic runs.
Record prompt processing separately from accepted output tokens/s, MTP draft
counts/acceptance, TTFT, peak VRAM and CPU misses. Preserve the original engine
and configuration for comparison. No speedup over the reported result is claimed.
