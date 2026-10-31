# Native Windows API startup

`tools/serve_sage.py` connects the pinned PolyStrata Python API to the built
`strata-poly.exe`. The API owns one resident inference process and speaks its
`--serve` stdin/stdout protocol. The NInfer-derived candidates run inside that
same process when selected.

This startup path has host tests. The IQ4_XS model has completed bounded Windows
API, GPU-oracle and full-logit comparisons; see [methodology and results](METHODOLOGY_AND_RESULTS.md).
This does not qualify every model, workload or context size. No 256K acceptance
is claimed. The original Q4_K_M identity below is distinct from the tested IQ4_XS artifact.

## Prepare the runtime and Python packages

Use Windows 11 x64, Python 3.10+ and the CUDA/MSVC development environment from
the [build instructions](../README.md#build-the-windows-inference-runtime).
Run these commands from the repository directory in the same Python environment
you will use to serve:

```powershell
python tools/bootstrap.py
python -m pip install -r .deps\PolyStrata\requirements.txt
.\tools\build_windows.ps1
```

The source revision is fixed by [sources.lock.json](../config/sources.lock.json).
The requirements file belongs to that pinned checkout and pins its Python
dependencies, including `numpy`, `jinja2`, `regex`, `pyyaml`, `requests`, `pillow`
and `psutil` and their listed dependencies. Installing these packages does not
run upstream `setup.py` or select/download a model. The API wrapper checks the
checkout's Git revision and never clones, fetches, installs packages or downloads
models itself.

Supply the full local `Sage-A1.5-Q4_K_M.gguf` yourself. Match its file size and
SHA-256 against [model.lock.json](../config/model.lock.json); a header prefix is
insufficient. For example:

```powershell
(Get-Item D:\models\Sage-A1.5-Q4_K_M.gguf).Length
Get-FileHash D:\models\Sage-A1.5-Q4_K_M.gguf -Algorithm SHA256
```

The expected size is 21,713,463,520 bytes and SHA-256 is
`a64fe0bbf936db3eb54f66f142c2b2eee569ade83673649f549a21daef8ca395`.
The wrapper validates local paths; it does not hash 21.7 GB on every launch.

## Start the strict baseline

```powershell
python tools/serve_sage.py --engine .\build-windows\strata\strata-poly.exe --model D:\models\Sage-A1.5-Q4_K_M.gguf --output-dir .\sage-api-baseline
```

Use the actual executable location if your build placed it elsewhere. Keep its
`strata-poly.exe` name: the upstream server uses the filename to select GGUF
engine behavior. The wrapper also accepts the historical `strata-glm` name.
Paths containing spaces can be quoted normally.

The wrapper exports a tokenizer pack directly from the supplied GGUF using
the pinned `tools/strata_tokenizer.py --gguf <file> --out <run>/pack`.
It requires the exported vocabulary, merges, tokenizer metadata and the GGUF's
own nonempty `chat_template.jinja`. If the GGUF has no template it stops; no
replacement tokenizer or generic template is chosen. This follows the upstream
[existing-architecture procedure](https://github.com/VecSzn/PolyStrata/blob/52c31cfcb55d7e42506a57757886af73f9547314/docs/BACKENDS.md#adding-a-model-of-an-architecture-that-is-there)
for Sage's `qwen35moe` architecture.

The saved `strata-sage.json` contains absolute paths for `exe`, `cwd`, `tokenizer`
and the model. Both the server and engine working directories are the prepared
PolyStrata directory so Python imports resolve from that checkout. The server
command is `python -m serve.server --engine strata --config <saved-config>
--host 127.0.0.1 --port 8080`; its `StrataEngine` adds `--serve` once when it
starts the executable.

Baseline engine arguments are `--tail-skip 0 --skip-weight 0 --no-draft --lookup 0
--window 1 --no-graphs --kv fp16 --max-context 8192`. The environment explicitly
sets `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_FREEZE_EXPERTS=1`,
`STRATA_SAGE_TRACE=0`, `STRATA_SAGE_RMSNORM=0`, `STRATA_SAGE_EXPERT_Q4K=0`,
`STRATA_SAGE_EXPERT_Q6K=0` and `STRATA_SAGE_EXPERT_IQ4XS=0`, overriding inherited
candidate settings. These values are also saved in the config for later starts.
Use `--max-context N` and `--threads N` to select a positive context allocation
and CPU thread count. Larger allocations require separate memory and correctness
qualification; the default is a starting point, not a measured capacity claim.

The config sets `fit_max_tokens` and `drop_oldest` to `false`. The API rejects
requests that exceed its context budget, including the engine's safety margin,
instead of silently shortening a requested answer or removing earlier turns.

`--output-dir` must name a new directory with an existing parent. Existing output
is never overwritten, including after a failed extraction. If omitted, a unique
`sage-api-<timestamp>-<id>` directory is created in your current directory. The
config and tokenizer persist after exit. To prepare a new run, choose another
directory. To restart an existing saved config without extracting again:

```powershell
$SageConfig = (Resolve-Path .\sage-api-baseline\strata-sage.json).Path
Push-Location .deps\PolyStrata
python -m serve.server --engine strata --config $SageConfig --host 127.0.0.1 --port 8080
Pop-Location
```

The wrapper returns the tokenizer or server process's exit code on failure. It
does not print API success before the actual server starts. Stop it with Ctrl+C.

## Verify from another PowerShell

After the server reports it is ready:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
Invoke-RestMethod http://127.0.0.1:8080/v1/models
$SageRequest = @{ model = "sage-a1.5"; messages = @(@{role = "user"; content = "Say hello."}); temperature = 0; max_tokens = 128; stream = $false } | ConvertTo-Json -Depth 8
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/chat/completions -ContentType "application/json" -Body $SageRequest
```

The OpenAI base URL is `http://127.0.0.1:8080/v1`, with model id `sage-a1.5`
(change it through `--model-name`). The upstream API also exposes
`POST /v1/messages` for Anthropic clients. If `STRATA_API_KEY` is set, clients
must send that key, for example with an `Authorization: Bearer ...` header.
The baseline binds only to this PC.

## Inspect a plan and select a candidate

On Windows or Linux, a dry run validates the engine/model paths and source pin,
then prints the exact commands, config and enforced environment. It writes
nothing and does not extract a tokenizer, start the engine or validate inference:

```powershell
python tools/serve_sage.py --engine .\build-windows\strata\strata-poly.exe --model D:\models\Sage-A1.5-Q4_K_M.gguf --output-dir .\sage-api-plan --dry-run
```

Real startup is Windows-only. Linux can use local Linux-style paths with
`--dry-run`; it cannot validate the existence of a Windows drive path.

Add `--ninfer-rmsnorm`, `--ninfer-expert-q4k`, `--ninfer-expert-q6k` and/or
`--ninfer-expert-iq4xs` to select the experimental
candidates explicitly. Each flag controls its own saved environment value.
Use a new output directory and compare against the same prompt and strict
baseline before claiming numerical equivalence or a performance gain.

## Reproduce the graph and draft profiles

The default is a diagnostic profile with graphs and drafting disabled. Add
`--cuda-graphs` for the measured graph-eligible no-draft profile. Add
`--draft-window 4` as well to use the model's own draft block with a verification
window containing one known token and up to three guesses. It is not four draft
tokens. The explicit draft option requires `--cuda-graphs`, accepts windows 2–4,
and keeps lookup disabled, FP16 KV and complete expert routing. Graph eligibility
does not by itself prove every call is captured.

For the supplied IQ4_XS model, use a new output directory and the built executable:

```powershell
python tools/serve_sage.py --engine .\build-optimized\strata\strata-poly.exe --model $SageModel --max-context 32768 --threads 16 --cuda-graphs --draft-window 4 --ninfer-expert-iq4xs --output-dir .\sage-iq4xs-draft
```

Set `$SageModel` to the actual local GGUF beforehand. Omit `--draft-window 4` for
the no-draft comparison; omit `--ninfer-expert-iq4xs` for native Strata with the
same decoding settings. Leave RMSNorm off: its full-model numerical differences
remain unresolved. See the [draft study](IQ4XS_MTP_RESULTS_2026-10-30.md) for
measured rates, acceptance and limitations. Allocating 32768 tokens does not mean
these short workloads occupied that context.

The launcher now persists the IQ4_XS selection and trace setting in saved configs.
Previously the initial process received the IQ4_XS selection, but the saved JSON
omitted it; restarting that JSON could inherit the opposite setting from another
shell. Regenerate old configs to make the saved choice explicit. The reported
benchmarks used explicit per-run configs and were not affected by this omission.

