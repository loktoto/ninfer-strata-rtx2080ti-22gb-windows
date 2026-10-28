# Native Windows baseline/candidate diagnostics

`tools/validate_model.py` captures the same pretokenized prompt and teacher-forced
continuation under the original and selected candidate kernels. Use the binary
built with all local patches and the exact GGUF in `config/model.lock.json`.

Create a JSON file with two nonempty arrays:

```json
{"prompt_ids": [1, 2, 3], "follow_ids": [4, 5]}
```

These are format examples, not meaningful Sage tokens. Obtain IDs using the
verified Sage tokenizer/template and choose a fixed continuation from a trusted
same-GGUF reference. Keep those inputs fixed across both runs. No token count
is inferred from text length. A 32K prompt test requires 32768 actual prompt IDs.

```powershell
python tools/validate_model.py --engine build-windows\strata\strata-poly.exe --model D:\models\Sage-A1.5-Q4_K_M.gguf --tokens-file D:\tests\sage-32k.json --output evidence\sage-32k-001 --candidate both --hash-model
```

The default context allocation is 32768 plus the exact continuation length.
For larger prompts supply `--max-context` with capacity for both prompt and
continuation. `--candidate rmsnorm`, `expert-q4k`, or `both` selects the candidate;
the baseline always forces both candidates off. Both runs enforce zero expert
skipping, FP16 KV, frozen expert residency, no graphs, no speculative decoding,
and the same seed. The original expert partition still applies.

The tool writes short `--tokens-file` / `--follow-file` engine commands so a full
prompt can pass Windows' command-line limit. The output directory must be new.
It retains input and binary hashes, the model lock, CSV hashes, exact command and
controlled environment, raw stdout/stderr, exit code, elapsed time, reported
prompt/generated counts, top probabilities and layer-checksum diagnostics.

The GGUF's magic and locked size are checked before execution. `--hash-model`
also checks the complete model SHA256. Without it, the report explicitly records
that full contents are unverified, avoiding a repeated 21.7 GB read on every run.

Short prompt consumption, short/incorrect forced continuation, malformed or
missing diagnostics, engine failure and timeout produce a failed result and
nonzero tool exit. Successful diagnostic capture is `completed_unqualified`.
`--dry-run` on Windows or Linux writes an `inventory_only` plan and does not run
inference. Real execution requires native Windows.

Teacher forcing replaces the model's selected token IDs. Equal forced outputs
therefore test the protocol and continuation length, not model accuracy. Printed
top probabilities are rounded/truncated; layer checksums lose information.
Neither establishes full-logit numerical parity or passes G1/G2. Diagnostic
dumping also changes timing, so this tool cannot establish throughput speedup.
Use independent GPU operator oracles and same-GGUF full-logit comparison for
numerical acceptance, plus a separate non-diagnostic benchmark for performance.

For complete-value comparison, add `--full-logits --logits-atol 0.0001
--logits-rtol 0.001`. These are explicit starting test tolerances, not established
Sage quality thresholds. Malformed, missing or mismatching dumps fail the run.
See LOGIT_VALIDATION.md for the format and independent comparator. G1/G2 still
require target hardware and operator/reference validation; matching one supplied
continuation is partial numerical evidence only.

Add `--require-dispatch` to require enqueue traces for every selected candidate.
Unsupported Q4_K paths, including all-cached device-ID dispatch, may legitimately
retain native kernels; choose a workload that actually exercises the cached
host-routed candidate when qualifying it. Traces mean host enqueue, potentially
during graph capture, not device execution or correctness. Strict runs override
inherited tracing settings for reproducibility.

Use `--candidate expert-q6k` for the NInfer-all-derived Q6_K candidate, or
`--candidate all` for RMSNorm/Q4_K/Q6_K together. `both` retains RMSNorm/Q4_K.
Baseline explicitly disables all three, including inherited Q6_K settings.
GPU numerical acceptance remains pending.
