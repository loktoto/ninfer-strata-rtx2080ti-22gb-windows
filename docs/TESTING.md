# Native Windows validation suite

This is an evidence-gathering procedure, not a claim that any hardware gate has passed. Current scope is Strata + NInfer on normal SM75; G5/G6 are deferred, and active gates are G1–G4/G7. Build the pinned, patched engine with token-file and full-logit support first. Prepare the exact engine executable, GGUF model, matching model lock, source lock, and a fixture manifest. The model and engine are hashed by baseline capture; the first diagnostic also verifies the full model hash against the lock. Source locks do not prove binary provenance. Keep binaries and model unchanged during the suite.

Create token JSON files containing exactly `prompt_ids` and `follow_ids` integer arrays from the model's actual tokenizer and representative tasks. Do not construct meaningless repeated IDs solely to reach a length. The five totals are **prompt plus requested output**, exactly 32768, 65536, 131072 and 262144 for the named long stages; the short total must be between 16 and 32767. Include a useful, externally reviewed task expectation for each fixture. For example, a manifest (paths are relative to the manifest) is:

```json
{"schema_version":1,"stages":[
  {"name":"short","tokens_file":"short.json","task_expectation":"Reviewed expected result for the short task"},
  {"name":"32k","tokens_file":"32k.json","task_expectation":"Reviewed expected result for the 32k task"},
  {"name":"64k","tokens_file":"64k.json","task_expectation":"Reviewed expected result for the 64k task"},
  {"name":"128k","tokens_file":"128k.json","task_expectation":"Reviewed expected result for the 128k task"},
  {"name":"256k","tokens_file":"256k.json","task_expectation":"Reviewed expected result for the 256k task"}
]}
```

First prepare a reviewable command inventory on any platform, then run the same command on native Windows after removing `--dry-run` and choosing a new output directory:

```powershell
python tools/run_validation_suite.py --engine C:\build\engine.exe --model C:\models\model.gguf --manifest C:\fixtures\manifest.json --candidate all --output C:\results\review --dry-run
python tools/run_validation_suite.py --engine C:\build\engine.exe --model C:\models\model.gguf --manifest C:\fixtures\manifest.json --candidate all --output C:\results\run1
```

`--candidate all` enables the RMSNorm, Q4_K and Q6_K candidates for comparison. The default `both` retains RMSNorm plus Q4_K for existing invocations. A selected candidate need not be exercised by every workload: run `validate_model.py --require-dispatch` separately with representative routed experts to require evidence of host enqueue for each selected path. Enqueue evidence does not establish GPU numerical correctness.

Full-logit tolerances are fixed by `--logits-atol` and `--logits-rtol` (defaults 0.0001 and 0.001); record and review them before running. Every stage captures baseline and candidate teacher-forced full logits and compares them, stopping before the next stage on a failure. Match of forced IDs is a protocol check, not proof of quality. Logs, raw logits, invocation parameters and per-run outputs are preserved. Real execution requires substantial time and storage, especially for full logits. Inspect the diagnostic report and actual source/build provenance. Do not treat execution completion as approval of the active gates.

Separate strict baseline and candidate performance runs use the same exact token inputs, without layer or logit dumps. Defaults are one warmup and three measured runs per configuration; change `--warmups` and `--repeats` before capture. Per-run wall times and engine prompt/decode rates are recorded, including warmups. Compare repeated measured runs and account for system conditions, model loading, cache effects and memory limits. These teacher-forced runs do not establish free-generation quality. The strict launcher disables MTP/draft and graphs; neither MTP nor dynamic graphs is tested or claimed. A full stage can still fail from resource limits, in which case subsequent stages remain unrun.

For optional local API smoke or repeated endurance requests, pass `--api-config path.json` with, for example, `{"url":"http://127.0.0.1:8080/v1/chat/completions","body":{"model":"model","messages":[{"role":"user","content":"Hello"}]},"expected_status":200,"expected_substring":"choices","requests":10}`. Requests are sequential, and actual responses and timings are saved. A status or substring assertion is not a semantic-quality evaluation. A failed request fails the suite. API tests occur after all inference stages and only localhost endpoints are permitted.

Task expectations are recorded for independent review; semantic quality remains `not_assessed`. Independently evaluate actual free-generation task outputs, long-context retrieval, streaming/concurrency and long-duration reliability against reviewed assertions. Refer to [llama.cpp benchmark documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench) for separate prompt-processing and token-generation measurements and [vLLM benchmarking documentation](https://docs.vllm.ai/en/latest/benchmarking/) for serving workload and latency methodology. These references do not validate this engine or its hardware.
