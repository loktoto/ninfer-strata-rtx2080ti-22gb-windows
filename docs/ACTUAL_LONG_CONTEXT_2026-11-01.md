# Actual-use long-context diagnosis — 2026-11-01

**The requested 100% decode uplift was not achieved.** Short-prompt results do not
describe the user's long, sampled tool conversation. All negative candidates
are retained here; none has been promoted as a faster default.

## What the supplied run shows

The running configuration was the previously measured optimized executable,
IQ4_XS expert enabled, FP16 KV, graphs eligible, MTP window 4, 16 threads,
262144 context capacity, temperature 1.0, top-p 0.95, top-k 20, presence penalty
1.5, thinking high and budget 6144. It was not the earlier greedy short-prompt test.

64 completed replies were parsed from the supplied log. Approximate input-context
bins, inferred from the displayed remaining output allowance, showed medians:
32–48K 68.5 tok/s (30 replies); 48–64K 64.9 (19); 64–68K 54.4 (9);
68–80K 30.6 (5). One earlier short reply was 95.9. These are different requests,
not controlled comparisons. The final incomplete reply stalled near 1580 tokens
and reached a displayed 9.3 tok/s; it is excluded from completed medians.

Before stopping the user-approved server, the GPU used 21985/22528 MiB and
available physical RAM was about 515 MiB. During diagnosis, a memory-counter
sample showed 252 MiB available and 11245 pages input/sec. This establishes host
memory pressure, not proof of GPU shared-memory spill or the unique cause of
the original stall. Expert-cache 100% hit does not establish absence of paging.

The live server was stopped with user authorization. Unrelated applications
were not terminated. No global driver or memory settings were changed.

## Bounded diagnostic

Six profiles, one start each, two requests each: **12 completed requests**.
Every profile received the same synthetic inventory prompt (70038 formatted
tokens), 81920 configured context, FP16 unless labelled Q8, 16 threads,
temperature 1.0/top-p .95/top-k 20/presence 1.5, seed 1, and a 128-token reply cap.

The capacity was deliberately bounded below the user's 262144 while retaining
approximately the occupied context at the observed slowdown. This was not a
replay of the user's private tool conversation, nor a 262K capacity qualification.

| Profile | Decode sample 1 | Decode sample 2 | Descriptive median tok/s |
| --- | ---: | ---: | ---: |
| A: original llama.cpp, FP16, no MTP | 62.25 | 63.66 | 62.96 |
| D: combined, FP16, MTP 4 | 62.5 | 59.5 | 61.00 |
| D: FP16, extra 4096 MiB reserve | 41.5 | 44.7 | 43.10 |
| D: experimental Q8 KV | 59.8 | 60.6 | 60.20 |
| D: FP16, two checkpoints | 62.1 | 59.6 | 60.85 |
| D: FP16, no MTP | 47.8 | 48.7 | 48.25 |

These are server-reported rates, not statistically established differences.
The two-checkpoint trial also explicitly disabled parked conversations, already
off by default. Extra reserve reduced the observed expert-cache hit rate to
86%, versus 100% in the other displayed Strata completions.

The original D's first cold request processed its prompt in 45.570 seconds
(1536.9 tok/s), versus A's 100.266 seconds (698.53 tok/s): approximately **2.20x
reported prefill throughput**. Cold request wall times were 47.516 versus
102.375 seconds. This does not establish 2x decode, tool-task completion,
equivalent answers or a general acceleration guarantee.

## Limits that prevent a 2x claim

- One process start per profile and only 128 output tokens. No balanced repeated
  A/D starts or confidence interval.
- Strata reused 70037 prompt tokens on request 2 despite cache_prompt=false.
  A's requests remained cold. Warm Strata wall/TTFT cannot be compared to cold A.
- The reply cap caused Strata to clamp thinking to 64 tokens and inject a close.
  llama.cpp handling differs. These are bounded configuration diagnostics, not
  equivalent full high-reasoning responses or quality acceptance tests.
- Stochastic outputs differ. Q8 changes KV precision and has not passed broad
  long-context quality validation; it was not enabled for normal use.
- RAM/VRAM minima include startup and prefill. They are not decode-only readings.
- Freshly restarted D did not reproduce the user's final stall. Restart recovery
  is not an optimization gain versus A, and a sustained multi-turn tool replay
  remains necessary to diagnose the late-session failure.

## Decision and next engineering target

Keep FP16 and MTP window 4. Do not promote the 4096 MiB reserve or Q8 trial.
Two checkpoints are an optional RAM-retention setting, not a measured speed win.
Their effect on long-session stability and re-reading older conversation turns
needs a longer test.

At this workload, the descriptive A median implies a 2x target near 125.91 tok/s;
current D is near 61.0. It needs roughly another 106% from its current observed
long-context rate. The earlier short-prompt +66% cannot be substituted here.

Further kernel work should first isolate warmed long-context attention, MTP
verification and transfers in a decode-only trace. Historical short-request
CUPTI sums do not establish this new workload's bottleneck. A sustained
multi-turn reproduction must record paging, RAM, VRAM and per-request timing.
Changing context, KV precision or thinking budget to manufacture a score is
not an acceptable matched comparison.

## Practical fix delivered

The missing root Start-Sage-A15-AntiLoop.ps1 now exists. It takes an explicit
-Config, creates a fresh configuration without overwriting the source, retains
the user's AntiLoop sampling/thinking controls, and supports optional
-MaxContext, -PromptCache, -VramReserveMiB and explicitly experimental Q8 KV.
Defaults preserve the input capacity and KV setting. ConfigureOnly
was checked both from the project directory and from Windows/System32 using
the absolute script path.

Example from the prepared PC (replace the config path only if yours differs):

~~~powershell
$root = 'D:\AI\ninfer-strata-test'
$config = "$root\sage-api-20261101-144511-aadea54c\strata-sage-antiloop.json"
& "$root\Start-Sage-A15-AntiLoop.ps1" -Config $config -MaxContext 81920 -ConfigureOnly
& "$root\Start-Sage-A15-AntiLoop.ps1" -Config $config -MaxContext 81920
~~~

This bounded profile fits the tested 70K prompt, not arbitrary future history.
Set the client's total output limit to 8192 if retaining the 6144 thinking
budget, leaving nominal room for an answer; both input and output must fit the
context. The log's allowance near 190K tokens was not limited by the thinking
budget. Summarize older history before the context is exhausted, accepting that
summarization changes retained information and is not a kernel speed gain.

Use -PromptCache 2 only as an optional memory-retention experiment. The script
does not auto-install, rebuild, validate a new model, or guarantee the absence
of loops. Its example launch was not left running after the GPU diagnostics.

## Evidence

- Engine SHA256: b1aa241711c97a708c61209a2baf4809e9fe175d2c2a7c449e201f788faf4ae6.
- llama-server SHA256: 11b1a4aea38359202647a40bb04a8c2300d856a1d5a4521c7806bd0c39003bde.
- Same previously identified Sage IQ4_XS GGUF; no model conversion.
- [All numeric samples and telemetry](../evidence/published-benchmarks/2026-11-01-actual-long-context.json).
- Reproduction: tools/diagnose_long_context.py --config <local-config> --output <new-folder>.
  Its default profiles are D-fp16, D-reserve, A-fp16 and D-q8. Additional
  --profiles D-lowram or D-no-draft select the final two probes.
- tools/export_actual_numeric.py exports measurements and hashes, excluding
  prompt/response text, commands and machine paths. Raw logs/configurations,
  synthetic prompt and the supplied user log remain in the local evidence archive.

