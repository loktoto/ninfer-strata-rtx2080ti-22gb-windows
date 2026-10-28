# Live cache and scheduler validation

The live scheduler is PolyStrata's engine, not the standalone `SlotPool` policy
model. The checkout is based on PolyStrata commit `52c31cfcb55d7e42506a57757886af73f9547314`.
Patch `0005-scheduler-cache-audit.patch` adds an optional audit of that engine's
actual metadata. Set `STRATA_SAGE_AUDIT_CACHE=1` before starting the engine. It
checks slot extents, overlapping owners, upload targets and the cached count at
initial allocation, upload publication/reservation, cache shrink/grow, and
resize. A violation throws with its boundary name. The audit has overhead and
is disabled by default. It does not observe GPU memory accesses or prove that a
CUDA event has completed; it is not a replacement for a GPU race test.

The ordering that needs testing is already present in the pinned engine:

* `follow` removes the outgoing expert's slot before the follower overwrites it;
  the caller synchronizes `st_` before scheduling a trade. The follower records
  an event after its host-to-device copy. `follow` only publishes that slot after
  `cudaEventQuery` succeeds. `settle_swaps` waits for requests to be sent and
  synchronizes the upload stream before publication.
* `shrink_cache` settles uploads and synchronizes compute, queues relocation on
  `st_`, then synchronizes again before VMM unmapping. `grow_cache` settles and
  synchronizes before mapping; its uploads use the compute stream. `trim_card_kv`
  synchronizes the device before unmapping. KV VMM growth maps at reserved virtual
  addresses; it is not a physical-address stability guarantee.
* Captured graphs are retained until engine destruction after stream synchronization.
  The all-cached device-routing graph uses a device slot table; `card_moe` refreshes
  the table when host slot pointers change and gates that path on `all_cached()`.
  These observations do not establish that every graph/KV lifetime case is safe.

The independent policy tests in `tests/slot_pool_test.cpp` use simulated
completion tokens; passing them is not evidence of a live CUDA event or eviction.
Integrating a second slot allocator would duplicate ownership and cannot establish
ordering unless it replaces the engine's existing scheduler. The audit therefore
checks the actual owner's state without introducing a second scheduler.

## Remaining GPU evidence

No GPU was accessible in this workspace. The audit was compiled but has not been
executed against a model. On the target GPU, retain logs, model hash, driver,
CUDA version, build revision and exact options for each test. Use a workload with
actual expert swaps and a constrained reserve so that the relevant boundaries are
reached; audit enablement alone does not prove that they ran.

1. Force or instrument a delayed follower copy; verify a not-ready event leaves
   the target unpublished, then verify publication and output only after completion.
   Repeat under Compute Sanitizer where supported.
2. Force shrink while an upload and compute are outstanding, check that the
   synchronization completes before relocation/unmapping, and compare outputs to
   a no-eviction reference. Force grow and resize and repeat.
3. Cross KV growth and trim boundaries with live sequences and compare retained KV
   contents and subsequent logits against an untrimmed reference. Include multiple
   sequences and allocation failure. The audit checks expert metadata, not KV data.
4. Run the same deterministic tokens with graphs on and off while forcing cache
   changes and KV growth/trim. Compare logits to an agreed numerical tolerance,
   exercise graph replay after slot table refresh and use Compute Sanitizer. Confirm
   that the intended graph path ran; a matching answer alone is insufficient.

Controlled delay/failure injection and end-to-end GPU tests are still needed;
none of the scenarios above is claimed to have passed.

## Reference implementations

[llama.cpp CUDA backend at 8ca4060](https://github.com/ggml-org/llama.cpp/blob/8ca4060b7e30e76baaa5670ee1ec1b760b7f8552/ggml/src/ggml-cuda/ggml-cuda.cu#L2734-L2737)
records a copy event on the source stream and makes the destination stream wait;
its [concurrent graph path](https://github.com/ggml-org/llama.cpp/blob/8ca4060b7e30e76baaa5670ee1ec1b760b7f8552/ggml/src/ggml-cuda/ggml-cuda.cu#L4684-L4688)
uses fork events to order streams. This demonstrates a cross-stream ordering
pattern, not proof of this engine's safety.
[vLLM's KV cache manager at edad8b0](https://github.com/vllm-project/vllm/blob/edad8b03c5285a1c12e7e4f83c1af709b319a01b/vllm/v1/core/kv_cache_manager.py#L379)
allocates blocks for requests and [frees them](https://github.com/vllm-project/vllm/blob/edad8b03c5285a1c12e7e4f83c1af709b319a01b/vllm/v1/core/kv_cache_manager.py#L628)
through its coordinator. Its block management is a different architecture and
is not evidence that PolyStrata's VMM or graph lifetimes have been validated.
