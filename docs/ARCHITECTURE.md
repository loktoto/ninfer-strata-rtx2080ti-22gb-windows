# Execution and acceptance boundaries

Strata remains the sole inference runtime: it reads GGUF, owns dense weights,
expert slots, CPU workers, KV/recurrent state, streams and graph lifetimes. The
NInfer-derived library accepts borrowed pointers and a caller stream; it owns no
model, allocator, tokenizer, server or CUDA context. Its first operation is FP32
RMSNorm, adapted to preserve Strata's activation representation.

This operation is useful for establishing the link/stream/numerical boundary. The Q4_K candidate also reads cached expert-slot pointers directly at the gate/up
and down projections. The Q6_K candidate now uses NInfer-all's byte decoder on
those same pointers. Device-routed all-cached ID dispatch and unsupported calls
retain original kernels. All candidates remain opt-in. GPU parity and
whole-model validation are required for G2; no Tensor Core speedup is claimed.

The independent `sage_scheduler` host library provides slot generation and reader
lease contracts plus checked KV budget reservations. It is not wired into Strata's
live CUDA scheduler yet. Completion tokens are supplied by the caller, not GPU
events created or polled by this library. Its tests validate host state transitions,
not CUDA race freedom. Future integration must translate actual CUDA event
completion into these transitions and retain live graph references before eviction.

The unlock repository is pinned research input, not a runtime library. It currently
implements host parsers and simulated lifecycle contracts. No Windows EFI unlocker
or measured post-driver speed change is available. Nothing here writes firmware,
changes TDR, changes boot configuration or assumes the GPU is unlocked. Its
GPL-2.0-only code is not copied or linked into the runtime.

## Sage identity and memory

`config/model.lock.json` pins the official InternScience Q4_K_M GGUF, including the
publisher's LFS SHA256 and size (21,713,463,520 bytes). The file has not been fully
downloaded or hash-verified locally. The reference config is separately pinned
and is checked against the bounded official GGUF header inventory in
`config/sage-gguf-inventory.json`. Only a 16 MiB prefix was downloaded; the complete
weight payload and its checksum remain unverified.

Reference geometry: Qwen3.5 MoE, 40 layers, width 2048, 256 routed experts, top-8,
expert width 512, 10 full-attention layers, 2 KV heads, head width 256. The reference
maximum position count is 262144; this is not an observed working context limit.
The GGUF declares 41 blocks and one next-token prediction layer, consistent with
40 main layers plus the draft layer. Its 753 tensors comprise 379 Q4_K, 64 Q6_K
and 310 F32 tensors. This inventory is why the integration retains mixed tensor
types and leaves draft execution disabled for baseline qualification.

For these full-attention dimensions, uncompressed FP16 KV alone is 20,480 bytes
per token: 0.625 / 1.25 / 2.5 / 5 GiB at 32K / 64K / 128K / 256K. These estimates
exclude recurrent state, alignment, scratch, dense weights, expert staging,
graphs, driver/display use and any draft state. They explain why dynamic expert
residency matters even when a roughly 20.2 GiB GGUF resembles a 22GB VRAM budget.

## Next integration gates

1. Build patched `strata-poly` on Windows SM75; inspect actual GGUF and pass short
   prompt/logit checks before a real 32768-token prompt plus output allowance.
2. Compare original versus NInfer-derived RMSNorm with independent operator oracle
   and teacher-forced model logits. Only enable after target evidence; measure
   complete inference latency before claiming an improvement.
3. Qualify and optimize the implemented GGUF Q4_K slot operations. NInfer's signed groupwise Q4 representation
   differs from Q4_K's 256-element block scales/minima and mixed tensor recipe.
   Never reinterpret the bytes or silently requantize. Any unpacking workspace
   must be bounded, owned and budgeted by Strata.
4. Integrate event-backed slot leases and KV reservations; test delayed CPU work,
   copies, eviction, graph pointer updates and allocation failure on the real GPU.
5. Qualify each context size for full prompt consumption, retrieval and document
   quality. Enable MTP only after inspecting GGUF draft tensors and baseline parity.
6. Implement/validate the Windows boot unlock separately; run matched MMA and
   whole-model A/B tests after driver takeover before using unlock in tuning.

All hardware gates currently remain not run. Hosted Windows compilation and host
unit tests cannot promote those gates.
