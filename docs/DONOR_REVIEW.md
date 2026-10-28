# NInfer donor review — 2026-10-28

## mr-september/ninfer-2080ti-22g

Reviewed master at [`8dfa5cc96620b646be4608ffb1246cb89df9a444`](https://github.com/mr-september/ninfer-2080ti-22g/tree/8dfa5cc96620b646be4608ffb1246cb89df9a444), confirmed against the remote branch on 2026-10-28. Its published [v0.2.0 Windows release](https://github.com/mr-september/ninfer-2080ti-22g/releases/tag/v0.2.0) is substantial work on the intended 22GB RTX 2080 Ti. It is a useful donor and Windows reference, but does not implement Sage-A1.5 GGUF MoE inference.

The author reports native Windows 11, CUDA 12.9 and driver 610.88 validation, 134 passing Windows tests, protocol-B perplexity 4.796799, matching logits and surviving device code against its previous release, and CLI/server execution from an extracted ZIP with a minimal PATH. These are upstream reports; we have not reproduced them. Release notes and current master documentation differ on DFlash2 artifact availability, and both leave end-to-end DFlash2 testing pending. Do not treat its feature names as completed Sage features.

| Component | Finding and integration decision |
| --- | --- |
| Model delivery | Current product is dense Qwen3.8-27B, with same-architecture Qwen3.6-27B support. Its README explicitly says the retained MoE code lacks working Turing kernels. It cannot replace Sage's runtime directly. |
| Quantized matrix kernels | Native Q4/Q5/Q6 use signed codes and FP16 scales per 64 weights in a v3 `.ninfer` container. These differ from GGUF Q4_K/Q6_K blocks. The converter has no GGUF reader; converting into this format would change the fixed-weight baseline. |
| Windows distribution | Adopted the portable executable/DLL packaging approach in `tools/package_windows.py`, extended with recursive import inspection, missing/conflicting DLL rejection and file hashes. See [WINDOWS_PACKAGE.md](WINDOWS_PACKAGE.md). Native ZIP execution is still pending. |
| Correctness and performance procedure | Reuse its insistence on full numerical comparisons, task quality, same-session performance trials and minimal-PATH package execution. Our exact-token/full-logit tools provide part of that procedure; real Sage perplexity and Windows validation remain pending. Its Qwen performance figures do not predict Sage performance. |
| Gated delta recurrence | Candidate for a later measured adaptation: 128-element state, four query/key values per lane, four value rows per warp, and register-resident state across tokens. Its interfaces use BF16 inputs/output and grouped head division. Strata uses FP32 intermediates, modulo head mapping, a different state layout, and fused normalization/gating. Preserve those semantics; direct replacement is incorrect. Strata already keeps state in registers across tokens, so a benefit must be demonstrated. No GDN donor code is integrated. |
| Serving and speculation | Its protocol tests and MTP validation are useful references. Copying its model-specific program/state ownership would require a separate Sage integration. Our existing Python API wrapper and strict non-MTP baseline remain the current implementation. |
| Hardware unlock | The working native Windows release does not establish a Windows TU102 unlock or its persistence after driver loading. G5/G6 remain unqualified. |

Pinned source references: [README](https://github.com/mr-september/ninfer-2080ti-22g/blob/8dfa5cc96620b646be4608ffb1246cb89df9a444/README.md), [Windows packager](https://github.com/mr-september/ninfer-2080ti-22g/blob/8dfa5cc96620b646be4608ffb1246cb89df9a444/scripts/package.ps1), [recurrence access/kernel](https://github.com/mr-september/ninfer-2080ti-22g/blob/8dfa5cc96620b646be4608ffb1246cb89df9a444/src/ops/linear_attention/gated_delta_net/recurrence/recurrent_io.cuh), [transition primitives](https://github.com/mr-september/ninfer-2080ti-22g/blob/8dfa5cc96620b646be4608ffb1246cb89df9a444/src/ops/linear_attention/gated_delta_net/kernels/transition.cuh), [reported Turing validation](https://github.com/mr-september/ninfer-2080ti-22g/blob/8dfa5cc96620b646be4608ffb1246cb89df9a444/docs/turing-port.md).

## NInfer-all

Reviewed [`iamwavecut/ninfer-all` at `71d3f8190772ef7089f8eda4e22371114fd5bbf9`](https://github.com/iamwavecut/ninfer-all/tree/71d3f8190772ef7089f8eda4e22371114fd5bbf9). It retains broader GGUF and MoE primitives than the dense-only mr-september product. Those primitives do not by themselves provide a complete Sage GGUF model path. Native MoE geometry and GGUF decoding are separate contracts.

The concrete reuse is now implemented: its `Decoder<GGML_TYPE_Q6_K>` byte interpretation is adapted into `kernels/src/q6k_slots.cu`, reading Strata-owned slots and existing Q8_1 activation scratch on the caller's stream. No second model bank, weight conversion or repacking is added. Sage's locked GGUF header includes Q6_K expert-down tensors, making this a relevant format alongside Q4_K. See [Q6K_DONOR.md](Q6K_DONOR.md) for rounding differences, independent oracle coverage and the unrun GPU qualification.

Source and component pins are in `config/sources.lock.json`. Keep the current single Strata owner and test donor kernels against the exact Sage file before enabling them by default. All three kernel candidates remain opt-in. Build/host evidence is recorded in [VALIDATION.md](VALIDATION.md); hardware acceptance status is recorded in [TEST_READINESS.md](TEST_READINESS.md).
