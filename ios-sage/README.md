# Sage A1.5 native iPhone: out-of-core MoE experiment

**Status: model expert paging is implemented/tested; Sage token inference is NOT implemented or measured on an iPhone.**

The CUDA-only Strata/NInfer engine cannot be moved directly to iOS. Instead, this independent iOS-compatible Swift module explores the user's proposal: **very low-bit GGUF expert tensors on local NAND, hot experts in a byte-limited RAM cache, cold experts on storage**.

Sage A1.5 text model: Qwen3.5 hybrid MoE, 40 decoder layers, 256 routed experts/layer, 8 chosen/token; shared expert and dense weights are not candidates for routed-expert eviction. Some GGUF manifests count an extra MTP block (41 total) which must not be mistaken for a normal decoder layer.

Model candidate: `mradermacher/Sage-A1.5-GGUF` Q2_K (~13.2 GB). This is *mixed* 2-bit-oriented quantization; it is not proof of acceptable answer quality. Current Swiftlet iPhone 35B demo uses a different 4-bit MLX quantization format and cannot consume Q2_K GGUF tensors without a correct Metal decoder.

## Reproduction

```sh
python3 ios-sage/tools/gguf_expert_index.py /path/to/Sage-A1.5.Q2_K.gguf --out /path/to/sage-experts.json --layers 40
cd ios-sage
python3 -m unittest discover -s tests -v
swift test
```

The indexing tool never loads tensor weights, only GGUF metadata. The Swift pager reads exact gate/down/up slices using `pread`, retains hot expert blobs under a RAM budget, and returns error on failed reads. Cache residency does **not** run the MoE router or compute logits. A model file must be SHA-256 verified independently: matching length is insufficient.

iOS integration: import the local Swift Package into an Xcode iPhone app (iOS 17+). For testing, store a complete GGUF locally, not as a cloud-only placeholder. Apply thermal/jetsam measurements before selecting RAM budget. Real text generation additionally requires the compatible Qwen3.5 Metal graph, mixed Q2_K kernels, tokenizer, exact router, and numerical verification.

**No model, API key, or personal data is included.** This path is separate from the working Windows Strata launcher.

References:
- Sage architecture: https://huggingface.co/InternScience/Sage-A1.5/blob/main/config.json
- Public Sage Q2_K: https://huggingface.co/mradermacher/Sage-A1.5-GGUF/tree/main
- Swiftlet iPhone Metal expert streaming: https://github.com/leonickson1/Swiftlet
- Routide iPhone study: https://github.com/MusaShams/Routide
- GGML block definitions: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
- Apple iOS memory enforcement: https://developer.apple.com/documentation/metal/reducing-the-memory-footprint-of-metal-apps
