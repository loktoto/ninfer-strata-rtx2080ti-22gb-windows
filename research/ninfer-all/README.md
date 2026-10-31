# Experimental NInfer-all CPU source-binding checks

This patch applies to NInfer-all commit `71d3f8190772ef7089f8eda4e22371114fd5bbf9`. It is **not a complete converter or a runnable inference path**. It adds expert-row LogicalSources, a CPU decoder and a mathematical oracle with tests. It does not alter GPU/runtime code, quantize model weights or include model data.

With Python 3.11, install NInfer's converter dependencies including CPU PyTorch, NumPy, safetensors and gguf into an isolated environment. From a checkout at the pinned commit:

```text
git apply --check sage-iq4xs-cpu-experiment.patch
git apply sage-iq4xs-cpu-experiment.patch
python -m unittest tests.convert.test_sage_gguf_experts tests.convert.test_sage_iq4xs_oracle -v
```

Without the optional input, real-model tests skip. Set `NINFER_TEST_SAGE_GGUF` to the explicitly selected Sage IQ4_XS GGUF file to run those checks. They read small samples plus one expert's three matrices; the artifact test creates a temporary artifact and uses a hard link. It needs a filesystem and execution environment that allow hard links. No fixture prompts, model bytes or private machine paths are included in this patch.

The represented-weight decoder and FP64 expert reference have not been compared with a Sage-shaped CUDA kernel. Q8_1 activations, BF16 intermediate storage, routing, graph capture and end-to-end model behavior need separate qualification.
