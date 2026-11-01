# NInfer standalone Sage progress — 31 October 2026

**C remains unmeasured as a complete model runtime.** A built Windows SM75 executable now runs a Sage-shaped NInfer expert operation against a numerical oracle. It does not load or generate from the full Sage model. D remains the separately measured Strata runtime with adapted NInfer kernels.

The work uses [NInfer-all at 71d3f819](https://github.com/iamwavecut/ninfer-all/tree/71d3f8190772ef7089f8eda4e22371114fd5bbf9). The pinned Qwen MoE binding does not accept its GGUF expert profile; the existing GGUF expert operator has Flash-Next dimensions (2560 hidden, 640 expert width, top 10). Sage needs 2048 hidden, 512 expert width, 256 experts and top 8.

## Source and artifact mapping

The experimental LogicalSource adapter preserves encoded IQ4_XS expert rows. There were 960 exact byte comparisons across all 40 text layers and all 120 expert banks, sampling expert IDs 0, 1, 127 and 255 and boundary rows. One real expert's gate/up/down matrices also survived artifact serialization and reread byte for byte. This is not a complete converted inference artifact.

The Python 3.11 CPU suite expanded from nine to 17 passing tests. It checks grouped gate/up bindings, GDN head permutations and input-column selection, full-attention encoded row mappings, and independent FP64 IQ4_XS/Q4_K/Q6_K decoders against the gguf CPU decoder. The hard-link artifact test required execution outside the managed sandbox. These checks do not qualify model output.

All 21,053,440 F32 router/shared-gate values in the 40 text layers were exactly representable in BF16. Conversely, decoding Q4_K GDN controls and then casting them to BF16 changed 3,818,585 of 3,932,160 values, maximum absolute change 0.00048732757568359375. A simple control-weight cast therefore changes the model's represented weights. A block-preserving control operation or a separately qualified precision change is still needed.

## Native operator milestone

The experimental operator shares the staged GGUF implementation with the donor while supplying explicit Sage dimensions. It does not call the Flash-Next fused decode path with incompatible dimensions. Both Sage and the original Flash-Next translation units compile for SM75. The extracted helper body was checked against the pinned original; a full Flash-Next runtime regression has not run.

The isolated CMake target builds real core, GGUF bridge and operator dependencies without FFmpeg or the serving engine. Initial linking exposed missing `device_launch.cpp` and `sm70_gemm.cu`; adding those upstream implementations resolved the missing symbols. No mathematical stubs were used. The Release build uses CUDA 13.0.88 and MSVC 19.44. It emitted CUDA header code-page warnings and a static/dynamic C-runtime linkage warning; this is an experimental test executable, not a packaged Windows runtime release.

On the RTX 2080 Ti 22 GB, the following synthetic encoded-weight checks passed:

| Tokens | Route | Relative L2 error | Maximum error / maximum reference magnitude |
| ---: | --- | ---: | ---: |
| 1 | Vector | 0.00898210 | 0.00593249 |
| 8 | Vector | 0.00970755 | 0.00824226 |
| 9 | Matrix | 0.00970093 | 0.01159010 |
| 9 | Vector | 0.01041270 | 0.01181550 |

The independent CPU oracle decodes IQ4_XS routed and Q4_K/Q6_K shared weights, computes products and SiLU in FP64, and sums the given routing weights. The GPU uses Q8_1 activations and BF16 intermediate storage; equality to this reference is not expected. Criteria were relative L2 below 0.04 and normalized maximum error below 0.16, inherited from the donor test, not adjusted after the result. Repeated eager execution was bit-identical. The fixture has 256 table entries backed by two distinct routed matrices, not 256 distinct real expert matrices. The nonresident flag selects the vector path over device-backed fixture weights; it does not test host/disk streaming. No operator speed claim is made.

## Remaining model integration

The graph extension also passed two replays for each of the four cases above,
with byte-identical eager/replay results and the workspace scope returning to
zero. The first rebuild missed changes in the included test fixture; adding an
explicit CMake object dependency caused it to rebuild, and only the executable
whose output lists all four graph checks is counted as graph evidence. This is
operator graph coverage, not a full-model graph qualification.

The next work is the complete converter recipe/frontend, GDN controls and attention mappings, Qwen routing and GGUF operand binding, model-owned weight tables and graph/workspace lifetimes, then full-model logits/quality and a matched A/B/C/D benchmark. MTP, host/disk experts and multi-GPU behavior are not supplied by this operator experiment. C's expected 40% uplift remains a hypothesis.

The [portable patches and reproduction commands](../research/ninfer-all/README.md) preserve the CPU adapter and operator work separately. No model weights or private launch paths are included in those patches.
