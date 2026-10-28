# GGUF Q6_K expert slots

The slot kernel consumes borrowed, row-major GGUF Q6_K blocks (210 bytes per
256 weights), Q8_1 activation blocks (36 bytes per 32 values), and produces
FP32 outputs on the caller's stream. It allocates, copies, and repacks nothing.
Q8_1's stored sum is ignored: Q6_K has no affine minimum correction.

The byte decoder in `kernels/src/q6k_slots.cu` is adapted from
NInfer-all `src/ops/linear/gguf/ggml_bridge_vec.cuh`,
`Decoder<GGML_TYPE_Q6_K>`, commit `71d3f8190772ef7089f8eda4e22371114fd5bbf9`.
That decoder selects a 128-element half and 32-element slice, combines the low
nibble from `ql` with two bits from `qh`, subtracts 32, and applies one signed
scale per 16 elements. This implementation uses its scalar byte equivalent;
blocks are read bytewise because their 210-byte stride does not preserve
four-byte alignment. No weight or activation conversion is performed.

NInfer's vector decoder computes each FP32 coefficient as `d * signed_scale`,
then multiplies those coefficients by two 16-element integer dot products,
adds them, and multiplies by the Q8_1 scale. This kernel follows that sequence
per 32-element slice. Strata's pinned native Q6_K MMVQ instead multiplies each
integer dot by its signed scale and Q8_1 scale before applying the weight
scale to a partial sum. These are the same real-number equation, but different
FP32 rounding and reduction orders; bitwise equality is not promised. With
finite FP16 `d` and int8 scales, `d * scale` is exactly representable in FP32,
so coefficient rounding itself introduces no discrepancy for valid Q6_K
blocks. Other products and sums can round differently.

`kernels/tests/q6k_slots_parity.cpp` constructs packed bytes from logical codes
and checks them through a bit-address oracle, then compares the CUDA result to
an independent long-double dot with a magnitude-scaled tolerance. It covers
signed scale and code extremes, eight distinct slots, output sentinels, graph
capture and rejection before enqueue, and Sage shapes 512→2048 and 2048→1024.
`SAGE_Q6K_CPU_ORACLE_ONLY` checks fixture and oracle construction alone; it is
not GPU qualification. CUDA execution and performance must be measured on the
target hardware before claiming parity or speed.
