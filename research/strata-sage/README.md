# Sage q36 Q4_K loader research patch

`q36-q4k-upload.patch` applies to xyehya/Strata revision `379f095027097937f28c5c009cacbbb66504d957` and uses ggml revision `3cf03257f219afbe7334045ff7c6a06ac68c627d`. It adds an FP32 upload conversion for stored Q4_K tensors and a focused CPU test. It does not change model weights or integrate NInfer. See [build and smoke results](../../docs/STRATA_SAGE_FORK_BUILD_2026-10-31.md) for configuration, validation and limits.
