# Windows RTX 2080 Ti 22 GB: Sage-A1.5 IQ4_XS smoke results

Test date: 2026-10-29 (Asia/Hong_Kong). These are development smoke results, not hardware acceptance or an MTP benchmark.

## Identity and configuration

- Repository source: `973fcd043d8906b6ac3be72cd9e1c13a3ed5c568`; files were retrieved through the connected GitHub account because local Git was not authenticated for the private repository.
- Pinned PolyStrata: `52c31cfcb55d7e42506a57757886af73f9547314`, with all seven integration patches applied; ggml: `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
- GPU: NVIDIA GeForce RTX 2080 Ti, SM 75, 22,528 MiB reported; NVIDIA driver 616.56, CUDA driver API 13.4. Windows x64, MSVC 19.44.35229 and CUDA compiler 13.0.88.
- Model: `Sage-A1.5-IQ4_XS-exps.gguf`, 19,322,710,880 bytes; full-file SHA-256 `be92beb4e0fda7b40fc4dd2705831095c27792d342833e9d06d9366850bf7069`. This hash identifies the file used; it is not a comparison with an independently published reference hash.
- GGUF v3 header: `qwen35moe`, 41 blocks, 2048 embedding width; 753 tensors: 120 IQ4_XS, 279 Q4_K, 44 Q6_K, 310 F32. The expert tensors include IQ4_XS, while `general.file_type` is 15 (Q4_K_M). Header inspection checks structure and tensor start offsets, not full tensor payload correctness.
- Built `strata-poly.exe` SHA-256: `8cfebf4f3625d86f2edb772dc7834af5cd37c7b2c9bd87d7f3e0321e957d5ec6`.

## Build and independent tests

The native Windows SM75 `strata-poly` build completed. The compiler-only CUDA setup described in the README does not include cuBLAS needed by the complete runtime; configuration initially failed because `CUDA::cublas` was missing. NVIDIA's CUDA 13.0.2 redistributable manifest specifies cuBLAS 13.1.0.3 for Windows; its archive SHA-256 `4ac4847bbe4f7709b244956fcfc32197a2954ee70b155cb67eebd9ee26f7e339` was verified and it was added to the local toolkit. For execution, its `bin/x64` directory was added to the process PATH. The first launch without that directory failed with missing DLL 0xC0000135; the corrected launch succeeded.

Independent GPU numerical checks on the target device:

| Test | Result | Maximum absolute error |
| --- | --- | --- |
| RMSNorm | PASS, 324 cases | 3.89987e-7 |
| Q4_K slots | PASS, 108 cases | 0 |
| Q6_K slots | PASS, 110 cases | 0.00221251, within the test's scale-dependent tolerance |

Nine C++ host CTest tests passed. Python unittest discovery ran 88 tests: 85 passed and three `test_bootstrap` tests errored because Git reported that the generated fixture patch did not apply. An isolated diagnostic that rewrote the generated patch fixtures from CRLF to LF made all three pass; the repository tests themselves were not modified. The real pinned integration patches were successfully applied during bootstrap.

The model's vocabulary, merges, tokenizer metadata and nonempty GGUF chat template were successfully exported.

## API smoke tests

Both baseline and RMSNorm opt-in ran on the same model with `--max-context 32768`, FP16 KV, no draft, no graphs, no expert skipping and frozen expert residency. Both returned `/health` status `ok`, `loaded: true`, and listed `sage-a1.5` with context 32768. A GPU memory sample after baseline readiness was 20,380 MiB used; this is one sample, not a measured peak.

For the same deterministic `Say hello.` request (temperature 0, 128 maximum output tokens), both returned 13 prompt tokens and 128 output tokens, with identical generated reasoning text. Both stopped at the length limit before producing a final answer. The API reported 54.7 output tokens/s for baseline and 53.6 for the RMSNorm run; these are single diagnostic measurements and do not establish a performance difference.

A separate RMSNorm request with a 512-token limit returned `Hello! How can I help you today?` and stopped normally after 257 output tokens. It reused 12 cached prompt tokens, so its timings are not comparable to the first requests.

## Limits and next steps

- The matching short text is smoke evidence only. It does not establish numerical parity, candidate dispatch/enqueue, output quality across prompts, or a speedup.
- The 32768 setting and health response do not establish that a full 32K prompt was processed or that full-context quality is correct.
- The strict launcher disables MTP; no comparison with the reported MTP4 setup or 100 tokens/s was made. IQ4_XS expert matrices follow Strata's native path; there is no integrated NInfer IQ4_XS expert candidate.
- Next, capture dispatch and numerical/logit evidence for the RMSNorm candidate, test longer actual prompt occupancy and free generation, and run repeated matched performance tests with the original MTP settings separately. The original Q4_K_M model lock must not be used to certify this different model.

Follow-up: [RMSNorm dispatch and full-logit comparison](IQ4XS_RMSNORM_INTEGRATION_2026-10-29.md) observed a repeatable difference that failed the selected tolerances; the candidate remains unqualified.
