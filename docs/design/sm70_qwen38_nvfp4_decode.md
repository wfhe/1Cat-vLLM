# SM70 Qwen3.8-27B-NVFP4 Decode Recovery

Date: 2026-08-23

## Decision

The accepted Qwen3.8-27B-NVFP4 TP4, no-MTP decode route keeps every eligible
FP8 and NVFP4 projection on an SM70 native path. Channel-FP8 projections use
the memory-neutral QPN8 decode kernels for the accepted gate/up, down, and
output shapes. Remaining channel-FP8 projections, including the LM head, use
TurboMind W8A16. NVFP4 projections use TurboMind W4A16. No accepted result in
this document credits a Marlin fallback.

The final route also adds E4M3 KV support to Flash-V100 XQA and selects a p64
partition for the exact batch-one, G6/D256 layout. This is the change that
makes the frozen native-sampler result comfortably exceed 70 tok/s. The p64
route is restricted to E4M3, batch one, G6/D256; the existing E5M2 and FP16
policies are unchanged.

## Frozen Contract

- Model: Qwen3.8-27B-NVFP4, config SHA256
  `1b3c71868d1299e52df6fc907deb202d5132b1ef0f72aae0ef6d15185dd53a5c`.
- Hardware/runtime: four V100-SXM2-32GB GPUs, TP4, Python 3.12, Torch
  2.10.0+cu128, CUDA 12.8.
- Engine: compressed-tensors, FP16 activation/output, E4M3 KV,
  `FLASH_ATTN_V100`, `max_model_len=262144`, `max_num_batched_tokens=8192`,
  `max_num_seqs=1`, prefix caching, aligned Mamba cache, chunked prefill, and
  `FULL_AND_PIECEWISE` CUDA graphs.
- Request: input 1024, natural output cap 256, no `ignore_eos`, no MTP.
- Sampling: temperature 1.0, top-p 0.95, top-k 20, request seed 20260815,
  engine seed 0. The accepted final run uses the unmodified native sampler.
- Pure decode uses the 255 steady token intervals and excludes TTFT/prefill.

## Route Map

| Layer family | TP-local shape | Accepted decode route |
|---|---|---|
| FP8 gate/up | K5120 x N8704 | fused QPN8 gate/SiLU/up |
| FP8 down | K4352 x N5120 | QPN8 split-16 |
| FP8 output | K1536 x N5120 | QPN8 split-12 |
| FP8 GDN input/full-attention QKV and LM head | model shapes | TurboMind W8A16 |
| NVFP4 gate/up | K5120 x N8704 | TurboMind N32, lookahead 1 |
| NVFP4 down | K4352 x N5120 | TurboMind N32, lookahead 2 |
| E4M3 G6/D256 attention | B1, page 1568 | Flash-V100 XQA p64 |

Channelwise `[N,1]` FP8 scales are admitted and packed without retaining a
second permanent weight copy. The existing QPN8 model/shape/concurrency gates
remain in force. The NVFP4 selector is exact-shape gated by
`VLLM_SM70_NVFP4_QWEN38_TP4_M1_FAST_SELECTOR`; setting it to zero restores
dynamic tuning. Setting `VLLM_FLASH_V100_DECODE_PARTITION_SIZE=128` restores
the previous E4M3 attention partition.

## Pure Decode Result

| Milestone | Steady decode | TPOT |
|---|---:|---:|
| Original mixed checkpoint route | 29.35 tok/s | about 34.07 ms |
| Channel-FP8 TurboMind | 57.285 tok/s | 17.457 ms |
| E4M3 XQA p256 | 62.615 tok/s | 15.971 ms |
| Exact channel-FP8 QPN8 | 65.522 tok/s | 15.262 ms |
| E4M3 XQA p128 | 67.880 tok/s | 14.732 ms |
| NVFP4 N32 selectors | 69.991 tok/s | 14.288 ms |
| Clean final binaries, native sampler, E4M3 p128 | 69.904 tok/s | 14.305 ms |
| Clean final binaries, native sampler, E4M3 p64 | **71.732 tok/s** | **13.941 ms** |

The final request generated all 256 tokens, contained no EOS, and finished by
length. Relative to the immediately preceding clean-p128 control, p64 saves
0.365 ms/token and improves pure decode by 2.61%. Both runs used physical GPUs
0-3 under an exclusive reservation, the same `_C`, model/config, sampling,
TurboMind/QPN8 selectors, and graph policy. Only the rebuilt Flash extension
and E4M3 partition selector differ.

Sampled token identity is not an attention quality criterion: one-output-ULP
changes can flip a low-margin random sample. The final p64 stream is coherent
and, independently, reproduces an earlier accepted full-length stream. The
operator gate below is the numerical criterion.

## Numerical and Operator Gates

The E4M3 XQA p64/p128 race uses page size 1568, G6/D256, checkpoint-style
E4M3 bytes, K/V scales 0.04/0.25, and the scalar paged decoder as reference.

| Sequence length | p64 | p128 | p64 gain | Maximum error |
|---:|---:|---:|---:|---:|
| 1025 | 46.633 us | 63.614 us | 16.981 us | 7.63e-6 |
| 1152 | 45.896 us | 59.346 us | 13.450 us | 7.63e-6 |
| 1280 | 46.072 us | 55.625 us | 9.553 us | 7.63e-6 |
| 2049 | 45.807 us | 56.751 us | 10.945 us | 7.63e-6 |

Every p64 and p128 output is within one representable FP16 output ULP of the
scalar reference. The focused GPU regression covers p64/p128/p256 with unit
and non-unit K/V scales and passes 6/6.

The retained NVFP4 selectors also passed independent FP32-oracle checks:
relative L2 is `2.701e-4` for gate/up and `2.924e-4` for down, with cosine
approximately one. The QPN8 source already passed its model-quality and
operator gates recorded in `sm70_qwen38_qpn8_decode.md`.

## Per-Token Profile

The short Nsight Systems node trace is composition evidence; its traced TPOT
is not used as the absolute speed result. Before p64, the steady critical rank
contains approximately:

| Component | Time/token |
|---|---:|
| QPN8 regular projections | 3.151 ms |
| NVFP4 gate/up | 2.740 ms |
| NVFP4 down | 1.702 ms |
| TP all-reduce | 1.676 ms |
| E4M3 XQA p128 | 0.892 ms |
| QPN8 fused gate/up | 0.506 ms |
| TurboMind FP8 LM head | 0.407 ms |

The p64 operator race predicts 0.15-0.27 ms/token attention savings across
the roughly 16 attention launches. The full model measures 0.365 ms/token;
the excess is within run-to-run system variation, so only the measured full
result is used for the final speed claim.

## Rejected Experiments

- NVFP4 N64 and K32 candidates lost to N32 and were removed. The retained
  exact shapes use split 3/swizzle 4 with shape-specific lookahead.
- QPN8 split 17/20/24 lost. Split 12 won only for K1536 x N5120, saving about
  12.7 us/token across the 64 output projections.
- A compact top-k20 Python sampler reduced traced sampler service, but its
  random-sampling trajectory was not deterministic against the native route.
  A fused CUDA top20 candidate had the same acceptance problem. Both were
  removed from production source; the final 71.732 tok/s result uses the
  native sampler.
- Replacing only the full-vocabulary sort while retaining both native
  full-vocabulary softmax operations saved 53.9 us in isolation, insufficient
  for a stable acceptance margin. A one-full-softmax hybrid stayed
  experimental because the p64 attention route solved the target without
  changing sampling math.

## Build, Tests, and Evidence

- Final `_C` SHA256:
  `e0ea14d0e40330b08a9951e67634e50b722540d5d1bbb48500f690323ab07624`.
- Final p64 Flash-V100 SHA256:
  `1fa566a28961d0585a9b6e1ee39e8fa637c4fe13ae894a9a9c07e09d850d02ca`.
- Focused CPU policy/dispatch tests: 12/12 passed (nine FP8/QPN8 and three
  E4M3 p64 policy cases).
- Focused E4M3 XQA GPU numerical tests: 6/6 passed.
- Ruff lint/format, Python byte compilation, shell syntax, and
  `git diff --check` pass for the changed files.
- Retained task evidence is under
  `.artifacts/qwen38_nvfp4_speed_20260823/`, notably the final p128/p64 JSON
  results, `e4m3_xqa_p64_vs_p128_clean.json`, and the parsed per-token Nsight
  tables under `profiles/`.
