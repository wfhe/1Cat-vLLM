# SM70 Qwen3.8-27B-NVFP4 Decode Recovery

Date: 2026-08-23; acceptance updated 2026-08-24

## Decision

The accepted Qwen3.8-27B-NVFP4 TP4, no-MTP decode route keeps every eligible
FP8 and NVFP4 projection on an SM70 native path. Channel-FP8 projections use
the memory-neutral QPN8 decode kernels for the accepted gate/up, down, and
output shapes. Remaining channel-FP8 projections, including the LM head, use
TurboMind W8A16. NVFP4 projections use TurboMind W4A16. No accepted result in
this document credits a Marlin fallback.

The recovery route also adds E4M3 KV support to Flash-V100 XQA and selects a
p64 partition for the exact batch-one, G6/D256 layout. This is the change that
makes the frozen native-sampler result exceed 70 tok/s. The p64 route is
restricted to E4M3, batch one, G6/D256; the existing E5M2 and FP16 policies
are unchanged. The later no-MTP 80 tok/s acceptance adds exact QPN4, TP4
all-reduce, QPN8 specialization, and an exact-Philox chunked sampler sidecar;
that acceptance is recorded at the end of this document.

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
  engine seed 0. The 2026-08-23 recovery baseline uses the unmodified native
  sampler. The 2026-08-24 acceptance uses the separately registered,
  exact-Philox chunked sampler described below.
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
| Clean pre-merge binaries, native sampler, E4M3 p64 | 71.732 tok/s | 13.941 ms |
| Merged-source confirmation, native sampler, E4M3 p64 | **71.342 tok/s** | **14.017 ms** |

Both p64 requests generated all 256 tokens, contained no EOS, and finished by
length. Relative to the immediately preceding clean-p128 control, the
pre-merge p64 run saves 0.365 ms/token and improves pure decode by 2.61%.
After merging `onecat/main` at `675a12dedc`, rebuilding Flash-V100, and
repeating the frozen request under an exclusive four-GPU reservation, p64
measures 71.342 tok/s at 14.017 ms/token. This is the conservative 2026-08-23
baseline and is superseded by the 2026-08-24 acceptance below.

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

The latest short Nsight Systems node trace uses the merged p64 route. It is
composition evidence; its 14.635 ms traced TPOT is not used as the absolute
speed result because the unprofiled frozen request measures 14.017 ms. The
trace captures 63 decode replays on each TP rank and uses the 61 middle steps
for steady statistics. Graph-node kernel coverage is 93.48%.

| Component | Time/token |
|---|---:|
| TurboMind NVFP4 gate/up | 2.762 ms |
| QPN8 split-16 projections | 2.234 ms |
| TurboMind NVFP4 down | 1.706 ms |
| TP all-reduce | 1.697 ms |
| QPN8 split-12 projections | 0.895 ms |
| E4M3 XQA p64 | 0.587 ms |
| QPN8 fused gate/up | 0.508 ms |
| TurboMind FP8 dense/LM head | 0.407 ms |

Across rank-token samples, the replay interval is 14.621 ms and GPU activity
union is 14.055 ms, or 96.136% of the interval. Idle gaps total 0.565 ms/token.
The stream still launches 1140.8 kernels per rank per token; half are shorter
than 5 us. The grid-limited static occupancy ceiling places 69.40% of service
below 25% occupancy and 28.50% at 25-50%. Nearly continuous GPU work therefore
does not imply high useful compute utilization: batch-one launch geometry and
the serial projection/communication chain remain the main limit.

Fifty-millisecond NVML samples report 99.47% mean GPU busy-window duty and
48.26% memory-active-window duty. Per-GPU power averages 175.9-180.2 W and
peaks at 221.4 W; runtime allocation peaks at 29.06-29.36 GiB/GPU. Model loading
accounts for 5.77 GiB/GPU, while the configured cache budget reports 21.29
GiB/GPU; the remaining allocation includes CUDA graphs, workspaces, state, and
runtime context.

NVML memory duty is not achieved HBM bandwidth. Current trace durations imply
effective minimum packed-weight rates of 451.2 GB/s for NVFP4 gate/up, 365.1
GB/s for NVFP4 down, and 717.4 GB/s for QPN8 down, with useful arithmetic rates
of 1.805, 1.461, and 1.435 TFLOP/s/GPU respectively. These omit scales,
activation/output traffic, caches, and implementation-internal work. A current
NCU capture was attempted, but the host rejected non-root performance-counter
access with `ERR_NVGPUCTRPERM`; exact current SM, Tensor Core, occupancy, and
DRAM counters therefore require administrative profiling permission. The
previously accepted QPN8 counter evidence remains in
`sm70_qwen38_qpn8_decode.md`, but it is not credited as current NVFP4 or p64
XQA counter evidence.

## Rejected Experiments

- NVFP4 N64 and K32 candidates lost to N32 and were removed. The retained
  exact shapes use split 3/swizzle 4 with shape-specific lookahead.
- QPN8 split 17/20/24 lost. Split 12 won only for K1536 x N5120, saving about
  12.7 us/token across the 64 output projections.
- The first compact top-k20 Python sampler and first fused CUDA top20
  candidate did not preserve native Philox sampling and were removed. Both
  2026-08-23 p64 results use the native sampler. They are distinct from the
  later canonicalized, generator-state-preserving implementation accepted on
  2026-08-24.
- Replacing only the full-vocabulary sort while retaining both native
  full-vocabulary softmax operations saved 53.9 us in isolation, insufficient
  for a stable acceptance margin. A one-full-softmax hybrid stayed
  experimental because the p64 attention route solved the target without
  changing sampling math.

## Build, Tests, and Evidence

- Final `_C` SHA256:
  `e0ea14d0e40330b08a9951e67634e50b722540d5d1bbb48500f690323ab07624`.
- Final p64 Flash-V100 SHA256:
  `b418fed86b9c1ab9297c8795c24732818239b9a3aaca5ec9efb60933853d8ce7`.
- Focused CPU policy/dispatch tests: 12/12 passed (nine FP8/QPN8 and three
  E4M3 p64 policy cases).
- Focused E4M3 XQA GPU numerical tests: 6/6 passed.
- Ruff lint/format, Python byte compilation, shell syntax, and
  `git diff --check` pass for the changed files.
- Retained task evidence is under
  `.artifacts/qwen38_nvfp4_speed_20260823/`, notably the final p128/p64 JSON
  results, the merged confirmation
  `final_merged_qwen38_nvfp4_tm_e4m3_xqa_p64_native_sampler_full_graph_i1k_o256.json`,
  `e4m3_xqa_p64_vs_p128_clean.json`, and the parsed per-token Nsight tables
  under `profiles/`.

## 2026-08-24 No-MTP 80 tok/s Acceptance

### Accepted deployment and route

The accepted deployment is deliberately a three-binary composition. It uses
the compatible primary `_C` module with SHA256
`a0a0cd9ddeccc73fa3d920c7a869450c4b33d001f97637c35b75b966d89ad36d`,
the production CMake C++17 sampler sidecar with SHA256
`cdbfdd87dfa9119e52acc88d5787063202561a879a63334145a5616344a549ae`,
and Flash-V100 p64 with SHA256
`b418fed86b9c1ab9297c8795c24732818239b9a3aaca5ec9efb60933853d8ce7`.
MTP remains disabled.

The exact FP8 gate/up, down, and output shapes stay on the proven-faster QPN8
route. GDN input, full-attention QKV, and the LM head stay on TurboMind W8A16.
The accepted batch-one NVFP4 shapes use the native QPN4 decode route; unsupported
or non-admitted cases retain the TurboMind fallback. Thus every eligible FP8
sublayer still uses TurboMind unless the frozen shape has a measured faster
QPN8 specialization.

The sampler sidecar replaces a full-vocabulary sampling fragment with an
80-chunk top-20 reduction while preserving canonical sort order, top-p math,
Philox draws, and generator state. It is built separately so sampler operator
registration does not relink the quality- and speed-frozen primary `_C`.
`setup.py` extracts either CPython-SOABI or abi3 sidecar names, and
`vllm/_sm70_ops.py` loads the bundled module or the explicitly configured
`VLLM_SM70_SAMPLER_LIBRARY`.

### Speed result

All absolute results use the frozen TP4, input-1024/output-256 contract and
255 steady decode intervals. TTFT and prefill are excluded.

| Binary/measurement | Steady decode | TPOT | Disposition |
|---|---:|---:|---|
| Merged native-sampler baseline | 71.342 tok/s | 14.017 ms | baseline |
| Hand sidecar repeat A2 | 80.177 tok/s | 12.472 ms | accepted |
| Hand sidecar repeat A3 | 80.164 tok/s | 12.474 ms | accepted |
| Hand sidecar repeat A4, physical GPUs 4-7 | 80.026 tok/s | 12.496 ms | accepted |
| Production CMake C++17 sidecar A2 | **80.624 tok/s** | **12.403 ms** | accepted |

The hand-sidecar accepted repeats preserve the frozen 256-token SHA256
`0b2d335ddce9b282e45eea1b6c86525bc61eeb5ba1655e8228e6ef3bd1ce823b`.
The production C++17 sidecar changes a low-margin full-model random trajectory,
so random token identity is not treated as the sole quality gate. A direct
cross-build sampler diagnostic covers 100 independent seeds and 256 sequential
draws: hand and production sidecars produce identical token selections and
identical final generator-state SHA256
`4257d205138503840fea92fe6b15ddfa276df82c315513da4bbd785393c98c96`.

### Output-quality gate

The quality contract is the frozen first 250 GSM8K test questions, five-shot
prompting, greedy generation, maximum 256 output tokens, and per-question
record retention. It is intentionally independent of the random performance
prompt.

| Route | Correct | Accuracy | Invalid |
|---|---:|---:|---:|
| Frozen pre-optimization baseline | 226/250 | 90.4% | 0 |
| Hand-sidecar candidate | 227/250 | 90.8% | 0 |
| Production CMake C++17 sidecar | **226/250** | **90.4%** | **0** |

The production result therefore equals the frozen accuracy baseline with no
invalid outputs. Its JSON contains all questions, generated texts, extracted
answers, labels, correctness flags, token IDs, and per-item output hashes. The
result file SHA256 is
`143b2b345b830a858bff2568f9cf51148c8185acb42bb129e2dfc2d421a196d2`.

### Accepted-route trace and resource use

The latest Nsight Systems trace uses the accepted compatible main library and
the behavior-equivalent hand sidecar. It captures 63 graph replays on each of
four ranks and reports the middle 61 steps. Its 13.465 ms node-traced TPOT is
composition evidence; the unprofiled 12.474 ms result remains the absolute
speed evidence. Graph-node kernel coverage is 94.88%.

| Critical component | GPU service/token |
|---|---:|
| QPN8 split-16 projections | 2.213 ms |
| QPN4 fused gate/up | 2.165 ms |
| TP4 pack32 all-reduce | 1.587 ms |
| QPN4 down | 1.441 ms |
| QPN8 output projection | 0.915 ms |
| Flash-V100 E4M3 XQA p64 | 0.618 ms |
| QPN8 fused gate/up | 0.496 ms |
| TurboMind FP8 dense/LM head | 0.410 ms |

The mean replay interval is 13.430 ms and GPU activity union is 12.887 ms,
or 95.958%; idle gaps total 0.543 ms/token and the largest mean gap is only
9.538 us. The route still launches 1061.9 kernels per rank per token. The
grid-limited occupancy ceiling assigns 38.46% of service below 25% occupancy,
47.86% at 25-50%, and 13.63% at 50-75%, showing that batch-one kernel geometry
and the serial projection/communication chain remain the main headroom.

NVML's coarse windows report 100% GPU-busy duty, 53.25% memory-active duty,
and 56.36% of the 300 W power limit. Runtime memory is about 28.97 GiB/GPU;
SM and memory clocks are 1530 and 877 MHz. These are duty indicators, not
achieved SM, Tensor Core, or HBM throughput. Nsight Compute counters remain
unavailable because the driver returns `ERR_NVGPUCTRPERM`. A payload-only
model estimate is about 491 GB/s/GPU and useful arithmetic is about 4.33
TFLOP/s across TP4, but neither value is a hardware-counter measurement.

### Reproducibility boundary and evidence

The production sidecar is reproducible through the CMake target
`_sm70_sampler_C` and its focused microbenchmark selects exactly the reference
tokens across five distributions, 100 explicit-seed trials, and 100 default
generator trials. It reduces the measured fragment from 102.427 to 62.972 us.
The final Python regression run passes all 94 tests in the sampler, NVFP4
admission, and SM70 TurboMind adapter files. Ruff lint/format, Python byte
compilation, the sidecar wheel-name gate, and `git diff --check` also pass.

Freshly relinking the primary `_C` was tested separately. Even after matching
the accepted CUDA cubins for QPN8, QPN4, and custom all-reduce, fresh main
libraries measured roughly 77-80 tok/s and changed the low-margin random stream.
Those main-library builds are rejected; do not substitute them for the
compatible `_C` SHA above until the remaining host/link reproducibility issue
is resolved. Keeping the sampler in its own sidecar is the accepted packaging
boundary.

Primary evidence paths are:

- `.artifacts/qwen38_nvfp4_speed_20260823/results/candidate_qpn4_fold_qpn8_m1_arpack32_official_cxx17_sidecar_chunk80_i1k_o256_a2.json`
- `.artifacts/qwen38_nvfp4_speed_20260823/results/qwen38_nvfp4_gsm8k_250_official_cxx17_sidecar.json`
- `.artifacts/qwen38_nvfp4_speed_20260823/results/sampler_stream_hand.json`
- `.artifacts/qwen38_nvfp4_speed_20260823/results/sampler_stream_cxx17.json`
- `.artifacts/qwen38_nvfp4_speed_20260823/profiles/candidate_qwen38_nvfp4_sidecar_chunk80_nsys_nvml_i1k_o64_per_token.md`
- `.artifacts/qwen38_nvfp4_speed_20260823/profiles/candidate_qwen38_nvfp4_sidecar_chunk80_resource_summary.md`
