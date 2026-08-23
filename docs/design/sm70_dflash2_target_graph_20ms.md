# SM70 DFlash2 target-verifier graph: 20 ms control log

## Scope and frozen baseline

- Date: 2026-08-23.
- Integration base: `c62c1dcd833458054daa12e134695aa19a4ac609`
  (`codex/v100-dflash2-gdn-metadata-20260822-122723`, Draft PR #257).
- Worktree:
  `/home/ymzx/桌面/1cat-vllm/worktrees/v100-dflash2-target-graph-20ms-20260823-084427`.
- Branch: `codex/v100-dflash2-target-graph-20ms-20260823-084427`.
- Task cache: `/data/minimax-h3/task-cache/v100-dflash2-target-graph-20ms`.
- Frozen workload: Qwen3.8-27B-FP8 target, official DFlash2 draft, TP4,
  batch one, block eight (seven draft tokens), Flash-V100, target E5M2 KV,
  draft FP16/auto KV, and CUDA Graph.
- Baseline node trace:
  `/data/minimax-h3/task-cache/v100-dflash2-pr257-gdn-metadata/profiles/latest-e2e/dflash2-fused-viewcache-b1-nodes-o128-v5.sqlite`.

The 30-round, four-rank steady-state baseline (one edge round removed from
each side) is:

| Phase | Critical-path wall | Nodes or launches per rank |
|---|---:|---:|
| Draft graph | 4.046 ms | - |
| Draft to target | 4.934 ms | 153 kernels |
| Target graph | **24.740 ms** | **2612 nodes** |
| Target to draft | 2.838 ms | - |
| Complete round | 36.300 ms | - |

The target-graph objective is at most 20.000 ms under this same diagnostic.
Acceptance requires an improvement in the complete round as well as identical
greedy tokens and acceptance trajectory. A later probabilistic quality gate
must preserve the accepted-length distribution and dataset quality.

## Resource-use decomposition

The target graph contains 23.365 ms of rank-average GPU kernel service inside
a 24.740 ms critical span, or 94.7% activity-envelope coverage. This rules out
a large idle bubble as the primary cause, but it does not imply efficient SM
or memory-pipeline utilization within each small kernel. Nsight Compute metrics
must remain separate from graph-span and Nsight Systems service time.

| Kernel category | Rank-average service | Launches per rank |
|---|---:|---:|
| TurboMind FP8 dense GEMM | 10.831 ms | 256 |
| Copy/cast elementwise | 3.387 ms | 941 |
| Other kernels | 2.869 ms | 240 |
| TP all-reduce/communication | 2.316 ms | 128 |
| Other Torch/Triton elementwise | 2.494 ms | 739 |
| RMSNorm/residual | 0.616 ms | 128 |
| LM-head/sample/TP gather | 0.416 ms | 48 |
| Dense GEMV/GEMM/compressor | 0.257 ms | 65 |
| Fill/mask | 0.179 ms | 67 |

The graph is therefore busy but fragmented: 1680 copy/cast/elementwise nodes
consume 5.881 ms, while the average node is only about 8.9 microseconds. The
20 ms target requires recovering at least 4.740 ms (19.2%); launch reduction
and work fusion are first-order requirements, not optional cleanup.

## Ordered experiments and stop gates

1. A/B the repaired, default-off DFlash2 packed GDN verifier with all other
   flags frozen. It preserves the current gating values, recurrent arithmetic
   order, and FP32-state contract while removing packed-QKV rearrangement and
   the final output copy. Record route-hit logs, graph wall, graph nodes,
   complete-round wall, exact output tokens, and acceptance trajectory.
2. If the packed route wins but remains above 20 ms, use its new node trace to
   isolate residual GDN state/copy traffic. Compare with SGLang-V100's fused
   recurrent verifier and all-layer state commit without importing unrelated
   scheduler or Eagle/MTP changes.
3. Profile the exact TP4 all-reduce shape. The current 2.316 ms/rank service is
   a second independent target; any replacement must preserve collective
   ordering and exact target hidden states.
4. Only after the verifier contract is stable, assess a DFlash2-safe variant of
   v100-skinny's small-M QPN8 projection work. It is not accepted on the basis
   of its non-speculative result.

Every failed experiment must be reverted or kept behind a default-off gate.
Do not report profiler-instrumented latency as unprofiled throughput. Do not
use target-only runs, and do not occupy a partial TP4 group or terminate an
unrelated process.

## External references

- SGLang-V100 fixed source and trace are compared locally at
  `/data/minimax-h3/task-cache/v100-dflash2-pr257-gdn-metadata/sources/sglang-v100`
  and
  `/data/models/v100-dflash2-20260820/sglang-audit/perf-rootcause/sglang-dflash2-single1-step20-v2.nsys-rep`.
- v100-skinny is pinned at `5b589c0dc81223e0ba65bcb3e755874723f8b515`;
  its 219.1 tok/s result is a Qwen3.8 mixed-NVFP4/FP8 MTP result, not a DFlash2
  target-verifier baseline.

## Results

### Accepted target-graph reduction

The matched TP4 node trace at source `500130882ba68fb7bac3a0b9e4eb5872647fb045`
uses the same model, draft, request, sampling, KV dtypes, and graph policy as the
frozen baseline:

- SQLite:
  `/data/minimax-h3/task-cache/v100-dflash2-target-graph-20ms/profiles/dflash2-fused-gdn-norm-split-gemma-dynamic-b1-nodes-o128-gpu0123-v3.sqlite`.
- Result JSON:
  `/data/minimax-h3/task-cache/v100-dflash2-target-graph-20ms/results/dflash2-fused-gdn-norm-split-gemma-dynamic-b1-nodes-o128-gpu0123-v3.json`.
- Four-rank synchronized target graph, with the first and last rounds removed:
  **19.317 ms mean, 19.322 ms p50, 19.444 ms p90, 19.477 ms p99**, and
  **1,257 nodes/rank**.
- Rank-average kernel service is 18.552 ms, or 96.04% of the critical span.
- Relative to the frozen 24.740 ms / 2,612-node baseline this removes
  **5.423 ms (21.92%)** and **1,355 nodes (51.88%)**. The p99 is also below
  the 20 ms objective.
- The benchmark endpoint inside this profiled run reports 125.060 steady
  decode token/s; this is trajectory evidence, not an unprofiled throughput
  claim. All 128 output
  token IDs and hash `fe0300...` match the baseline exactly. Probabilistic
  acceptance changed slightly from 4.129 (31 draft rounds) to 4.000 (32 draft
  rounds), so the Gemma suffix fusion remains a Type-B candidate pending a
  distribution-quality gate rather than being accepted from one trajectory.

The reduction is cumulative: the one-pass GDN output norm first reached
22.646 ms / 2,084 nodes; fused GDN split materialization reached 21.625 ms /
1,892 nodes; the dynamic-shape-correct Gemma residual RMS fusion reached the
19.317 ms / 1,257-node result above. A prior `v2` launch omitted the installed
`flash_attn_v100_cuda` extension directory from `PYTHONPATH` and failed during
graph capture; it is not a performance result.

### Remaining graph-boundary cost

Across 31 steady boundaries per rank, the draft-graph end to target-graph
start interval is 5.776 ms mean (5.582 ms p50). Only 1.304 ms is covered by
non-graph GPU kernels, leaving about 4.47 ms of launch, synchronization, and
CPU queueing bubbles. Every boundary submits 153 kernels:

- one TP `cross_device_reduce_1stage`: 0.785 ms;
- 20 `DeviceScanKernel`, 20 `DeviceScanInitKernel`, 20 `compute_cuda_kernel`,
  and 21 `indexSelectSmallIndex` calls: 0.276 ms service but much larger
  serialized submission cost;
- remaining input, slot, block-table, GDN metadata, and elementwise kernels:
  about 0.243 ms service.

The scan/select work arrives in five serial groups with scalar D2H ordering
points. Therefore the next boundary objective is not a faster individual
3-microsecond kernel. It is to construct persistent target metadata once and
capture or batch-submit the five repeated groups while preserving every
buffer/update dependency.

The same parser applied to the retained SGLang-V100 trace gives a useful
matched structural target. Its draft-to-target interval is 3.398 ms mean with
30 non-graph kernels and 0.115 ms GPU service, versus this branch's 5.776 ms,
153 kernels, and 1.304 ms service. SGLang still pays a 2.715 ms host
`cudaStreamSynchronize`, so it is not a zero-overhead endpoint. The immediate
vLLM gap is nevertheless concrete: remove 123 repeated launches and move the
roughly 0.785 ms TP reduction adjacent to target input/embedding preparation
into the target replay dependency chain. This comparison comes from
`/data/models/v100-dflash2-20260820/sglang-audit/perf-rootcause/sglang-dflash2-single1-step20-v2.sqlite`.

### Preliminary probabilistic quality pair

A fixed 16-question sequential GSM8K pair used graph TP4, target E5M2 KV,
draft FP16 KV, block eight, and official `temperature=1.0/top_p=.95/top_k=20`
sampling. The control artifact is
`quality/gsm8k-16-control-gemma-off.json`; the candidate artifact is
`quality/gsm8k-16-candidate-gemma-on.json` under the task cache.

- Control: 68.75% accuracy, aggregate acceptance length 4.470.
- Candidate: 75.00% accuracy, aggregate acceptance length 4.693.
- Eight of 16 full output-token trajectories match. Eight diverge under
  probabilistic sampling, as expected from the accepted one-FP16-ULP numeric
  bound; this small sample cannot establish a quality improvement.

The Gemma fusion stays default-off until paired prompt perplexity and broader
dataset gates show no regression. The GDN norm and split stages retain their
Type-A classification and can be defaulted independently of that decision.

The first deterministic distribution probe scores 1,850 prompt tokens from 32
fixed GSM8K questions in eager target prefill (Graph is irrelevant to prompt
logprobs). Weighted perplexity is 4.20119 for the decomposed control and
4.20172 for the Gemma fusion, a +0.00053 / +0.013% change. All 32 next-token
argmaxes match. Mean absolute prompt-logprob difference is 0.00173 and the
worst token is 0.02524. This is small enough to continue dataset testing but
confirms that the kernel is Type B, so it remains default-off.

### Fused Flash-V100 small-query metadata candidate

The five repeated scan groups are generated by
`FlashAttnV100MetadataBuilder._update_smallq_decode_metadata`: each of five
target KV groups performs four `repeat_interleave` scans, then materializes and
copies block-table and sequence-length temporaries. The
`VLLM_SM70_DFLASH2_FUSED_SMALLQ_METADATA` candidate writes the persistent
block-table, sequence-length, and query-boundary buffers directly with one
Triton launch per group. It is limited to a DFlash2 target on SM70; draft,
DDTree, Eagle, MTP, and CPU metadata retain the existing path.

- Exact V100 tests pass for B1 and mixed B3/B4 layouts, negative block IDs,
  zero-length padded requests, and graph-token padding.
- The unchanged CPU persistent-buffer, padding, and overflow tests pass.
- A realistic `q=8`, three block columns, five-group microbenchmark reports
  1.278 ms legacy versus 0.114 ms fused wall time and 1.235 versus 0.100 ms GPU
  service: **1.164 ms saved / 11.2x** for this isolated constructor.

The matched TP4 end-to-end trace accepts the candidate as a Type-A default-on
optimization (explicit environment value `0` remains an opt-out):

- Synchronized draft-to-target falls from 5.720 ms to 1.911 ms (-66.6%).
- Per-rank non-graph work falls from 153 nodes / 1.304 ms GPU service to 23
  nodes / 0.195 ms. All 20 `DeviceScan`, 20 `DeviceScanInit`, and 20 generic
  scan-compute launches disappear; `indexSelectSmallIndex` falls from 21 to 1.
- The five replacement launches total 0.016 ms of GPU service.
- The full speculative round falls from 31.791 ms to 27.951 ms (-12.1%), while
  the target graph remains effectively unchanged at 19.261 ms / 1,257 nodes.
- Profiled single-request steady decode rises from 125.06 to 142.37 token/s
  (+13.8%). The 128-token output hash, aggregate acceptance length 4.0, and
  per-position accepted counts `[28, 23, 18, 14, 8, 3, 2]` match exactly.

The accepted trace is
`profiles/dflash2-fused-gdn-norm-split-gemma-smallqmeta-b1-nodes-o128-gpu0123-v1.sqlite`
under the task cache. Its 1.911 ms synchronized boundary is also below the
3.398 ms boundary in the retained SGLang-V100 audit trace, though that external
trace is a structural reference rather than a fully matched throughput run.

### Next boundary: target-to-draft

The accepted trace leaves a stable 2.738 ms target-to-next-draft interval on
rank 0 (33 steady intervals). A representative `M=8` interval and the aggregate
kernel service separate it as follows:

| Stage | Representative wall / service |
| --- | ---: |
| Dense target LM head, local FP16 shard | 0.947 ms |
| TP full-vocabulary all-gather | 0.139 ms |
| Full-vocabulary top-k/top-p | 0.606 ms |
| Rejection statistics, rejection, and resample | 0.225 ms |
| Target-hidden-to-draft KV precompute | about 0.60 ms |
| Post-update and next-draft metadata | about 0.22 ms |

The next candidate must therefore attack the first four rows as one semantic
unit. For the fixed no-penalty `top_k=20, top_p=0.95` contract, each TP rank can
compute its exact local top 20, exchange only 20 `(score, token-id)` pairs per
row, merge the global top 20, and perform rejection/residual sampling on that
sparse support. The repository already contains the SM70 TurboMind FP16
LM-head top-20 epilogue, but the `_C` binary linked into this worktree predates
that op. The first TP4 microbenchmark was therefore rejected at route checking
without recording a timing; an incremental SM70 build is required before this
candidate can pass the microbenchmark gate.
