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

Pending.
