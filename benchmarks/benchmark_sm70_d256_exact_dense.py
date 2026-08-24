# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the exact SM70 D256 dense-prefill operator.

Run the baseline and a candidate extension in separate processes.  The optional
reference tensor makes the second run report elementwise differences without
loading two Torch libraries that register the same operator namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
from pathlib import Path
from typing import Any

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-len", type=int, default=8000)
    parser.add_argument("--kv-len", type=int, default=8000)
    parser.add_argument("--heads-q", type=int, default=6)
    parser.add_argument("--heads-kv", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--reference-out", type=Path)
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args()


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _event_trials(
    launch: Any, *, warmup: int, iters: int, trials: int
) -> dict[str, Any]:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            launch()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end) / iters))
    return {
        "median_ms": float(statistics.median(samples)),
        "mean_ms": float(statistics.mean(samples)),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires SM70/V100")
    if args.head_dim != 256:
        raise ValueError("The exact operator requires head_dim=256")
    if args.query_len % 32 or args.kv_len % 32:
        raise ValueError("Query and KV lengths must be divisible by 32")
    if args.heads_q % args.heads_kv:
        raise ValueError("heads_q must be divisible by heads_kv")
    if args.write_reference and args.reference_out is None:
        raise ValueError("--write-reference requires --reference-out")

    interface = importlib.import_module("vllm.vllm_flash_attn.flash_attn_interface")
    del interface
    extension = importlib.import_module("vllm.vllm_flash_attn._vllm_fa2_C")
    op = torch.ops._vllm_fa2_C.sm70_d256_splitd_n32_dense_fwd

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    query = torch.randn(
        (1, args.query_len, args.heads_q, args.head_dim),
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    key = torch.randn(
        (1, args.kv_len, args.heads_kv, args.head_dim),
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    value = torch.randn(
        (1, args.kv_len, args.heads_kv, args.head_dim),
        dtype=torch.float16,
        device=device,
        generator=generator,
    )
    output = torch.empty_like(query)
    scale = args.head_dim**-0.5

    def launch() -> None:
        op(query, key, value, output, scale, True)

    timing = _event_trials(
        launch,
        warmup=args.warmup,
        iters=args.iters,
        trials=args.trials,
    )
    launch()
    torch.cuda.synchronize()
    output_cpu = output.cpu()

    comparison = None
    if args.reference_out is not None and args.reference_out.exists():
        reference = torch.load(
            args.reference_out,
            map_location="cpu",
            weights_only=True,
        )
        if reference.shape != output_cpu.shape or reference.dtype != output_cpu.dtype:
            raise RuntimeError("Reference shape or dtype does not match output")
        difference = (reference.float() - output_cpu.float()).abs()
        comparison = {
            "reference_sha256": _digest(reference),
            "max_abs_diff": float(difference.max().item()),
            "mean_abs_diff": float(difference.mean().item()),
            "equal_elements": int((reference == output_cpu).sum().item()),
            "elements": output_cpu.numel(),
        }
    if args.write_reference:
        assert args.reference_out is not None
        args.reference_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_cpu, args.reference_out)

    payload = {
        "environment": {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "extension": str(Path(extension.__file__).resolve()),
            "query_len": args.query_len,
            "kv_len": args.kv_len,
            "heads_q": args.heads_q,
            "heads_kv": args.heads_kv,
            "head_dim": args.head_dim,
            "dtype": "float16",
            "causal": True,
            "softmax_scale": scale,
            "seed": args.seed,
            "warmup": args.warmup,
            "iters": args.iters,
            "trials": args.trials,
        },
        "timing": timing,
        "output_sha256": _digest(output_cpu),
        "comparison": comparison,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
