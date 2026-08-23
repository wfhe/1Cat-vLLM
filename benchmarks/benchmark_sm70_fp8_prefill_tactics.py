# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark TurboMind tactics on Qwen3.8 FP8 prefill projections.

This diagnostic loads the real TP-local fused QKVZ and QKV weights used by
Qwen3.8-27B-FP8. Run each dispatch policy in a fresh process: TurboMind caches
the first exact GEMM descriptor, so switching from ``default`` to ``measure``
inside one process would not actually tune the already-cached shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch
import vllm._C  # noqa: F401
from safetensors import safe_open

from vllm import _sm70_ops as sm70_ops

Case = Literal["gdn_qkvz", "full_qkv"]
Policy = Literal["default", "measure", "resident-f16", "resident-f16-measure"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/ymzx/models/Qwen3.8-27B-FP8"),
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument(
        "--cases",
        choices=("gdn_qkvz", "full_qkv"),
        nargs="+",
        default=["gdn_qkvz", "full_qkv"],
    )
    parser.add_argument("--m", type=int, nargs="+", default=[8000])
    parser.add_argument(
        "--policy",
        choices=("default", "measure", "resident-f16", "resident-f16-measure"),
        required=True,
    )
    parser.add_argument(
        "--safe-fast",
        choices=("off", "preserve-splits", "preserve-split-count"),
        default="off",
        help="Optionally constrain measured tactics to the default split contract.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def _load_keys(model: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(weight_map[key], []).append(key)

    result: dict[str, torch.Tensor] = {}
    for filename, file_keys in by_file.items():
        with safe_open(model / filename, framework="pt", device="cpu") as handle:
            for key in file_keys:
                result[key] = handle.get_tensor(key)
    return result


def _column_shard(
    weight: torch.Tensor,
    scales: torch.Tensor,
    tp_size: int,
    tp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(weight.shape[0])
    if n % tp_size:
        raise ValueError(f"N={n} is not divisible by TP={tp_size}.")
    begin = tp_rank * (n // tp_size)
    end = begin + n // tp_size
    scale_begin = begin // 128
    scale_end = math.ceil(end / 128)
    return (
        weight.view(torch.uint8)[begin:end].contiguous(),
        scales[scale_begin:scale_end].contiguous(),
    )


def _load_case(
    model: Path,
    case: Case,
    tp_size: int,
    tp_rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    root = "model.language_model.layers"
    if case == "gdn_qkvz":
        prefixes = [
            f"{root}.1.linear_attn.in_proj_qkv",
            f"{root}.1.linear_attn.in_proj_z",
        ]
    elif case == "full_qkv":
        prefixes = [
            f"{root}.3.self_attn.q_proj",
            f"{root}.3.self_attn.k_proj",
            f"{root}.3.self_attn.v_proj",
        ]
    else:
        raise ValueError(f"Unknown case: {case}")

    keys = [
        key
        for prefix in prefixes
        for key in (f"{prefix}.weight", f"{prefix}.weight_scale_inv")
    ]
    loaded = _load_keys(model, keys)
    raw_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    fp8_dtype: torch.dtype | None = None
    for prefix in prefixes:
        weight = loaded[f"{prefix}.weight"]
        scales = loaded[f"{prefix}.weight_scale_inv"].float()
        if fp8_dtype is None:
            fp8_dtype = weight.dtype
        elif fp8_dtype != weight.dtype:
            raise ValueError("Fused projections must use one FP8 dtype.")
        raw, scale = _column_shard(weight, scales, tp_size, tp_rank)
        raw_parts.append(raw)
        scale_parts.append(scale)

    assert fp8_dtype is not None
    raw = torch.cat(raw_parts, dim=0)
    scales = torch.cat(scale_parts, dim=0)
    qweight = raw.to(device).view(fp8_dtype).contiguous()
    return qweight, scales.to(device).contiguous(), prefixes


def _configure_policy(policy: Policy, safe_fast: str, max_m: int) -> None:
    os.environ["VLLM_SM70_FP8_DENSE_TUNE_MAX_M"] = str(max_m)
    os.environ["VLLM_SM70_FP8_TUNE_SMALL_SHAPES"] = "1" if policy == "measure" else "0"
    os.environ["VLLM_SM70_F16_DENSE_TUNE_MAX_M"] = str(max_m)
    os.environ["VLLM_SM70_AWQ_TUNE_SMALL_SHAPES"] = (
        "1" if policy == "resident-f16-measure" else "0"
    )
    os.environ["VLLM_SM70_FP8_SAFE_FAST_SELECTOR"] = "0" if safe_fast == "off" else "1"
    os.environ["VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS"] = (
        "1" if safe_fast == "preserve-splits" else "0"
    )
    os.environ["VLLM_SM70_FP8_PRESERVE_DEFAULT_SPLITS_ONLY"] = (
        "1" if safe_fast == "preserve-split-count" else "0"
    )


def _event_trials(
    launch: Callable[[], None], warmup: int, iters: int, trials: int
) -> dict[str, float]:
    for _ in range(warmup):
        launch()
    torch.accelerator.synchronize()

    samples_ms: list[float] = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            launch()
        end.record()
        end.synchronize()
        samples_ms.append(float(start.elapsed_time(end) / iters))
    ordered = sorted(samples_ms)
    return {
        "median_ms": float(statistics.median(samples_ms)),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_case(
    args: argparse.Namespace, case: Case, m: int, device: torch.device
) -> dict[str, Any]:
    qweight, scales, prefixes = _load_case(
        args.model, case, args.tp_size, args.tp_rank, device
    )
    n, k = (int(dim) for dim in qweight.shape)
    if (
        k != 5120
        or (case == "gdn_qkvz" and n != 4096)
        or (case == "full_qkv" and n != 3584)
    ):
        raise ValueError(f"Unexpected {case} TP-local shape N={n}, K={k}.")

    tm_weight, tm_scales, meta = sm70_ops.fp8_sm70_prepare(qweight, scales, 128)
    k_ld, q_ld = (int(value) for value in meta.tolist())
    generator = torch.Generator(device=device).manual_seed(args.seed + m)
    inputs = torch.randn(
        (m, k), generator=generator, device=device, dtype=torch.float16
    ).mul_(0.1)
    output = torch.empty((m, n), device=device, dtype=torch.float16)
    resident_bytes = 0
    if args.policy in ("resident-f16", "resident-f16-measure"):
        dense_weight = torch.empty((k, n), device=device, dtype=torch.float16)
        sm70_ops.fp8_sm70_dequantize_out(dense_weight, tm_weight, tm_scales, 128)
        source_f16_weight = dense_weight.t().contiguous()
        f16_tm_weight, f16_meta = sm70_ops.sm70_f16_prepare(source_f16_weight)
        f16_k_ld = int(f16_meta[0].item())
        resident_bytes = int(f16_tm_weight.numel() * f16_tm_weight.element_size())

        def launch(
            output: torch.Tensor = output,
            inputs: torch.Tensor = inputs,
            f16_tm_weight: torch.Tensor = f16_tm_weight,
        ) -> None:
            sm70_ops.sm70_f16_gemm_out(output, inputs, f16_tm_weight, f16_k_ld, False)

    else:

        def launch(
            output: torch.Tensor = output,
            inputs: torch.Tensor = inputs,
            tm_weight: torch.Tensor = tm_weight,
            tm_scales: torch.Tensor = tm_scales,
        ) -> None:
            sm70_ops.fp8_gemm_sm70_out(
                output, inputs, tm_weight, tm_scales, 128, k_ld, q_ld, False
            )

    timing = _event_trials(launch, args.warmup, args.iters, args.trials)
    launch()
    torch.accelerator.synchronize(device)
    seconds = timing["median_ms"] * 1e-3
    result = {
        "case": case,
        "prefixes": prefixes,
        "m": m,
        "n": n,
        "k": k,
        "tm_meta": {"k_ld": k_ld, "q_ld": q_ld},
        "timing": timing,
        "useful_tflops": float(2 * m * n * k / seconds / 1e12),
        "resident_f16_bytes": resident_bytes,
        "output_sha256": _digest(output),
        "output_finite": bool(torch.isfinite(output).all().item()),
    }
    del qweight, scales, tm_weight, tm_scales, inputs, output
    torch.accelerator.empty_cache()
    return result


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device(args.device)
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise RuntimeError("This benchmark requires an SM70/V100 GPU.")
    if args.tp_size <= 0 or not 0 <= args.tp_rank < args.tp_size:
        raise ValueError("Invalid TP size/rank.")
    if any(m < 1 for m in args.m):
        raise ValueError("Every M must be positive.")
    if args.warmup < 1 or args.iters < 1 or args.trials < 1:
        raise ValueError("Warmup, iterations, and trials must be positive.")

    torch.accelerator.set_device_index(device.index or 0)
    _configure_policy(args.policy, args.safe_fast, max(args.m))
    results = [_run_case(args, case, m, device) for case in args.cases for m in args.m]
    extension_path = Path(vllm._C.__file__)
    payload = {
        "environment": {
            "model": str(args.model),
            "vllm_c_extension": str(extension_path),
            "vllm_c_extension_sha256": _file_digest(extension_path),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tp_size": args.tp_size,
            "tp_rank": args.tp_rank,
            "policy": args.policy,
            "safe_fast": args.safe_fast,
            "warmup": args.warmup,
            "iters": args.iters,
            "trials": args.trials,
            "seed": args.seed,
            "tm_gemm_tune": os.environ.get("TM_GEMM_TUNE"),
            "fast_targets": os.environ.get("VLLM_SM70_AWQ_TP2_FAST_TARGETS"),
            "qwen38_prefill_fast_selector": os.environ.get(
                "VLLM_SM70_FP8_QWEN38_PREFILL_FAST_SELECTOR"
            ),
        },
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
