# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.fp8 import (
    _SM70_FP8_PREFILL_DENSE_MIN_M,
    _SM70_FP8_PREFILL_DENSE_WORKSPACE_BYTES,
    _SM70_FP8_QWEN38_PREFILL_M,
    Fp8LinearMethod,
    _get_sm70_fp8_prefill_exact_dense_workspace,
    _is_sm70_fp8_prefill_exact_dense_layer,
    _is_sm70_fp8_qwen38_prefill_layer,
    _sm70_fp8_prefill_dense_workspaces,
    _sm70_fp8_prefill_visible_dense_mm,
)


def test_fp8_prefill_exact_dense_is_default_on(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_FP8_PREFILL_EXACT_DENSE", raising=False)
    envs.disable_envs_cache()
    try:
        assert envs.VLLM_SM70_FP8_PREFILL_EXACT_DENSE
    finally:
        envs.disable_envs_cache()


def test_fp8_prefill_visible_dense_mm_is_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_FP8_PREFILL_VISIBLE_DENSE_MM", raising=False)
    envs.disable_envs_cache()
    try:
        assert not envs.VLLM_SM70_FP8_PREFILL_VISIBLE_DENSE_MM
    finally:
        envs.disable_envs_cache()


def test_fp8_prefill_visible_dense_mm_is_long_prefill_only(monkeypatch):
    workspace = torch.empty(24, dtype=torch.float16)
    weight = torch.empty((4, 6), dtype=torch.uint8)
    scales = torch.empty((1, 6), dtype=torch.float16)
    calls = []

    monkeypatch.setenv("VLLM_SM70_FP8_PREFILL_VISIBLE_DENSE_MM", "1")
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.sm70_ops.fp8_sm70_dequantize_out",
        lambda out, qweight, factors, group_size: (
            calls.append((qweight, factors, group_size)),
            out.fill_(1),
        ),
    )
    envs.disable_envs_cache()
    _sm70_fp8_prefill_dense_workspaces[(0, torch.float16)] = workspace
    try:
        long_prefill = _sm70_fp8_prefill_visible_dense_mm(
            torch.ones((_SM70_FP8_PREFILL_DENSE_MIN_M, 4), dtype=torch.float16),
            weight,
            scales,
            workspace.data_ptr(),
            gated_silu=False,
            min_prefill_m=_SM70_FP8_PREFILL_DENSE_MIN_M,
        )
        tail = _sm70_fp8_prefill_visible_dense_mm(
            torch.ones((1, 4), dtype=torch.float16),
            weight,
            scales,
            workspace.data_ptr(),
            gated_silu=False,
            min_prefill_m=_SM70_FP8_PREFILL_DENSE_MIN_M,
        )
        cached = _sm70_fp8_prefill_visible_dense_mm(
            torch.ones((_SM70_FP8_QWEN38_PREFILL_M, 4), dtype=torch.float16),
            weight,
            scales,
            workspace.data_ptr(),
            gated_silu=False,
            min_prefill_m=_SM70_FP8_PREFILL_DENSE_MIN_M,
        )
    finally:
        _sm70_fp8_prefill_dense_workspaces.clear()
        envs.disable_envs_cache()

    assert long_prefill is not None
    assert long_prefill.shape == (_SM70_FP8_PREFILL_DENSE_MIN_M, 6)
    assert tail is None
    assert cached is not None
    assert len(calls) == 2
    assert calls[0] == (weight, scales, 128)
    assert calls[1] == (weight, scales, 128)


def test_fp8_qwen38_prescaled_prefill_is_default_on(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_FP8_QWEN38_PREFILL_PRESCALED", raising=False)
    envs.disable_envs_cache()
    try:
        assert envs.VLLM_SM70_FP8_QWEN38_PREFILL_PRESCALED
    finally:
        envs.disable_envs_cache()


def test_fp8_qwen38_cutlass_prefill_is_default_on(monkeypatch):
    monkeypatch.delenv("VLLM_SM70_FP8_QWEN38_PREFILL_CUTLASS", raising=False)
    envs.disable_envs_cache()
    try:
        assert envs.VLLM_SM70_FP8_QWEN38_PREFILL_CUTLASS
    finally:
        envs.disable_envs_cache()


def test_fp8_prefill_exact_dense_shape_gate_is_narrow():
    layer = SimpleNamespace(
        tp_size=4,
        prefix="model.language_model.layers.1.mlp.gate_up_proj",
        weight=SimpleNamespace(shape=(5120, 8704)),
    )

    assert _is_sm70_fp8_prefill_exact_dense_layer(layer)

    layer.tp_size = 2
    assert not _is_sm70_fp8_prefill_exact_dense_layer(layer)
    layer.tp_size = 4
    layer.prefix = "model.language_model.layers.1.self_attn.qkv_proj"
    layer.weight = SimpleNamespace(shape=(5120, 3584))
    assert _is_sm70_fp8_prefill_exact_dense_layer(layer)
    layer.prefix = "model.language_model.layers.1.linear_attn.in_proj_qkvz"
    layer.weight = SimpleNamespace(shape=(5120, 4096))
    assert _is_sm70_fp8_prefill_exact_dense_layer(layer)
    layer.prefix = "model.language_model.layers.1.mlp.down_proj"
    layer.weight = SimpleNamespace(shape=(4352, 5120))
    assert _is_sm70_fp8_prefill_exact_dense_layer(layer)
    layer.prefix = "model.language_model.layers.3.self_attn.o_proj"
    layer.weight = SimpleNamespace(shape=(1536, 5120))
    assert _is_sm70_fp8_prefill_exact_dense_layer(layer)
    layer.prefix = "model.language_model.layers.1.mlp.gate_up_proj"
    layer.weight = SimpleNamespace(shape=(5120, 8192))
    assert not _is_sm70_fp8_prefill_exact_dense_layer(layer)


def test_fp8_qwen38_prefill_shape_gate_is_narrow():
    layer = SimpleNamespace(
        tp_size=4,
        prefix="model.language_model.layers.1.linear_attn.in_proj_qkvz",
        weight=SimpleNamespace(shape=(5120, 4096)),
    )

    assert _is_sm70_fp8_qwen38_prefill_layer(layer)
    layer.prefix = "model.language_model.layers.3.self_attn.qkv_proj"
    layer.weight = SimpleNamespace(shape=(5120, 3584))
    assert _is_sm70_fp8_qwen38_prefill_layer(layer)

    layer.tp_size = 2
    assert not _is_sm70_fp8_qwen38_prefill_layer(layer)
    layer.tp_size = 4
    layer.weight = SimpleNamespace(shape=(5120, 4096))
    assert not _is_sm70_fp8_qwen38_prefill_layer(layer)


def test_fp8_prefill_exact_dense_workspace_is_bounded():
    assert _SM70_FP8_PREFILL_DENSE_WORKSPACE_BYTES == 85 * 1024**2


def test_fp8_prefill_exact_dense_workspace_is_reused(monkeypatch):
    workspace = torch.empty(1, dtype=torch.float16)
    allocations = []

    def fake_empty(shape, *, dtype, device):
        allocations.append((shape, dtype, device))
        return workspace

    _sm70_fp8_prefill_dense_workspaces.clear()
    monkeypatch.setattr(torch, "empty", fake_empty)
    weight = SimpleNamespace(device=torch.device("cuda:0"))

    try:
        first = _get_sm70_fp8_prefill_exact_dense_workspace(weight)
        second = _get_sm70_fp8_prefill_exact_dense_workspace(weight)

        assert first is workspace
        assert second is workspace
        assert len(allocations) == 1
    finally:
        _sm70_fp8_prefill_dense_workspaces.clear()


def test_fp8_prefill_dispatch_reaches_runtime_op_for_small_and_large_m(monkeypatch):
    calls = []

    def fake_dispatch(
        out,
        dense_weight_ptr,
        input,
        qweight,
        scales,
        group_size,
        k_ld,
        q_ld,
        gated_silu,
        min_prefill_m,
    ):
        assert dense_weight_ptr == 42
        calls.append((input.shape[0], min_prefill_m, gated_silu))
        out.zero_()

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.sm70_ops."
        "fp8_gemm_sm70_prefill_dispatch_out",
        fake_dispatch,
    )
    layer = SimpleNamespace(
        sm70_fp8_turbomind=True,
        sm70_fp8_bmm=False,
        output_size_per_partition=6,
        weight=torch.empty((4, 6), dtype=torch.uint8),
        weight_scale_inv=torch.empty((1, 6), dtype=torch.float16),
        sm70_fp8_k_ld=4,
        sm70_fp8_q_ld=6,
        sm70_fp8_prefill_exact_dense_workspace_ptr=42,
        sm70_fp8_prefill_exact_dense_min_m=_SM70_FP8_QWEN38_PREFILL_M,
    )
    method = SimpleNamespace()

    for m in (1, _SM70_FP8_PREFILL_DENSE_MIN_M):
        output = Fp8LinearMethod.apply(
            method, layer, torch.empty((m, 4), dtype=torch.float16)
        )
        assert output.shape == (m, 6)

    assert calls == [
        (1, _SM70_FP8_QWEN38_PREFILL_M, False),
        (
            _SM70_FP8_PREFILL_DENSE_MIN_M,
            _SM70_FP8_QWEN38_PREFILL_M,
            False,
        ),
    ]


def test_fp8_qwen38_prescaled_scales_only_reach_exact_8k_route(monkeypatch):
    calls = []

    def fake_default(out, input, qweight, scales, *args):
        calls.append(("default", input.shape[0], scales))
        out.zero_()

    def fake_qwen38(out, input, qweight, scales, *args):
        calls.append(("qwen38", input.shape[0], scales))
        out.zero_()

    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.sm70_ops.fp8_gemm_sm70_out",
        fake_default,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8.sm70_ops."
        "fp8_gemm_sm70_qwen38_prefill_out",
        fake_qwen38,
    )
    normal_scales = torch.empty((1, 6), dtype=torch.float16)
    prescaled_scales = torch.empty((1, 6), dtype=torch.float16)
    layer = SimpleNamespace(
        sm70_fp8_turbomind=True,
        sm70_fp8_bmm=False,
        output_size_per_partition=6,
        weight=torch.empty((4, 6), dtype=torch.uint8),
        weight_scale_inv=normal_scales,
        sm70_fp8_qwen38_prefill_scales=prescaled_scales,
        sm70_fp8_k_ld=4,
        sm70_fp8_q_ld=6,
    )
    method = SimpleNamespace()

    monkeypatch.setenv("VLLM_SM70_FP8_QWEN38_PREFILL_FAST_SELECTOR", "1")
    monkeypatch.setenv("VLLM_SM70_FP8_QWEN38_PREFILL_PRESCALED", "1")
    envs.disable_envs_cache()
    try:
        Fp8LinearMethod.apply(method, layer, torch.empty((1, 4), dtype=torch.float16))
        Fp8LinearMethod.apply(
            method,
            layer,
            torch.empty((_SM70_FP8_QWEN38_PREFILL_M, 4), dtype=torch.float16),
        )
        monkeypatch.setenv("VLLM_SM70_FP8_QWEN38_PREFILL_PRESCALED", "0")
        envs.disable_envs_cache()
        Fp8LinearMethod.apply(
            method,
            layer,
            torch.empty((_SM70_FP8_QWEN38_PREFILL_M, 4), dtype=torch.float16),
        )
    finally:
        envs.disable_envs_cache()

    assert [(route, m) for route, m, _ in calls] == [
        ("default", 1),
        ("qwen38", _SM70_FP8_QWEN38_PREFILL_M),
        ("default", _SM70_FP8_QWEN38_PREFILL_M),
    ]
    assert calls[0][2] is normal_scales
    assert calls[1][2] is prescaled_scales
    assert calls[2][2] is normal_scales
