# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the sampled-token OOB sentinel kernel (SM70 DFlash2 L1 floor).

The kernel clamps out-of-range (< 0 or >= vocab_size) sampled token ids in
place and records (value, req_id, slot) for the first 8 violations. Only the
first ``num_sampled[req]`` entries of each row are valid; the rest of the
buffer is uninitialized fill and must be left untouched.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.gpu.input_batch import sampled_tokens_oob_sentinel

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="sentinel kernel requires a CUDA/ROCm device",
)

VOCAB_SIZE = 64
EVENT_SIZE = 1 + 3 * 8


def _run(sent: torch.Tensor, num_sampled: torch.Tensor) -> tuple[int, torch.Tensor]:
    event = torch.zeros(EVENT_SIZE, dtype=torch.int32, device="cuda")
    count = sampled_tokens_oob_sentinel(sent, num_sampled, VOCAB_SIZE, event)
    return count, event


@torch.inference_mode()
def test_clean_batch_untouched() -> None:
    set_random_seed(0)
    sent = torch.randint(0, VOCAB_SIZE, (4, 8), device="cuda", dtype=torch.int32)
    valid = sent.clone()
    num_sampled = torch.tensor([0, 1, 3, 8], device="cuda", dtype=torch.int32)
    count, _ = _run(sent, num_sampled)
    assert count == 0
    assert torch.equal(sent, valid)


@torch.inference_mode()
def test_exactly_vocab_size_clamped() -> None:
    # The Exp A2 root-cause signature: the sampler emits exactly V.
    sent = torch.zeros(2, 8, device="cuda", dtype=torch.int32)
    sent[0, 7] = VOCAB_SIZE  # bonus slot, full-rejection case
    num_sampled = torch.tensor([8, 1], device="cuda", dtype=torch.int32)
    count, event = _run(sent, num_sampled)
    assert count == 1
    assert sent[0, 7].item() == VOCAB_SIZE - 1
    assert event[1].item() == VOCAB_SIZE  # recorded raw value
    assert event[2].item() == 0
    assert event[3].item() == 7


@torch.inference_mode()
def test_negative_clamped_to_zero() -> None:
    sent = torch.zeros(1, 8, device="cuda", dtype=torch.int32)
    sent[0, 2] = -5
    num_sampled = torch.tensor([3], device="cuda", dtype=torch.int32)
    count, event = _run(sent, num_sampled)
    assert count == 1
    assert sent[0, 2].item() == 0
    assert event[1].item() == -5


@torch.inference_mode()
def test_invalid_slots_untouched() -> None:
    # Slots beyond num_sampled hold uninitialized fill (e.g. INT32_MAX)
    # and must neither be clamped nor counted.
    sent = torch.zeros(2, 8, device="cuda", dtype=torch.int32)
    sent[0, 4] = 2**31 - 1  # INT32_MAX fill, slot 4 not sampled
    sent[1, 5] = -1  # negative fill, slot 5 not sampled
    num_sampled = torch.tensor([4, 5], device="cuda", dtype=torch.int32)
    count, _ = _run(sent, num_sampled)
    assert count == 0
    assert sent[0, 4].item() == 2**31 - 1
    assert sent[1, 5].item() == -1


@torch.inference_mode()
def test_multiple_oob_across_requests() -> None:
    sent = torch.zeros(4, 8, device="cuda", dtype=torch.int32)
    sent[1, 0] = VOCAB_SIZE
    sent[2, 5] = VOCAB_SIZE + 12345
    sent[3, 7] = -1
    num_sampled = torch.tensor([8, 1, 6, 8], device="cuda", dtype=torch.int32)
    count, event = _run(sent, num_sampled)
    assert count == 3
    assert sent[1, 0].item() == VOCAB_SIZE - 1
    assert sent[2, 5].item() == VOCAB_SIZE - 1
    assert sent[3, 7].item() == 0
    # Records in (value, req, slot) form, in scan order.
    assert event[1].item() == VOCAB_SIZE
    assert event[2].item() == 1
    assert event[3].item() == 0
    assert event[4].item() == VOCAB_SIZE + 12345
    assert event[5].item() == 2
    assert event[6].item() == 5
    assert event[7].item() == -1
    assert event[8].item() == 3
    assert event[9].item() == 7


@torch.inference_mode()
def test_more_than_eight_oob_records_capped() -> None:
    sent = torch.full((2, 8), VOCAB_SIZE, device="cuda", dtype=torch.int32)
    num_sampled = torch.tensor([8, 8], device="cuda", dtype=torch.int32)
    count, event = _run(sent, num_sampled)
    assert count == 16  # all counted
    assert event[0].item() == 16
    assert sent.min().item() == VOCAB_SIZE - 1
    # Only the first 8 recorded: req 0, slots 0..7.
    assert event[1 + 3 * 7].item() == VOCAB_SIZE  # 8th record value
    assert event[2 + 3 * 7].item() == 0  # 8th record req
    assert event[3 + 3 * 7].item() == 7  # 8th record slot


@torch.inference_mode()
def test_boundary_values_valid() -> None:
    sent = torch.zeros(1, 8, device="cuda", dtype=torch.int32)
    sent[0, 0] = 0
    sent[0, 1] = VOCAB_SIZE - 1
    expected = sent.clone()
    num_sampled = torch.tensor([2], device="cuda", dtype=torch.int32)
    count, _ = _run(sent, num_sampled)
    assert count == 0
    assert torch.equal(sent, expected)


@torch.inference_mode()
def test_empty_batch() -> None:
    sent = torch.zeros(0, 8, device="cuda", dtype=torch.int32)
    num_sampled = torch.zeros(0, device="cuda", dtype=torch.int32)
    event = torch.zeros(EVENT_SIZE, dtype=torch.int32, device="cuda")
    count = sampled_tokens_oob_sentinel(sent, num_sampled, VOCAB_SIZE, event)
    assert count == 0


@torch.inference_mode()
def test_int64_input() -> None:
    sent = torch.zeros(2, 8, device="cuda", dtype=torch.int64)
    sent[0, 3] = VOCAB_SIZE
    num_sampled = torch.tensor([4, 1], device="cuda", dtype=torch.int32)
    count, event = _run(sent, num_sampled)
    assert count == 1
    assert sent[0, 3].item() == VOCAB_SIZE - 1
    assert event[1].item() == VOCAB_SIZE
