# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 DFlash2 selector defense-in-depth regression test.

The conc>=2 crash on SM70 (V100) is an out-of-range gather index in the
selector's codebook gathers: an OOB anchor/candidate id makes
``predecessor_table[ids]`` / ``successor_table[ids]`` fault with a
device-side assert and kill the engine. The defense-in-depth fix clamps
ids to ``[-vocab, vocab-1]`` inside ``CandidateSelector.forward``; this
test pins the semantics:

* legal ids (0..vocab-1 and the -1 invalid-draft sentinel) are untouched,
  so scores are bit-identical to the unguarded code path;
* garbage ids (>= vocab or < -vocab) no longer raise and land on valid
  codebook rows (the -1 wrap maps -vocab -> row 0, -1 -> row vocab-1).
"""

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.qwen3_dflash2 import (
    CandidateSelector,
    _score_edges,
)

VOCAB = 97
RANK = 16
TOP_K = 4
HIDDEN = 32
B, L = 2, 3  # batch, steps (anchor row + L-1 candidate rows)


def _make_selector() -> CandidateSelector:
    """Bypass __init__ (ReplicatedLinear needs a vllm context)."""
    sel = CandidateSelector.__new__(CandidateSelector)
    nn.Module.__init__(sel)
    # Bypassed __init__: the compile wrapper needs this attr to take the
    # eager path (also avoids torch.compile in a CPU test env).
    sel.do_not_compile = True
    g = torch.Generator().manual_seed(0)
    sel.top_k = TOP_K
    sel.vocab_size = VOCAB
    sel.predecessor_codebook = nn.Parameter(
        torch.randn(VOCAB, RANK, generator=g), requires_grad=False
    )
    sel.successor_codebook = nn.Parameter(
        torch.randn(VOCAB, RANK, generator=g), requires_grad=False
    )
    lin = nn.Linear(HIDDEN, RANK, bias=False)
    with torch.no_grad():
        for p in lin.parameters():
            p.normal_(0, 0.1, generator=g)
    sel.hidden_projection = lin
    return sel


def _inputs(cand: torch.Tensor, anchor: torch.Tensor) -> tuple:
    unary = torch.randn(B, L, TOP_K, generator=torch.Generator().manual_seed(1))
    hidden = torch.randn(B, L, HIDDEN, generator=torch.Generator().manual_seed(2))
    return cand, unary, hidden, anchor


def test_legal_ids_unchanged():
    sel = _make_selector()
    cand = torch.arange(B * L * TOP_K, dtype=torch.int64).reshape(B, L, TOP_K) % VOCAB
    anchor = torch.tensor([5, VOCAB - 1], dtype=torch.int32)
    cand, unary, hidden, anchor = _inputs(cand, anchor)
    out = sel(cand, unary, hidden, anchor)

    ref = _score_edges(
        sel.predecessor_codebook,
        sel.successor_codebook,
        cand,
        unary,
        sel.hidden_projection(hidden),
        anchor,
        TOP_K,
    )
    assert torch.equal(out, ref)


def test_minus_one_sentinel_maps_to_vocab_minus_one():
    sel = _make_selector()
    cand = torch.full((B, L, TOP_K), -1, dtype=torch.int64)
    anchor = torch.full((B,), -1, dtype=torch.int32)
    cand, unary, hidden, anchor = _inputs(cand, anchor)
    out = sel(cand, unary, hidden, anchor)

    # -1 must keep wrapping to row vocab-1 exactly as before the guard.
    ref = _score_edges(
        sel.predecessor_codebook,
        sel.successor_codebook,
        cand,
        unary,
        sel.hidden_projection(hidden),
        anchor,
        TOP_K,
    )
    assert torch.equal(out, ref)


@pytest.mark.parametrize(
    "bad_cand, bad_anchor",
    [
        (VOCAB, 0),            # exactly vocab (first OOB positive)
        (VOCAB + 1, 0),        # 08-22 crash class: >= vocab garbage
        (1_000_000, 0),        # huge positive garbage
        (-VOCAB - 1, 0),       # below -vocab (wraps negative to < -vocab)
        (-1_000_000, 0),       # huge negative garbage
        (0, VOCAB),            # anchor OOB
        (0, -1_000_000),       # anchor huge negative
    ],
)
def test_oob_ids_do_not_raise_and_match_clamped(bad_cand, bad_anchor):
    sel = _make_selector()
    cand = torch.full((B, L, TOP_K), 7, dtype=torch.int64)
    cand[0, 0, 0] = bad_cand
    anchor = torch.full((B,), 11, dtype=torch.int32)
    anchor[1] = bad_anchor
    cand, unary, hidden, anchor = _inputs(cand, anchor)

    out = sel(cand, unary, hidden, anchor)  # must not raise (eager OOB = IndexError)
    assert torch.isfinite(out).all()

    # Must equal the unguarded math applied to the clamped ids.
    v = VOCAB
    ref = _score_edges(
        sel.predecessor_codebook,
        sel.successor_codebook,
        cand.clamp(min=-v, max=v - 1),
        unary,
        sel.hidden_projection(hidden),
        anchor.clamp(min=-v, max=v - 1),
        TOP_K,
    )
    assert torch.equal(out, ref)


def test_oob_gather_would_fail_without_guard():
    """Sanity: the exact OOB value DOES fault a raw gather (test premise)."""
    codebook = torch.randn(VOCAB, RANK)
    ids = torch.tensor([VOCAB + 1], dtype=torch.int64)
    with pytest.raises(IndexError):
        codebook[ids]
