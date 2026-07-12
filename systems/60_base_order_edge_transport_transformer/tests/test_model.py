from __future__ import annotations

from pathlib import Path
import sys

import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.edge_transport import BaseOrderEdgeTransportTransformer  # noqa: E402


def values() -> dict[str, torch.Tensor]:
    return {
        "base_scores": torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64),
        "evidence": torch.tensor([[0.0, 3.0, 0.0]], dtype=torch.float64),
        "candidate_mask": torch.ones((1, 3), dtype=torch.bool),
        "canonical_order": torch.tensor([[0, 1, 2]]),
    }


def test_zero_evidence_is_exact_base() -> None:
    model = BaseOrderEdgeTransportTransformer()
    batch = values()
    batch["evidence"].zero_()
    output = model(**batch)
    assert torch.equal(output.scores, batch["base_scores"])
    assert torch.equal(output.correction, torch.zeros_like(output.correction))


def test_one_sided_transport_is_conservative_and_capacity_bounded() -> None:
    model = BaseOrderEdgeTransportTransformer()
    output = model(**values())
    assert output.transport[0, 0] > 0
    assert output.transport[0, 1] == 0
    assert torch.all(output.transport.abs() <= output.base_gap + 1e-12)
    assert torch.allclose(output.correction.sum(dim=-1), torch.zeros(1, dtype=torch.float64))


def test_one_sided_does_not_strengthen_supported_base_edge() -> None:
    model = BaseOrderEdgeTransportTransformer()
    batch = values()
    batch["evidence"] = torch.tensor([[3.0, 0.0, -1.0]], dtype=torch.float64)
    output = model(**batch)
    assert torch.equal(output.transport, torch.zeros_like(output.transport))


def test_signed_and_hard_are_distinct_controls() -> None:
    model = BaseOrderEdgeTransportTransformer()
    batch = values()
    signed = model(**batch, mode="signed")
    hard = model(**batch, mode="hard")
    assert not torch.equal(signed.transport, hard.transport)
    assert hard.transport[0, 0] == hard.base_gap[0, 0]


def test_candidate_permutation_is_equivariant_with_recomputed_order() -> None:
    model = BaseOrderEdgeTransportTransformer()
    batch = values()
    order = torch.tensor([2, 0, 1])
    inverse = torch.argsort(order)
    moved = {
        "base_scores": batch["base_scores"][:, order],
        "evidence": batch["evidence"][:, order],
        "candidate_mask": batch["candidate_mask"][:, order],
        "canonical_order": inverse[batch["canonical_order"]],
    }
    first = model(**batch)
    second = model(**moved)
    assert torch.equal(first.scores, second.scores[:, inverse])


def test_operator_has_no_parameters() -> None:
    model = BaseOrderEdgeTransportTransformer()
    assert model.parameter_count() == 0
