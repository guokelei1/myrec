from __future__ import annotations

import sys
from pathlib import Path

import torch

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.cceb import CCEBProbeRanker, CandidateContrastiveEvidenceBlock
from train.losses import masked_listwise_loss


def _inputs(*, history_present: bool = True) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    batch, candidates, events, dimension = 2, 3, 2, 4
    history_mask = torch.tensor(
        [[history_present, history_present], [history_present, False]],
        dtype=torch.bool,
    )
    return {
        "query": torch.randn(batch, dimension),
        "candidates": torch.randn(batch, candidates, dimension),
        "history": torch.randn(batch, events, dimension),
        "candidate_mask": torch.tensor(
            [[True, True, True], [True, True, False]], dtype=torch.bool
        ),
        "history_mask": history_mask,
        "exact_match": torch.zeros(
            batch, candidates, events, dtype=torch.bool
        ),
    }


def _block(*, dead_zone: float = 0.0) -> CandidateContrastiveEvidenceBlock:
    torch.manual_seed(11)
    return CandidateContrastiveEvidenceBlock(
        input_dim=4,
        evidence_dim=4,
        dead_zone=dead_zone,
        exact_bias_init=1.0,
        residual_scale_max=0.5,
        dropout=0.0,
    ).eval()


def test_no_history_is_exact_noop() -> None:
    model = _block()
    values = _inputs(history_present=False)
    output = model(**values)
    assert torch.equal(output.updated_candidates, values["candidates"])
    assert torch.count_nonzero(output.update) == 0
    assert torch.count_nonzero(output.evidence_weights) == 0
    assert torch.count_nonzero(output.evidence_l1_mass) == 0


def test_candidate_common_evidence_is_rejected() -> None:
    model = _block()
    values = _inputs()
    shared = torch.randn(2, 1, 4)
    values["candidates"] = shared.expand(-1, 3, -1).clone()
    values["candidate_mask"] = torch.ones(2, 3, dtype=torch.bool)
    output = model(**values)
    assert torch.count_nonzero(output.contrast) == 0
    assert torch.count_nonzero(output.update) == 0
    assert torch.equal(output.updated_candidates, values["candidates"])


def test_candidate_permutation_is_equivariant() -> None:
    model = _block()
    with torch.no_grad():
        model.raw_residual_scale.fill_(0.4)
    values = _inputs()
    values["exact_match"][0, 0, 0] = True
    original = model(**values)
    permutation = torch.tensor([2, 0, 1])
    permuted_values = dict(values)
    permuted_values["candidates"] = values["candidates"][:, permutation]
    permuted_values["candidate_mask"] = values["candidate_mask"][:, permutation]
    permuted_values["exact_match"] = values["exact_match"][:, permutation]
    permuted = model(**permuted_values)
    assert torch.allclose(
        permuted.updated_candidates,
        original.updated_candidates[:, permutation],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        permuted.evidence_weights,
        original.evidence_weights[:, permutation],
        atol=1e-6,
        rtol=1e-6,
    )


def test_evidence_mass_is_strictly_below_one_and_padding_is_zero() -> None:
    model = _block()
    values = _inputs()
    output = model(**values)
    valid = values["candidate_mask"]
    assert bool((output.evidence_l1_mass[valid] < 1.0).all())
    assert torch.count_nonzero(output.evidence_l1_mass[~valid]) == 0
    assert torch.count_nonzero(output.update[~valid]) == 0
    assert torch.count_nonzero(
        output.evidence_weights[:, :, 1][~values["history_mask"][:, None, :].expand(-1, 3, -1)[:, :, 1]]
    ) == 0


def test_exact_identity_is_a_positive_atom_in_same_alignment() -> None:
    model = _block()
    values = _inputs()
    values["candidate_mask"] = torch.ones(2, 3, dtype=torch.bool)
    with torch.no_grad():
        model.query_projection.weight.zero_()
        model.candidate_projection.weight.zero_()
        model.history_key_projection.weight.zero_()
    without_exact = model(**values)
    values["exact_match"][0, 1, 0] = True
    with_exact = model(**values)
    assert float(without_exact.raw_alignment.detach().abs().max()) == 0.0
    assert with_exact.raw_alignment[0, 1, 0] > 0
    assert with_exact.contrast[0, 1, 0] > 0
    assert with_exact.contrast[0, 1, 0] > with_exact.contrast[0, 0, 0]


def test_single_candidate_safely_has_no_cross_item_update() -> None:
    model = _block()
    values = _inputs()
    values["candidate_mask"] = torch.tensor(
        [[True, False, False], [True, False, False]], dtype=torch.bool
    )
    values["exact_match"][0, 0, 0] = True
    output = model(**values)
    assert torch.count_nonzero(output.contrast) == 0
    assert torch.count_nonzero(output.update) == 0


def test_present_evidence_has_finite_nonzero_gradients() -> None:
    model = _block().train()
    with torch.no_grad():
        model.raw_residual_scale.fill_(0.4)
    values = _inputs()
    values["exact_match"][0, 0, 0] = True
    output = model(**values)
    loss = output.updated_candidates.square().mean() + output.evidence_l1_mass.mean()
    loss.backward()
    required = [
        model.query_projection.weight,
        model.candidate_projection.weight,
        model.history_key_projection.weight,
        model.history_value_projection.weight,
        model.output_projection.weight,
        model.raw_residual_scale,
    ]
    for parameter in required:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0.0


def test_probe_no_history_full_loss_is_finite_and_exact_base() -> None:
    torch.manual_seed(13)
    model = CCEBProbeRanker(
        input_dim=4,
        evidence_dim=4,
        dead_zone=0.0,
        dropout=0.0,
    )
    values = _inputs(history_present=False)
    base_scores = torch.randn(2, 3, dtype=torch.float64)
    output = model(**values, base_scores=base_scores)
    assert torch.equal(output.scores, base_scores)
    relevance = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    loss = masked_listwise_loss(
        output.scores, relevance, values["candidate_mask"]
    )
    assert torch.isfinite(loss)
    loss.backward()


def test_invalid_all_masked_candidate_request_fails_closed() -> None:
    model = _block()
    values = _inputs()
    values["candidate_mask"][0] = False
    try:
        model(**values)
    except ValueError as error:
        assert "at least one valid candidate" in str(error)
    else:
        raise AssertionError("all-masked candidate request must fail closed")
