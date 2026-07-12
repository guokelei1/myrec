from pathlib import Path
import sys

import torch

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.joint_context import StrongAnchorJointContextTransformer  # noqa: E402


def model() -> StrongAnchorJointContextTransformer:
    torch.manual_seed(53)
    return StrongAnchorJointContextTransformer(
        input_dim=8, hidden_dim=16, heads=4, layers=2, ffn_dim=32,
        dropout=0.0, max_history=4,
    ).eval()


def inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(54)
    return {
        "query": torch.randn(2, 8),
        "history": torch.randn(2, 3, 8),
        "history_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "candidates": torch.randn(2, 5, 8),
        "candidate_mask": torch.tensor([[True] * 4 + [False], [True] * 5]),
        "base_scores": torch.randn(2, 5),
    }


def test_nohistory_is_exact_base() -> None:
    values = inputs()
    values["history_mask"] = torch.zeros_like(values["history_mask"])
    out = model()(**values)
    expected = values["base_scores"].masked_fill(~values["candidate_mask"], 0.0)
    assert torch.equal(out.scores, expected)
    assert torch.count_nonzero(out.correction) == 0


def test_candidate_permutation_equivariance() -> None:
    values = inputs()
    ranker = model()
    out = ranker(**values)
    permutation = torch.tensor([4, 2, 0, 3, 1])
    changed = dict(values)
    for name in ("candidates", "candidate_mask", "base_scores"):
        changed[name] = changed[name][:, permutation]
    permuted = ranker(**changed)
    assert torch.allclose(out.scores[:, permutation], permuted.scores, atol=2e-6)


def test_cross_candidate_edge_is_functionally_available() -> None:
    values = inputs()
    ranker = model()
    joint = ranker(**values)
    independent = ranker(**values, independent_candidates=True)
    assert not torch.allclose(joint.scores, independent.scores)


def test_directed_mask_contract() -> None:
    mask = model().attention_mask(
        history_slots=2, candidate_slots=3,
        independent_candidates=False, device=torch.device("cpu"),
    )
    # Query/history cannot read candidates; candidates read all context/list.
    assert bool(mask[:3, 3:].all())
    assert not bool(mask[3:, :].any())
    independent = model().attention_mask(
        history_slots=2, candidate_slots=3,
        independent_candidates=True, device=torch.device("cpu"),
    )
    candidate_block = independent[3:, 3:]
    assert torch.equal(candidate_block, ~torch.eye(3, dtype=torch.bool))


def test_gradients_are_finite() -> None:
    values = inputs()
    ranker = model().train()
    loss = ranker(**values).scores.sum()
    loss.backward()
    gradients = [value.grad for value in ranker.parameters() if value.grad is not None]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
