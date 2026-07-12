from pathlib import Path
import sys

import torch

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model import BehavioralSemanticTransformer  # noqa: E402


def make() -> BehavioralSemanticTransformer:
    torch.manual_seed(3)
    return BehavioralSemanticTransformer(
        input_dim=16,
        width=32,
        heads=4,
        layers=2,
        ff_multiplier=2,
        max_history=5,
        temperature=0.1,
    )


def test_nohistory_is_exact_zero_and_candidate_permutation_equivariant() -> None:
    torch.manual_seed(5)
    model = make().eval()
    history = torch.randn(4, 5, 16)
    candidates = torch.randn(4, 7, 16)
    empty = torch.zeros(4, 5, dtype=torch.bool)
    assert torch.equal(model.score(history, empty, candidates), torch.zeros(4, 7))
    mask = torch.ones(4, 5, dtype=torch.bool)
    score = model.score(history, mask, candidates)
    permutation = torch.tensor([6, 5, 4, 3, 2, 1, 0])
    permuted = model.score(history, mask, candidates[:, permutation])
    assert torch.allclose(score, permuted[:, permutation], atol=1e-6, rtol=0.0)


def test_positions_make_order_observable() -> None:
    torch.manual_seed(7)
    model = make().eval()
    history = torch.randn(3, 5, 16)
    mask = torch.ones(3, 5, dtype=torch.bool)
    first = model.encode_history(history, mask)
    second = model.encode_history(history.flip(1), mask)
    assert not torch.allclose(first, second)


def test_backward_activates_all_groups() -> None:
    torch.manual_seed(11)
    model = make().train()
    history = torch.randn(4, 5, 16)
    candidates = torch.randn(4, 8, 16)
    loss = model.score(history, torch.ones(4, 5, dtype=torch.bool), candidates).square().mean()
    loss.backward()
    names = {name for name, value in model.named_parameters() if value.grad is not None and value.grad.ne(0).any()}
    assert any(name.startswith("item_projection.") for name in names)
    assert any(name.startswith("transformer.") for name in names)
    assert "read_token" in names and "position" in names
    assert any(name.startswith("output_norm.") for name in names)
