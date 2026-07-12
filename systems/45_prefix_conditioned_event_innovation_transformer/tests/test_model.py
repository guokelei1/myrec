from pathlib import Path
import sys

import torch

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model import MODES, PrefixConditionedEventInnovationTransformer  # noqa: E402


def make(mode: str = "innovation") -> PrefixConditionedEventInnovationTransformer:
    torch.manual_seed(7)
    return PrefixConditionedEventInnovationTransformer(
        input_dim=8,
        width=16,
        heads=4,
        ff_multiplier=2,
        max_history=4,
        mode=mode,
    ).eval()


def inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(11)
    return (
        torch.randn(3, 8),
        torch.randn(3, 5, 8),
        torch.randn(3, 4, 8),
        torch.ones(3, 4, dtype=torch.bool),
    )


def test_modes_have_equal_parameters_and_paired_initialization() -> None:
    states = []
    counts = []
    for mode in MODES:
        model = make(mode)
        states.append({name: value.clone() for name, value in model.state_dict().items()})
        counts.append(sum(value.numel() for value in model.parameters()))
    assert len(set(counts)) == 1
    for name in states[0]:
        assert all(torch.equal(states[0][name], state[name]) for state in states[1:])


def test_exact_fallbacks() -> None:
    model = make()
    query, candidates, history, mask = inputs()
    no_history = model.forward_components(query, candidates, history, torch.zeros_like(mask))
    assert torch.equal(no_history.score, no_history.base)
    absent = model.forward_components(
        torch.zeros_like(query),
        candidates,
        history,
        mask,
        query_present=torch.zeros(len(query), dtype=torch.bool),
    )
    assert torch.equal(absent.correction, torch.zeros_like(absent.correction))
    item_only = torch.randn(3, 5)
    repeated = model.rank(
        query,
        candidates,
        history,
        mask,
        repeat_present=torch.ones(3, dtype=torch.bool),
        item_only_scores=item_only,
    )
    assert torch.equal(repeated, item_only)


def test_null_event_has_exact_zero_innovation() -> None:
    model = make()
    state = model.initial_state[None].expand(2, -1)
    null = model.null_event[None].expand_as(state)
    factual = model._transition_once(state, null, 0)
    counterfactual = model._transition_once(state, null, 0)
    assert torch.equal(factual - counterfactual, torch.zeros_like(factual))


def test_innovation_depends_on_prefix_and_candidates_are_equivariant() -> None:
    model = make()
    query, candidates, history, mask = inputs()
    changed = history.clone()
    changed[:, 0] = changed[:, 0] + 3.0
    first = model.encode_events(history, mask)[0]
    second = model.encode_events(changed, mask)[0]
    assert not torch.allclose(first[:, 1], second[:, 1])
    score = model.rank(query, candidates, history, mask)
    order = torch.tensor([4, 3, 2, 1, 0])
    permuted = model.rank(query, candidates[:, order], history, mask)
    assert torch.allclose(score, permuted[:, order], atol=1e-6, rtol=0.0)


def test_primary_backward_is_finite_and_active() -> None:
    model = make().train()
    query, candidates, history, mask = inputs()
    loss = model.rank(query, candidates, history, mask).square().mean()
    loss.backward()
    groups = {
        name.split(".")[0]
        for name, value in model.named_parameters()
        if value.grad is not None and torch.isfinite(value.grad).all() and value.grad.ne(0).any()
    }
    assert {"transition", "base_transformer", "evidence_transformer", "query_projection", "candidate_projection", "event_projection"} <= groups
