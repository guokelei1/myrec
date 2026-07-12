from __future__ import annotations

import hashlib

import torch

from model.halfspace_value import (
    EVENTWISE_HALFSPACE,
    MODES,
    RAY_ONLY,
    HalfspaceCertifiedValueTransformer,
    project_to_score_halfspace,
)


def _model(mode: str, seed: int = 17) -> HalfspaceCertifiedValueTransformer:
    return HalfspaceCertifiedValueTransformer(
        dim=16,
        inner_dim=8,
        heads=2,
        ffn_dim=12,
        temperature=0.7,
        global_scale=1.0,
        candidate_scale=1.0,
        seed=seed,
        mode=mode,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(123)
    return (
        torch.randn(16, generator=generator),
        torch.randn(7, 16, generator=generator),
        torch.randn(6, 16, generator=generator),
    )


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_halfspace_projection_matches_hand_example() -> None:
    values = torch.tensor([[-2.0, 3.0], [1.0, 4.0]])
    normals = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    actual = project_to_score_halfspace(values, normals)
    expected = torch.tensor([[0.0, 3.0], [1.0, 4.0]])
    assert torch.allclose(actual, expected, atol=1e-7, rtol=0.0)
    assert torch.all((actual * normals).sum(-1) >= -1e-7)


def test_projection_has_same_aggregate_witness() -> None:
    a = torch.tensor([2.0, 1.0])
    first = torch.stack((a, -a))
    second = torch.zeros_like(first)
    normal = torch.tensor([1.0, 0.0]).expand_as(first)
    assert torch.equal(first.mean(0), second.mean(0))
    first_projected = project_to_score_halfspace(first, normal).mean(0)
    second_projected = project_to_score_halfspace(second, normal).mean(0)
    assert not torch.equal(first_projected, second_projected)
    assert torch.allclose(first_projected, torch.tensor([1.0, 0.0]))


def test_modes_are_capacity_and_initialization_matched() -> None:
    models = [_model(mode) for mode in MODES]
    assert len({model.trainable_parameter_count() for model in models}) == 1
    assert len({_state_hash(model) for model in models}) == 1


def test_exact_fallbacks_and_candidate_permutation() -> None:
    model = _model(EVENTWISE_HALFSPACE)
    query, history, candidates = _inputs()
    assert torch.equal(model(query, history[:0], candidates), torch.zeros(len(candidates)))
    assert torch.equal(
        model(query, history, candidates, query_present=False),
        torch.zeros(len(candidates)),
    )
    assert torch.equal(
        model(query, history, candidates, repeat_present=True),
        torch.zeros(len(candidates)),
    )
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    reference = model(query, history, candidates)[permutation]
    actual = model(query, history, candidates[permutation])
    assert torch.allclose(reference, actual, atol=1e-7, rtol=0.0)


def test_certificate_zero_edges_and_ray_equivalence() -> None:
    primary = _model(EVENTWISE_HALFSPACE)
    ray = _model(RAY_ONLY)
    query, history, candidates = _inputs()
    primary_state = primary.components(query, history, candidates)
    ray_state = ray.components(query, history, candidates)

    assert float(primary_state["projected_readout"].min()) >= -1e-7
    unsupported = primary_state["support"] == 0
    assert bool(unsupported.any())
    assert torch.equal(
        primary_state["edge_value"][unsupported],
        torch.zeros_like(primary_state["edge_value"][unsupported]),
    )
    expected = torch.relu(primary_state["raw_readout"])
    assert torch.allclose(primary_state["projected_readout"], expected, atol=1e-7)
    assert torch.allclose(ray_state["projected_readout"], expected, atol=1e-7)
    assert not torch.equal(primary_state["pair_value"], ray_state["pair_value"])


def test_modes_are_operator_distinct_with_shared_active_ffn() -> None:
    query, history, candidates = _inputs()
    models = [_model(mode) for mode in MODES]
    generator = torch.Generator().manual_seed(999)
    shared_down = torch.randn(models[0].ffn_down.weight.shape, generator=generator) * 0.02
    for model in models:
        with torch.no_grad():
            model.ffn_down.weight.copy_(shared_down)
    outputs = [model(query, history, candidates) for model in models]
    for output in outputs:
        assert bool(torch.isfinite(output).all())
    for left in range(len(outputs)):
        for right in range(left + 1, len(outputs)):
            assert not torch.equal(outputs[left], outputs[right])


def test_finite_gradients_reach_every_projection_and_ffn() -> None:
    model = _model(EVENTWISE_HALFSPACE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    query, history, candidates = _inputs()
    target = torch.linspace(-0.5, 0.5, len(candidates))
    active: set[str] = set()
    for _ in range(3):
        output = model(query, history, candidates)
        loss = (output - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert bool(torch.isfinite(parameter.grad).all())
                if bool(parameter.grad.ne(0).any()):
                    active.add(name)
        optimizer.step()
    required = {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "out_proj.weight",
        "ffn_norm.weight",
        "ffn_norm.bias",
        "ffn_up.weight",
        "ffn_down.weight",
    }
    assert required <= active
