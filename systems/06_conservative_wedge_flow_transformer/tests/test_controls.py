from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.controls import CenteredCrossAttentionProbeRanker  # noqa: E402
from model.complexity import dominant_probe_flops  # noqa: E402
from model.wedge_flow import ConservativeWedgeFlowProbeRanker  # noqa: E402
from train.losses import masked_listwise_loss  # noqa: E402


def _inputs(history_count: int = 3) -> dict[str, torch.Tensor]:
    torch.manual_seed(307)
    return {
        "query": torch.randn(2, 8),
        "candidates": torch.randn(2, 4, 8),
        "history": torch.randn(2, history_count, 8),
        "candidate_mask": torch.tensor(
            [[True, True, True, True], [True, True, True, False]]
        ),
        "history_mask": torch.tensor(
            [[True] * history_count, [True] + [False] * (history_count - 1)]
        ),
        "history_prior": torch.ones(2, history_count),
        "base_scores": torch.randn(2, 4, dtype=torch.float32),
    }


def _flow(mode: str) -> ConservativeWedgeFlowProbeRanker:
    torch.manual_seed(311)
    return ConservativeWedgeFlowProbeRanker(
        8, 4, score_delta_max=0.7, trust_mode=mode
    )


def _centered() -> CenteredCrossAttentionProbeRanker:
    torch.manual_seed(313)
    return CenteredCrossAttentionProbeRanker(8, 4, score_delta_max=0.7)


def _open(model: torch.nn.Module, fraction: float = 0.8) -> None:
    with torch.no_grad():
        model.raw_residual_scale.fill_(math.atanh(fraction))


def test_minimal_controls_are_capacity_matched_and_start_at_exact_base() -> None:
    inputs = _inputs()
    primary = _flow("local_hodge")
    models = [
        primary,
        _flow("untrusted"),
        _flow("global_hodge"),
        _centered(),
    ]
    primary_count = primary.parameter_count()
    for model in models:
        difference = abs(model.parameter_count() - primary_count) / primary_count
        assert difference <= 0.02
        output = model.eval()(**inputs)
        assert torch.equal(output.scores, inputs["base_scores"])
    direct = _flow("direct_learned")
    assert torch.equal(direct.eval()(**inputs).scores, inputs["base_scores"])

    registered_primary = ConservativeWedgeFlowProbeRanker(512, 32)
    registered_direct = ConservativeWedgeFlowProbeRanker(
        512, 32, trust_mode="direct_learned"
    )
    registered_centered = CenteredCrossAttentionProbeRanker(512, 32)
    for control in (registered_direct, registered_centered):
        difference = abs(
            control.parameter_count() - registered_primary.parameter_count()
        ) / registered_primary.parameter_count()
        assert difference <= 0.02


def test_centered_attention_has_matched_frozen_dominant_flops() -> None:
    for candidates in (5, 48, 256):
        for history in (0, 1, 6, 20):
            primary = dominant_probe_flops(
                variant="local_hodge",
                input_dim=512,
                evidence_dim=32,
                candidates=candidates,
                history=history,
            )
            centered = dominant_probe_flops(
                variant="centered_cross_attention",
                input_dim=512,
                evidence_dim=32,
                candidates=candidates,
                history=history,
                centered_compute_rounds=4,
            )
            assert centered == primary


def test_flow_controls_have_registered_trust_reductions() -> None:
    inputs = _inputs()
    outputs = {}
    for mode in ("local_hodge", "untrusted", "global_hodge", "direct_learned"):
        model = _flow(mode).eval()
        _open(model)
        outputs[mode] = model(**inputs)
    active = inputs["candidate_mask"][:, :, None] & inputs["history_mask"][:, None, :]
    assert torch.equal(
        outputs["untrusted"].candidate_event_trust[active],
        torch.ones_like(outputs["untrusted"].candidate_event_trust[active]),
    )
    global_trust = outputs["global_hodge"].candidate_event_trust
    for batch in range(2):
        for event in range(inputs["history_mask"].shape[1]):
            if inputs["history_mask"][batch, event]:
                values = global_trust[batch, inputs["candidate_mask"][batch], event]
                assert torch.allclose(values, values[:1].expand_as(values))
    direct = outputs["direct_learned"].candidate_event_trust[active]
    assert bool(((direct > 0.0) & (direct < 1.0)).all().item())
    assert torch.allclose(
        outputs["local_hodge"].event_potential,
        outputs["untrusted"].event_potential,
        atol=0.0,
        rtol=0.0,
    )
    assert torch.allclose(
        outputs["local_hodge"].event_potential,
        outputs["global_hodge"].event_potential,
        atol=0.0,
        rtol=0.0,
    )
    expected_untrusted = torch.where(
        active,
        0.5 * outputs["untrusted"].event_potential,
        torch.zeros_like(outputs["untrusted"].event_potential),
    )
    assert torch.allclose(
        outputs["untrusted"].trusted_event_divergence,
        expected_untrusted,
        atol=1e-7,
        rtol=1e-6,
    )
    global_fraction = outputs["global_hodge"].gradient_energy / (
        outputs["global_hodge"].gradient_energy
        + outputs["global_hodge"].cycle_energy
        + 1e-12
    )
    expected_global = torch.where(
        active,
        0.5
        * global_fraction[:, None, :]
        * outputs["global_hodge"].event_potential,
        torch.zeros_like(outputs["global_hodge"].event_potential),
    )
    assert torch.allclose(
        outputs["global_hodge"].trusted_event_divergence,
        expected_global,
        atol=1e-7,
        rtol=1e-6,
    )
    for output in outputs.values():
        for row in range(2):
            mask = inputs["candidate_mask"][row]
            assert abs(float(output.divergence[row, mask].sum().detach())) < 1e-6
        assert (
            float(output.conservative_score_delta.abs().max().detach())
            <= 0.7 + 1e-7
        )


def test_direct_gate_receives_ranking_gradient() -> None:
    model = _flow("direct_learned").train()
    _open(model, 0.4)
    output = model(**_inputs())
    labels = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    loss = masked_listwise_loss(output.scores, labels, _inputs()["candidate_mask"])
    loss.backward()
    assert model.direct_gate_projection is not None
    gradient = model.direct_gate_projection.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_centered_attention_is_zero_sum_bounded_and_permutation_equivariant() -> None:
    model = _centered().eval()
    _open(model)
    inputs = _inputs()
    output = model(**inputs)
    for row in range(2):
        mask = inputs["candidate_mask"][row]
        assert (
            abs(float(output.conservative_score_delta[row, mask].sum().detach()))
            < 1e-6
        )
    assert (
        float(output.conservative_score_delta.abs().max().detach())
        <= 0.7 + 1e-7
    )

    permutation = torch.tensor([2, 0, 3, 1])
    permuted = dict(inputs)
    for key in ("candidates", "candidate_mask", "base_scores"):
        permuted[key] = inputs[key][:, permutation]
    changed = model(**permuted)
    assert torch.allclose(changed.scores, output.scores[:, permutation], atol=1e-7)


def test_centered_attention_common_candidates_and_no_history_are_exact_base() -> None:
    model = _centered().eval()
    _open(model)
    inputs = _inputs()
    shared = torch.randn(2, 1, 8)
    inputs["candidates"] = shared.expand(-1, 4, -1).clone()
    common = model(**inputs)
    assert torch.equal(common.scores, inputs["base_scores"])

    empty = _inputs(0)
    no_history = model(**empty)
    assert torch.equal(no_history.scores, empty["base_scores"])


def test_centered_attention_two_steps_reach_attention_path() -> None:
    model = _centered().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    inputs = _inputs()
    labels = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    reached = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(**inputs)
        loss = masked_listwise_loss(output.scores, labels, inputs["candidate_mask"])
        loss.backward()
        reached.append(
            {
                name.split(".")[0]
                for name, parameter in model.named_parameters()
                if parameter.grad is not None and bool((parameter.grad != 0).any())
            }
        )
        optimizer.step()
    assert reached[0] == {"raw_residual_scale"}
    assert {
        "query_projection",
        "candidate_projection",
        "history_projection",
        "attention_projection",
        "value_projection",
        "raw_residual_scale",
    }.issubset(reached[1])
