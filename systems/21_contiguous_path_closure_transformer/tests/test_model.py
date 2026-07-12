from __future__ import annotations

from pathlib import Path
import sys

import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.path_closure import MODES, PathClosureProbe


def make_model(mode: str = "contiguous_path", *, input_dim: int = 4, projection_dim: int = 4) -> PathClosureProbe:
    return PathClosureProbe(
        input_dim=input_dim,
        projection_dim=projection_dim,
        max_history=4,
        max_horizon=3,
        evidence_temperature=0.1,
        score_delta_max=1.0,
        mode=mode,
    )


def inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    return {
        "query": torch.randn(2, 4),
        "candidates": torch.randn(2, 5, 4),
        "history": torch.randn(2, 4, 4),
        "candidate_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool),
        "history_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool),
        "base_scores": torch.randn(2, 5),
    }


def test_directed_path_separates_closing_candidate_but_unordered_does_not() -> None:
    query = torch.zeros(1, 2)
    candidates = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    history = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
    kwargs = {
        "query": query,
        "candidates": candidates,
        "history": history,
        "candidate_mask": torch.ones(1, 2, dtype=torch.bool),
        "history_mask": torch.ones(1, 3, dtype=torch.bool),
        "base_scores": torch.zeros(1, 2),
    }
    outputs = {}
    for mode in ("contiguous_path", "unordered_pair"):
        model = make_model(mode, input_dim=2, projection_dim=2)
        with torch.no_grad():
            model.state_projection.weight.copy_(torch.eye(2))
            model.relation_projection.weight.copy_(torch.eye(2))
            model.anchor_projection.weight.copy_(torch.eye(2))
            model.anchor_log_gain.fill_(-100.0)
            model.residual_logit.fill_(4.0)
        outputs[mode] = model(**kwargs)
    assert outputs["contiguous_path"].scores[0, 0] > outputs["contiguous_path"].scores[0, 1]
    torch.testing.assert_close(
        outputs["unordered_pair"].scores[0, 0],
        outputs["unordered_pair"].scores[0, 1],
        atol=1e-6,
        rtol=0.0,
    )


def test_multistep_segment_changes_primary_relative_to_one_step() -> None:
    kwargs = {
        "query": torch.zeros(1, 2),
        "candidates": torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]]),
        "history": torch.tensor([[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]),
        "candidate_mask": torch.ones(1, 2, dtype=torch.bool),
        "history_mask": torch.ones(1, 3, dtype=torch.bool),
        "base_scores": torch.zeros(1, 2),
    }
    models = {mode: make_model(mode, input_dim=2, projection_dim=2) for mode in ("contiguous_path", "one_step")}
    initial = models["contiguous_path"].state_dict()
    models["one_step"].load_state_dict(initial)
    with torch.no_grad():
        for model in models.values():
            model.state_projection.weight.copy_(torch.eye(2))
            model.relation_projection.weight.copy_(torch.eye(2))
            model.anchor_projection.weight.copy_(torch.eye(2))
            model.anchor_log_gain.fill_(-100.0)
    primary = models["contiguous_path"](**kwargs).evidence
    one_step = models["one_step"](**kwargs).evidence
    assert not torch.allclose(primary, one_step)


def test_pooled_history_is_permutation_invariant_and_path_is_sensitive() -> None:
    batch = inputs()
    permutation = torch.tensor([2, 0, 3, 1])
    shuffled = dict(batch)
    shuffled["history"] = batch["history"][:, permutation]
    shuffled["history_mask"] = batch["history_mask"][:, permutation]
    pooled = make_model("pooled_history")
    path = make_model("contiguous_path")
    path.load_state_dict(pooled.state_dict())
    torch.testing.assert_close(pooled(**batch).scores, pooled(**shuffled).scores, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(path(**batch).evidence, path(**shuffled).evidence)


def test_candidate_permutation_equivariance_and_centering() -> None:
    model = make_model()
    batch = inputs()
    output = model(**batch)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    changed = dict(batch)
    changed["candidates"] = batch["candidates"][:, permutation]
    changed["candidate_mask"] = batch["candidate_mask"][:, permutation]
    changed["base_scores"] = batch["base_scores"][:, permutation]
    permuted = model(**changed)
    torch.testing.assert_close(permuted.scores, output.scores[:, permutation], atol=1e-6, rtol=1e-6)
    sums = (output.deltas * batch["candidate_mask"]).sum(dim=-1)
    torch.testing.assert_close(sums, torch.zeros_like(sums), atol=1e-7, rtol=0.0)


def test_nohistory_and_query_absent_are_bitwise_base() -> None:
    for mode in MODES:
        model = make_model(mode)
        batch = inputs()
        batch["history_mask"] = torch.zeros_like(batch["history_mask"])
        nohistory = model(**batch)
        assert torch.equal(nohistory.scores, batch["base_scores"])
        batch = inputs()
        absent = model(**batch, query_present=torch.zeros(2, dtype=torch.bool))
        assert torch.equal(absent.scores, batch["base_scores"])


def test_modes_have_identical_parameters_and_initial_state() -> None:
    torch.manual_seed(91)
    template = make_model()
    state = template.state_dict()
    signatures = []
    for mode in MODES:
        model = make_model(mode)
        model.load_state_dict(state)
        signatures.append([(name, tuple(value.shape)) for name, value in model.named_parameters()])
        for name, value in model.state_dict().items():
            assert torch.equal(value, state[name])
    assert all(signature == signatures[0] for signature in signatures)


def test_primary_gradients_reach_load_bearing_paths() -> None:
    model = make_model()
    output = model(**inputs())
    loss = output.scores[0, 0] - output.scores[0, 1]
    loss.backward()
    gradients = {name: value.grad for name, value in model.named_parameters()}
    for name in (
        "state_projection.weight",
        "relation_projection.weight",
        "anchor_projection.weight",
        "direction_log_gain",
        "anchor_log_gain",
        "evidence_log_gain",
        "residual_logit",
    ):
        assert gradients[name] is not None
        assert torch.isfinite(gradients[name]).all()
        assert gradients[name].ne(0).any()
