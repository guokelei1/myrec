from __future__ import annotations

from pathlib import Path
import sys

import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.eft import EFTLayer, EvidenceFiltrationRanker, MODES, PrefixRMSNorm


def make_model(mode: str = "filtration") -> EvidenceFiltrationRanker:
    return EvidenceFiltrationRanker(
        input_dim=8,
        anchor_dim=4,
        recurrence_dim=4,
        transfer_dim=4,
        history_slots=3,
        layers=2,
        heads_per_block=1,
        ffn_multiplier=2,
        dropout=0.0,
        transfer_delta_max=1.0,
        recurrence_scale_min=2.5,
        mode=mode,
    )


def model_inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(31)
    identity = torch.zeros(2, 4, 3, dtype=torch.bool)
    identity[0, 1, 2] = True
    return {
        "query": torch.randn(2, 8),
        "candidates": torch.randn(2, 4, 8),
        "history": torch.randn(2, 3, 8),
        "identity": identity,
        "event_strength": torch.ones(2, 3),
        "history_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        "candidate_mask": torch.ones(2, 4, dtype=torch.bool),
        "base_scores": torch.randn(2, 4),
    }


def jacobian_block_norm(layer: EFTLayer, mode: str, output_block: int, input_block: int) -> float:
    value = torch.randn(1, 3, 6, requires_grad=True)
    output = layer(value, torch.ones(1, 3, dtype=torch.bool), mode)
    slices = (slice(0, 2), slice(2, 4), slice(4, 6))
    gradient = torch.autograd.grad(output[..., slices[output_block]].sum(), value)[0]
    return float(gradient[..., slices[input_block]].norm())


def test_filtration_has_protected_zero_jacobians_and_one_way_read() -> None:
    torch.manual_seed(5)
    layer = EFTLayer((2, 2, 2), heads_per_block=1, ffn_multiplier=2, dropout=0.0)
    assert jacobian_block_norm(layer, "filtration", 0, 1) == 0.0
    assert jacobian_block_norm(layer, "filtration", 0, 2) == 0.0
    assert jacobian_block_norm(layer, "filtration", 1, 2) == 0.0
    assert jacobian_block_norm(layer, "filtration", 2, 1) > 0.0


def test_dense_mixing_and_dense_norm_violate_protected_direction() -> None:
    torch.manual_seed(7)
    layer = EFTLayer((2, 2, 2), heads_per_block=1, ffn_multiplier=2, dropout=0.0)
    assert jacobian_block_norm(layer, "dense", 0, 2) > 0.0
    norm = PrefixRMSNorm((2, 2, 2))
    value = torch.randn(1, 1, 6, requires_grad=True)
    dense = norm(value, "dense")[..., :2].sum()
    gradient = torch.autograd.grad(dense, value)[0]
    assert gradient[..., 4:].norm() > 0.0
    value = value.detach().requires_grad_(True)
    filtered = norm(value, "filtration")[..., :2].sum()
    gradient = torch.autograd.grad(filtered, value)[0]
    assert gradient[..., 2:].eq(0).all()


def test_identity_initializes_only_recurrence_quotient() -> None:
    model = make_model()
    inputs = model_inputs()
    with_identity = model._tokens(
        inputs["query"],
        inputs["candidates"],
        inputs["history"],
        inputs["identity"],
        inputs["event_strength"],
        inputs["history_mask"],
    )[0]
    without_identity = model._tokens(
        inputs["query"],
        inputs["candidates"],
        inputs["history"],
        torch.zeros_like(inputs["identity"]),
        inputs["event_strength"],
        inputs["history_mask"],
    )[0]
    difference = with_identity - without_identity
    anchor, recurrence, transfer = difference.split(model.block_dims, dim=-1)
    assert anchor.eq(0).all()
    assert recurrence.ne(0).any()
    assert transfer.eq(0).all()


def test_nohistory_and_query_absent_return_base_bitwise() -> None:
    for mode in MODES:
        model = make_model(mode)
        inputs = model_inputs()
        inputs["history_mask"] = torch.zeros_like(inputs["history_mask"])
        inputs["identity"] = torch.zeros_like(inputs["identity"])
        output = model(**inputs)
        assert torch.equal(output.scores, inputs["base_scores"])
        inputs = model_inputs()
        absent = model(**inputs, query_present=torch.zeros(2, dtype=torch.bool))
        assert torch.equal(absent.scores, inputs["base_scores"])


def test_candidate_permutation_equivariance_and_transfer_centering() -> None:
    model = make_model()
    inputs = model_inputs()
    output = model(**inputs)
    permutation = torch.tensor([2, 0, 3, 1])
    changed = dict(inputs)
    for name in ("candidates", "identity", "candidate_mask", "base_scores"):
        changed[name] = inputs[name][:, permutation]
    permuted = model(**changed)
    torch.testing.assert_close(permuted.scores, output.scores[:, permutation], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        output.transfer_delta.sum(dim=-1),
        torch.zeros(2),
        atol=1e-6,
        rtol=0.0,
    )


def test_modes_have_identical_parameters_and_initial_state() -> None:
    torch.manual_seed(73)
    template = make_model()
    state = template.state_dict()
    signatures = []
    for mode in MODES:
        model = make_model(mode)
        model.load_state_dict(state)
        signatures.append([(name, tuple(value.shape)) for name, value in model.named_parameters()])
        assert all(torch.equal(value, state[name]) for name, value in model.state_dict().items())
    assert all(signature == signatures[0] for signature in signatures)


def test_gradients_reach_all_load_bearing_components() -> None:
    model = make_model()
    output = model(**model_inputs())
    loss = output.scores[0, 1] - output.scores[0, 0]
    loss.backward()
    required = (
        "anchor_input.weight",
        "transfer_input.weight",
        "recurrence_atom",
        "layers.0.attention.qkv_1.weight",
        "layers.0.attention.qkv_2.weight",
        "layers.0.ffn.in_1.weight",
        "layers.0.ffn.in_2.weight",
        "recurrence_readout.weight",
        "transfer_readout.weight",
        "recurrence_log_scale",
    )
    gradients = dict(model.named_parameters())
    for name in required:
        gradient = gradients[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.ne(0).any()


def test_zero_transfer_readout_removes_nonidentity_write_but_keeps_recurrence_path() -> None:
    model = make_model()
    inputs = model_inputs()
    with torch.no_grad():
        model.transfer_readout.weight.zero_()
    output = model(**inputs)
    assert output.transfer_delta.eq(0).all()
    assert output.recurrence_delta[0, 1].ne(0)
