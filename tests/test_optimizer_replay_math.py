from __future__ import annotations

import pytest

from myrec.mechanism.optimizer_replay_math import (
    Q2_PARAMETER_FAMILIES,
    adamw_exact_delta,
    clip_gradients,
    gradient_pair_summary,
    lora_function_delta,
    lora_singular_values,
    parameter_order_digest,
    q2_parameter_family,
    vector_cosine,
    vector_summary,
)


def test_adamw_replay_matches_real_torch_step_elementwise():
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    optimizer = torch.optim.AdamW(
        [parameter], lr=0.03, betas=(0.8, 0.9), eps=1e-7, weight_decay=0.2
    )
    parameter.grad = torch.tensor([0.4, -0.1], dtype=torch.float64)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = parameter.detach().clone()
    gradient = torch.tensor([-0.2, 0.3], dtype=torch.float64)
    state = optimizer.state[parameter]
    group = optimizer.param_groups[0]
    replay = adamw_exact_delta(before, gradient, state, group)
    parameter.grad = gradient.clone()
    optimizer.step()
    torch.testing.assert_close(
        parameter.detach() - before, replay["total_delta"], rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        replay["total_delta"],
        replay["moment_delta"] + replay["weight_decay_delta"],
        rtol=0,
        atol=0,
    )


def test_lora_function_delta_recomposes_and_svd_is_rank_bounded():
    torch = pytest.importorskip("torch")
    a = torch.randn(3, 5)
    b = torch.randn(7, 3)
    delta_a = torch.randn_like(a) * 0.01
    delta_b = torch.randn_like(b) * 0.01
    result = lora_function_delta(a, b, delta_a, delta_b)
    torch.testing.assert_close(
        result["joint"],
        result["a_only"] + result["b_only"] + result["interaction"],
    )
    assert result["recomposition_max_abs_error"] < 1e-5
    svd = lora_singular_values(a, b)
    assert len(svd["singular_values_descending"]) == 3
    assert svd["effective_rank"] <= 3


def test_q2_family_partition_and_vector_accounting():
    torch = pytest.importorskip("torch")
    names = {
        "model.embed_tokens.weight": "embedding_readout",
        "model.layers.0.self_attn.q_proj.weight": "attention_q",
        "model.layers.0.self_attn.k_proj.weight": "attention_k",
        "model.layers.0.self_attn.v_proj.weight": "attention_v",
        "model.layers.0.self_attn.o_proj.weight": "attention_o",
        "model.layers.0.self_attn.q_norm.weight": "rmsnorm",
        "model.layers.0.mlp.gate_proj.weight": "mlp_gate",
        "model.layers.0.mlp.up_proj.weight": "mlp_up",
        "model.layers.0.mlp.down_proj.weight": "mlp_down",
    }
    assert {q2_parameter_family(name) for name in names} == set(Q2_PARAMETER_FAMILIES)
    vectors = {name: torch.tensor([index + 1.0]) for index, name in enumerate(names)}
    summary = vector_summary(vectors, family_by_name=names)
    assert summary["squared_norm"] == pytest.approx(sum((i + 1) ** 2 for i in range(9)))
    assert sum(summary["family_share"].values()) == pytest.approx(1.0)
    assert vector_cosine(vectors, vectors) == pytest.approx(1.0)


def test_global_clip_uses_registered_torch_coefficient():
    torch = pytest.importorskip("torch")
    clipped, coefficient, norm = clip_gradients({"x": torch.tensor([3.0, 4.0])}, 1.0)
    assert norm == pytest.approx(5.0)
    assert coefficient == pytest.approx(1.0 / 5.000001)
    assert clipped["x"].norm().item() == pytest.approx(5.0 / 5.000001)


def test_parameter_order_digest_binds_index_name_shape_and_dtype():
    torch = pytest.importorskip("torch")
    rows = [("a", torch.nn.Parameter(torch.zeros(2))), ("b", torch.nn.Parameter(torch.zeros(3)))]
    assert parameter_order_digest(rows) != parameter_order_digest(list(reversed(rows)))


def test_chunked_gradient_pair_summary_is_hand_computed():
    torch = pytest.importorskip("torch")
    result = gradient_pair_summary(
        ["a", "b"],
        [torch.tensor([3.0, 4.0]), torch.tensor([0.0])],
        [torch.tensor([4.0, -3.0]), torch.tensor([0.0])],
        family_by_name={"a": "x", "b": "y"},
        chunk_size=1,
    )
    assert result["left_norm"] == pytest.approx(5.0)
    assert result["right_norm"] == pytest.approx(5.0)
    assert result["cosine"] == pytest.approx(0.0)
    assert result["left_family_share"] == {"x": 1.0, "y": 0.0}
