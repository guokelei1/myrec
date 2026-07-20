import importlib.util
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analyze_deep_dive_q2_parameter_updates.py"
SPEC = importlib.util.spec_from_file_location("q2_parameter_update_analysis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parameter_family_classification_covers_attention_mlp_and_norms() -> None:
    assert MODULE._classify_key("model.layers.3.self_attn.q_proj.weight") == (3, "q_proj")
    assert MODULE._classify_key("model.layers.27.mlp.gate_proj.weight") == (
        27,
        "mlp_gate_proj",
    )
    assert MODULE._classify_key("model.layers.0.input_layernorm.weight") == (
        0,
        "input_rmsnorm",
    )
    assert MODULE._classify_key("model.embed_tokens.weight") == (
        None,
        "tied_embedding_readout",
    )


def test_tensor_update_statistics_are_hand_computed() -> None:
    base = torch.tensor([1.0, 2.0])
    final = torch.tensor([2.0, 0.0])
    stats = MODULE._tensor_statistics(base, final, chunk_elements=1)
    result = MODULE._finalize_accumulator(stats)
    assert result["parameter_count"] == 2
    assert np.isclose(result["base_frobenius"], np.sqrt(5.0))
    assert np.isclose(result["final_frobenius"], 2.0)
    assert np.isclose(result["update_frobenius"], np.sqrt(5.0))
    assert np.isclose(result["mean_absolute_update"], 1.5)
    assert result["exact_zero_fraction"] == 0.0
    assert np.isclose(result["base_update_cosine"], -3.0 / 5.0)


def test_accumulator_energy_share_uses_supplied_scope() -> None:
    first = MODULE._tensor_statistics(torch.zeros(2), torch.ones(2), 2)
    second = MODULE._tensor_statistics(torch.zeros(2), torch.tensor([2.0, 2.0]), 2)
    total = MODULE._empty_accumulator()
    MODULE._add_accumulator(total, first)
    MODULE._add_accumulator(total, second)
    result = MODULE._finalize_accumulator(first, total["update_sq_sum"])
    assert np.isclose(result["update_energy_share"], 2.0 / 10.0)


def test_pearson_is_exact_for_affine_and_constant_vectors() -> None:
    assert np.isclose(MODULE._pearson([1.0, 2.0, 3.0], [4.0, 6.0, 8.0]), 1.0)
    assert MODULE._pearson([1.0, 1.0], [2.0, 3.0]) is None
