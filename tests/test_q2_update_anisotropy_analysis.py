import importlib.util
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analyze_deep_dive_q2_update_anisotropy.py"
SPEC = importlib.util.spec_from_file_location("q2_update_anisotropy_analysis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_uniform_energy_has_full_normalized_effective_counts() -> None:
    metrics = MODULE._distribution_metrics(np.ones(4))
    assert metrics["normalized_participation_ratio"] == 1.0
    assert np.isclose(metrics["normalized_entropy_effective_count"], 1.0)
    assert np.isclose(metrics["top_10pct_energy_share"], 0.25)
    assert metrics["maximum_to_mean_energy_ratio"] == 1.0


def test_single_channel_energy_is_maximally_concentrated() -> None:
    metrics = MODULE._distribution_metrics(np.asarray([4.0, 0.0, 0.0, 0.0]))
    assert np.isclose(metrics["normalized_participation_ratio"], 0.25)
    assert np.isclose(metrics["normalized_entropy_effective_count"], 0.25)
    assert metrics["top_1pct_energy_share"] == 1.0
    assert metrics["maximum_to_mean_energy_ratio"] == 4.0
    assert metrics["zero_energy_fraction"] == 0.75


def test_query_head_grouping_sums_exact_energy() -> None:
    update = torch.zeros((2048, 1024))
    update[:128] = 1.0
    kind, energy = MODULE._semantic_energy("q_proj", update)
    assert kind == "query_attention_head"
    assert energy.shape == (16,)
    assert energy[0].item() == 128 * 1024
    assert torch.count_nonzero(energy[1:]).item() == 0


def test_o_projection_groups_input_head_columns() -> None:
    update = torch.zeros((1024, 2048))
    update[:, 128:256] = 2.0
    kind, energy = MODULE._semantic_energy("o_proj", update)
    assert kind == "query_attention_head"
    assert energy.shape == (16,)
    assert energy[1].item() == 1024 * 128 * 4
    assert torch.count_nonzero(torch.cat((energy[:1], energy[2:]))).item() == 0


def test_layer_key_classification_preserves_component() -> None:
    assert MODULE._classify_layer_key("model.layers.12.mlp.down_proj.weight") == (
        12,
        "mlp_down_proj",
    )
    assert MODULE._classify_layer_key("model.norm.weight") is None
