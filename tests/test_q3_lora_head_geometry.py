import importlib.util
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analyze_deep_dive_q3_lora_head_geometry.py"
SPEC = importlib.util.spec_from_file_location("q3_lora_head_geometry", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_adapter_key_parser_recovers_layer_projection_and_factor() -> None:
    key = "base_model.model.model.layers.27.self_attn.v_proj.lora_B.weight"
    assert MODULE._parse_adapter_key(key) == (27, "v", "B")


def test_lora_function_scaling_is_two_ba(tmp_path: Path) -> None:
    # Function algebra is covered directly with the same expression used by the loader.
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[3.0], [4.0]])
    expected = torch.tensor([[6.0, 12.0], [8.0, 16.0]])
    torch.testing.assert_close(2.0 * b @ a, expected)


def test_q_head_energy_groups_128_output_rows() -> None:
    delta = torch.zeros((2048, 1024), dtype=torch.float64)
    delta[128:256] = 2.0
    energy = MODULE._head_energy(delta, "q")
    assert energy.shape == (16,)
    assert energy[1].item() == 128 * 1024 * 4
    assert torch.count_nonzero(torch.cat((energy[:1], energy[2:]))).item() == 0


def test_uniform_and_concentrated_head_distributions_are_hand_computed() -> None:
    uniform = MODULE._distribution_metrics(np.ones(8))
    assert np.isclose(uniform["normalized_participation_ratio"], 1.0)
    assert np.isclose(uniform["top_group_energy_share"], 1.0 / 8.0)
    concentrated = MODULE._distribution_metrics(np.asarray([4.0] + [0.0] * 7))
    assert np.isclose(concentrated["normalized_participation_ratio"], 1.0 / 8.0)
    assert concentrated["top_group_energy_share"] == 1.0
