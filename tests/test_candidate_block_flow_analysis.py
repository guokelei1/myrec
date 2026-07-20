import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analyze_deep_dive_candidate_block_flow.py"
SPEC = importlib.util.spec_from_file_location("candidate_block_flow_analysis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _pad_states(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    values = np.zeros((2, 29, 1024), dtype=np.float64)
    values[:, 0, 0] = first
    values[:, 1:, 0] = second[:, None]
    return values


def test_component_flow_energy_identity_is_hand_computed() -> None:
    # x=2, update=1, output=3: 9 = 4 + 1 + 4.
    values = np.zeros((29, 1024), dtype=np.float64)
    values[0, 0] = 2.0
    values[1:, 0] = 3.0
    flow = MODULE._component_flow(values, state_axis=0)
    assert np.isclose(flow["input_mse"][0], 4.0 / 1024.0)
    assert np.isclose(flow["update_mse"][0], 1.0 / 1024.0)
    assert np.isclose(flow["output_mse"][0], 9.0 / 1024.0)
    assert np.isclose(flow["interaction_cross_mse"][0], 4.0 / 1024.0)
    assert np.isclose(flow["input_update_cosine"][0], 1.0)


def test_candidate_relative_attenuation_has_negative_block_interference() -> None:
    delta = _pad_states(np.asarray([-1.0, 1.0]), np.asarray([-0.5, 0.5]))
    flow = MODULE._block_flow(delta)["candidate_relative"]
    assert np.isclose(flow["input_update_cosine"][0], -1.0)
    assert flow["interaction_cross_mse"][0] < 0.0
    assert flow["energy_decreased"][0]
    assert np.isclose(flow["output_input_rms_ratio"][0], 0.5)


def test_common_and_candidate_relative_updates_are_separated() -> None:
    # Both candidates receive +1, so only the common component changes.
    delta = _pad_states(np.asarray([1.0, 3.0]), np.asarray([2.0, 4.0]))
    flow = MODULE._block_flow(delta)
    assert flow["common"]["update_mse"][0] > 0.0
    assert np.isclose(flow["candidate_relative"]["update_mse"][0], 0.0)
    assert np.isclose(flow["update_common_energy_fraction"][0], 1.0)


def test_zero_input_ratios_are_undefined_not_infinite() -> None:
    delta = _pad_states(np.asarray([0.0, 0.0]), np.asarray([-1.0, 1.0]))
    flow = MODULE._block_flow(delta)["candidate_relative"]
    assert np.isnan(flow["input_update_cosine"][0])
    assert np.isnan(flow["output_input_rms_ratio"][0])
    assert np.isfinite(flow["energy_change"]).all()


def test_region_summary_skips_undefined_block_zero_cosine() -> None:
    rows = []
    for model_key in MODULE.MODELS:
        for fold_name in ("all", "0", "1"):
            for block in MODULE.BLOCKS:
                row = {
                    "model_key": model_key,
                    "normalized_query_fold": fold_name,
                    "block_zero_based": block,
                }
                for metric in (
                    "mean_output_common_energy_fraction",
                    "mean_update_common_energy_fraction",
                    "mean_common_energy_change",
                    "mean_common_interaction_cross_mse",
                    "mean_common_input_update_cosine",
                    "fraction_requests_common_energy_decreased",
                    "mean_candidate_relative_energy_change",
                    "mean_candidate_relative_interaction_cross_mse",
                    "mean_candidate_relative_input_update_cosine",
                    "mean_candidate_relative_update_projection_coefficient",
                    "fraction_requests_candidate_relative_energy_decreased",
                ):
                    row[metric] = None if block == 0 and "cosine" in metric else 1.0
                rows.append(row)
    regions = MODULE._build_region_rows(rows)
    assert len(regions) == 24
    first = regions[0]
    assert first["mean_common_input_update_cosine"] == 1.0
