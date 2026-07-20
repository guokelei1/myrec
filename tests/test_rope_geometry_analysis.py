import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "analyze_deep_dive_rope_geometry.py"
SPEC = importlib.util.spec_from_file_location("rope_geometry_analysis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_candidate_geometry_counts_phase_order_crossing_by_hand() -> None:
    # History keys are at 0,1,2,3; Q positions are 4,5; H=4.
    # After compression the relative phases are:
    # Q=4: [0,-1,-2,-3], Q=5: [1,0,-1,-2].
    result = MODULE._candidate_geometry(
        history_start=0,
        history_end=4,
        readout_positions=(4, 5),
        sequence_length=7,
        candidate_start=4,
        prompt_tokens=5,
        prompt_at_max_boundary=False,
        native_paths=2,
    )
    assert np.isclose(result["natural_closest_distance"], 1.5)
    assert np.isclose(result["natural_center_distance"], 3.0)
    assert np.isclose(result["natural_farthest_distance"], 4.5)
    assert np.isclose(result["compressed_center_distance"], -1.0)
    assert np.isclose(result["expanded_center_distance"], 7.0)
    assert np.isclose(result["compression_negative_edge_fraction"], 5.0 / 8.0)
    assert np.isclose(result["compression_zero_edge_fraction"], 2.0 / 8.0)
    assert np.isclose(result["compression_nonpositive_edge_fraction"], 7.0 / 8.0)
    # Of distances [1,2,3,4] and [2,3,4,5], five become absolutely closer,
    # two are unchanged (d=H/2=2), and one becomes farther (d=1).
    assert np.isclose(result["compression_absolute_closer_edge_fraction"], 5.0 / 8.0)
    assert np.isclose(result["compression_absolute_equal_edge_fraction"], 2.0 / 8.0)
    assert np.isclose(result["compression_absolute_farther_edge_fraction"], 1.0 / 8.0)
    assert np.isclose(result["compressed_absolute_distance"], 10.0 / 8.0)
    assert result["readout_q_negative_phase_fraction"] == 0.0
    assert result["history_k_beyond_readout_fraction"] == 1.0


def test_candidate_geometry_without_overshoot_stays_positive() -> None:
    result = MODULE._candidate_geometry(
        history_start=2,
        history_end=5,
        readout_positions=(8,),
        sequence_length=10,
        candidate_start=6,
        prompt_tokens=9,
        prompt_at_max_boundary=True,
        native_paths=1,
    )
    assert result["history_tokens"] == 3.0
    assert result["candidate_gap_tokens"] == 1.0
    assert result["natural_closest_distance"] == 4.0
    assert result["natural_center_distance"] == 5.0
    assert result["natural_farthest_distance"] == 6.0
    assert result["compression_negative_edge_fraction"] == 0.0
    assert result["compression_zero_edge_fraction"] == 0.0
    assert result["compressed_center_distance"] == 2.0
    assert result["compressed_absolute_distance"] == 2.0
    assert result["compression_absolute_closer_edge_fraction"] == 1.0
    assert result["prompt_at_max_boundary_fraction"] == 1.0


def test_request_geometry_uses_equal_candidate_mean() -> None:
    first = MODULE._candidate_geometry(
        history_start=2,
        history_end=4,
        readout_positions=(6,),
        sequence_length=8,
        candidate_start=5,
        prompt_tokens=7,
        prompt_at_max_boundary=False,
        native_paths=1,
    )
    second = MODULE._candidate_geometry(
        history_start=2,
        history_end=4,
        readout_positions=(8,),
        sequence_length=10,
        candidate_start=5,
        prompt_tokens=9,
        prompt_at_max_boundary=True,
        native_paths=1,
    )
    result = MODULE._request_geometry([first, second])
    assert result["candidate_count"] == 2
    assert result["prompt_tokens_mean"] == 8.0
    assert result["natural_center_distance_mean"] == 4.5
    assert result["prompt_at_max_boundary_fraction"] == 0.5


def test_nearest_rank_and_median_are_hand_auditable() -> None:
    summary = MODULE._summary([4.0, 1.0, 3.0, 2.0])
    assert summary["median"] == 2.5
    assert summary["p25"] == 1.0
    assert summary["p75"] == 3.0
    assert summary["mean"] == 2.5


def test_absolute_integer_range_sum_crosses_zero() -> None:
    assert MODULE._absolute_integer_range_sum(-3, 2) == 9
    assert MODULE._absolute_integer_range_sum(2, 4) == 9
    assert MODULE._absolute_integer_range_sum(-4, -2) == 9


def test_rope_rotation_geometry_has_identity_at_zero() -> None:
    cosine, relative_l2 = MODULE._rope_rotation_geometry(0)
    assert cosine == 1.0
    assert relative_l2 == 0.0
    cosine, relative_l2 = MODULE._rope_rotation_geometry(333)
    assert -1.0 <= cosine <= 1.0
    assert 0.0 <= relative_l2 <= 2.0
