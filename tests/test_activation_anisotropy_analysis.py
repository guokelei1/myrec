import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_deep_dive_activation_anisotropy.py"
)
SPEC = importlib.util.spec_from_file_location("activation_anisotropy", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_channel_participation_is_one_for_uniform_energy() -> None:
    values = np.ones(MODULE.HIDDEN_SIZE)
    metrics = MODULE._channel_energy_metrics(values)
    assert np.isclose(metrics["channel_participation_ratio"], 1.0)
    assert np.isclose(
        metrics["top_10pct_channel_energy_share"],
        np.ceil(MODULE.HIDDEN_SIZE * 0.10) / MODULE.HIDDEN_SIZE,
    )


def test_channel_participation_is_inverse_width_for_one_channel() -> None:
    values = np.zeros(MODULE.HIDDEN_SIZE)
    values[7] = 4.0
    metrics = MODULE._channel_energy_metrics(values)
    assert np.isclose(
        metrics["channel_participation_ratio"], 1.0 / MODULE.HIDDEN_SIZE
    )
    assert metrics["top_1pct_channel_energy_share"] == 1.0


def test_pairwise_cosine_from_unit_sum_is_hand_computed() -> None:
    # Unit vectors [1,0], [1,0], [0,1] have ordered off-diagonal cosines
    # 1,0,1,0,0,0, whose mean is 1/3.
    summed = np.asarray([2.0, 1.0])
    assert np.isclose(MODULE._mean_pairwise_cosine_from_sum(summed, 3), 1.0 / 3.0)
    assert MODULE._mean_pairwise_cosine_from_sum(np.asarray([1.0, 0.0]), 1) is None


def test_profile_cosine_and_safe_divide_handle_zero() -> None:
    assert np.isclose(
        MODULE._profile_cosine(np.asarray([1.0, 0.0]), np.asarray([2.0, 0.0])),
        1.0,
    )
    assert MODULE._profile_cosine(np.zeros(2), np.ones(2)) is None
    assert MODULE._safe_divide(1.0, 0.0) is None


def test_accumulator_preserves_channelwise_common_residual_identity() -> None:
    accumulator = MODULE._empty_accumulator()
    common = np.zeros((len(MODULE.STATES), MODULE.HIDDEN_SIZE))
    residual = np.zeros_like(common)
    common[:, 0] = 2.0
    residual[:, 1] = 3.0
    total = common**2 + residual**2
    MODULE._update_accumulator(
        accumulator,
        total_energy=total,
        common=common,
        common_energy=common**2,
        residual_energy=residual**2,
        history=common,
        candidate_count=2,
    )
    rows = MODULE._finalize_accumulator("q2", "all", accumulator)
    assert len(rows) == len(MODULE.STATES)
    assert np.isclose(rows[1]["common_energy_fraction"], 4.0 / 13.0)
    assert np.isclose(rows[1]["common_residual_channel_energy_cosine"], 0.0)
