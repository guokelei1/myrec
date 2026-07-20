from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "analyze_deep_dive_embedding_readout.py"
)
SPEC = importlib.util.spec_from_file_location("embedding_readout", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_row_update_metrics_are_hand_computed() -> None:
    torch = pytest.importorskip("torch")
    base = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
    final = torch.tensor([[3.0, 4.0], [0.0, 1.0]])
    result = MODULE._row_update_metrics(base, final, chunk_size=1)
    np.testing.assert_allclose(result["base_l2"], [5.0, 1.0])
    np.testing.assert_allclose(result["update_l2"], [0.0, np.sqrt(2.0)])
    np.testing.assert_allclose(result["update_rms"], [0.0, 1.0])
    np.testing.assert_allclose(result["base_final_cosine"], [1.0, 0.0])
    assert result["exactly_unchanged"].tolist() == [True, False]


def test_role_summary_uses_occurrence_weights() -> None:
    metrics = {
        "vocabulary_rows": 3,
        "hidden_size": 2,
        "update_l2": np.asarray([0.0, np.sqrt(2.0), 2.0 * np.sqrt(2.0)]),
        "update_rms": np.asarray([0.0, 1.0, 2.0]),
        "relative_update_l2": np.asarray([0.0, 0.25, 0.5]),
        "base_final_cosine": np.asarray([1.0, 0.9, 0.8]),
        "exactly_unchanged": np.asarray([True, False, False]),
    }
    result = MODULE._role_summary(Counter({1: 3, 2: 1}), metrics)
    assert result["occurrences"] == 4
    assert result["unique_token_ids"] == 2
    assert result["occurrence_weighted_mean_row_update_rms"] == pytest.approx(1.25)
    assert result["occurrence_weighted_update_rms"] == pytest.approx(
        np.sqrt((3.0 + 4.0) / 4.0)
    )
    assert result["occurrence_weighted_relative_update_l2"] == pytest.approx(0.3125)


def test_direction_metrics_detect_rotation_and_common_stability() -> None:
    torch = pytest.importorskip("torch")
    base = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    final = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    result = MODULE._direction_metrics(
        base,
        final,
        yes_token_id=0,
        no_token_id=1,
        update_rms_distribution=np.asarray([0.0, 1.0, 2.0]),
    )
    assert result["base_final_direction_cosine"] == pytest.approx(-1.0)
    assert result["base_final_common_cosine"] == pytest.approx(1.0)
    assert result["direction_update_relative_to_base"] == pytest.approx(2.0)
