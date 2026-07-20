from __future__ import annotations

import numpy as np
import pytest

from myrec.mechanism.attention_edge_scoring import (
    _aggregate_paths,
    _neutralize_paths,
)
from myrec.mechanism.deep_dive_assignments import CONTENT_NEUTRAL_TOKEN_ID


def test_q3_path_aggregation_is_yes_minus_no():
    paths = [{"weight": 1.0}, {"weight": -1.0}]
    observed = _aggregate_paths(
        paths,
        [np.asarray([2.0, 3.0]), np.asarray([0.5, 4.0])],
    )
    np.testing.assert_allclose(observed, [1.5, -1.0])


def test_neutralization_changes_only_frozen_history_span():
    torch = pytest.importorskip("torch")
    path = {
        "ids": torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
        "starts": torch.tensor([1, 2]),
        "ends": torch.tensor([3, 4]),
        "weight": 1.0,
    }
    result = _neutralize_paths([path])[0]
    assert result["ids"].tolist() == [
        [1, CONTENT_NEUTRAL_TOKEN_ID, CONTENT_NEUTRAL_TOKEN_ID, 4, 5],
        [6, 7, CONTENT_NEUTRAL_TOKEN_ID, CONTENT_NEUTRAL_TOKEN_ID, 10],
    ]
    assert path["ids"].tolist() == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
