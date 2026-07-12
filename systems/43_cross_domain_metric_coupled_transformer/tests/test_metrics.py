from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "src"))

from myrec.eval.metrics import ndcg_at_k  # noqa: E402
from train.gate_metrics import bootstrap, clicked_direction  # noqa: E402


def test_shared_ndcg_hand_value_and_gate_helpers() -> None:
    ranked = ["negative", "positive", "other"]
    expected = (1.0 / math.log2(3.0)) / (1.0 / math.log2(2.0))
    assert ndcg_at_k(ranked, {"positive"}, 10) == expected
    values = np.asarray([1.0, 2.0, 3.0])
    report = bootstrap(values, samples=1000, seed=7)
    assert report["mean"] == 2.0
    direction = clicked_direction(
        [np.asarray([0.5, -0.5]), np.asarray([1.0, 0.0, -1.0])],
        [np.asarray([1, 0]), np.asarray([0, 1, 0])],
    )
    assert np.allclose(direction, [1.0, 0.0])
