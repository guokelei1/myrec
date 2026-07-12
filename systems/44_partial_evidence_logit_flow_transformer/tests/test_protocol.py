from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "src"))

from myrec.eval.metrics import ndcg_at_k  # noqa: E402
from probe.run_design_gate import make_rows  # noqa: E402


def config() -> dict:
    return yaml.safe_load((ROOT / "configs/design_gate.yaml").read_text(encoding="utf-8"))


def test_generator_is_deterministic_and_planted_signal_is_candidate_local() -> None:
    first = make_rows(config(), 8, 123)
    second = make_rows(config(), 8, 123)
    for name in first.__dataclass_fields__:
        assert np.array_equal(getattr(first, name), getattr(second, name))
    for row in range(8):
        signal = first.clean_history[row, first.signal_position[row]]
        target = first.target[row]
        query = first.query[row]
        candidates = first.candidates[row]
        transported = (query + signal) / np.linalg.norm(query + signal)
        surplus = candidates @ transported - candidates @ query
        assert int(np.argmax(surplus)) == int(target)
        assert surplus[target] > 0


def test_config_is_data_free_and_shared_metric_hand_value() -> None:
    value = config()
    registered = "\n".join(str(item).lower() for item in value["paths"].values())
    for forbidden in ("data/", "records_", "qrels", "candidate_labels", "runs/"):
        assert forbidden not in registered
    assert value["authorization"] == {
        "data_free_synthetic_only": True,
        "repository_data": False,
        "train_labels": False,
        "dev": False,
        "test": False,
    }
    expected = 1.0 / math.log2(3.0)
    assert ndcg_at_k(["negative", "positive"], {"positive"}, 10) == expected
