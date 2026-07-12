from __future__ import annotations

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.data_bridge import C74Store  # noqa: E402


def config() -> dict:
    return yaml.safe_load((ROOT / "configs/kuai_lm_probe.yaml").read_text())


def test_authorization_is_label_staged() -> None:
    row = config()
    assert row["authorization"]["validation_labels_after_A0_only"] is True
    assert row["authorization"]["fresh_features_scores_labels"] is False
    assert row["authorization"]["dev"] is False
    assert row["authorization"]["test"] is False
    assert row["authorization"]["qrels"] is False


def test_store_split_matches_registered_boundary_without_labels() -> None:
    row = config(); store = C74Store(row, REPO)
    manifest = store.split_manifest()
    assert manifest["train_requests"] == 4800
    assert manifest["validation_requests"] == 1200
    assert manifest["overlap"] == 0
    assert store._labels is None
