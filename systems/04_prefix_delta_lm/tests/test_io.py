from __future__ import annotations

import json
from pathlib import Path

import pytest

from cpdlr.io import (
    assert_candidate_manifest,
    assert_label_free_record,
    assert_train_only_path,
)


def test_candidate_manifest_hash_is_asserted(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"entries": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate manifest hash mismatch"):
        assert_candidate_manifest(path, "0" * 64)


@pytest.mark.parametrize(
    "path",
    ["qrels_dev.jsonl", "records_test.jsonl", "runs/x/metrics.json"],
)
def test_training_rejects_label_or_heldout_paths(path: str) -> None:
    with pytest.raises(ValueError, match="frozen label/split boundary"):
        assert_train_only_path(path)


def test_label_free_scoring_rejects_candidate_labels() -> None:
    with pytest.raises(ValueError, match="label-bearing candidate"):
        assert_label_free_record({"candidates": [{"item_id": "a", "clicked": 1}]})
