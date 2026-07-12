from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.features import collect_label_free_features  # noqa: E402
from train.selection import materialize_selection  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_feature_collection_uses_only_frozen_blind_roles(tmp_path: Path) -> None:
    records = []
    for index in range(8):
        records.append(
            {
                "request_id": f"r{index}",
                "user_id": f"u{index}",
                "session_id": f"s{index}",
                "ts": 10,
                "query": f"query {index}",
                "history": [
                    {
                        "item_id": f"h{index}",
                        "title": f"history {index}",
                        "brand": "brand",
                        "cat": ["cat", "", ""],
                        "ts": 1,
                    }
                ],
                "candidates": [
                    {
                        "item_id": f"c{index}_{candidate}",
                        "title": f"candidate {candidate}",
                        "brand": "brand",
                        "cat": ["cat", "", ""],
                    }
                    for candidate in range(3)
                ],
                "masks": {"history_present": True},
            }
        )
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    candidates = tmp_path / "candidates.json"
    c0 = tmp_path / "c0.json"
    _write_json(manifest, {"dataset_id": "amazon_c4"})
    _write_json(candidates, {"dataset_id": "amazon_c4"})
    _write_json(c0, {"overall_status": "passed", "checks": {"dev_test_records_label_free": True}})
    selection_path = tmp_path / "selection.json"
    selection = materialize_selection(
        records_path=records_path,
        standardized_manifest_path=manifest,
        candidate_manifest_path=candidates,
        c0_report_path=c0,
        output_path=selection_path,
        seed=38,
        role_counts={"fit": 2, "internal_A": 2, "delayed_B": 1, "escrow": 1},
        length_bins=[1, 2],
    )
    feature_root = tmp_path / "features"
    output = collect_label_free_features(
        records_path=records_path,
        selection_path=selection_path,
        output_root=feature_root,
    )
    assert output["requests"] == 4
    assert output["label_access"]["records_train_labels_opened"] is False
    indices = np.load(feature_root / "feature_request_indices.npy")
    expected = selection["roles"]["fit"]["indices"] + selection["roles"]["internal_A"]["indices"]
    assert indices.tolist() == expected
    offsets = np.load(feature_root / "candidate_offsets.npy")
    assert offsets.tolist() == [0, 3, 6, 9, 12]
    assert (feature_root / "items.jsonl").read_text(encoding="utf-8").count("\n") > 0
