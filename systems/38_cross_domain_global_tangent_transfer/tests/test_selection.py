from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.selection import load_blind_records, materialize_selection  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_records(path: Path, *, labeled: bool = False) -> None:
    rows = []
    for index in range(12):
        candidates = [
            {"item_id": f"i{index}_{candidate}", "title": "text"}
            for candidate in range(4)
        ]
        if labeled:
            candidates[0]["clicked"] = 1
        rows.append(
            {
                "request_id": f"r{index}",
                "user_id": f"u{index % 5}",
                "session_id": f"s{index}",
                "ts": 100,
                "query": "query",
                "history": [
                    {"item_id": f"h{index}_{event}", "ts": event + 1, "title": "history"}
                    for event in range(1 + index % 3)
                ],
                "candidates": candidates,
                "masks": {"history_present": True},
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_blind_loader_rejects_candidate_labels(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    _write_records(path, labeled=True)
    with pytest.raises(PermissionError):
        load_blind_records(path)


def test_selection_is_deterministic_and_label_free(tmp_path: Path) -> None:
    records = tmp_path / "records_train_blind.jsonl"
    _write_records(records)
    manifest = tmp_path / "manifest.json"
    candidates = tmp_path / "candidate_manifest.json"
    report = tmp_path / "c0.json"
    _write_json(manifest, {"dataset_id": "amazon_c4"})
    _write_json(candidates, {"dataset_id": "amazon_c4"})
    _write_json(
        report,
        {
            "overall_status": "passed",
            "checks": {"dev_test_records_label_free": True},
        },
    )
    kwargs = {
        "records_path": records,
        "standardized_manifest_path": manifest,
        "candidate_manifest_path": candidates,
        "c0_report_path": report,
        "seed": 38,
        "role_counts": {"fit": 4, "internal_A": 2, "delayed_B": 2, "escrow": 2},
        "length_bins": [1, 2, 3, 5],
    }
    first = materialize_selection(output_path=tmp_path / "first.json", **kwargs)
    second = materialize_selection(output_path=tmp_path / "second.json", **kwargs)
    assert first == second
    assert first["wrong_donor_audit"]["coverage_fraction"] == 1.0
    assert first["wrong_donor_audit"]["same_length_bin_fraction"] == 1.0
    assert first["wrong_donor_audit"]["same_user_assignments"] == 0
    assert first["label_access"]["records_train_labels_opened"] is False
    selected = sum(len(first["roles"][role]["indices"]) for role in first["roles"])
    assert selected == 10
