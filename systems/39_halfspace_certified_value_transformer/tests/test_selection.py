from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.selection import (  # noqa: E402
    load_blind_records,
    materialize_selection,
    sha256_file,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_records(path: Path, *, labeled: bool = False, rows: int = 20) -> None:
    values = []
    for index in range(rows):
        candidates = [
            {"item_id": f"i{index}_{candidate}", "title": "text"}
            for candidate in range(4)
        ]
        if labeled:
            candidates[0]["clicked"] = 1
        values.append(
            {
                "request_id": f"r{index}",
                "user_id": f"u{index}",
                "session_id": f"s{index}",
                "ts": 100,
                "query": "query",
                "history": [
                    {
                        "item_id": f"h{index}_{event}",
                        "ts": event + 1,
                        "title": "history",
                    }
                    for event in range(1 + index % 3)
                ],
                "candidates": candidates,
                "masks": {"history_present": True},
            }
        )
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def _predecessor(path: Path) -> str:
    value = {
        "candidate_id": "c38",
        "train_requests": 20,
        "roles": {
            "fit": {"indices": [0, 1, 2, 3]},
            "internal_A": {"indices": [4, 5]},
            "delayed_B": {"indices": [6, 7]},
            "escrow": {"indices": [8, 9]},
        },
        "unused_indices": list(range(10, 20)),
    }
    _write_json(path, value)
    return sha256_file(path)


def test_blind_loader_rejects_candidate_labels(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    _write_records(path, labeled=True)
    with pytest.raises(PermissionError):
        load_blind_records(path)


def test_selection_reuses_only_fit_and_draws_a_from_unused(tmp_path: Path) -> None:
    records = tmp_path / "records_train_blind.jsonl"
    _write_records(records)
    manifest = tmp_path / "manifest.json"
    candidates = tmp_path / "candidate_manifest.json"
    c0 = tmp_path / "c0.json"
    predecessor = tmp_path / "c38_selection.json"
    _write_json(manifest, {"dataset_id": "amazon_c4"})
    _write_json(candidates, {"dataset_id": "amazon_c4"})
    _write_json(
        c0,
        {
            "overall_status": "passed",
            "checks": {"dev_test_records_label_free": True},
        },
    )
    predecessor_sha = _predecessor(predecessor)
    kwargs = {
        "records_path": records,
        "standardized_manifest_path": manifest,
        "candidate_manifest_path": candidates,
        "c0_report_path": c0,
        "predecessor_selection_path": predecessor,
        "predecessor_selection_sha256": predecessor_sha,
        "seed": 39,
        "internal_a_requests": 6,
        "length_bins": [1, 2, 3, 5],
    }
    first = materialize_selection(output_path=tmp_path / "first.json", **kwargs)
    second = materialize_selection(output_path=tmp_path / "second.json", **kwargs)
    assert first == second
    assert first["roles"]["fit"]["indices"] == [0, 1, 2, 3]
    assert set(first["roles"]["internal_A"]["indices"]) <= set(range(10, 20))
    assert len(first["reserve_indices"]) == 4
    assert first["outcome_isolation"]["overlap_with_c38_internal_A_delayed_B_escrow"] == 0
    assert first["wrong_donor_audit"] == {
        "requests": 10,
        "coverage_fraction": 1.0,
        "same_length_bin_fraction": 1.0,
        "same_user_assignments": 0,
    }
    assert first["label_access"]["records_train_labels_opened"] is False
