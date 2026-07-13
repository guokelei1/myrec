import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.analysis.r0_scope import (  # noqa: E402
    mde_from_reference_ci,
    required_units_for_mde,
    split_user_overlap,
    summarize_records,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_scope_summary_uses_only_label_free_objects(tmp_path: Path) -> None:
    path = tmp_path / "records_train.jsonl"
    write_jsonl(
        path,
        [
            {
                "request_id": "r1",
                "user_id": "u1",
                "ts": 10,
                "query": "shoe",
                "history": [
                    {"item_id": "i1", "title": "old shoe", "event": "click", "ts": 4}
                ],
                "candidates": [
                    {"item_id": "i1", "title": "shoe", "clicked": 1},
                    {"item_id": "i2", "title": "boot", "clicked": 0},
                ],
            },
            {
                "request_id": "r2",
                "user_id": "u2",
                "ts": 20,
                "query": "bag",
                "history": [],
                "candidates": [{"item_id": "i3", "cat": ["bags"]}],
            },
        ],
    )
    summary, users = summarize_records([path])
    assert users == {"u1", "u2"}
    assert summary["requests"] == 2
    assert summary["repeat_present_rate"] == 0.5
    assert summary["candidate_text_coverage"] == 1.0
    assert summary["strictly_prior_violations"] == 0


def test_scope_rejects_qrels_path(tmp_path: Path) -> None:
    path = tmp_path / "qrels_dev.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PermissionError):
        summarize_records([path])


def test_power_scaling_and_user_overlap() -> None:
    first = mde_from_reference_ci([-0.01, 0.01], 100, 100)
    second = mde_from_reference_ci([-0.01, 0.01], 100, 400)
    assert second == pytest.approx(first / 2)
    required = required_units_for_mde([-0.01, 0.01], 100, first)
    assert required == 100
    overlap = split_user_overlap({"a", "b"}, {"b", "c"})
    assert overlap["overlap_users"] == 1
    assert overlap["user_disjoint"] is False

