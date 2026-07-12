from __future__ import annotations

import json
from pathlib import Path

from myrec.data.amazon_c4_protocol_audit import check_train_blind_equivalence


def test_train_blind_equivalence_allows_only_label_removal(tmp_path: Path) -> None:
    labeled = tmp_path / "records_train.jsonl"
    blind = tmp_path / "records_train_blind.jsonl"
    row = {
        "request_id": "r1",
        "query": "q",
        "history": [],
        "candidates": [
            {"item_id": "a", "title": "A", "clicked": 1, "purchased": 1},
            {"item_id": "b", "title": "B", "clicked": 0, "purchased": 0},
        ],
    }
    labeled.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    for candidate in row["candidates"]:
        candidate.pop("clicked")
        candidate.pop("purchased")
    blind.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    assert check_train_blind_equivalence(labeled, blind)["status"] == "passed"
    row["query"] = "changed"
    blind.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    assert check_train_blind_equivalence(labeled, blind)["status"] == "failed"
