from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.data.kuaisearch_source_audit import (
    _resolve_source_path,
    audit_kuaisearch_source,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _recall(
    user: int,
    session: int,
    time: int,
    candidates: list[int],
    clicked: list[int],
) -> dict:
    return {
        "user_id": user,
        "session_id": session,
        "query": "red shoes",
        "time_index": time,
        "impressed_item_ids": candidates,
        "clicked_item_ids": clicked,
        "purchased_item_ids": [],
        "split": "train",
    }


def _rank_rows(recall: dict, history: list[int]) -> list[dict]:
    return [
        {
            "user_id": recall["user_id"],
            "session_id": recall["session_id"],
            "query": recall["query"],
            "time_index": recall["time_index"],
            "target_item_id": item_id,
            "recently_clicked_item_ids": history,
            "recently_purchased_item_ids": [],
            "is_clicked": int(item_id in recall["clicked_item_ids"]),
            "is_purchased": 0,
        }
        for item_id in recall["impressed_item_ids"]
    ]


class KuaiSearchSourceAuditTest(unittest.TestCase):
    def test_rejects_mixed_lite_and_full_source_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jsonl(root / "recall" / "train.jsonl", [{"split": "train"}])
            _write_jsonl(
                root / "recall_lite" / "train.jsonl", [{"split": "train"}]
            )
            with self.assertRaisesRegex(FileNotFoundError, "exactly one"):
                _resolve_source_path(root, "recall")

    def test_audit_separates_prior_history_and_finds_collision_opportunity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            recall_rows = [
                _recall(1, 11, 1, [90, 91], [90]),
                _recall(2, 21, 1, [92, 93], [92]),
                _recall(1, 12, 2, [10, 20, 30], [10]),
                _recall(2, 22, 2, [10, 20, 40], [20]),
            ]
            _write_jsonl(raw / "recall_lite" / "train.jsonl", recall_rows)
            rank_rows: list[dict] = []
            rank_rows.extend(_rank_rows(recall_rows[0], []))
            rank_rows.extend(_rank_rows(recall_rows[1], []))
            rank_rows.extend(_rank_rows(recall_rows[2], [90]))
            rank_rows.extend(_rank_rows(recall_rows[3], [92]))
            _write_jsonl(raw / "rank_lite" / "train.jsonl", rank_rows)

            report_path = root / "report.json"
            report = audit_kuaisearch_source(
                raw,
                report_path,
                collision_query_limit=10,
                collision_requests_per_query=10,
                rank_history_sample_size=10,
            )

            self.assertEqual(report["recall"]["rows"], 4)
            self.assertFalse(
                report["integrity"]["evaluation_source_behavior_fields_accessed"]
            )
            self.assertEqual(
                report["reconstructed_prior_history"]["history_present_requests"],
                2,
            )
            self.assertEqual(
                report["reconstructed_prior_history"]["strict_nonrepeat_requests"],
                2,
            )
            self.assertEqual(
                report["exact_query_collision_scout"][
                    "cross_user_pairs_with_at_least_two_shared_strict_candidates"
                ],
                1,
            )
            self.assertEqual(
                report["exact_query_collision_scout"]["eligible_pairs_per_query"][
                    "max"
                ],
                1,
            )
            self.assertEqual(
                report["raw_rank_recent_history_sample"]["found_requests"], 2
            )
            self.assertEqual(
                report["raw_rank_recent_history_sample"]["classification"][
                    "past_supported"
                ],
                2,
            )
            self.assertTrue(report_path.exists())

    def test_evaluation_source_rows_are_excluded_before_behavior_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            train = _recall(1, 11, 1, [10, 20], [10])
            held_out = {
                "split": "test",
                "clicked_item_ids": "must-not-be-interpreted-as-a-list",
                "purchased_item_ids": "must-not-be-interpreted-as-a-list",
            }
            _write_jsonl(
                raw / "recall_lite" / "train.jsonl",
                [train, held_out],
            )
            _write_jsonl(
                raw / "rank_lite" / "train.jsonl",
                _rank_rows(train, []),
            )
            report = audit_kuaisearch_source(
                raw,
                root / "report.json",
                collision_query_limit=2,
                collision_requests_per_query=2,
                rank_history_sample_size=1,
            )
            self.assertEqual(report["recall"]["rows"], 1)
            self.assertEqual(report["recall"]["excluded_source_rows"], 1)
            self.assertEqual(
                report["recall"]["source_split_counts_seen"],
                {"test": 1, "train": 1},
            )

    def test_rejects_request_to_include_test_source_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "locked to split=train"):
                audit_kuaisearch_source(
                    Path(tmp),
                    Path(tmp) / "report.json",
                    included_source_splits=("train", "test"),
                )


if __name__ == "__main__":
    unittest.main()
