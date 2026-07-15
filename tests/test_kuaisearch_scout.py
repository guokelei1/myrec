from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.data.contracts import audit_standardized_file
from myrec.data.kuaisearch_scout import (
    _resolve_source_path,
    build_kuaisearch_lite_scout,
)
from myrec.utils.jsonl import iter_jsonl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class KuaiSearchScoutTest(unittest.TestCase):
    def test_resolves_full_source_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jsonl(root / "recall" / "train.jsonl", [{"split": "train"}])
            path, variant = _resolve_source_path(root, "recall")
            self.assertEqual(path, root / "recall" / "train.jsonl")
            self.assertEqual(variant, "full")

    def test_builds_time_split_label_isolated_scout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            recall_rows = []
            item_ids = set()
            for index in range(14):
                candidates = [100 + index, 200 + index]
                item_ids.update(candidates)
                recall_rows.append(
                    {
                        "user_id": index % 2,
                        "session_id": index,
                        "query": "shared query" if index >= 8 else f"query {index}",
                        "time_index": index + 1,
                        "impressed_item_ids": candidates,
                        "clicked_item_ids": [candidates[0]],
                        "purchased_item_ids": [candidates[0]] if index % 5 == 0 else [],
                        "split": "train",
                    }
                )
            recall_rows.append(
                {
                    "split": "test",
                    "clicked_item_ids": "closed",
                    "purchased_item_ids": "closed",
                }
            )
            _write_jsonl(raw / "recall_lite" / "train.jsonl", recall_rows)
            _write_jsonl(
                raw / "items_lite" / "train.jsonl",
                [
                    {
                        "item_id": item_id,
                        "item_title": f"item {item_id}",
                        "brand_name": "brand",
                        "category_level1_name": "cat",
                    }
                    for item_id in sorted(item_ids)
                ],
            )

            output = root / "standardized"
            report = build_kuaisearch_lite_scout(
                raw,
                output,
                root / "report.json",
                max_requests=10,
                dev_fraction=0.3,
                max_history_len=3,
                include_history_query=True,
            )

            self.assertEqual(report["selection"]["selected_requests"], 10)
            self.assertEqual(report["source"]["excluded_source_split_counts"], {"test": 1})
            dev_rows = list(iter_jsonl(output / "records_dev.jsonl"))
            self.assertTrue(dev_rows)
            self.assertTrue(all("clicked" not in candidate for row in dev_rows for candidate in row["candidates"]))
            self.assertTrue(
                all(event["ts"] < row["ts"] for row in dev_rows for event in row["history"])
            )
            self.assertTrue(
                all("query" in event for row in dev_rows for event in row["history"])
            )
            train_audit = audit_standardized_file(output / "records_train.jsonl", "train")
            dev_audit = audit_standardized_file(output / "records_dev.jsonl", "dev")
            self.assertEqual(
                train_audit["request_count"] + dev_audit["request_count"], 10
            )
            self.assertEqual(
                len(list(iter_jsonl(output / "qrels_dev.jsonl"))),
                dev_audit["request_count"],
            )


if __name__ == "__main__":
    unittest.main()
