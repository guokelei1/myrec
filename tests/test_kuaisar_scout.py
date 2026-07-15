from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.data.kuaisar_scout import build_kuaisar_small_scout
from myrec.utils.jsonl import iter_jsonl


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class KuaiSarScoutTest(unittest.TestCase):
    def test_builds_user_input_time_split_with_causal_cross_service_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "KuaiSAR_final"
            src_rows = []
            item_ids = set()
            for index in range(14):
                user_id = str(index % 2)
                timestamp = 1_000 + index * 10
                for position in range(3):
                    item_id = str(1000 + index * 3 + position)
                    item_ids.add(item_id)
                    src_rows.append(
                        {
                            "keyword": "[11, 12]",
                            "item_id": item_id,
                            "click_cnt": int(position == 0),
                            "search_session_id": str(index + 1),
                            "item_type": "VIDEO",
                            "user_id": user_id,
                            "search_session_timestamp": timestamp,
                            "search_source": "USER_INPUT",
                            "search_session_time": "2023-05-22 00:00:00",
                        }
                    )
            _write_csv(
                raw / "src_inter.csv",
                [
                    "keyword",
                    "item_id",
                    "click_cnt",
                    "search_session_id",
                    "item_type",
                    "user_id",
                    "search_session_timestamp",
                    "search_source",
                    "search_session_time",
                ],
                src_rows,
            )
            rec_item = "9999"
            item_ids.add(rec_item)
            _write_csv(
                raw / "rec_inter.csv",
                [
                    "user_id",
                    "item_id",
                    "duration_ms",
                    "playing_time",
                    "timestamp",
                    "forward",
                    "like",
                    "follow",
                    "search_item_related",
                    "search",
                    "click",
                    "time",
                ],
                [
                    {
                        "user_id": "0",
                        "item_id": rec_item,
                        "duration_ms": 1,
                        "playing_time": 1,
                        "timestamp": 995,
                        "forward": 0,
                        "like": 0,
                        "follow": 0,
                        "search_item_related": 0,
                        "search": 0,
                        "click": 1,
                        "time": "2023-05-22 00:00:00",
                    }
                ],
            )
            _write_csv(
                raw / "item_features.csv",
                [
                    "item_id",
                    "caption",
                    "first_level_category_name_en",
                    "second_level_category_name_en",
                    "third_level_category_name_en",
                    "fourth_level_category_name_en",
                ],
                [
                    {
                        "item_id": item_id,
                        "caption": "[11, 20]",
                        "first_level_category_name_en": "Funny",
                        "second_level_category_name_en": "Joke",
                        "third_level_category_name_en": "empty",
                        "fourth_level_category_name_en": "empty",
                    }
                    for item_id in sorted(item_ids)
                ],
            )
            output = root / "standardized"
            result = build_kuaisar_small_scout(
                root,
                output,
                root / "report.json",
                max_requests=10,
                dev_fraction=0.3,
                max_history_len=4,
            )
            self.assertTrue(result["admission_checks"]["strict_causal_history"])
            self.assertTrue(result["admission_checks"]["candidate_text_coverage_at_least_95pct"])
            self.assertFalse(result["admission_passed"])
            self.assertEqual(result["selection"]["selected_requests"], 10)
            dev = list(iter_jsonl(output / "records_dev.jsonl"))
            self.assertTrue(dev)
            self.assertTrue(all(row["query"] == "w11 w12" for row in dev))
            self.assertTrue(all(row["history"] for row in dev))
            self.assertTrue(
                all(event["ts"] < row["ts"] for row in dev for event in row["history"])
            )
            self.assertTrue(
                all(
                    "clicked" not in candidate
                    for row in dev
                    for candidate in row["candidates"]
                )
            )
            self.assertEqual(len(list(iter_jsonl(output / "qrels_dev.jsonl"))), len(dev))


if __name__ == "__main__":
    unittest.main()
