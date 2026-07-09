import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.kuaisearch_materializer import materialize_official_ranking_format


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class KuaiSearchMaterializerTest(unittest.TestCase):
    def test_materializes_official_files_with_age_mapping_and_last_time_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            out = root / "out"
            _write_jsonl(
                raw / "rank_lite" / "train.jsonl",
                [
                    {
                        "user_id": 1,
                        "session_id": 10,
                        "time_index": 10,
                        "recently_clicked_item_ids": [101],
                        "query": "q1",
                        "target_item_id": 100,
                        "is_clicked": 1,
                        "is_purchased": 0,
                        "split": "train",
                    },
                    {
                        "user_id": 2,
                        "session_id": 11,
                        "time_index": 20,
                        "recently_clicked_item_ids": [102],
                        "query": "q2",
                        "target_item_id": 101,
                        "is_clicked": 0,
                        "is_purchased": 0,
                        "split": "train",
                    },
                ],
            )
            _write_jsonl(
                raw / "items_lite" / "train.jsonl",
                [
                    {
                        "item_id": 100,
                        "item_title": "item 100",
                        "category_level1_id": 1,
                        "category_level2_id": 2,
                        "category_level3_id": 3,
                    },
                    {
                        "item_id": 101,
                        "item_title": "item 101",
                        "category_level1_id": 1,
                        "category_level2_id": 2,
                        "category_level3_id": 4,
                    },
                    {
                        "item_id": 102,
                        "item_title": "item 102",
                        "category_level1_id": 1,
                        "category_level2_id": 2,
                        "category_level3_id": 5,
                    },
                ],
            )
            _write_jsonl(
                raw / "users_lite" / "train.jsonl",
                [{"user_id": 1, "gender": "F", "age_bucket": "24-30"}],
            )

            manifest = materialize_official_ranking_format(
                raw,
                out,
                test_fraction=0.5,
                min_target_coverage=1.0,
            )

            self.assertEqual(manifest["status"], "passed")
            self.assertEqual(manifest["target_item_coverage"]["coverage_rate"], 1.0)
            rank_rows = _read_jsonl(out / "data" / "rank.jsonl")
            self.assertEqual([row["split"] for row in rank_rows], ["train", "test"])
            users = {row["user_id"]: row for row in _read_jsonl(out / "data" / "users.jsonl")}
            self.assertEqual(users[1]["age"], "24-30")
            self.assertEqual(users[2]["age"], "31-40")
            self.assertEqual(users[2]["gender"], "M")
            self.assertTrue(users[2]["synthetic_missing_user"])
            self.assertEqual(manifest["users"]["synthetic_missing_users"], 1)

    def test_reports_failed_status_when_target_coverage_is_low(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            out = root / "out"
            _write_jsonl(
                raw / "rank_lite" / "train.jsonl",
                [
                    {
                        "user_id": 1,
                        "session_id": 10,
                        "time_index": 10,
                        "recently_clicked_item_ids": [],
                        "query": "q",
                        "target_item_id": 999,
                        "is_clicked": 0,
                        "is_purchased": 0,
                    }
                ],
            )
            _write_jsonl(raw / "items_lite" / "train.jsonl", [])
            _write_jsonl(raw / "users_lite" / "train.jsonl", [])

            manifest = materialize_official_ranking_format(
                raw,
                out,
                min_target_coverage=1.0,
            )

            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["target_item_coverage"]["coverage_rate"], 0.0)
            self.assertEqual(manifest["rank"]["missing_target_rows"], 1)


if __name__ == "__main__":
    unittest.main()
