from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from myrec.baselines.kuaisearch_official_adapter import materialize_b5o_stageb_format
from myrec.utils.jsonl import write_jsonl


class KuaiSearchOfficialAdapterTest(unittest.TestCase):
    def test_materializer_writes_train_and_label_free_score_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            std = root / "std"
            std.mkdir()
            train_record = {
                "request_id": "r1",
                "session_id": "s1",
                "user_id": "7",
                "query": "手机壳",
                "ts": 10,
                "history": [
                    {"event": "click", "item_id": "11", "cat": ["数码", "配件", "壳"], "title": "旧壳"},
                    {"event": "purchase", "item_id": "12", "cat": ["数码", "配件", "膜"], "title": "旧膜"},
                ],
                "candidates": [
                    {"clicked": 1, "purchased": 0, "item_id": "21", "cat": ["数码", "配件", "壳"], "title": "手机壳 A"},
                    {"clicked": 0, "purchased": 0, "item_id": "22", "cat": ["数码", "配件", "壳"], "title": "手机壳 B"},
                ],
            }
            dev_record = {
                "request_id": "r2",
                "session_id": "s2",
                "user_id": "8",
                "query": "耳机",
                "ts": 11,
                "history": [],
                "candidates": [
                    {"item_id": "31", "cat": ["数码", "影音", "耳机"], "title": "耳机 A"},
                ],
            }
            catalog_rows = [
                {"item_id": str(item_id), "cat": ["数码", "配件", "壳"], "title": f"item {item_id}"}
                for item_id in (11, 12, 21, 22)
            ] + [{"item_id": "31", "cat": ["数码", "影音", "耳机"], "title": "耳机 A"}]
            write_jsonl(std / "records_train.jsonl", [train_record])
            write_jsonl(std / "records_dev.jsonl", [dev_record])
            write_jsonl(std / "item_catalog.jsonl", catalog_rows)
            (std / "candidate_manifest.json").write_text(
                json.dumps({"dataset_id": "kuaisearch", "dataset_version": "tiny", "entries": []}),
                encoding="utf-8",
            )

            manifest = materialize_b5o_stageb_format(std, root / "out")

            rank_rows = _read_jsonl(root / "out" / "data" / "rank.jsonl")
            score_rows = _read_jsonl(root / "out" / "data" / "score_dev.jsonl")
            users = _read_jsonl(root / "out" / "data" / "users.jsonl")

            self.assertEqual(manifest["qrels_read"], False)
            self.assertEqual(len(rank_rows), 2)
            self.assertEqual(rank_rows[0]["session_id"], "r1")
            self.assertEqual(rank_rows[0]["original_session_id"], "s1")
            self.assertEqual(rank_rows[0]["recently_clicked_item_ids"], [11])
            self.assertEqual(rank_rows[0]["recently_purchased_item_ids"], [12])
            self.assertEqual(score_rows[0]["session_id"], "r2")
            self.assertEqual(score_rows[0]["is_clicked"], 0)
            self.assertEqual(score_rows[0]["is_purchased"], 0)
            self.assertEqual({row["age"] for row in users}, {"31-40"})
            self.assertEqual({row["gender"] for row in users}, {"M"})


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


if __name__ == "__main__":
    unittest.main()
