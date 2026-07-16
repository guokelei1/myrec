import json
import tempfile
import unittest
from pathlib import Path

from myrec.baselines.instructrec import (
    build_instructrec_examples,
    candidate_text,
    serialize_instructrec_prompt,
)


class InstructRecTest(unittest.TestCase):
    def test_candidate_text_and_prompt_keep_query_history_and_full_slate(self):
        record = {
            "query": "蓝色水杯",
            "history": [
                {
                    "event": "purchase",
                    "title": "白色水杯",
                    "brand": "A",
                    "cat": ["家居", "杯子"],
                }
            ],
            "candidates": [
                {"item_id": "a", "title": "蓝色水杯", "brand": "A", "cat": ["家居"]},
                {"item_id": "b", "title": "红色水壶", "brand": "B", "cat": ["厨房"]},
            ],
        }
        prompt = serialize_instructrec_prompt(record, record["history"], history_budget=6)
        self.assertIn("蓝色水杯", prompt)
        self.assertIn("白色水杯", prompt)
        self.assertIn("红色水壶", prompt)
        self.assertEqual(candidate_text(record["candidates"][0]), "蓝色水杯 | A | 家居")

    def test_build_examples_reads_only_train_qrels_and_supports_qc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "records.jsonl"
            qrels = root / "qrels.jsonl"
            record = {
                "request_id": "r1",
                "query": "杯子",
                "history": [],
                "candidates": [
                    {"item_id": "a", "title": "杯子", "brand": "", "cat": []},
                    {"item_id": "b", "title": "水壶", "brand": "", "cat": []},
                ],
            }
            records.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            qrels.write_text(json.dumps({"request_id": "r1", "clicked": ["a"]}) + "\n", encoding="utf-8")
            examples, stats = build_instructrec_examples(
                records,
                qrels,
                input_mode="qc",
                history_budget=6,
                seed=1,
            )
            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0].target, "杯子")
            self.assertIn("(empty)", examples[0].prompt)
            self.assertEqual(stats["labeled_requests"], 1)


if __name__ == "__main__":
    unittest.main()
