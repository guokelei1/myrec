import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.data.batch2b_interactions import export_train_interactions_from_path
from myrec.baselines.recbole_adapter import write_recbole_atomic_interactions


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Batch2bInteractionsTest(unittest.TestCase):
    def test_export_deduplicates_sorts_and_prefers_purchase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "records_train.jsonl"
            output = root / "interactions_train.jsonl"
            report = root / "manifest.json"
            _write_jsonl(
                train,
                [
                    {
                        "request_id": "r2",
                        "ts": 20,
                        "user_id": "u1",
                        "history": [
                            {"event": "click", "item_id": "i2", "ts": 10},
                            {"event": "purchase", "item_id": "i2", "ts": 10},
                        ],
                        "candidates": [
                            {"clicked": 1, "item_id": "i3", "purchased": 0},
                            {"clicked": 1, "item_id": "i4", "purchased": 1},
                        ],
                    },
                    {
                        "request_id": "r1",
                        "ts": 5,
                        "user_id": "u1",
                        "history": [],
                        "candidates": [{"clicked": 1, "item_id": "i1", "purchased": 0}],
                    },
                ],
            )

            manifest = export_train_interactions_from_path(train, output, report)
            rows = _read_jsonl(output)

            self.assertEqual(
                [(row["user_id"], row["event_time"], row["item_id"]) for row in rows],
                [("u1", 5, "i1"), ("u1", 10, "i2"), ("u1", 20, "i3"), ("u1", 20, "i4")],
            )
            self.assertEqual(rows[1]["event_type"], "purchase")
            self.assertEqual(rows[1]["sources"], ["history"])
            self.assertEqual(rows[3]["event_type"], "purchase")
            self.assertEqual(manifest["unique_interactions"], 4)
            self.assertEqual(manifest["duplicate_keys"], 1)
            self.assertFalse(manifest["qrels_read"])
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["artifact_sha256"], manifest["artifact_sha256"])

    def test_rejects_non_train_records_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev = root / "records_dev.jsonl"
            dev.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "train-only"):
                export_train_interactions_from_path(dev, root / "out.jsonl")

    def test_recbole_atomic_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            interactions = root / "interactions_train.jsonl"
            output_root = root / "atomic"
            _write_jsonl(
                interactions,
                [
                    {
                        "event_time": 2,
                        "event_type": "purchase",
                        "item_id": "i2",
                        "request_id": "r2",
                        "sources": ["request_positive"],
                        "user_id": "u1",
                    },
                    {
                        "event_time": 1,
                        "event_type": "click",
                        "item_id": "i1",
                        "request_id": "r1",
                        "sources": ["history"],
                        "user_id": "u1",
                    },
                ],
            )

            manifest = write_recbole_atomic_interactions(interactions, output_root, "toy")
            inter_path = output_root / "toy" / "toy.inter"
            lines = inter_path.read_text(encoding="utf-8").splitlines()

            self.assertEqual(lines[0], "user_id:token\titem_id:token\ttimestamp:float")
            self.assertEqual(lines[1:], ["u1\ti2\t2", "u1\ti1\t1"])
            self.assertEqual(manifest["rows"], 2)
            self.assertEqual(manifest["unique_items"], 2)
            self.assertEqual(manifest["event_counts"], {"click": 1, "purchase": 1})


if __name__ == "__main__":
    unittest.main()
