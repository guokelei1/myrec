import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.eval.compare import compare_per_request_metrics


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


class CompareTest(unittest.TestCase):
    def test_subset_compare_filters_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "a.jsonl"
            run_b = root / "b.jsonl"
            subset = root / "subset.txt"
            output = root / "compare.json"
            _write_jsonl(
                run_a,
                [
                    {"request_id": "r1", "ndcg@10": 0.5},
                    {"request_id": "r2", "ndcg@10": 0.7},
                    {"request_id": "r3", "ndcg@10": 0.1},
                ],
            )
            _write_jsonl(
                run_b,
                [
                    {"request_id": "r1", "ndcg@10": 0.4},
                    {"request_id": "r2", "ndcg@10": 0.2},
                    {"request_id": "r3", "ndcg@10": 0.9},
                ],
            )
            subset.write_text("r1\nr2\n", encoding="utf-8")

            result = compare_per_request_metrics(
                run_a,
                run_b,
                output,
                samples=100,
                seed=1,
                request_ids_path=subset,
            )

            self.assertEqual(result["num_requests"], 2)
            self.assertAlmostEqual(result["delta"], 0.3)
            self.assertEqual(result["request_ids_path"], str(subset))

    def test_subset_compare_requires_full_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "a.jsonl"
            run_b = root / "b.jsonl"
            subset = root / "subset.txt"
            output = root / "compare.json"
            _write_jsonl(run_a, [{"request_id": "r1", "ndcg@10": 0.5}])
            _write_jsonl(run_b, [{"request_id": "r1", "ndcg@10": 0.4}])
            subset.write_text("r1\nr2\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "subset coverage mismatch"):
                compare_per_request_metrics(
                    run_a,
                    run_b,
                    output,
                    samples=10,
                    seed=1,
                    request_ids_path=subset,
                )


if __name__ == "__main__":
    unittest.main()
