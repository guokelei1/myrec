from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.eval.response_direction_intervention_surfaces import (
    summarize_response_direction_intervention_surfaces,
)


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class ResponseDirectionInterventionSurfacesTest(unittest.TestCase):
    def test_empty_label_free_surface_is_reported_and_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            per_request = root / "per_request.jsonl"
            records = root / "records.jsonl"
            surfaces = root / "surfaces"
            surfaces.mkdir()
            _write_jsonl(
                per_request,
                [
                    {
                        "request_id": "r1",
                        "direction_eligible": True,
                        "actual_ndcg@10": 0.7,
                        "null_ndcg@10": 0.5,
                        "random_mean_ndcg@10": 0.55,
                        "aligned_ndcg@10": 0.9,
                        "anti_aligned_ndcg@10": 0.3,
                        "actual_minus_null_ndcg@10": 0.2,
                        "actual_minus_random_ndcg@10": 0.15,
                        "aligned_minus_actual_ndcg@10": 0.2,
                        "aligned_minus_null_ndcg@10": 0.4,
                    }
                ],
            )
            _write_jsonl(
                records,
                [{"request_id": "r1", "user_id": "u1", "query": "q"}],
            )
            (surfaces / "all.txt").write_text("r1\n", encoding="utf-8")
            (surfaces / "empty.txt").write_text("", encoding="utf-8")
            result = summarize_response_direction_intervention_surfaces(
                per_request,
                records,
                surfaces,
                root / "report.json",
                bootstrap_samples=10,
                seed=7,
            )
            self.assertEqual(result["empty_surfaces"], ["empty"])
            self.assertEqual(result["surfaces"]["all"]["num_requests"], 1)


if __name__ == "__main__":
    unittest.main()
