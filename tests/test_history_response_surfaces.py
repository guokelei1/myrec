from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.eval.history_response_surfaces import summarize_history_response_surfaces


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class HistoryResponseSurfacesTest(unittest.TestCase):
    def test_summarizes_surface_with_cluster_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            per_request = root / "per_request.jsonl"
            records = root / "records.jsonl"
            surfaces = root / "surfaces"
            surfaces.mkdir()
            rows = []
            for index, delta in enumerate((0.2, -0.1), start=1):
                rows.append(
                    {
                        "request_id": f"r{index}",
                        "response_active": True,
                        "direction_preferred_pairs": 1,
                        "active_pair_fraction": 1.0,
                        "direction_pair_coverage": 1.0,
                        "pairwise_directional_accuracy": float(delta > 0),
                        "signed_delta_alignment": 1.0 if delta > 0 else -1.0,
                        "true_minus_null_ndcg@10": delta,
                        "true_ndcg@10": 0.5 + delta,
                        "common_energy": 1.0,
                        "differential_energy": 1.0,
                        "true_minus_wrong_ndcg@10": delta / 2,
                        "true_over_wrong_signed_alignment": delta,
                    }
                )
            _write_jsonl(per_request, rows)
            _write_jsonl(
                records,
                [
                    {"request_id": "r1", "user_id": "u1", "query": "Q"},
                    {"request_id": "r2", "user_id": "u2", "query": "Q"},
                ],
            )
            (surfaces / "all.txt").write_text("r1\nr2\n", encoding="utf-8")
            (surfaces / "empty.txt").write_text("", encoding="utf-8")
            result = summarize_history_response_surfaces(
                per_request,
                records,
                surfaces,
                root / "report.json",
                utility_epsilon=0.0,
                bootstrap_samples=20,
                seed=7,
            )
            summary = result["surfaces"]["all"]
            self.assertEqual(summary["num_requests"], 2)
            self.assertEqual(summary["bootstrap_ci95"]["query_cluster"]["num_clusters"], 1)
            self.assertEqual(result["empty_surfaces"], ["empty"])
            self.assertTrue((root / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
