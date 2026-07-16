from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.eval.label_mode_evaluator import evaluate_label_mode_score_run


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class LabelModeEvaluatorTest(unittest.TestCase):
    def test_reports_all_and_positive_conditional_graded_ndcg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standardized = root / "standardized"
            runs = root / "runs"
            manifest = standardized / "candidate_manifest.json"
            _write_json(
                manifest,
                {
                    "entries": [
                        {"split": "dev", "request_id": request_id, "candidate_item_ids": ["a", "b"]}
                        for request_id in ("positive", "no-positive")
                    ]
                },
            )
            _write_jsonl(
                standardized / "qrels_dev.jsonl",
                [
                    {"request_id": "positive", "relevance": {"a": 2, "b": 0}},
                    {"request_id": "no-positive", "relevance": {}},
                ],
            )
            score_dir = runs / "scores"
            _write_json(
                score_dir / "metadata.json",
                {
                    "candidate_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "split": "dev",
                },
            )
            _write_jsonl(
                score_dir / "scores.jsonl",
                [
                    {"request_id": request_id, "candidate_item_id": item_id, "score": score}
                    for request_id in ("positive", "no-positive")
                    for item_id, score in (("a", 1.0), ("b", 0.0))
                ],
            )
            log = root / "dev_eval_log.jsonl"
            result = evaluate_label_mode_score_run(
                "analysis",
                "scores",
                "dev",
                manifest,
                standardized,
                label_mode="graded",
                runs_dir=runs,
                dev_eval_log_path=log,
            )
            self.assertEqual(result["num_requests"], 2)
            self.assertEqual(result["num_positive_eligible_requests"], 1)
            self.assertAlmostEqual(result["ndcg@10_all_requests"], 0.5)
            self.assertAlmostEqual(result["ndcg@10_positive_requests"], 1.0)
            self.assertEqual(len(log.read_text(encoding="utf-8").splitlines()), 1)

    def test_confirmation_evaluation_appends_to_supplied_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standardized = root / "standardized"
            runs = root / "runs"
            manifest = standardized / "candidate_manifest.json"
            _write_json(
                manifest,
                {
                    "entries": [
                        {
                            "split": "confirmation",
                            "request_id": "r",
                            "candidate_item_ids": ["a", "b"],
                        }
                    ]
                },
            )
            _write_jsonl(
                standardized / "qrels_confirmation.jsonl",
                [{"request_id": "r", "relevance": {"a": 1, "b": 0}}],
            )
            score_dir = runs / "scores"
            _write_json(
                score_dir / "metadata.json",
                {
                    "candidate_manifest_sha256": hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest(),
                    "split": "confirmation",
                },
            )
            _write_jsonl(
                score_dir / "scores.jsonl",
                [
                    {"request_id": "r", "candidate_item_id": "a", "score": 1.0},
                    {"request_id": "r", "candidate_item_id": "b", "score": 0.0},
                ],
            )
            log = root / "confirmation_eval_log.jsonl"
            evaluate_label_mode_score_run(
                "analysis",
                "scores",
                "confirmation",
                manifest,
                standardized,
                label_mode="graded",
                runs_dir=runs,
                dev_eval_log_path=log,
            )
            row = json.loads(log.read_text(encoding="utf-8"))
            self.assertEqual(row["split"], "confirmation")

    def test_internal_tuning_evaluation_does_not_append_dev_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standardized = root / "standardized"
            runs = root / "runs"
            manifest = standardized / "candidate_manifest.json"
            _write_json(
                manifest,
                {
                    "entries": [
                        {
                            "split": "internal",
                            "request_id": "r",
                            "candidate_item_ids": ["a", "b"],
                        }
                    ]
                },
            )
            _write_jsonl(
                standardized / "qrels_internal.jsonl",
                [{"request_id": "r", "relevance": {"a": 1, "b": 0}}],
            )
            score_dir = runs / "scores"
            _write_json(
                score_dir / "metadata.json",
                {
                    "candidate_manifest_sha256": hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest(),
                    "split": "internal",
                },
            )
            _write_jsonl(
                score_dir / "scores.jsonl",
                [
                    {"request_id": "r", "candidate_item_id": "a", "score": 1.0},
                    {"request_id": "r", "candidate_item_id": "b", "score": 0.0},
                ],
            )
            log = root / "dev_eval_log.jsonl"
            result = evaluate_label_mode_score_run(
                "analysis",
                "scores",
                "internal",
                manifest,
                standardized,
                label_mode="graded",
                runs_dir=runs,
                dev_eval_log_path=log,
            )
            self.assertEqual(result["split"], "internal")
            self.assertAlmostEqual(result["ndcg@10_positive_requests"], 1.0)
            self.assertFalse(log.exists())


if __name__ == "__main__":
    unittest.main()
