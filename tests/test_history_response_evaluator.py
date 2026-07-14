from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.eval.history_response_evaluator import evaluate_history_response_runs


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HistoryResponseEvaluatorTest(unittest.TestCase):
    def test_bundle_identity_and_dev_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            standardized = root / "standardized"
            manifest = standardized / "candidate_manifest.json"
            request_manifest = standardized / "request_manifest.json"
            _write_json(
                manifest,
                {
                    "entries": [
                        {"split": "dev", "request_id": "r1", "candidate_item_ids": ["a", "b"]}
                    ]
                },
            )
            _write_jsonl(
                standardized / "qrels_dev.jsonl",
                [{"request_id": "r1", "clicked": ["a"], "purchased": []}],
            )
            _write_json(
                request_manifest,
                {"entries": [{"split": "dev", "request_id": "r1", "query_sha256": "q"}]},
            )
            candidate_hash = _sha(manifest)
            common = {
                "candidate_manifest_sha256": candidate_hash,
                "checkpoint_id": "checkpoint-1",
                "dataset_id": "tiny",
                "dataset_version": "v1",
                "request_manifest_sha256": _sha(request_manifest),
                "scoring_signature": {"max_length": 32, "head": "rank"},
                "split": "dev",
            }
            for condition, scores in {
                "true": {"a": 2.0, "b": 0.0},
                "null": {"a": 0.0, "b": 1.0},
                "wrong": {"a": -1.0, "b": 2.0},
            }.items():
                run_dir = runs / condition
                _write_json(
                    run_dir / "metadata.json",
                    {
                        **common,
                        "history_assignment_sha256": f"history-{condition}",
                        "history_condition": condition,
                    },
                )
                _write_jsonl(
                    run_dir / "scores.jsonl",
                    [
                        {"request_id": "r1", "candidate_item_id": item_id, "score": score}
                        for item_id, score in scores.items()
                    ],
                )

            log = root / "reports" / "dev_eval_log.jsonl"
            result = evaluate_history_response_runs(
                analysis_run_id="analysis",
                true_run_id="true",
                null_run_id="null",
                wrong_run_id="wrong",
                split="dev",
                label_mode="click",
                candidate_manifest_path=manifest,
                standardized_dir=standardized,
                activity_epsilon=0.1,
                utility_epsilon=0.0,
                runs_dir=runs,
                dev_eval_log_path=log,
            )
            self.assertEqual(result["num_requests"], 1)
            self.assertGreater(result["mean_true_minus_null_ndcg@10"], 0.0)
            self.assertGreater(result["mean_true_minus_wrong_ndcg@10"], 0.0)
            self.assertTrue((runs / "analysis" / "per_request_history_response.jsonl").exists())
            self.assertEqual(len(log.read_text(encoding="utf-8").splitlines()), 1)

    def test_rejects_different_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            standardized = root / "standardized"
            manifest = standardized / "candidate_manifest.json"
            request_manifest = standardized / "request_manifest.json"
            _write_json(
                manifest,
                {"entries": [{"split": "dev", "request_id": "r", "candidate_item_ids": ["a", "b"]}]},
            )
            _write_jsonl(
                standardized / "qrels_dev.jsonl",
                [{"request_id": "r", "clicked": ["a"], "purchased": []}],
            )
            _write_json(
                request_manifest,
                {"entries": [{"split": "dev", "request_id": "r", "query_sha256": "q"}]},
            )
            candidate_hash = _sha(manifest)
            for condition, checkpoint in (("true", "c1"), ("null", "c2")):
                run_dir = runs / condition
                _write_json(
                    run_dir / "metadata.json",
                    {
                        "candidate_manifest_sha256": candidate_hash,
                        "checkpoint_id": checkpoint,
                        "dataset_id": "tiny",
                        "dataset_version": "v1",
                        "history_assignment_sha256": f"history-{condition}",
                        "history_condition": condition,
                        "request_manifest_sha256": _sha(request_manifest),
                        "scoring_signature": "same",
                        "split": "dev",
                    },
                )
                _write_jsonl(
                    run_dir / "scores.jsonl",
                    [
                        {"request_id": "r", "candidate_item_id": "a", "score": 1.0},
                        {"request_id": "r", "candidate_item_id": "b", "score": 0.0},
                    ],
                )
            with self.assertRaisesRegex(ValueError, "checkpoint_id"):
                evaluate_history_response_runs(
                    analysis_run_id="analysis",
                    true_run_id="true",
                    null_run_id="null",
                    split="dev",
                    candidate_manifest_path=manifest,
                    standardized_dir=standardized,
                    activity_epsilon=0.1,
                    utility_epsilon=0.0,
                    runs_dir=runs,
                )


if __name__ == "__main__":
    unittest.main()
