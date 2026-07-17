from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.data.history_assignments import materialize_history_assignments
from myrec.utils.jsonl import iter_jsonl


class HistoryAssignmentTest(unittest.TestCase):
    def test_wrong_history_is_causal_cross_user_and_target_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            rows = [
                _record("a", "u1", 20, "q", ["target-a"], [("h1", 10)]),
                _record("b", "u2", 18, "q", ["target-b"], [("h2", 8)]),
                _record("c", "u3", 12, "other", ["target-c"], [("h3", 11)]),
                _record("n", "u4", 20, "q", ["target-n"], []),
            ]
            with records.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            report = materialize_history_assignments(
                records,
                root / "assignments",
                root / "report.json",
            )
            wrong = {
                row["request_id"]: row
                for row in iter_jsonl(root / "assignments" / "wrong.jsonl")
            }
            self.assertEqual(wrong["a"]["donor_user_id"], "u2")
            self.assertTrue(all(event["ts"] < 20 for event in wrong["a"]["history"]))
            self.assertNotIn("target-a", {event["item_id"] for event in wrong["a"]["history"]})
            self.assertEqual(wrong["n"]["history"], [])
            self.assertEqual(report["target_candidate_leakage_violations"], 0)
            self.assertEqual(report["history_not_strictly_before_target_violations"], 0)

    def test_uses_earlier_split_exact_query_donor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            targets = root / "dev.jsonl"
            donors = root / "train.jsonl"
            targets.write_text(
                json.dumps(_record("target", "u1", 30, "same q", ["x"], [("h1", 20)]))
                + "\n",
                encoding="utf-8",
            )
            donors.write_text(
                json.dumps(_record("donor", "u2", 15, "same q", ["y"], [("h2", 10)]))
                + "\n",
                encoding="utf-8",
            )
            report = materialize_history_assignments(
                targets,
                root / "assignments",
                root / "report.json",
                donor_records_path=donors,
            )
            wrong = list(iter_jsonl(root / "assignments" / "wrong.jsonl"))[0]
            self.assertEqual(wrong["match_type"], "exact_query_other_user")
            self.assertEqual(wrong["donor_request_id"], "donor")
            self.assertEqual(report["donor_pool_requests"], 2)

    def test_global_shortlist_is_bounded_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            rows = [
                _record("target", "u0", 100, "target q", ["x"], [("ht", 90)]),
                *[
                    _record(f"d{i}", f"u{i}", 80, f"q{i}", [f"c{i}"], [(f"h{i}", 70)])
                    for i in range(1, 21)
                ],
            ]
            records.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            donors = []
            for suffix in ("a", "b"):
                report = materialize_history_assignments(
                    records,
                    root / f"assignments_{suffix}",
                    root / f"report_{suffix}.json",
                    global_donor_shortlist_size=5,
                )
                donors.append(
                    list(iter_jsonl(root / f"assignments_{suffix}" / "wrong.jsonl"))[0][
                        "donor_request_id"
                    ]
                )
                self.assertEqual(report["matching"]["global_donor_shortlist_size"], 5)
            self.assertEqual(donors[0], donors[1])

    def test_registered_v12_confirmation_requires_release_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records_confirmation.jsonl"
            records.write_text(
                json.dumps(
                    _record("target", "u1", 30, "q", ["x"], [("h", 20)])
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "dataset_version": (
                            "full_confirm_preceding40k_newholdout4k_v12"
                        )
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "require.*release lock"):
                materialize_history_assignments(
                    records,
                    root / "assignments",
                    root / "report.json",
                )
            self.assertFalse((root / "assignments").exists())


def _record(request_id, user_id, ts, query, candidates, history):
    return {
        "request_id": request_id,
        "user_id": user_id,
        "ts": ts,
        "query": query,
        "candidates": [{"item_id": item_id} for item_id in candidates],
        "history": [
            {"item_id": item_id, "event": "click", "ts": event_ts}
            for item_id, event_ts in history
        ],
    }


if __name__ == "__main__":
    unittest.main()
