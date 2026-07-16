from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.eval.target_aware_surfaces import (
    ALL_REQUEST_PARTITION,
    OBSERVED_POSITIVE_PARTITION,
    build_target_aware_surface_memberships,
)


class TargetAwareSurfaceTest(unittest.TestCase):
    def test_hand_computed_target_and_candidate_overlap_partitions(self) -> None:
        rows = [
            {
                "request_id": "target-repeat",
                "history": [{"item_id": "p"}],
                "candidates": [{"item_id": "p"}, {"item_id": "n"}],
            },
            {
                "request_id": "other-overlap",
                "history": [{"item_id": "n"}],
                "candidates": [{"item_id": "p"}, {"item_id": "n"}],
            },
            {
                "request_id": "no-overlap",
                "history": [{"item_id": "h"}],
                "candidates": [{"item_id": "p"}, {"item_id": "n"}],
            },
            {
                "request_id": "no-history",
                "history": [],
                "candidates": [{"item_id": "p"}, {"item_id": "n"}],
            },
            {
                "request_id": "no-positive",
                "history": [{"item_id": "n"}],
                "candidates": [{"item_id": "p"}, {"item_id": "n"}],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            records = Path(tmp) / "records_dev.jsonl"
            records.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            candidates = {
                row["request_id"]: [candidate["item_id"] for candidate in row["candidates"]]
                for row in rows
            }
            gains = {
                row["request_id"]: (
                    {} if row["request_id"] == "no-positive" else {"p": 1.0}
                )
                for row in rows
            }
            members = build_target_aware_surface_memberships(
                records, candidates, gains
            )

        self.assertEqual(members["target_repeat"], {"target-repeat"})
        self.assertEqual(
            members["target_nonrepeat_other_candidate_overlap"], {"other-overlap"}
        )
        self.assertEqual(
            members["target_nonrepeat_no_candidate_overlap"], {"no-overlap"}
        )
        self.assertEqual(members["target_nonrepeat_no_history"], {"no-history"})
        self.assertEqual(members["no_observed_positive"], {"no-positive"})
        self.assertEqual(
            members["candidate_overlap"],
            {"target-repeat", "other-overlap", "no-positive"},
        )
        self.assertEqual(members["no_candidate_overlap_history_present"], {"no-overlap"})
        self.assertEqual(members["no_history"], {"no-history"})
        self.assertEqual(
            set().union(*(members[name] for name in OBSERVED_POSITIVE_PARTITION)),
            members["observed_positive"],
        )
        self.assertEqual(
            set().union(*(members[name] for name in ALL_REQUEST_PARTITION)),
            members["all"],
        )

    def test_rejects_record_candidate_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = Path(tmp) / "records_dev.jsonl"
            records.write_text(
                json.dumps(
                    {
                        "request_id": "r",
                        "history": [],
                        "candidates": [{"item_id": "a"}, {"item_id": "x"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                build_target_aware_surface_memberships(
                    records, {"r": ["a", "b"]}, {"r": {"a": 1.0}}
                )


if __name__ == "__main__":
    unittest.main()
