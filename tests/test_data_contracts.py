from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.data.contracts import audit_standardized_file, validate_standardized_record


def _record() -> dict:
    return {
        "request_id": "r1",
        "user_id": "u1",
        "session_id": "s1",
        "ts": 20,
        "query": "red shoes",
        "history": [{"item_id": "h1", "title": "old", "event": "click", "ts": 10}],
        "candidates": [
            {"item_id": "a", "title": "A", "brand": "x", "cat": []},
            {"item_id": "b", "title": "B", "brand": "y", "cat": []},
        ],
        "masks": {"history_present": True, "text_coverage": 1.0},
    }


class DataContractTest(unittest.TestCase):
    def test_valid_label_free_dev_record(self):
        validate_standardized_record(_record(), "dev")

    def test_rejects_dev_label_leakage(self):
        row = _record()
        row["candidates"][0]["clicked"] = 1
        with self.assertRaisesRegex(ValueError, "labels leaked"):
            validate_standardized_record(row, "dev")

    def test_rejects_noncausal_history(self):
        row = _record()
        row["history"][0]["ts"] = row["ts"]
        with self.assertRaisesRegex(ValueError, "strictly before"):
            validate_standardized_record(row, "dev")

    def test_audit_counts_strict_nonrepeat_and_repeated_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records_dev.jsonl"
            left = _record()
            right = _record()
            right["request_id"] = "r2"
            right["user_id"] = "u2"
            right["session_id"] = "s2"
            with path.open("w", encoding="utf-8") as handle:
                for row in (left, right):
                    handle.write(json.dumps(row) + "\n")
            result = audit_standardized_file(path, "dev")
            self.assertEqual(result["request_count"], 2)
            self.assertEqual(result["strict_nonrepeat_requests"], 2)
            self.assertEqual(result["repeated_query_requests"], 2)


if __name__ == "__main__":
    unittest.main()
