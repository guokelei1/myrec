from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.data.request_surfaces import materialize_request_surfaces


class RequestSurfaceTest(unittest.TestCase):
    def test_materializes_disjoint_history_surfaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records.jsonl"
            rows = [
                {
                    "request_id": "n",
                    "query": "shared query",
                    "history": [],
                    "candidates": [{"item_id": "a"}, {"item_id": "b"}],
                    "masks": {"history_present": False, "strict_nonrepeat": False},
                },
                {
                    "request_id": "s",
                    "query": "shared query",
                    "history": [{"item_id": "h"}],
                    "candidates": [{"item_id": "a"}, {"item_id": "b"}],
                    "masks": {"history_present": True, "strict_nonrepeat": True},
                },
                {
                    "request_id": "r",
                    "query": "unique query",
                    "history": [{"item_id": "a"}],
                    "candidates": [{"item_id": "a"}, {"item_id": "b"}],
                    "masks": {"history_present": True, "strict_nonrepeat": False},
                },
            ]
            with records.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            result = materialize_request_surfaces(records, root / "surfaces")
            self.assertEqual(result["files"]["all"]["requests"], 3)
            self.assertEqual(result["files"]["history_present"]["requests"], 2)
            self.assertEqual(result["files"]["strict_nonrepeat"]["requests"], 1)
            self.assertEqual(result["files"]["repeat"]["requests"], 1)
            self.assertEqual(result["files"]["no_history"]["requests"], 1)
            self.assertEqual(result["files"]["repeated_query"]["requests"], 2)
            self.assertEqual(
                result["files"]["repeated_query_history_present"]["requests"], 1
            )
            self.assertEqual(
                result["files"]["repeated_query_strict_nonrepeat"]["requests"], 1
            )
            self.assertEqual(result["files"]["repeated_query_repeat"]["requests"], 0)
            self.assertEqual(result["files"]["singleton_query"]["requests"], 1)


if __name__ == "__main__":
    unittest.main()
