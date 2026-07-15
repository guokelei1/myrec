from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.data.assignment_surfaces import materialize_assignment_surfaces


class AssignmentSurfacesTest(unittest.TestCase):
    def test_groups_request_ids_by_match_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "wrong.jsonl"
            rows = [
                {"request_id": "a", "match_type": "exact_query_other_user"},
                {"request_id": "b", "match_type": "global_other_user"},
                {"request_id": "c", "match_type": "global_other_user"},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = materialize_assignment_surfaces(path, root / "surfaces")
            self.assertEqual(result["files"]["all"]["requests"], 3)
            self.assertEqual(
                result["files"]["exact_query_other_user"]["requests"], 1
            )
            self.assertEqual(result["files"]["global_other_user"]["requests"], 2)

    def test_materializes_label_free_intersections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "wrong.jsonl"
            path.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in [
                        {"request_id": "a", "match_type": "exact_query_other_user"},
                        {"request_id": "b", "match_type": "exact_query_other_user"},
                        {"request_id": "c", "match_type": "global_other_user"},
                    ]
                ),
                encoding="utf-8",
            )
            strict = root / "strict_nonrepeat.txt"
            strict.write_text("b\nc\n", encoding="utf-8")
            result = materialize_assignment_surfaces(
                path,
                root / "surfaces",
                intersection_surfaces={"strict_nonrepeat": strict},
            )
            self.assertEqual(
                result["files"]["exact_query_other_user__strict_nonrepeat"][
                    "requests"
                ],
                1,
            )
            self.assertEqual(
                result["files"]["global_other_user__strict_nonrepeat"]["requests"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
