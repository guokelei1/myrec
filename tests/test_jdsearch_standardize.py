from __future__ import annotations

import sys
import tempfile
import unittest
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.data.jdsearch_standardize import (
    EXPECTED_BEHAVIOR_HEADER,
    EXPECTED_PRODUCT_HEADER,
    TERM_SEPARATOR,
    build_jdsearch_scout,
)
from myrec.utils.jsonl import iter_jsonl


class JDSearchStandardizeTest(unittest.TestCase):
    def test_builds_aligned_label_isolated_scout_with_history_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            repo = root / "repo"
            raw.mkdir()
            repo.mkdir()
            behavior_lines = [EXPECTED_BEHAVIOR_HEADER]
            item_ids = set()
            for index in range(14):
                candidates = [str(1000 + index * 3 + offset) for offset in range(3)]
                histories = [str(5000 + index), str(6000 + index)]
                item_ids.update(candidates + histories)
                behavior_lines.append(
                    "\t".join(
                        [
                            f"11{TERM_SEPARATOR}12",
                            "_".join(candidates),
                            "3.0_0.0_0.0",
                            f"21{TERM_SEPARATOR}22_-1",
                            "_".join(histories),
                            "ORD_CLICK",
                            "0_2_3",
                        ]
                    )
                )
            behavior_text = "\n".join(behavior_lines) + "\n"
            (raw / "user_behavior_data.txt").write_text(behavior_text, encoding="utf-8")
            (repo / "user_behavior_data_sample.txt").write_text(behavior_text, encoding="utf-8")
            product_lines = [EXPECTED_PRODUCT_HEADER]
            for item_id in sorted(item_ids):
                product_lines.append(
                    "\t".join(
                        [
                            item_id,
                            f"11{TERM_SEPARATOR}30",
                            "1",
                            "40",
                            "1",
                            "50",
                            "2",
                            "60",
                            "3",
                            "70",
                            "4",
                            "80",
                            "9",
                        ]
                    )
                )
            product_text = "\n".join(product_lines) + "\n"
            (raw / "product_meta_data.txt").write_text(product_text, encoding="utf-8")
            (repo / "product_meta_data_sample.txt").write_text(
                "\n".join(product_lines[:2]) + "\n", encoding="utf-8"
            )
            output = root / "standardized"
            result = build_jdsearch_scout(
                raw,
                output,
                root / "report.json",
                official_repository_dir=repo,
                max_requests=10,
                dev_fraction=0.3,
                max_history_len=2,
            )
            self.assertTrue(result["admission_checks"]["aligned_source_lists"])
            self.assertFalse(result["admission_passed"])
            dev = list(iter_jsonl(output / "records_dev.jsonl"))
            self.assertTrue(dev)
            self.assertTrue(all(row["history"] for row in dev))
            self.assertTrue(all(row["history"][0]["query"] == "w21 w22" for row in dev))
            self.assertTrue(all(row["history"][0]["event"] == "purchase" for row in dev))
            self.assertTrue(all(row["history"][1]["event"] == "click" for row in dev))
            self.assertTrue(
                all(
                    "clicked" not in candidate
                    for row in dev
                    for candidate in row["candidates"]
                )
            )
            self.assertTrue(
                all(event["ts"] < row["ts"] for row in dev for event in row["history"])
            )
            seed = 20260714
            for row in dev:
                item_ids = [candidate["item_id"] for candidate in row["candidates"]]
                expected = sorted(
                    item_ids,
                    key=lambda item_id: (
                        int.from_bytes(
                            hashlib.sha256(
                                f"{seed}|candidate_order|{row['request_id']}|{item_id}".encode()
                            ).digest(),
                            "big",
                        ),
                        item_id,
                    ),
                )
                self.assertEqual(item_ids, expected)
                self.assertEqual(
                    [candidate["source_position"] for candidate in row["candidates"]],
                    list(range(len(item_ids))),
                )
            self.assertEqual(result["outputs"]["candidate_order_violations"], 0)


if __name__ == "__main__":
    unittest.main()
