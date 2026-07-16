from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from myrec.baselines.fixed_residual_anchor import write_fixed_residual_anchor_scores
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


class FixedResidualAnchorTest(unittest.TestCase):
    def _source(self, root: Path, name: str, values: list[float], manifest: Path) -> Path:
        run = root / name
        run.mkdir()
        rows = [
            {"request_id": "r1", "candidate_item_id": item, "score": value}
            for item, value in zip(("a", "b"), values, strict=True)
        ]
        (run / "scores.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (run / "metadata.json").write_text(
            json.dumps({
                "candidate_manifest_sha256": sha256_file(manifest),
                "checkpoint_id": "checkpoint",
                "dataset_id": "dataset",
                "dataset_version": "v1",
                "history_assignment_sha256": f"assignment-{name}",
                "history_assignments_path": f"{name}.jsonl",
                "request_manifest_sha256": "requests",
                "scoring_signature": {"model": "fixture"},
                "split": "dev",
            }),
            encoding="utf-8",
        )
        return run

    def test_hand_computed_residual_and_exact_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "candidate_manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            protocol = root / "protocol.md"
            protocol.write_text("frozen\n", encoding="utf-8")
            qc = self._source(root, "qc", [2.0, 1.0], manifest)
            null = self._source(root, "null", [0.5, 0.25], manifest)
            true = self._source(root, "true", [0.9, 0.05], manifest)
            output = root / "out_true"
            metadata = write_fixed_residual_anchor_scores(
                qc, true, null, output, manifest,
                coefficient=0.5, history_condition="true", method_id="anchor",
                protocol_path=protocol,
            )
            rows = list(iter_jsonl(output / "scores.jsonl"))
            self.assertEqual([row["candidate_item_id"] for row in rows], ["a", "b"])
            self.assertAlmostEqual(rows[0]["score"], 2.2)
            self.assertAlmostEqual(rows[1]["score"], 0.9)
            self.assertFalse(metadata["qrels_read"])
            self.assertEqual(metadata["dataset_id"], "dataset")
            self.assertEqual(metadata["history_assignment_sha256"], "assignment-true")
            self.assertTrue(metadata["checkpoint_id"].startswith("fixed-residual-anchor@"))

            null_output = root / "out_null"
            write_fixed_residual_anchor_scores(
                qc, null, null, null_output, manifest,
                coefficient=1.0, history_condition="null", method_id="anchor",
                protocol_path=protocol,
            )
            null_rows = list(iter_jsonl(null_output / "scores.jsonl"))
            self.assertEqual([row["score"] for row in null_rows], [2.0, 1.0])

    def test_candidate_manifest_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "candidate_manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            protocol = root / "protocol.md"
            protocol.write_text("frozen\n", encoding="utf-8")
            qc = self._source(root, "qc", [1.0, 0.0], manifest)
            null = self._source(root, "null", [0.0, 0.0], manifest)
            true = self._source(root, "true", [0.1, -0.1], manifest)
            (true / "metadata.json").write_text(
                json.dumps({
                    "candidate_manifest_sha256": "wrong",
                    "dataset_id": "dataset",
                    "dataset_version": "v1",
                    "request_manifest_sha256": "requests",
                    "split": "dev",
                }), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "candidate manifest mismatch"):
                write_fixed_residual_anchor_scores(
                    qc, true, null, root / "out", manifest,
                    coefficient=1.0, history_condition="true", method_id="anchor",
                    protocol_path=protocol,
                )


if __name__ == "__main__":
    unittest.main()
