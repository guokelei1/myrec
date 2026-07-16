from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.prodsearch_adapter import (
    _prodsearch_counterfactual_metadata,
    build_official_command,
    materialize_prodsearch_format,
    prepare_prodsearch_shared_history_view,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _item(item_id: str, *, clicked: int | None = None) -> dict:
    row = {"item_id": item_id, "title": f"title {item_id}", "brand": "b", "cat": ["c"]}
    if clicked is not None:
        row["clicked"] = clicked
    return row


class ProdSearchAdapterTest(unittest.TestCase):
    def test_official_command_uses_exact_interaction_query_binding(self):
        command = build_official_command(
            baseline_dir="baselines/pps_classic/prodsearch_tem",
            materialized_root="materialized",
            save_dir="output",
            model="zam",
            seed=7,
            embedding_size=8,
            learning_rate=1e-3,
            max_train_epoch=1,
            batch_size=2,
            valid_batch_size=2,
            valid_candidate_size=3,
            test_candidate_size=3,
            candidate_batch_size=3,
            num_workers=0,
        )
        flag_index = command.index("--use_review_query_idx")
        self.assertEqual(command[flag_index + 1], "true")

    def test_counterfactual_metadata_changes_only_history_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standardized = root / "standardized"
            standardized.mkdir()
            candidate_manifest = standardized / "candidate_manifest.json"
            candidate_manifest.write_text(
                json.dumps({"entries": []}),
                encoding="utf-8",
            )
            (standardized / "manifest.json").write_text(
                json.dumps({"dataset_id": "fixture", "dataset_version": "v1"}),
                encoding="utf-8",
            )
            (standardized / "request_manifest.json").write_text(
                json.dumps({"dataset_version": "v1"}), encoding="utf-8"
            )
            checkpoint = root / "model_best.ckpt"
            checkpoint.write_bytes(b"checkpoint")

            metadata = []
            for condition, history_bytes in (("true", b"true"), ("null", b"null")):
                materialized = root / condition
                (materialized / "data").mkdir(parents=True)
                (materialized / "data" / "u_r_seq.txt.gz").write_bytes(history_bytes)
                metadata.append(
                    _prodsearch_counterfactual_metadata(
                        materialized_root=materialized,
                        candidate_manifest_path=candidate_manifest,
                        checkpoint_path=checkpoint,
                        model="zam",
                        embedding_size=128,
                        batch_size=256,
                        valid_batch_size=24,
                        valid_candidate_size=100,
                        test_candidate_size=100,
                        candidate_batch_size=100,
                        history_condition=condition,
                    )
                )

            true, null = metadata
            self.assertEqual(true["checkpoint_id"], null["checkpoint_id"])
            self.assertEqual(true["dataset_id"], "fixture")
            self.assertEqual(true["dataset_version"], "v1")
            self.assertEqual(true["scoring_signature"], null["scoring_signature"])
            self.assertEqual(
                true["candidate_manifest_sha256"], null["candidate_manifest_sha256"]
            )
            self.assertEqual(
                true["request_manifest_sha256"], null["request_manifest_sha256"]
            )
            self.assertNotEqual(
                true["history_assignment_sha256"], null["history_assignment_sha256"]
            )
            self.assertEqual(true["history_condition"], "true")
            self.assertEqual(null["history_condition"], "null")

            confirmation = _prodsearch_counterfactual_metadata(
                materialized_root=root / "true",
                candidate_manifest_path=candidate_manifest,
                checkpoint_path=checkpoint,
                model="zam",
                embedding_size=128,
                batch_size=256,
                valid_batch_size=24,
                valid_candidate_size=100,
                test_candidate_size=100,
                candidate_batch_size=100,
                history_condition="true",
                split="confirmation",
            )
            self.assertEqual(confirmation["split"], "confirmation")

    def test_true_and_null_materializations_change_only_effective_dev_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standardized = root / "standardized"
            standardized.mkdir()
            _write_jsonl(
                standardized / "records_train.jsonl",
                [
                    {
                        "request_id": "tr1",
                        "query": "phone",
                        "ts": 10,
                        "history": [_item("h")],
                        "candidates": [_item("p1", clicked=1), _item("p2", clicked=0)],
                    },
                    {
                        "request_id": "tr2",
                        "query": "case",
                        "ts": 20,
                        "history": [_item("p1")],
                        "candidates": [_item("p2", clicked=1), _item("p3", clicked=0)],
                    },
                ],
            )
            _write_jsonl(
                standardized / "records_dev.jsonl",
                [
                    {
                        "request_id": "dev1",
                        "query": "charger",
                        "ts": 30,
                        "history": [_item("h")],
                        "candidates": [_item("p1"), _item("p3")],
                    }
                ],
            )
            (standardized / "candidate_manifest.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "split": "dev",
                                "request_id": "dev1",
                                "candidate_item_ids": ["p1", "p3"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            true = materialize_prodsearch_format(
                standardized,
                root / "true",
                seed=7,
                valid_examples=1,
                valid_candidate_size=2,
                test_candidate_size=2,
                dev_history_condition="true",
            )
            null = materialize_prodsearch_format(
                standardized,
                root / "null",
                seed=7,
                valid_examples=1,
                valid_candidate_size=2,
                test_candidate_size=2,
                dev_history_condition="null",
            )

            self.assertEqual(true["counts"], null["counts"])
            self.assertEqual(true["mapping"]["dev_original_history_events"], 1)
            self.assertEqual(true["mapping"]["dev_effective_history_events"], 1)
            self.assertEqual(null["mapping"]["dev_original_history_events"], 1)
            self.assertEqual(null["mapping"]["dev_effective_history_events"], 0)
            self.assertEqual(
                (root / "true" / "dev_request_map.jsonl").read_text(encoding="utf-8"),
                (root / "null" / "dev_request_map.jsonl").read_text(encoding="utf-8"),
            )
            for relative in (
                "data/product.txt.gz",
                "data/users.txt.gz",
                "data/review_text.txt.gz",
                "split/query.txt.gz",
                "split/train.txt.gz",
                "split/train_id.txt.gz",
                "split/valid_id.txt.gz",
                "split/test_id.txt.gz",
            ):
                self.assertEqual(
                    (root / "true" / relative).read_bytes(),
                    (root / "null" / relative).read_bytes(),
                    relative,
                )
            with gzip.open(root / "true" / "data" / "u_r_seq.txt.gz", "rt") as handle:
                true_last = handle.read().splitlines()[-1].split()
            with gzip.open(root / "null" / "data" / "u_r_seq.txt.gz", "rt") as handle:
                null_last = handle.read().splitlines()[-1].split()
            self.assertEqual(len(true_last), 2)
            self.assertEqual(len(null_last), 1)

    def test_shared_counterfactual_view_reuses_catalog_and_maps_wrong_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standardized = root / "standardized"
            standardized.mkdir()
            _write_jsonl(
                standardized / "records_train.jsonl",
                [
                    {
                        "request_id": "tr1",
                        "query": "phone",
                        "ts": 10,
                        "history": [_item("h")],
                        "candidates": [_item("p1", clicked=1), _item("p2", clicked=0)],
                    },
                    {
                        "request_id": "tr2",
                        "query": "case",
                        "ts": 20,
                        "history": [_item("p1")],
                        "candidates": [_item("p2", clicked=1), _item("p1", clicked=0)],
                    },
                ],
            )
            _write_jsonl(
                standardized / "records_dev.jsonl",
                [
                    {
                        "request_id": "dev1",
                        "query": "charger",
                        "ts": 30,
                        "history": [_item("h")],
                        "candidates": [_item("p1"), _item("p2")],
                    }
                ],
            )
            (standardized / "candidate_manifest.json").write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "split": "dev",
                                "request_id": "dev1",
                                "candidate_item_ids": ["p1", "p2"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (standardized / "manifest.json").write_text(
                json.dumps({"dataset_id": "fixture", "dataset_version": "v1"}),
                encoding="utf-8",
            )
            (standardized / "request_manifest.json").write_text(
                json.dumps({"dataset_version": "v1"}), encoding="utf-8"
            )
            true_root = root / "true"
            wrong_root = root / "wrong"
            materialize_prodsearch_format(
                standardized,
                true_root,
                seed=7,
                valid_examples=1,
                valid_candidate_size=2,
                test_candidate_size=2,
                dev_history_condition="true",
            )
            materialize_prodsearch_format(
                standardized,
                wrong_root,
                seed=7,
                valid_examples=1,
                valid_candidate_size=2,
                test_candidate_size=2,
                dev_history_condition="null",
            )
            with gzip.open(wrong_root / "data/product.txt.gz", "rt") as handle:
                wrong_products = [line.rstrip("\n") for line in handle]
            with gzip.open(wrong_root / "data/product.txt.gz", "wt") as handle:
                for product_id in reversed(wrong_products):
                    handle.write(product_id + "\n")
            with gzip.open(wrong_root / "data/u_r_seq.txt.gz", "rt") as handle:
                wrong_sequences = [line.split() for line in handle]
            wrong_sequences[-1] = ["0", wrong_sequences[-1][-1]]
            with gzip.open(wrong_root / "data/u_r_seq.txt.gz", "wt") as handle:
                for row in wrong_sequences:
                    handle.write(" ".join(row) + "\n")
            shared_root = root / "shared"
            manifest = prepare_prodsearch_shared_history_view(
                catalog_root=true_root,
                history_root=wrong_root,
                output_root=shared_root,
                history_condition="wrong",
            )
            self.assertTrue(manifest["shared_catalog"])
            self.assertEqual(
                (true_root / "data/product.txt.gz").read_bytes(),
                (shared_root / "data/product.txt.gz").read_bytes(),
            )
            with gzip.open(shared_root / "data/u_r_seq.txt.gz", "rt") as handle:
                rows = [line.split() for line in handle]
            with gzip.open(true_root / "data/u_r_seq.txt.gz", "rt") as handle:
                true_rows = [line.split() for line in handle]
            self.assertEqual(rows[:-1], true_rows[:-1])
            self.assertEqual(rows[-1][-1], true_rows[-1][-1])
            self.assertEqual(rows[-1][:-1], [str(len(true_rows[0]) - 1)])


if __name__ == "__main__":
    unittest.main()
