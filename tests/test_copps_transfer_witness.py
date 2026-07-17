from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.copps_transfer_witness import (  # noqa: E402
    _assert_frozen_dev_scoring_inputs,
    _assert_frozen_scoring_population,
    _load_history_assignment_manifest,
    _load_history_assignments,
    _validate_assigned_history,
    CatalogItem,
    FORBIDDEN_MODEL_INPUTS,
    build_training_requests,
    CoPPSTransferWitness,
    history_view_contrastive_loss,
    project_visible_record,
    select_semantic_replacement,
    serialize_visible_item,
    train_copps_transfer_witness,
    write_copps_transfer_witness_scores,
)
from myrec.baselines.frozen_text_features import (  # noqa: E402
    FrozenTextFeatureStore,
    build_store_fingerprint,
    collect_visible_content_texts,
    finalize_fingerprint,
    serialize_item_semantic_content,
)
from myrec.utils.hashing import sha256_file, sha256_text  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _event(item_id: str, title: str, *, query: str = "old query") -> dict:
    return {
        "item_id": item_id,
        "title": title,
        "brand": "brand",
        "cat": ["root", "leaf"],
        "event": "click",
        "query": query,
        "ts": 1,
    }


def _candidate(item_id: str, title: str, **labels: int) -> dict:
    return {
        "item_id": item_id,
        "title": title,
        "brand": "brand",
        "cat": ["root", "leaf"],
        **labels,
    }


def _record(
    request_id: str,
    query: str,
    history: list[dict],
    candidates: list[dict],
) -> dict:
    return {
        "request_id": request_id,
        "user_id": f"u-{request_id}",
        "session_id": f"s-{request_id}",
        "ts": 10,
        "query": query,
        "history": history,
        "candidates": candidates,
        "masks": {"history_present": bool(history), "text_coverage": 1.0},
    }


def _materialize_test_store(
    root: Path,
    records: list[Path],
    *,
    dimension: int = 8,
    name: str = "features",
    base_store: Path | None = None,
) -> Path:
    store = root / name
    store.mkdir()
    texts = collect_visible_content_texts(records)
    hashes = [sha256_text(text) for text in texts]
    vectors = np.stack(
        [
            np.random.default_rng(
                int.from_bytes(
                    bytes.fromhex(sha256_text(f"20260714|{digest}"))[:8], "big"
                )
            ).normal(size=dimension)
            for digest in hashes
        ]
    ).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(store / "vectors.npy", vectors.astype(np.float16))
    (store / "index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hash_to_row": {digest: index for index, digest in enumerate(hashes)},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    encoder_fingerprint = finalize_fingerprint(
        {
            "schema_version": 1,
            "model_name_or_path": "BAAI/bge-small-zh-v1.5",
            "resolved_revision": "fixture",
            "artifact_files": [{"path": "fixture", "sha256": "0" * 64}],
            "encoding_recipe": {
                "pooling": "last_hidden_state_token_0_cls",
                "l2_normalize": True,
                "max_length": 128,
                "batch_size": 64,
                "requested_inference_dtype": "float32",
                "effective_compute_dtype": "float32",
                "device_identity": {
                    "device_type": "cpu",
                    "autocast_enabled": False,
                },
                "package_versions": {
                    "numpy": np.__version__,
                    "sentence_transformers": "fixture",
                    "torch": torch.__version__,
                    "transformers": "fixture",
                },
            },
        }
    )
    base_fingerprint = None
    base_metadata_sha = None
    ancestry: list[dict] = []
    reused_rows = 0
    if base_store is not None:
        base = FrozenTextFeatureStore(base_store, require_fingerprints=True)
        self_hashes = set(hashes)
        if not set(base.hash_to_row) <= self_hashes:
            raise ValueError("fixture scoring store is not a text superset")
        base_fingerprint = base.store_fingerprint_sha256
        base_metadata_sha = sha256_file(base_store / "metadata.json")
        ancestry = [
            {
                "store_fingerprint_sha256": base_fingerprint,
                "metadata_sha256": base_metadata_sha,
                "relation": "direct_bitwise_row_reuse",
                "text_count": int(base.metadata["text_count"]),
            },
            *base.metadata.get("store_ancestry", []),
        ]
        reused_rows = int(base.metadata["text_count"])
    metadata = {
        "schema_version": 1,
        "feature_contract": "frozen_transformer_cls_l2_v1",
        "visible_text_contract": "query_context_and_canonical_item_semantics_v2",
        "model_name_or_path": "BAAI/bge-small-zh-v1.5",
        "qrels_read": False,
        "record_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in records
        ],
        "encoder_fingerprint": encoder_fingerprint,
        "index_sha256": sha256_file(store / "index.json"),
        "vectors_sha256": sha256_file(store / "vectors.npy"),
        "text_count": len(texts),
        "hidden_size": dimension,
        "storage_dtype": "float16",
        "base_store_path": str(base_store) if base_store is not None else None,
        "base_store_fingerprint_sha256": base_fingerprint,
        "base_store_metadata_sha256": base_metadata_sha,
        "reused_text_rows": reused_rows,
        "new_text_rows": len(texts) - reused_rows,
        "store_ancestry": ancestry,
    }
    metadata["store_fingerprint"] = build_store_fingerprint(metadata)
    (store / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    return store


def _write_manifests(
    standardized: Path,
    *,
    train_rows: list[dict],
    dev_rows: list[dict] | None = None,
) -> None:
    split_rows = {"train": train_rows, "dev": dev_rows or []}
    (standardized / "manifest.json").write_text(
        json.dumps({"dataset_id": "kuaisearch", "dataset_version": "tiny"}),
        encoding="utf-8",
    )
    candidate_entries = []
    request_entries = []
    for split, rows in split_rows.items():
        for row in rows:
            candidate_ids = [item["item_id"] for item in row["candidates"]]
            candidate_entries.append(
                {
                    "split": split,
                    "request_id": row["request_id"],
                    "candidate_item_ids": candidate_ids,
                }
            )
            request_entries.append(
                {
                    "split": split,
                    "request_id": row["request_id"],
                    "query_sha256": sha256_text(row["query"]),
                    "candidate_item_ids_sha256": sha256_text(
                        json.dumps(candidate_ids, separators=(",", ":"))
                    ),
                }
            )
    (standardized / "candidate_manifest.json").write_text(
        json.dumps(
            {"dataset_version": "tiny", "entries": candidate_entries},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (standardized / "request_manifest.json").write_text(
        json.dumps(
            {"dataset_version": "tiny", "entries": request_entries},
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_assignment_bundle(
    root: Path,
    *,
    records_path: Path,
    rows_by_condition: dict[str, list[dict]],
) -> dict[str, Path]:
    root.mkdir()
    paths: dict[str, Path] = {}
    files: dict[str, dict[str, str]] = {}
    counts = {len(rows) for rows in rows_by_condition.values()}
    if len(counts) != 1:
        raise ValueError("assignment fixture conditions must have equal coverage")
    for condition, rows in rows_by_condition.items():
        path = root / f"{condition}.jsonl"
        _write_jsonl(path, rows)
        paths[condition] = path
        files[condition] = {"path": str(path), "sha256": sha256_file(path)}
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "qrels_read": False,
                "model_scores_read": False,
                "source_records_path": str(records_path),
                "source_records_sha256": sha256_file(records_path),
                "requests": counts.pop(),
                "seed": 20260714,
                "evidence_mode": "test",
                "target_candidate_leakage_violations": 0,
                "history_not_strictly_before_target_violations": 0,
                "files": files,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return paths


class CoPPSTransferWitnessTest(unittest.TestCase):
    def test_explicit_projection_removes_every_label_field(self):
        raw = _record(
            "r",
            "shoe",
            [dict(_event("h", "history shoe"), relevance=3, label=1)],
            [
                _candidate(
                    "c",
                    "candidate shoe",
                    clicked=1,
                    purchased=1,
                    relevance=3,
                    target=1,
                )
            ],
        )
        raw["labels"] = ["c"]
        raw["masks"]["target"] = "c"
        projected = project_visible_record(raw)
        serialized = json.dumps(projected, ensure_ascii=False)
        for forbidden in FORBIDDEN_MODEL_INPUTS:
            self.assertNotIn(f'"{forbidden}"', serialized)
        self.assertEqual(projected["candidates"][0]["item_id"], "c")
        self.assertEqual(projected["history"][0]["event"], "click")

    def test_assignment_semantics_reject_mismatched_null_and_unsafe_wrong_history(self):
        raw = _record(
            "d1",
            "shoe",
            [_event("h", "true history")],
            [_candidate("x", "candidate x"), _candidate("y", "candidate y")],
        )
        record = project_visible_record(raw)
        true_assignment = {
            "history": record["history"],
            "donor_request_id": "d1",
            "donor_user_id": None,
        }
        self.assertEqual(
            _validate_assigned_history(
                raw, record, true_assignment, history_condition="true"
            ),
            record["history"],
        )
        mismatched_true = {
            **true_assignment,
            "history": [_event("other", "not the source history")],
        }
        with self.assertRaisesRegex(ValueError, "true assignment differs"):
            _validate_assigned_history(
                raw, record, mismatched_true, history_condition="true"
            )
        with self.assertRaisesRegex(ValueError, "null assignment is non-empty"):
            _validate_assigned_history(
                raw, record, true_assignment, history_condition="null"
            )

        valid_wrong = {
            "history": [_event("donor", "safe donor")],
            "donor_request_id": "t1",
            "donor_user_id": "u-other",
        }
        self.assertEqual(
            len(
                _validate_assigned_history(
                    raw, record, valid_wrong, history_condition="wrong"
                )
            ),
            1,
        )
        unsafe = {
            "candidate": {**valid_wrong, "history": [_event("x", "leaked candidate")]},
            "future": {
                **valid_wrong,
                "history": [{**_event("donor", "future donor"), "ts": raw["ts"]}],
            },
            "same_user": {**valid_wrong, "donor_user_id": raw["user_id"]},
            "same_request": {**valid_wrong, "donor_request_id": raw["request_id"]},
        }
        patterns = {
            "candidate": "target candidate",
            "future": "not causal",
            "same_user": "not cross-user",
            "same_request": "reused the target request",
        }
        for name, assignment in unsafe.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, patterns[name]
            ):
                _validate_assigned_history(
                    raw, record, assignment, history_condition="wrong"
                )

    def test_assignment_manifest_pins_sha_source_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records_dev.jsonl"
            row = _record(
                "d1",
                "shoe",
                [_event("h", "history")],
                [_candidate("x", "one"), _candidate("y", "two")],
            )
            _write_jsonl(records, [row])
            paths = _write_assignment_bundle(
                root / "assignments",
                records_path=records,
                rows_by_condition={
                    "true": [
                        {
                            "request_id": "d1",
                            "assignment": "true",
                            "history": row["history"],
                        }
                    ]
                },
            )
            assignments = _load_history_assignments(
                paths["true"], expected_condition="true"
            )
            verified = _load_history_assignment_manifest(
                paths["true"],
                expected_condition="true",
                records_path=records,
                assignment_count=len(assignments),
            )
            self.assertEqual(verified["sha256"], sha256_file(root / "assignments" / "manifest.json"))

            with paths["true"].open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "differs from its manifest"):
                _load_history_assignment_manifest(
                    paths["true"],
                    expected_condition="true",
                    records_path=records,
                    assignment_count=1,
                )

            # Restore the file identity, then prove the independent count gate.
            _write_jsonl(
                paths["true"],
                [
                    {
                        "request_id": "d1",
                        "assignment": "true",
                        "history": row["history"],
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "request count mismatch"):
                _load_history_assignment_manifest(
                    paths["true"],
                    expected_condition="true",
                    records_path=records,
                    assignment_count=2,
                )

    def test_query_attention_uses_unscaled_normalized_cosine(self):
        model = CoPPSTransferWitness(content_dim=2, projection_dim=2)
        with torch.no_grad():
            model.content_projection.weight.copy_(torch.eye(2))
        query = torch.tensor([[1.0, 0.0]])
        history = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        mask = torch.tensor([[True, True]])
        profile = model.history_profile(query, history, mask)
        weights = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)
        expected = torch.nn.functional.normalize(
            torch.tensor([[weights[0], weights[1]]]), dim=-1
        )
        torch.testing.assert_close(profile, expected)

    def test_feature_store_rejects_vector_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records_train.jsonl"
            _write_jsonl(
                records,
                [
                    _record(
                        "r",
                        "shoe",
                        [_event("h", "history")],
                        [_candidate("p", "positive"), _candidate("n", "negative")],
                    )
                ],
            )
            store = _materialize_test_store(root, [records])
            FrozenTextFeatureStore(store, require_fingerprints=True)
            vectors = np.load(store / "vectors.npy", mmap_mode="r+")
            vectors[0, 0] = np.float16(float(vectors[0, 0]) + 0.25)
            vectors.flush()
            with self.assertRaisesRegex(ValueError, "vectors hash"):
                FrozenTextFeatureStore(store, require_fingerprints=True)

    def test_frozen_dev_scoring_inputs_pin_every_qrels_free_population_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            standardized = Path(tmp) / "standardized"
            standardized.mkdir()
            paths = {
                "manifest.json": standardized / "manifest.json",
                "candidate_manifest.json": standardized / "candidate_manifest.json",
                "request_manifest.json": standardized / "request_manifest.json",
                "records_dev.jsonl": standardized / "records_dev.jsonl",
            }
            paths["manifest.json"].write_text(
                json.dumps({"dataset_id": "kuaisearch", "dataset_version": "tiny"}),
                encoding="utf-8",
            )
            paths["candidate_manifest.json"].write_text("{}", encoding="utf-8")
            paths["request_manifest.json"].write_text("{}", encoding="utf-8")
            paths["records_dev.jsonl"].write_text("{}\n", encoding="utf-8")
            # A malformed qrels file must remain irrelevant to this verifier.
            (standardized / "qrels_dev.jsonl").write_text(
                "must-not-be-opened\n", encoding="utf-8"
            )
            development = {
                "manifest_sha256": sha256_file(paths["manifest.json"]),
                "candidate_manifest_sha256": sha256_file(
                    paths["candidate_manifest.json"]
                ),
                "request_manifest_sha256": sha256_file(paths["request_manifest.json"]),
                "records_dev_sha256": sha256_file(paths["records_dev.jsonl"]),
            }
            config = {
                "dataset": {
                    "dataset_id": "kuaisearch",
                    "dataset_version": "tiny",
                    "standardized_dir": str(standardized),
                },
                "protocol": {"sha256": "a" * 64},
                "_protocol_payload": {
                    "data": {"development_population": development}
                },
            }
            result = _assert_frozen_dev_scoring_inputs(
                config=config,
                standardized_dir=standardized,
                records_path=paths["records_dev.jsonl"],
                dataset_manifest={
                    "dataset_id": "kuaisearch",
                    "dataset_version": "tiny",
                },
            )
            self.assertTrue(result["passed"])
            self.assertFalse(result["qrels_opened"])
            paths["records_dev.jsonl"].write_text("{\"tampered\":true}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "records_dev.jsonl"):
                _assert_frozen_dev_scoring_inputs(
                    config=config,
                    standardized_dir=standardized,
                    records_path=paths["records_dev.jsonl"],
                    dataset_manifest={
                        "dataset_id": "kuaisearch",
                        "dataset_version": "tiny",
                    },
                )

    def test_holdout_scoring_binds_released_w0_identity_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standardized = root / "holdout"
            standardized.mkdir()
            records = standardized / "records_confirmation.jsonl"
            records.write_text("{}\n", encoding="utf-8")
            model_path = root / "model.pt"
            model_path.write_bytes(b"w0-model")
            metadata_path = root / "metadata.json"
            metadata_path.write_text("{}", encoding="utf-8")
            model_sha = sha256_file(model_path)
            metadata_sha = sha256_file(metadata_path)
            checkpoint_id = f"w0_copps_style_transfer_witness@{model_sha[:20]}"
            config = {
                "_config_sha256": "3" * 64,
                "protocol": {"path": "frozen-protocol.yaml", "sha256": "4" * 64},
                "_protocol_payload": {
                    "data": {
                        "development_population": {"dataset_version": "development"}
                    }
                },
            }
            frozen = {
                "identity_manifest_sha256": "5" * 64,
                "checkpoint_id": checkpoint_id,
                "checkpoint_sha256": model_sha,
                "checkpoint_files": [
                    {
                        "name": "model.pt",
                        "path": str(model_path),
                        "sha256": model_sha,
                        "size_bytes": model_path.stat().st_size,
                    }
                ],
                "config_sha256": config["_config_sha256"],
                "training_metadata_path": str(metadata_path),
                "training_metadata_sha256": metadata_sha,
                "implementation_digest": "6" * 64,
                "protocol_sha256": config["protocol"]["sha256"],
            }
            audit = {
                "checkpoint_identities": {
                    "w0_copps_style_transfer_witness": frozen
                },
                "integrity_lock_sha256": "7" * 64,
                "manifest_sha256": "8" * 64,
                "post_selection_recipe_checkpoint_lock_sha256": "9" * 64,
                "protocol_sha256": config["protocol"]["sha256"],
                "qrels_opened": False,
            }
            checkpoint = {"checkpoint_id": checkpoint_id}
            with mock.patch(
                "myrec.data.kuaisearch_holdout.verify_published_holdout",
                return_value=audit,
            ) as verifier:
                result = _assert_frozen_scoring_population(
                    config=config,
                    standardized_dir=standardized,
                    records_path=records,
                    dataset_manifest={
                        "dataset_version": "full_confirm_preceding40k_newholdout4k_v12"
                    },
                    split="confirmation",
                    history_condition="wrong",
                    checkpoint=checkpoint,
                    checkpoint_metadata_path=metadata_path,
                    model_path=model_path,
                    model_sha256=model_sha,
                    implementation_digest="6" * 64,
                )
            self.assertEqual(
                result,
                {
                    "checkpoint_identity_manifest_sha256": "5" * 64,
                    "checkpoint_id": checkpoint_id,
                    "integrity_lock_sha256": "7" * 64,
                    "manifest_sha256": "8" * 64,
                    "post_selection_recipe_checkpoint_lock_sha256": "9" * 64,
                    "protocol_sha256": "4" * 64,
                    "qrels_opened": False,
                    "verified_before_model_load": True,
                },
            )
            verifier.assert_called_once_with(
                standardized,
                protocol_path="frozen-protocol.yaml",
                open_qrels=False,
            )

            frozen["implementation_digest"] = "0" * 64
            with mock.patch(
                "myrec.data.kuaisearch_holdout.verify_published_holdout",
                return_value=audit,
            ), self.assertRaisesRegex(ValueError, "implementation_digest"):
                _assert_frozen_scoring_population(
                    config=config,
                    standardized_dir=standardized,
                    records_path=records,
                    dataset_manifest={
                        "dataset_version": "full_confirm_preceding40k_newholdout4k_v12"
                    },
                    split="confirmation",
                    history_condition="true",
                    checkpoint=checkpoint,
                    checkpoint_metadata_path=metadata_path,
                    model_path=model_path,
                    model_sha256=model_sha,
                    implementation_digest="6" * 64,
                )

    def test_semantic_replacement_is_same_category_and_excludes_whole_slate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(
                root / "vectors.npy",
                np.asarray(
                    [
                        [1.0, 0.0],
                        [0.99, 0.01],
                        [0.8, 0.2],
                        [0.0, 1.0],
                    ],
                    dtype=np.float16,
                ),
            )
            (root / "index.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "hash_to_row": {f"dummy-{index}": index for index in range(4)},
                    }
                ),
                encoding="utf-8",
            )
            (root / "metadata.json").write_text(
                json.dumps({"feature_contract": "frozen_transformer_cls_l2_v1"}),
                encoding="utf-8",
            )
            store = FrozenTextFeatureStore(root)
            category = ("root", "leaf")
            source = CatalogItem("source", 0, category)
            pool = (
                source,
                CatalogItem("candidate-in-slate", 1, category),
                CatalogItem("allowed", 2, category),
            )
            replacement = select_semantic_replacement(
                source,
                pool,
                store,
                excluded_item_ids={"source", "candidate-in-slate"},
            )
            self.assertIsNotNone(replacement)
            assert replacement is not None
            self.assertEqual(replacement.item_id, "allowed")
            self.assertEqual(replacement.category, source.category)

    def test_two_views_replace_ceil_thirty_percent_and_use_train_qrels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records_train.jsonl"
            qrels = root / "qrels_train.jsonl"
            histories = [_event(f"h{i}", f"history {i}") for i in range(4)]
            rows = [
                _record(
                    "r1",
                    "red shoe",
                    histories,
                    [
                        _candidate("p1", "positive", clicked=0),
                        _candidate("n1", "negative", clicked=1),
                    ],
                ),
                _record(
                    "r2",
                    "blue shoe",
                    [_event("donor-h", "donor history")],
                    [
                        _candidate("p2", "donor positive"),
                        _candidate("n2", "donor negative"),
                        _candidate("d1", "donor one"),
                        _candidate("d2", "donor two"),
                        _candidate("d3", "donor three"),
                    ],
                ),
            ]
            _write_jsonl(records, rows)
            _write_jsonl(
                qrels,
                [
                    {"request_id": "r1", "clicked": ["p1"], "purchased": [], "relevance": {}},
                    {"request_id": "r2", "clicked": ["p2"], "purchased": [], "relevance": {}},
                ],
            )
            store_dir = _materialize_test_store(root, [records])
            store = FrozenTextFeatureStore(store_dir, require_fingerprints=True)
            examples, stats = build_training_requests(records, qrels, store)
            self.assertEqual(len(examples), 2)
            self.assertEqual(stats["selected_history_events"]["a"], 3)
            self.assertEqual(stats["selected_history_events"]["b"], 3)
            self.assertEqual(
                stats["replacement_contract"]["replacement_ratio"], 0.3
            )
            self.assertEqual(stats["replacement_contract"]["views_per_request"], 2)
            first = next(row for row in examples if row.request_id == "r1")
            self.assertLessEqual(len(first.augmented_history_rows_a), 4)
            self.assertLessEqual(len(first.augmented_history_rows_b), 4)
            # Embedded record labels contradict qrels; qrels_train remains authoritative.
            self.assertEqual(first.positive_mask, (True, False))
            contextual_row = store.hash_to_row[
                sha256_text(serialize_visible_item(histories[0], history=True))
            ]
            semantic_row = store.hash_to_row[
                sha256_text(serialize_item_semantic_content(histories[0]))
            ]
            self.assertEqual(first.history_rows[0], contextual_row)
            self.assertNotEqual(contextual_row, semantic_row)
            canonical_rows = {
                store.hash_to_row[sha256_text(serialize_item_semantic_content(item))]
                for record in rows
                for item in [*record["history"], *record["candidates"]]
            }
            self.assertTrue(set(first.augmented_history_rows_a) <= canonical_rows)
            self.assertTrue(set(first.augmented_history_rows_b) <= canonical_rows)
            self.assertIn("identical_nonempty_augmented_views", stats)
            self.assertIn("contrastive_eligible_requests", stats)

    def test_replacement_excludes_history_outside_retained_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = root / "records_train.jsonl"
            qrels = root / "qrels_train.jsonl"
            old = _event("old-outside-budget", "old semantic twin")
            retained = [_event(f"h{i}", f"retained {i}") for i in range(6)]
            rows = [
                _record(
                    "r1",
                    "shoe",
                    [old, *retained],
                    [_candidate("p1", "positive"), _candidate("n1", "negative")],
                ),
                _record(
                    "r2",
                    "shoe donor",
                    [_event("donor-history", "donor history")],
                    [
                        _candidate("p2", "positive two"),
                        _candidate("n2", "negative two"),
                        _candidate("allowed-donor", "allowed donor"),
                    ],
                ),
            ]
            _write_jsonl(records, rows)
            _write_jsonl(
                qrels,
                [
                    {"request_id": "r1", "clicked": ["p1"], "purchased": [], "relevance": {}},
                    {"request_id": "r2", "clicked": ["p2"], "purchased": [], "relevance": {}},
                ],
            )
            store_dir = _materialize_test_store(root, [records])
            store = FrozenTextFeatureStore(store_dir, require_fingerprints=True)
            old_row = store.hash_to_row[
                sha256_text(serialize_item_semantic_content(old))
            ]
            examples, _ = build_training_requests(records, qrels, store)
            first = next(row for row in examples if row.request_id == "r1")
            self.assertNotIn(old_row, first.augmented_history_rows_a)
            self.assertNotIn(old_row, first.augmented_history_rows_b)

    def test_symmetric_contrastive_loss_prefers_aligned_view_pairs(self):
        original = torch.eye(3)
        aligned = history_view_contrastive_loss(
            original, original.clone(), torch.ones(3, dtype=torch.bool), temperature=0.1
        )
        permuted = history_view_contrastive_loss(
            original,
            original[[1, 2, 0]],
            torch.ones(3, dtype=torch.bool),
            temperature=0.1,
        )
        self.assertTrue(math.isfinite(float(aligned)))
        self.assertLess(float(aligned), float(permuted))

    def test_production_scoring_requires_a_frozen_config(self):
        with self.assertRaisesRegex(ValueError, "requires a frozen config_path"):
            write_copps_transfer_witness_scores(
                "missing-standardized",
                "missing-features",
                "missing-checkpoint",
                "missing-assignments.jsonl",
                "20260716_kuaisearch_w0_missing_config",
                history_condition="true",
                device="cpu",
                config_path=None,
            )

    def test_train_and_full_null_wrong_scoring_are_label_isolated_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standardized = root / "standardized"
            standardized.mkdir()
            train_rows = [
                _record(
                    "t1",
                    "red shoe",
                    [_event("h1", "red history")],
                    [
                        _candidate("p1", "red positive", clicked=0),
                        _candidate("n1", "red negative", clicked=1),
                    ],
                ),
                _record(
                    "t2",
                    "blue shoe",
                    [_event("h2", "blue history")],
                    [
                        _candidate("p2", "blue positive", clicked=0),
                        _candidate("n2", "blue negative", clicked=1),
                    ],
                ),
            ]
            dev_rows = [
                _record(
                    "d1",
                    " green shoe ",
                    [_event("hd", "green history")],
                    [_candidate("x", "green one"), _candidate("y", "green two")],
                )
            ]
            records_train = standardized / "records_train.jsonl"
            records_dev = standardized / "records_dev.jsonl"
            _write_jsonl(records_train, train_rows)
            _write_jsonl(records_dev, dev_rows)
            _write_jsonl(
                standardized / "qrels_train.jsonl",
                [
                    {"request_id": "t1", "clicked": ["p1"], "purchased": [], "relevance": {}},
                    {"request_id": "t2", "clicked": ["p2"], "purchased": [], "relevance": {}},
                ],
            )
            # Deliberately invalid dev qrels proves neither training nor scoring opens it.
            (standardized / "qrels_dev.jsonl").write_text("not-json\n", encoding="utf-8")
            (standardized / "manifest.json").write_text(
                json.dumps({"dataset_id": "kuaisearch", "dataset_version": "tiny"}),
                encoding="utf-8",
            )
            candidate_manifest = {
                "dataset_version": "tiny",
                "entries": [
                    *[
                        {
                            "split": "train",
                            "request_id": row["request_id"],
                            "candidate_item_ids": [
                                candidate["item_id"] for candidate in row["candidates"]
                            ],
                        }
                        for row in train_rows
                    ],
                    {
                        "split": "dev",
                        "request_id": "d1",
                        "candidate_item_ids": ["x", "y"],
                    },
                ],
            }
            (standardized / "candidate_manifest.json").write_text(
                json.dumps(candidate_manifest, sort_keys=True), encoding="utf-8"
            )
            (standardized / "request_manifest.json").write_text(
                json.dumps(
                    {
                        "dataset_version": "tiny",
                        "entries": [
                            {
                                "split": "train",
                                "request_id": row["request_id"],
                                "query_sha256": sha256_text(row["query"]),
                                "candidate_item_ids_sha256": sha256_text(
                                    json.dumps(
                                        [item["item_id"] for item in row["candidates"]],
                                        separators=(",", ":"),
                                    )
                                ),
                            }
                            for row in train_rows
                        ]
                        + [
                            {
                                "split": "dev",
                                "request_id": "d1",
                                "query_sha256": sha256_text(dev_rows[0]["query"]),
                                "candidate_item_ids_sha256": sha256_text(
                                    json.dumps(["x", "y"], separators=(",", ":"))
                                ),
                            },
                        ],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            train_store_dir = _materialize_test_store(
                root, [records_train], name="train_features"
            )
            scoring_store_dir = _materialize_test_store(
                root,
                [records_train, records_dev],
                name="scoring_features",
                base_store=train_store_dir,
            )
            train_store = FrozenTextFeatureStore(
                train_store_dir, require_fingerprints=True
            )
            scoring_store = FrozenTextFeatureStore(
                scoring_store_dir, require_fingerprints=True
            )
            self.assertEqual(
                train_store.encoder_fingerprint_sha256,
                scoring_store.encoder_fingerprint_sha256,
            )
            self.assertNotEqual(
                train_store.store_fingerprint_sha256,
                scoring_store.store_fingerprint_sha256,
            )
            runs = root / "runs"
            model_dir = root / "model"
            training = train_copps_transfer_witness(
                standardized,
                train_store_dir,
                model_dir,
                "20260716_kuaisearch_w0_tiny_train",
                runs_dir=runs,
                device="cpu",
            )
            self.assertEqual(training["status"], "completed")
            self.assertEqual(training["seed"], 20260714)
            self.assertEqual(training["evidence_mode"], "first_round_pilot")
            self.assertEqual(
                training["implementation_digest"],
                training["implementation_identity"]["digest"],
            )
            self.assertEqual(training["qrels_scope"], "qrels_train_only")
            self.assertFalse(training["dev_qrels_read"])
            resumed = train_copps_transfer_witness(
                standardized,
                train_store_dir,
                model_dir,
                "20260716_kuaisearch_w0_tiny_resume",
                runs_dir=runs,
                device="cpu",
                resume=True,
            )
            self.assertEqual(resumed["status"], "completed")
            self.assertTrue(resumed["training"]["resumed"])
            self.assertEqual(
                resumed["training"]["optimizer_steps"],
                training["training"]["optimizer_steps"],
            )

            assignment_rows = {
                "true": [
                    {
                        "request_id": "d1",
                        "assignment": "true",
                        "donor_request_id": "d1",
                        "history": dev_rows[0]["history"],
                    }
                ],
                "null": [
                    {"request_id": "d1", "assignment": "null", "history": []}
                ],
                "wrong": [
                    {
                        "request_id": "d1",
                        "assignment": "wrong",
                        "donor_request_id": "t1",
                        "donor_user_id": "u-t1",
                        "history": train_rows[0]["history"],
                    }
                ],
            }
            assignment_paths = _write_assignment_bundle(
                root / "assignments",
                records_path=records_dev,
                rows_by_condition=assignment_rows,
            )
            score_metadata: dict[str, dict] = {}
            for condition, assignment in assignment_paths.items():
                metadata = write_copps_transfer_witness_scores(
                    standardized,
                    scoring_store_dir,
                    model_dir,
                    assignment,
                    f"20260716_kuaisearch_w0_tiny_{condition}",
                    history_condition=condition,
                    runs_dir=runs,
                    device="cpu",
                    config_path=None,
                    _test_only_allow_unfrozen_config=True,
                )
                score_metadata[condition] = metadata
                self.assertFalse(metadata["qrels_read"])
                self.assertEqual(metadata["seed"], 20260714)
                self.assertEqual(metadata["evidence_mode"], "first_round_pilot")
                self.assertIsNone(metadata["holdout_integrity"])
                self.assertTrue(metadata["coverage"]["finite_scores"])
                self.assertTrue(metadata["coverage"]["complete_candidate_coverage"])
                self.assertGreater(
                    metadata["score_non_degeneracy"][
                        "nonconstant_requests_at_1e_8"
                    ],
                    0,
                )
                self.assertEqual(
                    metadata["score_non_degeneracy"]["threshold"], 1.0e-8
                )
                self.assertTrue(metadata["history_assignment_semantics_verified"])
                self.assertEqual(
                    metadata["feature_store_compatibility"]["mode"],
                    "bitwise_reuse_descendant_superset",
                )
                self.assertEqual(
                    metadata["history_assignment_manifest_sha256"],
                    sha256_file(assignment.parent / "manifest.json"),
                )
                self.assertEqual(metadata["score_rows"], 2)
                self.assertEqual(
                    metadata["candidate_manifest_sha256"],
                    sha256_file(standardized / "candidate_manifest.json"),
                )
                self.assertEqual(
                    metadata["request_manifest_sha256"],
                    sha256_file(standardized / "request_manifest.json"),
                )
                self.assertEqual(
                    metadata["scores_sha256"],
                    sha256_file(runs / f"20260716_kuaisearch_w0_tiny_{condition}" / "scores.jsonl"),
                )

            scoring_signature = score_metadata["true"]["scoring_signature"]
            self.assertEqual(scoring_signature["batch_requests"], 128)
            self.assertEqual(
                scoring_signature["checkpoint_id"], training["checkpoint_id"]
            )
            self.assertEqual(
                scoring_signature["checkpoint_model_sha256"],
                training["model_sha256"],
            )
            self.assertIsNone(scoring_signature["config_sha256"])
            self.assertIsNone(scoring_signature["protocol_sha256"])
            self.assertEqual(
                scoring_signature["implementation_digest"],
                training["implementation_digest"],
            )
            self.assertEqual(scoring_signature["inference_dtype"], "float32")
            self.assertTrue(scoring_signature["request_aligned_batches"])
            self.assertEqual(
                scoring_signature["runtime_identity"]["device"], "cpu"
            )
            self.assertEqual(
                score_metadata["true"]["scoring_signature"],
                score_metadata["null"]["scoring_signature"],
            )
            self.assertEqual(
                score_metadata["true"]["scoring_signature"],
                score_metadata["wrong"]["scoring_signature"],
            )

            changed_batch_metadata = write_copps_transfer_witness_scores(
                standardized,
                scoring_store_dir,
                model_dir,
                assignment_paths["true"],
                "20260716_kuaisearch_w0_changed_batch",
                history_condition="true",
                runs_dir=runs,
                device="cpu",
                batch_requests=1,
                config_path=None,
                _test_only_allow_unfrozen_config=True,
            )
            self.assertEqual(
                changed_batch_metadata["scoring_signature"]["batch_requests"], 1
            )
            self.assertNotEqual(
                changed_batch_metadata["scoring_signature"], scoring_signature
            )

            same_encoder_without_ancestry = _materialize_test_store(
                root,
                [records_train, records_dev],
                name="same_encoder_without_ancestry",
            )
            with self.assertRaisesRegex(ValueError, "bitwise-reuse descendant"):
                write_copps_transfer_witness_scores(
                    standardized,
                    same_encoder_without_ancestry,
                    model_dir,
                    assignment_paths["true"],
                    "20260716_kuaisearch_w0_no_ancestry",
                    history_condition="true",
                    runs_dir=runs,
                    device="cpu",
                    config_path=None,
                    _test_only_allow_unfrozen_config=True,
                )

            tampered_ancestry_store = root / "tampered_ancestry_features"
            shutil.copytree(scoring_store_dir, tampered_ancestry_store)
            ancestry_metadata_path = tampered_ancestry_store / "metadata.json"
            ancestry_metadata = json.loads(
                ancestry_metadata_path.read_text(encoding="utf-8")
            )
            fake_fingerprint = "1" * 64
            fake_metadata_sha = "2" * 64
            ancestry_metadata["base_store_fingerprint_sha256"] = fake_fingerprint
            ancestry_metadata["base_store_metadata_sha256"] = fake_metadata_sha
            ancestry_metadata["store_ancestry"] = [
                {
                    "store_fingerprint_sha256": fake_fingerprint,
                    "metadata_sha256": fake_metadata_sha,
                    "relation": "direct_bitwise_row_reuse",
                    "text_count": ancestry_metadata["reused_text_rows"],
                }
            ]
            ancestry_metadata["store_fingerprint"] = build_store_fingerprint(
                ancestry_metadata
            )
            ancestry_metadata_path.write_text(
                json.dumps(ancestry_metadata, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "bitwise-reuse descendant"):
                write_copps_transfer_witness_scores(
                    standardized,
                    tampered_ancestry_store,
                    model_dir,
                    assignment_paths["true"],
                    "20260716_kuaisearch_w0_tampered_ancestry",
                    history_condition="true",
                    runs_dir=runs,
                    device="cpu",
                    config_path=None,
                    _test_only_allow_unfrozen_config=True,
                )

            forged_reuse_store = root / "forged_reuse_features"
            shutil.copytree(scoring_store_dir, forged_reuse_store)
            forged_vectors = np.load(
                forged_reuse_store / "vectors.npy", mmap_mode="r+"
            )
            shared_digest = next(iter(train_store.hash_to_row))
            forged_row = scoring_store.hash_to_row[shared_digest]
            forged_vectors[forged_row, 0] = np.float16(
                float(forged_vectors[forged_row, 0]) + 0.5
            )
            forged_vectors.flush()
            forged_metadata_path = forged_reuse_store / "metadata.json"
            forged_metadata = json.loads(
                forged_metadata_path.read_text(encoding="utf-8")
            )
            forged_metadata["vectors_sha256"] = sha256_file(
                forged_reuse_store / "vectors.npy"
            )
            forged_metadata["store_fingerprint"] = build_store_fingerprint(
                forged_metadata
            )
            forged_metadata_path.write_text(
                json.dumps(forged_metadata, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "bitwise reuse training rows"):
                write_copps_transfer_witness_scores(
                    standardized,
                    forged_reuse_store,
                    model_dir,
                    assignment_paths["true"],
                    "20260716_kuaisearch_w0_forged_reuse",
                    history_condition="true",
                    runs_dir=runs,
                    device="cpu",
                    config_path=None,
                    _test_only_allow_unfrozen_config=True,
                )

            tampered_model = root / "tampered_model"
            shutil.copytree(model_dir, tampered_model)
            with (tampered_model / "model.pt").open("ab") as handle:
                handle.write(b"tamper")
            assignment = assignment_paths["true"]
            with self.assertRaisesRegex(ValueError, "model SHA-256"):
                write_copps_transfer_witness_scores(
                    standardized,
                    scoring_store_dir,
                    tampered_model,
                    assignment,
                    "20260716_kuaisearch_w0_tampered_model",
                    history_condition="true",
                    runs_dir=runs,
                    device="cpu",
                    config_path=None,
                    _test_only_allow_unfrozen_config=True,
                )

            degenerate_model = root / "degenerate_model"
            shutil.copytree(model_dir, degenerate_model)
            degenerate_weights_path = degenerate_model / "model.pt"
            degenerate_weights = torch.load(
                degenerate_weights_path, map_location="cpu", weights_only=True
            )
            for key in degenerate_weights:
                degenerate_weights[key].zero_()
            torch.save(degenerate_weights, degenerate_weights_path)
            degenerate_metadata_path = degenerate_model / "metadata.json"
            degenerate_metadata = json.loads(
                degenerate_metadata_path.read_text(encoding="utf-8")
            )
            degenerate_sha = sha256_file(degenerate_weights_path)
            degenerate_metadata["model_sha256"] = degenerate_sha
            degenerate_metadata["checkpoint_id"] = (
                f"w0_copps_style_transfer_witness@{degenerate_sha[:20]}"
            )
            degenerate_metadata_path.write_text(
                json.dumps(degenerate_metadata, sort_keys=True), encoding="utf-8"
            )
            degenerate_run_id = "20260716_kuaisearch_w0_degenerate_scores"
            with self.assertRaisesRegex(ValueError, "globally degenerate"):
                write_copps_transfer_witness_scores(
                    standardized,
                    scoring_store_dir,
                    degenerate_model,
                    assignment,
                    degenerate_run_id,
                    history_condition="true",
                    runs_dir=runs,
                    device="cpu",
                    config_path=None,
                    _test_only_allow_unfrozen_config=True,
                )
            self.assertFalse((runs / degenerate_run_id).exists())

            mismatched_implementation = root / "mismatched_implementation_model"
            shutil.copytree(model_dir, mismatched_implementation)
            mismatched_metadata_path = mismatched_implementation / "metadata.json"
            mismatched_metadata = json.loads(
                mismatched_metadata_path.read_text(encoding="utf-8")
            )
            mismatched_metadata["implementation_digest"] = "0" * 64
            mismatched_metadata_path.write_text(
                json.dumps(mismatched_metadata, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "implementation"):
                write_copps_transfer_witness_scores(
                    standardized,
                    scoring_store_dir,
                    mismatched_implementation,
                    assignment,
                    "20260716_kuaisearch_w0_wrong_implementation",
                    history_condition="true",
                    runs_dir=runs,
                    device="cpu",
                    config_path=None,
                    _test_only_allow_unfrozen_config=True,
                )

            wrong_encoder_store = root / "wrong_encoder_features"
            shutil.copytree(scoring_store_dir, wrong_encoder_store)
            wrong_metadata_path = wrong_encoder_store / "metadata.json"
            wrong_metadata = json.loads(wrong_metadata_path.read_text(encoding="utf-8"))
            encoder_payload = {
                key: value
                for key, value in wrong_metadata["encoder_fingerprint"].items()
                if key != "sha256"
            }
            encoder_payload["resolved_revision"] = "different-fixture-revision"
            wrong_metadata["encoder_fingerprint"] = finalize_fingerprint(
                encoder_payload
            )
            wrong_metadata["store_fingerprint"] = build_store_fingerprint(
                wrong_metadata
            )
            wrong_metadata_path.write_text(
                json.dumps(wrong_metadata, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "encoder fingerprint"):
                write_copps_transfer_witness_scores(
                    standardized,
                    wrong_encoder_store,
                    model_dir,
                    assignment,
                    "20260716_kuaisearch_w0_wrong_encoder",
                    history_condition="true",
                    runs_dir=runs,
                    device="cpu",
                    config_path=None,
                    _test_only_allow_unfrozen_config=True,
                )

    def test_mid_epoch_resume_matches_uninterrupted_training_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            standardized = root / "standardized"
            standardized.mkdir()
            train_rows = [
                _record(
                    f"t{index:03d}",
                    f"query {index}",
                    [_event(f"h{index:03d}", f"history {index}")],
                    [
                        _candidate(f"p{index:03d}", f"positive {index}"),
                        _candidate(f"n{index:03d}", f"negative {index}"),
                    ],
                )
                for index in range(129)
            ]
            records_train = standardized / "records_train.jsonl"
            _write_jsonl(records_train, train_rows)
            _write_jsonl(
                standardized / "qrels_train.jsonl",
                [
                    {
                        "request_id": row["request_id"],
                        "clicked": [row["candidates"][0]["item_id"]],
                        "purchased": [],
                        "relevance": {},
                    }
                    for row in train_rows
                ],
            )
            _write_manifests(standardized, train_rows=train_rows)
            store = _materialize_test_store(root, [records_train])
            runs = root / "runs"
            full_model = root / "full_model"
            resumed_model = root / "resumed_model"

            uninterrupted = train_copps_transfer_witness(
                standardized,
                store,
                full_model,
                "20260716_kuaisearch_w0_resume_full",
                runs_dir=runs,
                device="cpu",
            )
            interrupted = train_copps_transfer_witness(
                standardized,
                store,
                resumed_model,
                "20260716_kuaisearch_w0_resume_part1",
                runs_dir=runs,
                device="cpu",
                max_optimizer_steps_this_job=1,
            )
            self.assertEqual(interrupted["status"], "pending_step_limit")
            self.assertEqual(interrupted["training"]["next_epoch"], 0)
            self.assertEqual(interrupted["training"]["next_batch_cursor"], 1)
            state_path = resumed_model / "training_state.pt"
            original_state = torch.load(
                state_path, map_location="cpu", weights_only=False
            )
            for field in (
                "config_sha256",
                "protocol_sha256",
                "implementation_digest",
            ):
                with self.subTest(resume_contract_field=field):
                    tampered_state = torch.load(
                        state_path, map_location="cpu", weights_only=False
                    )
                    tampered_state["training_contract"][field] = "0" * 64
                    torch.save(tampered_state, state_path)
                    with self.assertRaisesRegex(
                        ValueError, "resume training contract differs"
                    ):
                        train_copps_transfer_witness(
                            standardized,
                            store,
                            resumed_model,
                            f"20260716_kuaisearch_w0_drift_{field}",
                            runs_dir=runs,
                            device="cpu",
                            resume=True,
                        )
                    torch.save(original_state, state_path)
            resumed = train_copps_transfer_witness(
                standardized,
                store,
                resumed_model,
                "20260716_kuaisearch_w0_resume_part2",
                runs_dir=runs,
                device="cpu",
                resume=True,
            )
            self.assertEqual(uninterrupted["status"], "completed")
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(
                resumed["training"]["run_lineage"],
                [
                    "20260716_kuaisearch_w0_resume_part1",
                    "20260716_kuaisearch_w0_resume_part2",
                ],
            )
            full_weights = torch.load(
                full_model / "model.pt", map_location="cpu", weights_only=True
            )
            resumed_weights = torch.load(
                resumed_model / "model.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(set(full_weights), set(resumed_weights))
            for key in full_weights:
                torch.testing.assert_close(
                    full_weights[key], resumed_weights[key], rtol=0.0, atol=0.0
                )
            full_state = torch.load(
                full_model / "training_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            resumed_state = torch.load(
                resumed_model / "training_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                full_state["optimizer_steps"], resumed_state["optimizer_steps"]
            )
            self.assertEqual(
                full_state["scheduler_state"], resumed_state["scheduler_state"]
            )
            self.assertEqual(full_state["losses"], resumed_state["losses"])


if __name__ == "__main__":
    unittest.main()
