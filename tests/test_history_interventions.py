from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.motivation_v12_contracts import HISTORY_INPUT_FIELDS  # noqa: E402
from myrec.mechanism.history_interventions import (  # noqa: E402
    CONDITION_IDS,
    HISTORY_BUDGET,
    MECHANISM_INTERVENTION_SEED,
    generate_history_interventions,
    materialize_history_interventions,
)
from myrec.utils.hashing import sha256_file  # noqa: E402


def _event(
    item_id: str,
    title: str,
    *,
    brand: str = "brand-a",
    cat: tuple[str, ...] = ("root-a", "leaf-a"),
    ts: int = 1,
    query: str = "prior query",
    **extra: object,
) -> dict:
    return {
        "item_id": item_id,
        "title": title,
        "brand": brand,
        "cat": list(cat),
        "event": "click",
        "query": query,
        "ts": ts,
        **extra,
    }


def _candidate(
    item_id: str,
    title: str = "candidate",
    *,
    brand: str = "candidate-brand",
    cat: tuple[str, ...] = ("candidate-root", "candidate-leaf"),
    **extra: object,
) -> dict:
    return {
        "item_id": item_id,
        "title": title,
        "brand": brand,
        "cat": list(cat),
        **extra,
    }


def _record(
    request_id: str,
    history: list[dict],
    candidates: list[dict],
    *,
    query: str = "needle query",
    ts: int = 100,
    **extra: object,
) -> dict:
    return {
        "request_id": request_id,
        "user_id": f"user-{request_id}",
        "session_id": f"session-{request_id}",
        "query": query,
        "ts": ts,
        "history": history,
        "candidates": candidates,
        **extra,
    }


class _KeywordFeatures:
    """Injectable normalized feature lookup; no BGE fixture files required."""

    def __call__(self, text: str) -> np.ndarray:
        lowered = text.casefold()
        if text == "query: needle query":
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if "rel-high" in lowered or "preserve-near" in lowered:
            return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if "rel-mid" in lowered:
            return np.asarray([0.7, 0.7, 0.0, 0.0], dtype=np.float32)
        if "break-low" in lowered:
            return np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if "break-low-two" in lowered:
            return np.asarray([-0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        if "preserve-far" in lowered:
            return np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        digest = bytes.fromhex(
            __import__("hashlib").sha256(text.encode("utf-8")).hexdigest()
        )
        value = np.asarray(
            [int(digest[index]) - 127.5 for index in range(4)], dtype=np.float32
        )
        if not bool(value.any()):
            value[0] = 1.0
        return value


def _minimal_train() -> list[dict]:
    return [
        _record(
            "train-1",
            [_event("train-donor", "unrelated donor", ts=1)],
            [_candidate("train-candidate")],
            query="train query",
            ts=10,
        )
    ]


class HistoryInterventionsTest(unittest.TestCase):
    def test_relevance_selection_and_shuffle_preserve_their_contracts(self):
        history = [
            _event(f"h{index}", f"rel-high {index}" if index in {0, 7} else f"noise {index}", ts=index + 1)
            for index in range(8)
        ]
        history[2]["title"] = "rel-mid"
        dev = [_record("dev-relevance", history, [_candidate("c1"), _candidate("c2")])]
        rows, audit = generate_history_interventions(
            _minimal_train(), dev, _KeywordFeatures()
        )

        relevant_ids = [
            row["item_id"] for row in rows["relevant_6"][0]["history"]
        ]
        irrelevant_ids = [
            row["item_id"] for row in rows["irrelevant_6"][0]["history"]
        ]
        self.assertEqual(len(relevant_ids), HISTORY_BUDGET)
        self.assertEqual(len(irrelevant_ids), HISTORY_BUDGET)
        self.assertIn("h0", relevant_ids)
        self.assertIn("h7", relevant_ids)
        self.assertEqual(
            [history.index(next(row for row in history if row["item_id"] == item_id)) for item_id in relevant_ids],
            sorted(history.index(next(row for row in history if row["item_id"] == item_id)) for item_id in relevant_ids),
        )
        self.assertNotEqual(set(relevant_ids), set(irrelevant_ids))

        recent_ids = [row["item_id"] for row in history[-HISTORY_BUDGET:]]
        shuffled_ids = [
            row["item_id"]
            for row in rows["recent_6_order_shuffle"][0]["history"]
        ]
        self.assertCountEqual(shuffled_ids, recent_ids)
        self.assertEqual(len(shuffled_ids), len(recent_ids))
        self.assertEqual(audit["integrity"]["forbidden_field_count"], 0)
        self.assertEqual(audit["integrity"]["causality_violation_count"], 0)

    def test_semantic_donors_are_excluded_causal_and_condition_specific(self):
        train_history = [
            # Invalid preserving choices: exact title, current candidate, original ID.
            _event("identical-title", "source preserve-near", brand="brand-a", ts=1),
            _event("candidate-2", "preserve-near candidate leak", brand="brand-a", ts=1),
            _event("source-1", "preserve-near original-id", brand="brand-a", ts=1),
            # The selected same-brand/category semantic neighbor is intentionally
            # later than dev so timestamp clamping is exercised.
            _event("valid-near", "preserve-near valid", brand="brand-a", ts=200),
            _event("valid-far", "preserve-far valid", brand="brand-a", ts=1),
            _event("category-fallback", "preserve-near fallback", brand="other-brand", ts=1),
            _event(
                "break-one",
                "break-low",
                brand="break-brand",
                cat=("root-b", "leaf-b"),
                ts=1,
            ),
            _event(
                "break-two",
                "break-low-two",
                brand="break-brand-two",
                cat=("root-c", "leaf-c"),
                ts=1,
            ),
        ]
        train = [
            _record(
                "train-donors",
                train_history,
                [_candidate("unused-train-candidate")],
                query="train",
                ts=300,
            )
        ]
        dev_history = [
            _event("source-1", "source preserve-near", brand="brand-a", ts=10),
            _event(
                "candidate-overlap",
                "source preserve-near overlap",
                brand="brand-a",
                ts=11,
            ),
        ]
        dev = [
            _record(
                "dev-semantic",
                dev_history,
                [_candidate("candidate-overlap"), _candidate("candidate-2")],
            )
        ]
        rows, audit = generate_history_interventions(train, dev, _KeywordFeatures())

        preserving = rows["semantic_preserving_different_id"][0]["history"]
        preserving_ids = [row["item_id"] for row in preserving]
        self.assertEqual(preserving_ids[0], "valid-near")
        self.assertEqual(preserving[0]["ts"], 99)
        self.assertFalse(
            {"candidate-2", "source-1", "candidate-overlap"}
            & set(preserving_ids)
        )
        self.assertNotEqual(preserving[0]["title"], dev_history[0]["title"])
        self.assertNotEqual(preserving[1]["title"], dev_history[1]["title"])

        breaking = rows["semantic_breaking_different_id"][0]["history"]
        for row in breaking:
            self.assertNotEqual(row["brand"], "brand-a")
            self.assertNotEqual(row["cat"][0], "root-a")

        overlap = rows["candidate_overlap_semantic_swap"][0]["history"]
        self.assertEqual(overlap[0], dev_history[0])
        self.assertNotIn(
            overlap[1]["item_id"],
            {"candidate-overlap", "candidate-2", "source-1"},
        )
        self.assertEqual(len(overlap), len(dev_history))

        self.assertEqual(audit["integrity"]["candidate_leakage_count"], 0)
        self.assertEqual(audit["integrity"]["causality_violation_count"], 0)
        self.assertGreater(
            audit["donor_audit"]["timestamp_adjusted_to_precede_target"], 0
        )
        self.assertEqual(
            audit["donor_audit"]["source_kinds"],
            {"train_history": audit["donor_audit"]["assignment_count"]},
        )

    def test_category_fallback_is_audited(self):
        train = [
            _record(
                "train-fallback",
                [
                    _event(
                        "fallback-donor",
                        "preserve-near donor",
                        brand="different-brand",
                    )
                ],
                [_candidate("train-candidate")],
                query="train",
                ts=10,
            )
        ]
        dev = [
            _record(
                "dev-fallback",
                [_event("source", "preserve-near source", brand="missing-brand")],
                [_candidate("c1"), _candidate("c2")],
            )
        ]
        rows, audit = generate_history_interventions(train, dev, _KeywordFeatures())
        self.assertEqual(
            rows["semantic_preserving_different_id"][0]["history"][0]["item_id"],
            "fallback-donor",
        )
        condition = audit["conditions"]["semantic_preserving_different_id"]
        self.assertEqual(condition["fallback_replacements"], 1)
        self.assertEqual(condition["fallback_rate"], 1.0)

    def test_label_injection_does_not_change_assignments_and_fields_are_whitelisted(self):
        train = _minimal_train()
        dev = [
            _record(
                "dev-labels",
                [_event("h1", "rel-high", clicked=1, relevance=3, label="x")],
                [
                    _candidate("c1", clicked=1, purchased=1, relevance=3),
                    _candidate("c2", label=0),
                ],
                target="c1",
            )
        ]
        clean_train = copy.deepcopy(train)
        clean_dev = copy.deepcopy(dev)
        for record in clean_train + clean_dev:
            record.pop("target", None)
            for event in record["history"]:
                for field in ("clicked", "purchased", "relevance", "label"):
                    event.pop(field, None)
            for candidate in record["candidates"]:
                for field in ("clicked", "purchased", "relevance", "label"):
                    candidate.pop(field, None)

        clean_rows, clean_audit = generate_history_interventions(
            clean_train, clean_dev, _KeywordFeatures()
        )
        injected_rows, injected_audit = generate_history_interventions(
            train, dev, _KeywordFeatures()
        )
        repeated_rows, repeated_audit = generate_history_interventions(
            train, dev, _KeywordFeatures()
        )
        self.assertEqual(clean_rows, injected_rows)
        self.assertEqual(injected_rows, repeated_rows)
        self.assertEqual(injected_audit, repeated_audit)
        self.assertEqual(clean_audit["query_candidate_binding_sha256"], injected_audit["query_candidate_binding_sha256"])
        for condition_id, assignments in injected_rows.items():
            self.assertIn(condition_id, CONDITION_IDS)
            for assignment in assignments:
                self.assertEqual(
                    set(assignment), {"request_id", "condition_id", "history"}
                )
                self.assertEqual(assignment["condition_id"], condition_id)
                self.assertLessEqual(len(assignment["history"]), HISTORY_BUDGET)
                for event in assignment["history"]:
                    self.assertLessEqual(set(event), set(HISTORY_INPUT_FIELDS))

    def test_noncausal_source_history_is_rejected(self):
        dev = [
            _record(
                "dev-noncausal",
                [_event("h", "history", ts=100)],
                [_candidate("c1"), _candidate("c2")],
                ts=100,
            )
        ]
        with self.assertRaisesRegex(ValueError, "not strictly causal"):
            generate_history_interventions(_minimal_train(), dev, _KeywordFeatures())

    def test_materialized_manifest_has_hashes_coverage_and_zero_boundary_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "records_train.jsonl"
            dev_path = root / "records_dev.jsonl"
            train = _minimal_train()
            dev = [
                _record(
                    "dev-materialize",
                    [_event("h1", "rel-high")],
                    [_candidate("c1"), _candidate("c2")],
                )
            ]
            _write_jsonl(train_path, train)
            _write_jsonl(dev_path, dev)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset_id": "kuaisearch",
                        "dataset_version": "fixture_internal_dev",
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (root / "candidate_manifest.json").write_text(
                json.dumps({"dataset_version": "fixture_internal_dev", "entries": []}),
                encoding="utf-8",
            )
            (root / "request_manifest.json").write_text(
                json.dumps({"dataset_version": "fixture_internal_dev", "entries": []}),
                encoding="utf-8",
            )
            feature_root = root / "features"
            feature_root.mkdir()
            (feature_root / "index.json").write_text("{}", encoding="utf-8")
            (feature_root / "vectors.npy").write_bytes(b"fixture")
            metadata = {
                "schema_version": 1,
                "feature_contract": "frozen_transformer_cls_l2_v1",
                "visible_text_contract": "query_context_and_canonical_item_semantics_v2",
                "qrels_read": False,
                "index_sha256": "1" * 64,
                "vectors_sha256": "2" * 64,
                "record_files": [
                    {"path": str(train_path), "sha256": sha256_file(train_path)},
                    {"path": str(dev_path), "sha256": sha256_file(dev_path)},
                ],
                "encoder_fingerprint": {"sha256": "3" * 64},
                "store_fingerprint": {"sha256": "4" * 64},
            }
            (feature_root / "metadata.json").write_text(
                json.dumps(metadata, sort_keys=True), encoding="utf-8"
            )

            output = root / "output"
            manifest = materialize_history_interventions(
                train_path,
                dev_path,
                feature_root,
                output,
                feature_lookup=_KeywordFeatures(),
            )
            self.assertEqual(manifest["seed"], MECHANISM_INTERVENTION_SEED)
            self.assertEqual(manifest["catalog_source"], "train_history_only")
            self.assertEqual(set(manifest["conditions"]), set(CONDITION_IDS))
            self.assertEqual(manifest["request_coverage"]["source_request_count"], 1)
            self.assertTrue(manifest["request_coverage"]["all_conditions_exact"])
            self.assertEqual(manifest["forbidden_field_count"], 0)
            self.assertEqual(manifest["candidate_leakage_count"], 0)
            self.assertEqual(manifest["causality_violation_count"], 0)
            self.assertFalse(manifest["qrels_read"])
            self.assertFalse(manifest["model_scores_read"])
            self.assertFalse(manifest["confirmation_records_read"])
            self.assertFalse(manifest["source_test_opened"])
            self.assertEqual(
                manifest["donor_catalog"]["source_scope"],
                "records_train.jsonl history events only",
            )
            for condition_id, entry in manifest["conditions"].items():
                path = Path(entry["path"])
                self.assertTrue(path.is_file())
                self.assertEqual(entry["sha256"], sha256_file(path))
                self.assertEqual(entry["count"], 1)
                self.assertEqual(entry["request_count"], 1)
                assignment = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    set(assignment), {"request_id", "condition_id", "history"}
                )
                self.assertEqual(assignment["condition_id"], condition_id)
            self.assertTrue((output / "manifest.json").is_file())

    def test_materializer_refuses_non_dev_population_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "records_train.jsonl"
            confirmation_path = root / "records_confirmation.jsonl"
            _write_jsonl(train_path, _minimal_train())
            _write_jsonl(confirmation_path, [])
            with self.assertRaisesRegex(ValueError, "records_dev.jsonl"):
                materialize_history_interventions(
                    train_path,
                    confirmation_path,
                    root / "missing-features",
                    root / "output",
                    feature_lookup=_KeywordFeatures(),
                )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
