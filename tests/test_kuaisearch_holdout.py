from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import myrec.data.kuaisearch_holdout as holdout_module
from myrec.data.contracts import audit_standardized_file
from myrec.data.history_assignments import (
    current_motivation_v12_assignment_implementation_identity,
    materialize_history_assignments,
    motivation_v12_assignment_recipe,
    verify_motivation_v12_history_assignments,
)
from myrec.data.kuaisearch_holdout import (
    PopulationState,
    EVALUATOR_ONLY_LABEL_AUDIT_FILENAME,
    V12_HOLDOUT_SELECTION_IMPLEMENTATION_FILES,
    _implementation_files_digest,
    _current_analysis_selection_implementation_identity,
    _cross_population_overlap_audit,
    _load_holdout_protocol,
    _load_post_selection_lock,
    materialize_motivation_v12_kuaisearch_holdout,
    verify_published_holdout,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _standardized_record(
    request_id: str,
    session_id: str,
    ts: int,
    split: str,
) -> dict:
    candidates = [
        {"item_id": f"{request_id}-a", "title": "a", "brand": "", "cat": []},
        {"item_id": f"{request_id}-b", "title": "b", "brand": "", "cat": []},
    ]
    if split == "train":
        candidates[0].update({"clicked": 1, "purchased": 0, "relevance": 1})
        candidates[1].update({"clicked": 0, "purchased": 0, "relevance": 0})
    return {
        "request_id": request_id,
        "user_id": f"user-{request_id}",
        "session_id": session_id,
        "ts": ts,
        "query": f"query {request_id}",
        "history": [],
        "candidates": candidates,
        "masks": {"history_present": False, "text_coverage": 1.0},
    }


def _write_standardized_population(
    root: Path, rows_by_split: dict[str, list[dict]], dataset_version: str
) -> None:
    candidate_entries = []
    request_entries = []
    for split, rows in rows_by_split.items():
        _write_jsonl(root / f"records_{split}.jsonl", rows)
        _write_jsonl(
            root / f"qrels_{split}.jsonl",
            [
                {
                    "request_id": row["request_id"],
                    "clicked": [row["candidates"][0]["item_id"]],
                    "purchased": [],
                    "relevance": {row["candidates"][0]["item_id"]: 1},
                }
                for row in rows
            ],
        )
        for row in rows:
            candidate_ids = [candidate["item_id"] for candidate in row["candidates"]]
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
    root.mkdir(parents=True, exist_ok=True)
    (root / "candidate_manifest.json").write_text(
        json.dumps({"dataset_version": dataset_version, "entries": candidate_entries}),
        encoding="utf-8",
    )
    (root / "request_manifest.json").write_text(
        json.dumps({"dataset_version": dataset_version, "entries": request_entries}),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps({"dataset_version": dataset_version}), encoding="utf-8"
    )


def _write_raw_source(raw: Path) -> None:
    rows = []
    item_ids = set()
    for ts in range(1, 41):
        user_id = 1_000 + ts
        candidates = [10_000 + ts, 20_000 + ts]
        clicked = [candidates[0]]
        if ts == 1:
            user_id, candidates, clicked = 1, [501, 601], [501]
        elif ts == 2:
            user_id, candidates, clicked = 2, [502, 602], [602]
        elif ts == 3:
            user_id, candidates, clicked = 3, [503, 603], [503]
        elif ts == 37:
            user_id, candidates, clicked = 1, [501, 637], [501]
        elif ts == 38:
            user_id, candidates, clicked = 2, [538, 602], [538]
        elif ts == 39:
            user_id, candidates, clicked = 3, [539, 639], [539]
        item_ids.update(candidates)
        rows.append(
            {
                "user_id": user_id,
                "session_id": f"raw-session-{ts}",
                "query": f"raw query {ts}",
                "time_index": ts,
                "impressed_item_ids": candidates,
                "clicked_item_ids": clicked,
                "purchased_item_ids": [],
                "split": "train",
            }
        )
    _write_jsonl(raw / "recall" / "train.jsonl", rows)
    with (raw / "recall" / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"split":"test","clicked_item_ids":not-json}\n')
    _write_jsonl(
        raw / "items" / "train.jsonl",
        [
            {
                "item_id": item_id,
                "item_title": f"item {item_id}",
                "brand_name": "brand",
                "category_level1_name": "cat",
            }
            for item_id in sorted(item_ids)
        ],
    )
    # A malformed sentinel proves that the materializer does not need source test.
    (raw / "recall" / "test.jsonl").write_text("not-json\n", encoding="utf-8")


def _write_protocol(path: Path, *, status: str = "pre_pilot_frozen") -> None:
    payload = {
        "schema_version": 1,
        "protocol_id": "tiny_v12_test",
        "status": status,
        "seed_policy": {"pilot_seed": 20260714},
        "common_training": {
            "checkpoint_selection": "final_finite_non_degenerate_completed_epoch"
        },
        "data": {
            "development_population": {
                "train_requests": 2,
                "internal_dev_requests": 1,
                "legacy_compatibility_requests": 1,
            },
            "new_holdout_rule": {
                "materialize_only_after_recipe_lock": True,
                "source_split": "train",
                "source_test_opened": False,
                "end_before_time_exclusive": 41,
                "selected_source_window_requests": 20,
                "confirmation_fraction": 0.20,
                "confirmation_requests": 4,
                "max_history_len": 5,
                "candidate_count": [2, 100],
                "include_history_query": True,
                "exclude_all_prior_request_and_session_populations": True,
                "selection_uses_model_outputs": False,
            },
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_release_lock(
    path: Path,
    *,
    protocol: Path,
    raw: Path,
    development: Path,
    scout: Path,
    status: str = "post_selection_frozen",
) -> None:
    method_ids = (
        "q0_qwen3_reranker_06b",
        "q1_instructrec_generalqwen",
        "q2_recranker_generalqwen",
        "q3_tallrec_generalqwen",
        "w0_copps_style_transfer_witness",
    )
    artifacts = path.parent / "frozen_artifacts"
    artifacts.mkdir(exist_ok=True)
    configs = {}
    checkpoints = {}
    protocol_sha256 = sha256_file(protocol)
    for method_id in method_ids:
        config = artifacts / f"{method_id}.config"
        config.write_text(f"config {method_id}\n", encoding="utf-8")
        config_sha256 = sha256_file(config)
        configs[method_id] = {"path": str(config), "sha256": config_sha256}
        method_dir = artifacts / method_id
        method_dir.mkdir()
        checkpoint_names = (
            ["model.pt"]
            if method_id == "w0_copps_style_transfer_witness"
            else ["config.json", "model.safetensors"]
        )
        checkpoint_files = []
        identity_checkpoint_files = []
        for checkpoint_name in checkpoint_names:
            checkpoint_artifact = method_dir / checkpoint_name
            checkpoint_artifact.write_text(
                f"inference artifact {method_id} {checkpoint_name}\n",
                encoding="utf-8",
            )
            checkpoint_file = {
                "name": checkpoint_name,
                "sha256": sha256_file(checkpoint_artifact),
                "size_bytes": checkpoint_artifact.stat().st_size,
            }
            checkpoint_files.append(checkpoint_file)
            identity_checkpoint_files.append(
                {"path": str(checkpoint_artifact), **checkpoint_file}
            )
        checkpoint_sha256 = (
            checkpoint_files[0]["sha256"]
            if method_id == "w0_copps_style_transfer_witness"
            else sha256_text(
                json.dumps(
                    checkpoint_files, sort_keys=True, separators=(",", ":")
                )
            )
        )
        checkpoint_id = f"{method_id}@{checkpoint_sha256[:20]}"
        implementation_digest = sha256_text(f"implementation {method_id}")
        training_metadata = {
            "method_id": method_id,
            "status": "completed",
            "protocol_sha256": protocol_sha256,
            "config_sha256": config_sha256,
            "seed": 20260714,
            "evidence_mode": "first_round_pilot",
            "checkpoint_id": checkpoint_id,
            "implementation_identity": {"digest": implementation_digest},
        }
        if method_id == "w0_copps_style_transfer_witness":
            training_metadata.update(
                {
                    "model_path": identity_checkpoint_files[0]["path"],
                    "model_sha256": checkpoint_files[0]["sha256"],
                }
            )
        else:
            training_metadata["checkpoint_weight_files"] = checkpoint_files
        training_metadata_path = method_dir / "training_metadata.json"
        training_metadata_path.write_text(
            json.dumps(training_metadata, sort_keys=True), encoding="utf-8"
        )
        checkpoint = artifacts / f"{method_id}.checkpoint_selection.json"
        checkpoint.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "motivation_v1_2_checkpoint_selection_identity",
                    "method_id": method_id,
                    "selection_frozen": True,
                    "selection_rule": (
                        "final_finite_non_degenerate_completed_epoch"
                    ),
                    "config_sha256": config_sha256,
                    "protocol_sha256": protocol_sha256,
                    "seed": 20260714,
                    "status": "completed",
                    "evidence_mode": "first_round_pilot",
                    "implementation_digest": implementation_digest,
                    "checkpoint_reference": str(method_dir),
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_files": identity_checkpoint_files,
                    "training_metadata_path": str(training_metadata_path),
                    "training_metadata_sha256": sha256_file(
                        training_metadata_path
                    ),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        checkpoints[method_id] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "artifact_type": "checkpoint_selection_identity_manifest",
        }
    input_paths = {
        "source_recall_train": raw / "recall" / "train.jsonl",
        "source_items_train": raw / "items" / "train.jsonl",
        **{
            f"development_{name}": development / filename
            for name, filename in {
                "manifest": "manifest.json",
                "candidate_manifest": "candidate_manifest.json",
                "request_manifest": "request_manifest.json",
                "records_train": "records_train.jsonl",
                "records_dev": "records_dev.jsonl",
                "records_confirmation": "records_confirmation.jsonl",
                "qrels_train": "qrels_train.jsonl",
                "qrels_dev": "qrels_dev.jsonl",
            }.items()
        },
        **{
            f"subsequent_scout_{name}": scout / filename
            for name, filename in {
                "manifest": "manifest.json",
                "candidate_manifest": "candidate_manifest.json",
                "request_manifest": "request_manifest.json",
                "records_train": "records_train.jsonl",
                "records_dev": "records_dev.jsonl",
            }.items()
        },
    }
    payload = {
        "schema_version": 1,
        "lock_id": "tiny_post_selection_lock",
        "status": status,
        "protocol_sha256": protocol_sha256,
        "holdout_materialization": {
            "authorized": True,
            "methods_frozen": True,
            "witness_frozen": True,
            "configs_frozen": True,
            "checkpoint_selection_frozen": True,
            "analysis_and_selection_rules_frozen": True,
            "history_assignment_recipe_and_generator_frozen": True,
        },
        "analysis_selection_implementation": (
            _current_analysis_selection_implementation_identity()
        ),
        "history_assignment_release": {
            "schema_version": 1,
            "recipe": motivation_v12_assignment_recipe(),
            "implementation": (
                current_motivation_v12_assignment_implementation_identity()
            ),
        },
        "frozen_configs": configs,
        "frozen_checkpoints": checkpoints,
        "input_sha256": {
            name: sha256_file(input_path)
            for name, input_path in input_paths.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    raw = tmp_path / "raw"
    _write_raw_source(raw)
    development = tmp_path / "development"
    retrospective_train = _standardized_record(
        "base-train-1", "base-session-1", 41, "train"
    )
    retrospective_train["user_id"] = "1"
    retrospective_train["history"] = [
        {
            "item_id": "501",
            "title": "item 501",
            "brand": "brand",
            "cat": ["cat"],
            "event": "click",
            "query": "raw query 37",
            "ts": 37,
        }
    ]
    retrospective_train["masks"]["history_present"] = True
    _write_standardized_population(
        development,
        {
            "train": [
                retrospective_train,
                _standardized_record("base-train-2", "base-session-2", 42, "train"),
            ],
            "dev": [_standardized_record("base-dev", "base-session-3", 43, "dev")],
            "confirmation": [
                _standardized_record(
                    "legacy-confirmation", "base-session-4", 44, "confirmation"
                )
            ],
        },
        "development_v11",
    )
    scout = tmp_path / "scout"
    _write_standardized_population(
        scout,
        {
            "train": [
                _standardized_record("scout-train", "scout-session-1", 45, "train")
            ],
            "dev": [_standardized_record("scout-dev", "scout-session-2", 46, "dev")],
        },
        "scout_v1",
    )
    protocol = tmp_path / "protocol.yaml"
    _write_protocol(protocol)
    lock = tmp_path / "post_selection_lock.json"
    _write_release_lock(
        lock,
        protocol=protocol,
        raw=raw,
        development=development,
        scout=scout,
    )
    return raw, development, scout, protocol, lock


def test_materializes_locked_disjoint_label_isolated_holdout(tmp_path: Path) -> None:
    raw, development, scout, protocol, lock = _fixture(tmp_path)
    output = tmp_path / "output"

    manifest = materialize_motivation_v12_kuaisearch_holdout(
        raw_dir=raw,
        development_dir=development,
        subsequent_scout_dir=scout,
        output_dir=output,
        protocol_path=protocol,
        recipe_checkpoint_lock_path=lock,
        dataset_version="tiny_v12",
        command_argv=[
            "materialize",
            "--protocol",
            str(protocol),
            "--recipe-checkpoint-lock",
            str(lock),
        ],
        enforce_registered_v12_recipe=False,
    )

    assert manifest["freeze_gate"]["protocol"]["sha256"] == sha256_file(protocol)
    assert manifest["freeze_gate"]["post_selection_recipe_checkpoint_lock"][
        "sha256"
    ] == sha256_file(lock)
    assert not (output / ".materialization_incomplete").exists()
    assert manifest["selection"]["source_window_requests"] == 20
    assert manifest["selection"]["discarded_earlier_buffer_requests"] == 16
    assert manifest["selection"]["confirmation_requests"] == 4
    assert manifest["source"]["separate_source_test_path_opened"] is False
    assert manifest["source"]["mixed_file_non_train_json_payloads_deserialized"] is False
    assert manifest["source"]["excluded_non_train_rows_inside_source_train_file"] == {
        "test": 1
    }
    assert manifest["population_isolation"]["overlap"][
        "all_request_overlaps_zero"
    ]
    assert manifest["population_isolation"]["overlap"][
        "all_session_overlaps_zero"
    ]
    assert manifest["population_isolation"]["time"]["all_boundaries_strict"]
    assert not manifest["population_isolation"]["time"]["forward_temporal_holdout"]
    assert manifest["population_isolation"][
        "retrospective_training_leakage_audit_sealed"
    ] is True
    assert "retrospective_training_leakage_audit" not in manifest[
        "population_isolation"
    ]
    sealed_payload = json.loads(
        (output / EVALUATOR_ONLY_LABEL_AUDIT_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    leakage = sealed_payload["retrospective_training_leakage_audit"]
    assert leakage["overlapping_users"] == 1
    assert leakage["holdout_requests_with_event_in_train_history"] == 1
    assert "retrospective" in manifest["scope_warning"]
    assert ".confirmation-" not in json.dumps(manifest)

    assert "materializer_only_power_audit" not in manifest
    power = sealed_payload["materializer_only_power_audit"]
    assert power["observed_positive"] == 4
    assert power["target_repeat"] == 1
    assert power["target_nonrepeat_other_candidate_overlap"] == 1
    assert power["target_nonrepeat_no_candidate_overlap"] == 1
    assert power["target_nonrepeat_no_history"] == 1
    assert power["aggregate_only"] is True
    assert sealed_payload["access_policy"][
        "model_and_scorer_access_forbidden"
    ] is True
    public_serialization = json.dumps(manifest, sort_keys=True)
    for forbidden in (
        "observed_positive",
        "target_repeat",
        "holdout_positive_event_instances",
        "confirmation_with_click",
        "confirmation_with_purchase",
    ):
        assert forbidden not in public_serialization

    assert (output / "records_train.jsonl").read_bytes() == (
        development / "records_train.jsonl"
    ).read_bytes()
    assert (output / "records_dev.jsonl").read_bytes() == (
        development / "records_dev.jsonl"
    ).read_bytes()
    confirmation_rows = list(iter_jsonl(output / "records_confirmation.jsonl"))
    assert len(confirmation_rows) == 4
    assert all(
        not ({"clicked", "purchased", "relevance"} & set(candidate))
        for row in confirmation_rows
        for candidate in row["candidates"]
    )
    assert all(
        event["ts"] < row["ts"]
        for row in confirmation_rows
        for event in row["history"]
    )
    assert len(list(iter_jsonl(output / "qrels_confirmation.jsonl"))) == 4
    integrity_lock = json.loads(
        (output / "confirmation_integrity_lock.json").read_text(encoding="utf-8")
    )
    assert integrity_lock["files"]["qrels_confirmation"]["sha256"] == sha256_file(
        output / "qrels_confirmation.jsonl"
    )
    assert integrity_lock["scope_warning"] == manifest["scope_warning"]
    assert integrity_lock["retrospective_time_direction"]["direction"] == (
        "retrospective_confirmation_before_training"
    )
    assert integrity_lock["retrospective_time_direction"][
        "forward_temporal_holdout"
    ] is False
    assert "retrospective_training_leakage_audit" not in integrity_lock
    assert integrity_lock["evaluator_only_sealed_label_audit"] == manifest[
        "evaluator_only_sealed_label_audit"
    ]
    assert integrity_lock["evaluator_only_sealed_label_audit"][
        "open_after_score_audit_only"
    ] is True
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == (
        manifest
    )
    assert manifest["freeze_gate"][
        "pre_publish_full_revalidation_completed"
    ] is True
    assert manifest["outputs"][
        "published_atomically_by_single_directory_rename"
    ] is True
    assert audit_standardized_file(
        output / "records_confirmation.jsonl", "confirmation"
    )["request_count"] == 4

    candidate_manifest = json.loads(
        (output / "candidate_manifest.json").read_text(encoding="utf-8")
    )
    request_manifest = json.loads(
        (output / "request_manifest.json").read_text(encoding="utf-8")
    )
    assert len(candidate_manifest["entries"]) == 7
    assert len(request_manifest["entries"]) == 7
    assert {entry["split"] for entry in candidate_manifest["entries"]} == {
        "train",
        "dev",
        "confirmation",
    }
    assert all(
        entry["request_id"] != "legacy-confirmation"
        for entry in candidate_manifest["entries"]
    )


def test_new_holdout_assignments_are_release_bound_and_reproducible(
    tmp_path: Path,
) -> None:
    raw, development, scout, protocol, lock = _fixture(tmp_path)
    output = tmp_path / "assignment_holdout"
    materialize_motivation_v12_kuaisearch_holdout(
        raw_dir=raw,
        development_dir=development,
        subsequent_scout_dir=scout,
        output_dir=output,
        protocol_path=protocol,
        recipe_checkpoint_lock_path=lock,
        dataset_version="tiny_v12",
        enforce_registered_v12_recipe=False,
    )
    assignments = tmp_path / "assignments"
    report = materialize_history_assignments(
        output / "records_confirmation.jsonl",
        assignments,
        assignments / "assignment_report.json",
        donor_records_path=development / "records_train.jsonl",
        seed=20260714,
        global_donor_shortlist_size=512,
        motivation_v12_release_lock_path=lock,
        motivation_v12_enforce_registered_recipe=False,
    )
    assert report["conditions"] == ["full", "null", "wrong"]
    assert report["condition_assignment_names"] == {
        "full": "true",
        "null": "null",
        "wrong": "wrong",
    }
    assert report["donor_pool_requests"] == 6
    binding = report["motivation_v12_release_binding"]
    assert binding["external_donor_records_role"] == "development_records_train"
    assert binding["target_population_included_in_wrong_user_donor_pool"] is True
    assert binding["recipe"]["seed"] == 20260714
    assert binding["recipe"]["global_donor_shortlist_size"] == 512
    assert binding["post_selection_release_lock_sha256"] == sha256_file(lock)
    verified = verify_motivation_v12_history_assignments(
        assignments / "manifest.json",
        standardized_dir=output,
        release_lock_path=lock,
        enforce_registered_v12_recipe=False,
    )
    assert verified["passed"] is True
    assert verified["deterministically_regenerated"] is True
    assert set(verified["files"]) == {"true", "null", "wrong"}

    wrong_path = assignments / "wrong.jsonl"
    manifest_path = assignments / "manifest.json"
    original_wrong = wrong_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    wrong_rows = list(iter_jsonl(wrong_path))
    wrong_rows[0]["donor_request_id"] = "posthoc-replacement"
    _write_jsonl(wrong_path, wrong_rows)
    tampered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered_manifest["files"]["wrong"]["sha256"] = sha256_file(wrong_path)
    manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="deterministic frozen generation"):
        verify_motivation_v12_history_assignments(
            manifest_path,
            standardized_dir=output,
            release_lock_path=lock,
            enforce_registered_v12_recipe=False,
        )
    wrong_path.write_bytes(original_wrong)
    manifest_path.write_bytes(original_manifest)

    for name, kwargs, message in (
        ("wrong_seed", {"seed": 7}, "seed differs"),
        (
            "wrong_shortlist",
            {"global_donor_shortlist_size": 511},
            "shortlist differs",
        ),
    ):
        rejected = tmp_path / name
        with pytest.raises(ValueError, match=message):
            materialize_history_assignments(
                output / "records_confirmation.jsonl",
                rejected,
                rejected / "report.json",
                donor_records_path=development / "records_train.jsonl",
                motivation_v12_release_lock_path=lock,
                motivation_v12_enforce_registered_recipe=False,
                **kwargs,
            )
        assert not rejected.exists()

    wrong_donor_output = tmp_path / "wrong_donor"
    with pytest.raises(ValueError, match="external donor must be development"):
        materialize_history_assignments(
            output / "records_confirmation.jsonl",
            wrong_donor_output,
            wrong_donor_output / "report.json",
            donor_records_path=output / "records_train.jsonl",
            motivation_v12_release_lock_path=lock,
            motivation_v12_enforce_registered_recipe=False,
        )
    assert not wrong_donor_output.exists()


def test_rejects_missing_or_nonfrozen_post_selection_lock(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="post-selection"):
        _load_post_selection_lock(
            tmp_path / "missing.json", protocol_sha256="0" * 64
        )

    lock = tmp_path / "draft.json"
    lock.write_text(
        json.dumps(
            {"schema_version": 1, "lock_id": "draft_lock", "status": "draft"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be final"):
        _load_post_selection_lock(lock, protocol_sha256="0" * 64)


def test_published_verifier_is_qrels_free_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, development, scout, protocol, lock = _fixture(tmp_path)
    output = tmp_path / "verified_output"
    materialize_motivation_v12_kuaisearch_holdout(
        raw_dir=raw,
        development_dir=development,
        subsequent_scout_dir=scout,
        output_dir=output,
        protocol_path=protocol,
        recipe_checkpoint_lock_path=lock,
        dataset_version="tiny_v12",
        enforce_registered_v12_recipe=False,
    )
    original_sha256_file = holdout_module.sha256_file
    original_read_json = holdout_module._read_json

    def reject_restricted_hash(path: Path) -> str:
        path = Path(path)
        if path.name.startswith("qrels_") or (
            path.name == EVALUATOR_ONLY_LABEL_AUDIT_FILENAME
        ):
            raise AssertionError(f"restricted artifact was opened by default: {path}")
        return original_sha256_file(path)

    def reject_sealed_deserialization(path: Path) -> dict:
        if Path(path).name == EVALUATOR_ONLY_LABEL_AUDIT_FILENAME:
            raise AssertionError(f"sealed audit was deserialized by default: {path}")
        return original_read_json(path)

    monkeypatch.setattr(holdout_module, "sha256_file", reject_restricted_hash)
    monkeypatch.setattr(holdout_module, "_read_json", reject_sealed_deserialization)
    audit = verify_published_holdout(
        output,
        protocol_path=protocol,
        recipe_checkpoint_lock_path=lock,
        open_qrels=False,
        enforce_registered_v12_recipe=False,
    )
    assert audit["passed"] is True
    assert audit["qrels_opened"] is False
    assert audit["qrels_audits"] is None
    assert audit["evaluator_only_sealed_label_audit_opened"] is False
    assert audit["evaluator_only_sealed_label_audit"] is None
    assert set(audit["checkpoint_identities"]) == {
        "q0_qwen3_reranker_06b",
        "q1_instructrec_generalqwen",
        "q2_recranker_generalqwen",
        "q3_tallrec_generalqwen",
        "w0_copps_style_transfer_witness",
    }
    q0_identity = audit["checkpoint_identities"]["q0_qwen3_reranker_06b"]
    assert q0_identity["checkpoint_id"].startswith(
        "q0_qwen3_reranker_06b@"
    )
    assert len(q0_identity["checkpoint_files"]) == 2
    assert q0_identity["status"] == "completed"
    assert q0_identity["evidence_mode"] == "first_round_pilot"
    assert all(
        len(q0_identity[field]) == 64
        for field in (
            "checkpoint_sha256",
            "config_sha256",
            "training_metadata_sha256",
            "implementation_digest",
        )
    )
    assert set(audit["verified_files"]).isdisjoint(
        {"qrels_train", "qrels_dev", "qrels_confirmation"}
    )

    monkeypatch.setattr(holdout_module, "sha256_file", original_sha256_file)
    monkeypatch.setattr(holdout_module, "_read_json", original_read_json)
    label_audit = verify_published_holdout(
        output,
        protocol_path=protocol,
        recipe_checkpoint_lock_path=lock,
        open_qrels=True,
        enforce_registered_v12_recipe=False,
    )
    assert label_audit["qrels_opened"] is True
    assert label_audit["qrels_audits"]["confirmation"]["request_count"] == 4
    assert label_audit["evaluator_only_sealed_label_audit_opened"] is True
    assert label_audit["evaluator_only_sealed_label_audit"][
        "access_policy_verified"
    ] is True
    assert label_audit["evaluator_only_sealed_label_audit"]["payload"][
        "materializer_only_power_audit"
    ]["target_repeat"] == 1

    sealed_path = output / EVALUATOR_ONLY_LABEL_AUDIT_FILENAME
    original_sealed_bytes = sealed_path.read_bytes()
    sealed_path.write_text("not-json\n", encoding="utf-8")
    sealed_still_qrels_free = verify_published_holdout(
        output,
        protocol_path=protocol,
        recipe_checkpoint_lock_path=lock,
        open_qrels=False,
        enforce_registered_v12_recipe=False,
    )
    assert sealed_still_qrels_free["passed"] is True
    with pytest.raises(
        ValueError, match="evaluator_only_sealed_label_audit differs"
    ):
        verify_published_holdout(
            output,
            protocol_path=protocol,
            recipe_checkpoint_lock_path=lock,
            open_qrels=True,
            enforce_registered_v12_recipe=False,
        )
    sealed_path.write_bytes(original_sealed_bytes)

    (output / "qrels_confirmation.jsonl").write_text(
        "not-json\n", encoding="utf-8"
    )
    still_qrels_free = verify_published_holdout(
        output,
        protocol_path=protocol,
        recipe_checkpoint_lock_path=lock,
        open_qrels=False,
        enforce_registered_v12_recipe=False,
    )
    assert still_qrels_free["passed"] is True
    with pytest.raises(ValueError, match="qrels_confirmation differs"):
        verify_published_holdout(
            output,
            protocol_path=protocol,
            recipe_checkpoint_lock_path=lock,
            open_qrels=True,
            enforce_registered_v12_recipe=False,
        )


def test_published_verifier_rejects_dataset_identity_drift(tmp_path: Path) -> None:
    raw, development, scout, protocol, lock = _fixture(tmp_path)
    output = tmp_path / "dataset_identity_output"
    materialize_motivation_v12_kuaisearch_holdout(
        raw_dir=raw,
        development_dir=development,
        subsequent_scout_dir=scout,
        output_dir=output,
        protocol_path=protocol,
        recipe_checkpoint_lock_path=lock,
        dataset_version="tiny_v12",
        enforce_registered_v12_recipe=False,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_id"] = "other"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_id must equal kuaisearch"):
        verify_published_holdout(
            output,
            protocol_path=protocol,
            recipe_checkpoint_lock_path=lock,
            enforce_registered_v12_recipe=False,
        )

    manifest["dataset_id"] = "kuaisearch"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="registered holdout dataset_version"):
        verify_published_holdout(
            output,
            protocol_path=protocol,
            recipe_checkpoint_lock_path=lock,
            enforce_registered_v12_recipe=True,
        )


def test_release_identity_rejects_recomputed_holdout_source_tampering(
    tmp_path: Path,
) -> None:
    raw, development, scout, protocol, lock = _fixture(tmp_path)
    del raw, development, scout
    payload = json.loads(lock.read_text(encoding="utf-8"))
    identity = payload["analysis_selection_implementation"]
    holdout = identity["holdout_selection"]
    holdout["files"][0]["sha256"] = "0" * 64
    holdout["digest"] = _implementation_files_digest(holdout["files"])
    identity["canonical_digest"] = sha256_text(
        json.dumps(
            {
                "evaluator": identity["evaluator"]["digest"],
                "holdout_selection": holdout["digest"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    lock.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="current holdout/selection implementation differs",
    ):
        _load_post_selection_lock(
            lock,
            protocol_sha256=sha256_file(protocol),
            checkpoint_selection_rule=(
                "final_finite_non_degenerate_completed_epoch"
            ),
        )
    assert set(
        entry["path"] for entry in holdout["files"]
    ) == set(V12_HOLDOUT_SELECTION_IMPLEMENTATION_FILES)


def test_verifier_rejects_label_statistics_moved_into_public_manifest(
    tmp_path: Path,
) -> None:
    raw, development, scout, protocol, lock = _fixture(tmp_path)
    output = tmp_path / "public_label_leak"
    materialize_motivation_v12_kuaisearch_holdout(
        raw_dir=raw,
        development_dir=development,
        subsequent_scout_dir=scout,
        output_dir=output,
        protocol_path=protocol,
        recipe_checkpoint_lock_path=lock,
        dataset_version="tiny_v12",
        enforce_registered_v12_recipe=False,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_writer_audit"]["target_repeat"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="exposes sealed label-derived field"):
        verify_published_holdout(
            output,
            protocol_path=protocol,
            recipe_checkpoint_lock_path=lock,
            open_qrels=False,
            enforce_registered_v12_recipe=False,
        )


def test_pre_pilot_protocol_still_requires_a_separate_release(tmp_path: Path) -> None:
    protocol = tmp_path / "pre_pilot.yaml"
    _write_protocol(protocol, status="pre_pilot_frozen")
    loaded = _load_holdout_protocol(
        protocol, enforce_registered_v12_recipe=False
    )
    assert loaded["payload"]["status"] == "pre_pilot_frozen"
    with pytest.raises(FileNotFoundError, match="post-selection"):
        _load_post_selection_lock(
            tmp_path / "not_created.json", protocol_sha256=loaded["sha256"]
        )


def test_cli_recipe_enforcement_rejects_alternate_holdout_size(tmp_path: Path) -> None:
    protocol = tmp_path / "tiny.yaml"
    _write_protocol(protocol)
    with pytest.raises(ValueError, match="registered Motivation V1.2"):
        _load_holdout_protocol(protocol, enforce_registered_v12_recipe=True)


def test_rejects_source_changed_after_post_selection_lock(tmp_path: Path) -> None:
    raw, development, scout, protocol, lock = _fixture(tmp_path)
    with (raw / "items" / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="input SHA mismatch: source_items_train"):
        materialize_motivation_v12_kuaisearch_holdout(
            raw_dir=raw,
            development_dir=development,
            subsequent_scout_dir=scout,
            output_dir=tmp_path / "must_not_exist",
            protocol_path=protocol,
            recipe_checkpoint_lock_path=lock,
            dataset_version="tiny_v12",
            enforce_registered_v12_recipe=False,
        )
    assert not (tmp_path / "must_not_exist").exists()


def test_cross_population_session_overlap_is_a_hard_failure(tmp_path: Path) -> None:
    def population(name: str, request_id: str) -> PopulationState:
        return PopulationState(
            name=name,
            split="confirmation",
            path=tmp_path / f"{name}.jsonl",
            request_ids=frozenset({request_id}),
            session_ids=frozenset({"shared-session"}),
            records={request_id: ("q", ("a", "b"))},
            time_min=1,
            time_max=1,
        )

    with pytest.raises(ValueError, match="identity overlap"):
        _cross_population_overlap_audit(
            [population("left", "left-request"), population("right", "right-request")]
        )


def test_crash_while_writing_full_manifest_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, development, scout, protocol, lock = _fixture(tmp_path)
    output = tmp_path / "crash_output"
    original_write_json = holdout_module.write_json

    def crash_on_full_manifest(path: Path, value: object) -> None:
        path = Path(path)
        if path.name == "manifest.json" and path.parent.name == "assembled_output":
            raise RuntimeError("injected manifest crash")
        original_write_json(path, value)

    monkeypatch.setattr(holdout_module, "write_json", crash_on_full_manifest)
    with pytest.raises(RuntimeError, match="injected manifest crash"):
        materialize_motivation_v12_kuaisearch_holdout(
            raw_dir=raw,
            development_dir=development,
            subsequent_scout_dir=scout,
            output_dir=output,
            protocol_path=protocol,
            recipe_checkpoint_lock_path=lock,
            dataset_version="tiny_v12",
            enforce_registered_v12_recipe=False,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.confirmation-*"))


def test_atomic_publish_receives_only_complete_marker_free_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, development, scout, protocol, lock = _fixture(tmp_path)
    output = tmp_path / "atomic_output"
    observed = {}

    def inspect_and_stop(staged_dir: Path, target_dir: Path) -> None:
        observed["called"] = True
        assert target_dir == output
        assert not target_dir.exists()
        assert (staged_dir / "manifest.json").is_file()
        assert (staged_dir / "confirmation_integrity_lock.json").is_file()
        assert (staged_dir / EVALUATOR_ONLY_LABEL_AUDIT_FILENAME).is_file()
        assert not (staged_dir / ".materialization_incomplete").exists()
        raise RuntimeError("stop before rename")

    monkeypatch.setattr(
        holdout_module, "_atomic_publish_directory", inspect_and_stop
    )
    with pytest.raises(RuntimeError, match="stop before rename"):
        materialize_motivation_v12_kuaisearch_holdout(
            raw_dir=raw,
            development_dir=development,
            subsequent_scout_dir=scout,
            output_dir=output,
            protocol_path=protocol,
            recipe_checkpoint_lock_path=lock,
            dataset_version="tiny_v12",
            enforce_registered_v12_recipe=False,
        )

    assert observed == {"called": True}
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.confirmation-*"))


@pytest.mark.parametrize(
    ("mutation_target", "error_match"),
    [
        ("source", "input SHA mismatch: source_items_train"),
        ("development", "input SHA mismatch: development_records_dev"),
        ("scout", "input SHA mismatch: subsequent_scout_records_dev"),
        ("config", "frozen artifact SHA mismatch"),
        ("checkpoint", "checkpoint selection identity SHA mismatch"),
        ("checkpoint_artifact", "checkpoint file SHA mismatch"),
        ("training_metadata", "checkpoint training metadata SHA mismatch"),
        ("protocol", "protocol changed during holdout materialization"),
        ("release_lock", "post-selection lock changed during holdout materialization"),
    ],
)
def test_last_revalidation_blocks_dependency_mutation_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_target: str,
    error_match: str,
) -> None:
    raw, development, scout, protocol, lock = _fixture(tmp_path)
    output = tmp_path / f"mutated_{mutation_target}"
    lock_payload = json.loads(lock.read_text(encoding="utf-8"))
    method_id = "q0_qwen3_reranker_06b"
    checkpoint_identity_path = Path(
        lock_payload["frozen_checkpoints"][method_id]["path"]
    )
    checkpoint_identity = json.loads(
        checkpoint_identity_path.read_text(encoding="utf-8")
    )
    mutation_paths = {
        "source": raw / "items" / "train.jsonl",
        "development": development / "records_dev.jsonl",
        "scout": scout / "records_dev.jsonl",
        "config": Path(lock_payload["frozen_configs"][method_id]["path"]),
        "checkpoint": checkpoint_identity_path,
        "checkpoint_artifact": Path(
            checkpoint_identity["checkpoint_files"][0]["path"]
        ),
        "training_metadata": Path(
            checkpoint_identity["training_metadata_path"]
        ),
        "protocol": protocol,
        "release_lock": lock,
    }
    original_write_json = holdout_module.write_json
    mutated = False

    def mutate_after_full_manifest(path: Path, value: object) -> None:
        nonlocal mutated
        path = Path(path)
        original_write_json(path, value)
        if (
            not mutated
            and path.name == "manifest.json"
            and path.parent.name == "assembled_output"
        ):
            original = mutation_paths[mutation_target].read_text(encoding="utf-8")
            replacement = (
                ("X" if original[:1] != "X" else "Y") + original[1:]
                if mutation_target == "checkpoint_artifact"
                else original + "\n"
            )
            mutation_paths[mutation_target].write_text(
                replacement, encoding="utf-8"
            )
            mutated = True

    monkeypatch.setattr(
        holdout_module, "write_json", mutate_after_full_manifest
    )
    with pytest.raises(ValueError, match=error_match):
        materialize_motivation_v12_kuaisearch_holdout(
            raw_dir=raw,
            development_dir=development,
            subsequent_scout_dir=scout,
            output_dir=output,
            protocol_path=protocol,
            recipe_checkpoint_lock_path=lock,
            dataset_version="tiny_v12",
            enforce_registered_v12_recipe=False,
        )

    assert mutated
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.confirmation-*"))


@pytest.mark.parametrize(
    ("field", "value", "error_match"),
    [
        ("schema_version", 2, "schema_version must equal 1"),
        ("lock_id", "  ", "lock_id must be non-empty"),
    ],
)
def test_post_selection_lock_requires_schema_and_identity(
    tmp_path: Path, field: str, value: object, error_match: str
) -> None:
    raw, development, scout, protocol, lock = _fixture(tmp_path)
    del raw, development, scout
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload[field] = value
    lock.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=error_match):
        _load_post_selection_lock(
            lock,
            protocol_sha256=sha256_file(protocol),
            checkpoint_selection_rule=(
                "final_finite_non_degenerate_completed_epoch"
            ),
        )


@pytest.mark.parametrize(
    ("identity_field", "identity_value"),
    [
        ("method_id", "wrong_method"),
        ("selection_rule", "best_observed_dev_metric"),
        ("selection_frozen", False),
        ("config_sha256", "0" * 64),
    ],
)
def test_checkpoint_selection_identity_semantics_are_locked(
    tmp_path: Path, identity_field: str, identity_value: object
) -> None:
    raw, development, scout, protocol, lock = _fixture(tmp_path)
    del raw, development, scout
    payload = json.loads(lock.read_text(encoding="utf-8"))
    method_id = "q0_qwen3_reranker_06b"
    artifact = payload["frozen_checkpoints"][method_id]
    identity_path = Path(artifact["path"])
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity[identity_field] = identity_value
    identity_path.write_text(json.dumps(identity, sort_keys=True), encoding="utf-8")
    artifact["sha256"] = sha256_file(identity_path)
    lock.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint selection identity mismatch"):
        _load_post_selection_lock(
            lock,
            protocol_sha256=sha256_file(protocol),
            checkpoint_selection_rule=(
                "final_finite_non_degenerate_completed_epoch"
            ),
        )
