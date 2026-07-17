from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.data.request_manifest import materialize_request_manifest
from myrec.eval.motivation_v12_evidence import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    audit_motivation_v12_score_bundle,
    build_motivation_v12_evidence,
    evaluate_motivation_v12_evidence,
    normalize_query_cluster,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import write_json, write_jsonl
import myrec.eval.motivation_v12_evidence as v12_evidence


def test_hand_computed_surfaces_bootstrap_and_weighted_contributions(
    tmp_path: Path,
) -> None:
    fixture = _write_bundle_fixture(tmp_path)
    report = evaluate_motivation_v12_evidence(
        analysis_run_id="analysis",
        full_run_id="full",
        null_run_id="null",
        split="dev",
        standardized_dir=fixture["standardized"],
        candidate_manifest_path=fixture["candidate_manifest"],
        activity_epsilon=0.01,
        utility_epsilon=0.0,
        label_mode="graded",
        expected_qrels_sha256=fixture["qrels_sha256"],
        protocol_path=fixture["protocol"],
        runs_dir=fixture["runs"],
        dev_eval_log_path=fixture["dev_log"],
    )

    delta = 1.0 - 1.0 / math.log2(3.0)
    assert report["bootstrap"] == {
        "cluster": "normalized_query",
        "normalization": "unicode_casefold_then_remove_all_whitespace",
        "samples": 5000,
        "seed": 20260715,
    }
    assert report["surfaces"]["overall"]["num_requests"] == 5
    assert report["surfaces"]["overall"]["num_query_clusters"] == 4
    assert report["surfaces"]["overall"]["full_minus_null_ndcg@10"][
        "mean"
    ] == pytest.approx(delta / 5.0)
    assert report["surfaces"]["recurrence"]["full_minus_null_ndcg@10"] == {
        "mean": pytest.approx(delta),
        "query_cluster_ci95": pytest.approx([delta, delta]),
    }
    assert report["surfaces"]["strict_transfer"]["full_minus_null_ndcg@10"][
        "mean"
    ] == pytest.approx(delta)
    assert report["surfaces"]["other_overlap"]["full_minus_null_ndcg@10"][
        "mean"
    ] == pytest.approx(-delta)
    assert report["surfaces"]["recurrence"]["full_minus_wrong_ndcg@10"] is None

    contributions = report["population_weighted_contributions"]
    assert contributions["all_mean"] == pytest.approx(delta / 5.0)
    assert contributions["reconstructed_mean"] == pytest.approx(delta / 5.0)
    assert contributions["surfaces"]["target_repeat"]["contribution"] == pytest.approx(
        delta / 5.0
    )
    assert contributions["surfaces"][
        "target_nonrepeat_other_candidate_overlap"
    ]["contribution"] == pytest.approx(-delta / 5.0)
    assert contributions["surfaces"][
        "target_nonrepeat_no_candidate_overlap"
    ]["contribution"] == pytest.approx(delta / 5.0)
    assert contributions["surfaces"]["target_nonrepeat_no_history"][
        "contribution"
    ] == 0.0
    assert contributions["surfaces"]["no_observed_positive"]["contribution"] == 0.0
    assert len(fixture["dev_log"].read_text(encoding="utf-8").splitlines()) == 1
    assert (fixture["runs"] / "analysis" / "metrics.json").exists()
    assert (
        fixture["runs"] / "analysis" / "pre_qrels_score_bundle_audit.json"
    ).exists()
    with (fixture["runs"] / "full" / "scores.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="scores changed after"):
        build_motivation_v12_evidence(
            analysis_run_id="analysis",
            standardized_dir=fixture["standardized"],
            runs_dir=fixture["runs"],
        )


def test_pre_qrels_audit_rejects_runtime_signature_mismatch(tmp_path: Path) -> None:
    fixture = _write_bundle_fixture(tmp_path)
    metadata_path = fixture["runs"] / "null" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["scoring_signature"]["runtime_versions"]["torch"] = "different"
    write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="scoring_signature"):
        audit_motivation_v12_score_bundle(
            full_run_id="full",
            null_run_id="null",
            split="dev",
            standardized_dir=fixture["standardized"],
            candidate_manifest_path=fixture["candidate_manifest"],
            protocol_path=fixture["protocol"],
            runs_dir=fixture["runs"],
        )


def test_pre_qrels_audit_accepts_q_scorer_nested_protocol_sha(tmp_path: Path) -> None:
    fixture = _write_bundle_fixture(tmp_path)
    protocol_sha256 = sha256_file(fixture["protocol"])
    for run_id in ("full", "null"):
        metadata_path = fixture["runs"] / run_id / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        del metadata["protocol_sha256"]
        metadata["scoring_signature"]["protocol_sha256"] = protocol_sha256
        write_json(metadata_path, metadata)

    audit = audit_motivation_v12_score_bundle(
        full_run_id="full",
        null_run_id="null",
        split="dev",
        standardized_dir=fixture["standardized"],
        candidate_manifest_path=fixture["candidate_manifest"],
        protocol_path=fixture["protocol"],
        runs_dir=fixture["runs"],
    )

    assert audit["qrels_read"] is False
    assert audit["protocol"]["sha256"] == protocol_sha256


def test_pre_qrels_audit_rejects_conflicting_protocol_sha_locations(
    tmp_path: Path,
) -> None:
    fixture = _write_bundle_fixture(tmp_path)
    for run_id in ("full", "null"):
        metadata_path = fixture["runs"] / run_id / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["scoring_signature"]["protocol_sha256"] = "0" * 64
        write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="conflicting protocol SHA"):
        audit_motivation_v12_score_bundle(
            full_run_id="full",
            null_run_id="null",
            split="dev",
            standardized_dir=fixture["standardized"],
            candidate_manifest_path=fixture["candidate_manifest"],
            protocol_path=fixture["protocol"],
            runs_dir=fixture["runs"],
        )


@pytest.mark.parametrize("nested_value", (None, "0" * 64))
def test_pre_qrels_audit_rejects_missing_or_wrong_nested_protocol_sha(
    tmp_path: Path, nested_value: str | None
) -> None:
    fixture = _write_bundle_fixture(tmp_path)
    for run_id in ("full", "null"):
        metadata_path = fixture["runs"] / run_id / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        del metadata["protocol_sha256"]
        if nested_value is None:
            metadata["scoring_signature"].pop("protocol_sha256", None)
        else:
            metadata["scoring_signature"]["protocol_sha256"] = nested_value
        write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="protocol SHA does not match"):
        audit_motivation_v12_score_bundle(
            full_run_id="full",
            null_run_id="null",
            split="dev",
            standardized_dir=fixture["standardized"],
            candidate_manifest_path=fixture["candidate_manifest"],
            protocol_path=fixture["protocol"],
            runs_dir=fixture["runs"],
        )


def test_pre_qrels_audit_rejects_a_globally_degenerate_condition(
    tmp_path: Path,
) -> None:
    fixture = _write_bundle_fixture(tmp_path)
    scores_path = fixture["runs"] / "null" / "scores.jsonl"
    rows = [
        {**json.loads(line), "score": 0.0}
        for line in scores_path.read_text(encoding="utf-8").splitlines()
    ]
    write_jsonl(scores_path, rows)
    metadata_path = fixture["runs"] / "null" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["scores_sha256"] = sha256_file(scores_path)
    metadata["score_non_degeneracy"]["nonconstant_requests_at_1e_8"] = 0
    write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="globally degenerate"):
        audit_motivation_v12_score_bundle(
            full_run_id="full",
            null_run_id="null",
            split="dev",
            standardized_dir=fixture["standardized"],
            candidate_manifest_path=fixture["candidate_manifest"],
            protocol_path=fixture["protocol"],
            runs_dir=fixture["runs"],
        )


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("candidate_manifest_sha256", "candidate manifest hash mismatch"),
        ("request_manifest_sha256", "request manifest hash mismatch"),
    ),
)
def test_pre_qrels_audit_rejects_manifest_hash_mismatch(
    tmp_path: Path, field: str, message: str
) -> None:
    fixture = _write_bundle_fixture(tmp_path)
    metadata_path = fixture["runs"] / "null" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = "not-the-frozen-hash"
    write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match=message):
        audit_motivation_v12_score_bundle(
            full_run_id="full",
            null_run_id="null",
            split="dev",
            standardized_dir=fixture["standardized"],
            candidate_manifest_path=fixture["candidate_manifest"],
            protocol_path=fixture["protocol"],
            runs_dir=fixture["runs"],
        )


def test_nonfinite_score_fails_before_malformed_qrels_is_opened(
    tmp_path: Path,
) -> None:
    fixture = _write_bundle_fixture(tmp_path)
    scores_path = fixture["runs"] / "full" / "scores.jsonl"
    rows = [
        json.loads(line)
        for line in scores_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["score"] = float("nan")
    write_jsonl(scores_path, rows)
    metadata_path = fixture["runs"] / "full" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["scores_sha256"] = sha256_file(scores_path)
    write_json(metadata_path, metadata)
    (fixture["standardized"] / "qrels_dev.jsonl").write_text(
        "this is not qrels json\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="non-finite score"):
        evaluate_motivation_v12_evidence(
            analysis_run_id="analysis",
            full_run_id="full",
            null_run_id="null",
            split="dev",
            standardized_dir=fixture["standardized"],
            candidate_manifest_path=fixture["candidate_manifest"],
            activity_epsilon=0.01,
            utility_epsilon=0.0,
            expected_qrels_sha256=fixture["qrels_sha256"],
            protocol_path=fixture["protocol"],
            runs_dir=fixture["runs"],
            dev_eval_log_path=fixture["dev_log"],
        )
    assert not (fixture["runs"] / "analysis").exists()
    assert not fixture["dev_log"].exists()


def test_population_gate_rejects_arbitrary_dataset_id_or_version(
    tmp_path: Path,
) -> None:
    fixture = _write_bundle_fixture(tmp_path)
    manifest_path = fixture["standardized"] / "manifest.json"
    write_json(
        manifest_path,
        {"dataset_id": "kuaisearch", "dataset_version": "invented_v12"},
    )
    with pytest.raises(ValueError, match="not allowlisted"):
        audit_motivation_v12_score_bundle(
            full_run_id="full",
            null_run_id="null",
            split="dev",
            standardized_dir=fixture["standardized"],
            candidate_manifest_path=fixture["candidate_manifest"],
            protocol_path=fixture["protocol"],
            runs_dir=fixture["runs"],
        )

    write_json(
        manifest_path,
        {"dataset_id": "not_kuaisearch", "dataset_version": "v1"},
    )
    with pytest.raises(ValueError, match="not allowlisted"):
        audit_motivation_v12_score_bundle(
            full_run_id="full",
            null_run_id="null",
            split="dev",
            standardized_dir=fixture["standardized"],
            candidate_manifest_path=fixture["candidate_manifest"],
            protocol_path=fixture["protocol"],
            runs_dir=fixture["runs"],
        )


def test_population_gate_rejects_data_merely_labeled_as_legacy_version(
    tmp_path: Path,
) -> None:
    fixture = _write_bundle_fixture(tmp_path)
    records_path = fixture["standardized"] / "records_dev.jsonl"
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    rows[0]["query"] = "different population wearing the frozen version label"
    write_jsonl(records_path, rows)

    with pytest.raises(ValueError, match="internal_dev records hash"):
        audit_motivation_v12_score_bundle(
            full_run_id="full",
            null_run_id="null",
            split="dev",
            standardized_dir=fixture["standardized"],
            candidate_manifest_path=fixture["candidate_manifest"],
            protocol_path=fixture["protocol"],
            runs_dir=fixture["runs"],
        )


def test_population_gate_enforces_split_and_condition_allowlists(
    tmp_path: Path,
) -> None:
    fixture = _write_bundle_fixture(tmp_path, include_wrong=True)
    with pytest.raises(ValueError, match="full/null only"):
        audit_motivation_v12_score_bundle(
            full_run_id="full",
            null_run_id="null",
            wrong_run_id="wrong",
            split="dev",
            standardized_dir=fixture["standardized"],
            candidate_manifest_path=fixture["candidate_manifest"],
            protocol_path=fixture["protocol"],
            runs_dir=fixture["runs"],
        )
    with pytest.raises(FileNotFoundError, match="records_confirmation"):
        audit_motivation_v12_score_bundle(
            full_run_id="full",
            null_run_id="null",
            split="confirmation",
            standardized_dir=fixture["standardized"],
            candidate_manifest_path=fixture["candidate_manifest"],
            protocol_path=fixture["protocol"],
            runs_dir=fixture["runs"],
        )


def test_legacy_confirmation_uses_frozen_legacy_records_and_qrels_hashes(
    tmp_path: Path,
) -> None:
    fixture = _write_bundle_fixture(tmp_path)
    _convert_fixture_to_legacy_confirmation(fixture)
    audit = audit_motivation_v12_score_bundle(
        full_run_id="full",
        null_run_id="null",
        split="confirmation",
        standardized_dir=fixture["standardized"],
        candidate_manifest_path=fixture["candidate_manifest"],
        protocol_path=fixture["protocol"],
        runs_dir=fixture["runs"],
    )
    gate = audit["population_gate"]
    assert gate["population_role"] == "legacy_compatibility"
    assert gate["conditions"] == ["full", "null"]
    assert gate["qrels_opened"] is False
    assert gate["qrels_sha256_protocol_field"] == (
        "qrels_legacy_compatibility_sha256"
    )
    assert Path(gate["records_path"]).name == "records_confirmation.jsonl"
    assert gate["verified_label_free_files"]["records"][
        "protocol_field"
    ] == "records_legacy_compatibility_sha256"


def test_evaluation_rejects_caller_qrels_override_from_protocol(
    tmp_path: Path,
) -> None:
    fixture = _write_bundle_fixture(tmp_path)
    with pytest.raises(ValueError, match="requested qrels SHA differs"):
        evaluate_motivation_v12_evidence(
            analysis_run_id="analysis",
            full_run_id="full",
            null_run_id="null",
            split="dev",
            standardized_dir=fixture["standardized"],
            candidate_manifest_path=fixture["candidate_manifest"],
            activity_epsilon=0.01,
            utility_epsilon=0.0,
            expected_qrels_sha256="f" * 64,
            protocol_path=fixture["protocol"],
            runs_dir=fixture["runs"],
            dev_eval_log_path=fixture["dev_log"],
        )
    assert not (fixture["runs"] / "analysis").exists()
    assert not fixture["dev_log"].exists()


def test_evaluation_rejects_protocol_drift_after_pre_qrels_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write_bundle_fixture(tmp_path)
    real_audit = v12_evidence.audit_motivation_v12_score_bundle

    def audit_then_drift(**kwargs: object) -> dict[str, object]:
        audit = real_audit(**kwargs)
        protocol = json.loads(fixture["protocol"].read_text(encoding="utf-8"))
        protocol["post_audit_drift"] = True
        write_json(fixture["protocol"], protocol)
        return audit

    monkeypatch.setattr(
        v12_evidence,
        "audit_motivation_v12_score_bundle",
        audit_then_drift,
    )
    with pytest.raises(ValueError, match="protocol changed after"):
        evaluate_motivation_v12_evidence(
            analysis_run_id="analysis",
            full_run_id="full",
            null_run_id="null",
            split="dev",
            standardized_dir=fixture["standardized"],
            candidate_manifest_path=fixture["candidate_manifest"],
            activity_epsilon=0.01,
            utility_epsilon=0.0,
            protocol_path=fixture["protocol"],
            runs_dir=fixture["runs"],
            dev_eval_log_path=fixture["dev_log"],
        )


def test_frozen_bootstrap_and_query_normalization_are_not_tunable(
    tmp_path: Path,
) -> None:
    assert normalize_query_cluster("  Red\tSHOE \n") == "redshoe"
    with pytest.raises(ValueError, match="bootstrap is frozen"):
        build_motivation_v12_evidence(
            analysis_run_id="missing",
            standardized_dir=tmp_path,
            runs_dir=tmp_path,
            bootstrap_samples=BOOTSTRAP_SAMPLES - 1,
            bootstrap_seed=BOOTSTRAP_SEED,
        )


def test_registered_holdout_audit_binds_qrels_free_publication_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _write_bundle_fixture(tmp_path, include_wrong=True)
    version = v12_evidence.V12_DATASET_VERSION
    records_dev = fixture["standardized"] / "records_dev.jsonl"
    (fixture["standardized"] / "records_confirmation.jsonl").write_text(
        records_dev.read_text(encoding="utf-8"), encoding="utf-8"
    )
    manifest_path = fixture["standardized"] / "manifest.json"
    write_json(
        manifest_path,
        {"dataset_id": "kuaisearch", "dataset_version": version},
    )
    candidate_path = fixture["candidate_manifest"]
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["dataset_version"] = version
    for entry in candidate["entries"]:
        entry["split"] = "confirmation"
    write_json(candidate_path, candidate)
    request_path = fixture["standardized"] / "request_manifest.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["dataset_version"] = version
    for entry in request["entries"]:
        entry["split"] = "confirmation"
    write_json(request_path, request)

    frozen = {
        "identity_manifest_sha256": "1" * 64,
        "checkpoint_id": "checkpoint-fixed",
        "config_sha256": "2" * 64,
        "implementation_digest": "3" * 64,
    }
    valid_evaluator_digest = (
        v12_evidence._current_evaluator_implementation_identity()["digest"]
    )
    holdout = {
        "analysis_selection_implementation": {
            "evaluator_digest": valid_evaluator_digest,
        },
        "checkpoint_identities": {"q_fixture": frozen},
        "integrity_lock_sha256": "4" * 64,
        "manifest_sha256": sha256_file(manifest_path),
        "post_selection_recipe_checkpoint_lock_sha256": "5" * 64,
        "post_selection_recipe_checkpoint_lock_path": str(
            tmp_path / "release_lock.json"
        ),
        "protocol_sha256": sha256_file(fixture["protocol"]),
        "qrels_opened": False,
    }
    assignment_dir = tmp_path / "assignments"
    assignment_files = {}
    for condition in ("true", "null", "wrong"):
        path = assignment_dir / f"{condition}.jsonl"
        write_jsonl(path, [{"assignment": condition, "request_id": "fixture"}])
        assignment_files[condition] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "requests": 5,
        }
    assignment_manifest = assignment_dir / "manifest.json"
    write_json(assignment_manifest, {"fixture": True})
    assignment_audit = {
        "schema_version": 1,
        "passed": True,
        "qrels_read": False,
        "model_scores_read": False,
        "manifest_path": str(assignment_manifest.resolve()),
        "manifest_sha256": sha256_file(assignment_manifest),
        "files": assignment_files,
        "request_count": 5,
        "deterministically_regenerated": True,
    }
    declared = {
        "checkpoint_identity_manifest_sha256": frozen[
            "identity_manifest_sha256"
        ],
        "checkpoint_id": frozen["checkpoint_id"],
        "integrity_lock_sha256": holdout["integrity_lock_sha256"],
        "manifest_sha256": holdout["manifest_sha256"],
        "post_selection_recipe_checkpoint_lock_sha256": holdout[
            "post_selection_recipe_checkpoint_lock_sha256"
        ],
        "protocol_sha256": holdout["protocol_sha256"],
        "qrels_opened": False,
        "verified_before_model_load": True,
    }
    for run_id in ("full", "null", "wrong"):
        metadata_path = fixture["runs"] / run_id / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "candidate_manifest_sha256": sha256_file(candidate_path),
                "config_sha256": frozen["config_sha256"],
                "dataset_version": version,
                "holdout_integrity": declared,
                "implementation_identity": {
                    "digest": frozen["implementation_digest"]
                },
                "history_assignment_manifest_path": str(assignment_manifest),
                "history_assignment_manifest_sha256": assignment_audit[
                    "manifest_sha256"
                ],
                "history_assignments_path": assignment_files[
                    "true" if run_id == "full" else run_id
                ]["path"],
                "history_assignment_sha256": assignment_files[
                    "true" if run_id == "full" else run_id
                ]["sha256"],
                "request_manifest_sha256": sha256_file(request_path),
                "protocol_sha256": sha256_file(fixture["protocol"]),
                "split": "confirmation",
            }
        )
        metadata["scoring_signature"] = {
            "head": "fixed",
            "holdout_integrity_lock_sha256": holdout[
                "integrity_lock_sha256"
            ],
            "holdout_release_lock_sha256": holdout[
                "post_selection_recipe_checkpoint_lock_sha256"
            ],
        }
        write_json(metadata_path, metadata)

    calls = []
    assignment_calls = []

    def fake_verify(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        assert kwargs.get("open_qrels") is False
        return holdout

    def fake_assignment_verify(
        *args: object, **kwargs: object
    ) -> dict[str, object]:
        assignment_calls.append((args, kwargs))
        assert kwargs["standardized_dir"] == fixture["standardized"]
        assert kwargs["release_lock_path"] == holdout[
            "post_selection_recipe_checkpoint_lock_path"
        ]
        return assignment_audit

    monkeypatch.setattr(v12_evidence, "verify_published_holdout", fake_verify)
    monkeypatch.setattr(
        v12_evidence,
        "verify_motivation_v12_history_assignments",
        fake_assignment_verify,
    )
    audit = audit_motivation_v12_score_bundle(
        full_run_id="full",
        null_run_id="null",
        wrong_run_id="wrong",
        split="confirmation",
        standardized_dir=fixture["standardized"],
        candidate_manifest_path=candidate_path,
        protocol_path=fixture["protocol"],
        runs_dir=fixture["runs"],
    )
    assert calls
    assert assignment_calls
    assert audit["holdout_integrity"]["qrels_opened"] is False
    assert audit["analysis_selection_implementation"] == holdout[
        "analysis_selection_implementation"
    ]
    assert audit["evaluator_implementation"]["files"] == sorted(
        audit["evaluator_implementation"]["files"],
        key=lambda value: value["path"],
    )

    null_metadata_path = fixture["runs"] / "null" / "metadata.json"
    null_metadata = json.loads(null_metadata_path.read_text(encoding="utf-8"))
    null_metadata["history_assignment_sha256"] = "0" * 64
    write_json(null_metadata_path, null_metadata)
    with pytest.raises(ValueError, match="not bound to the released null"):
        audit_motivation_v12_score_bundle(
            full_run_id="full",
            null_run_id="null",
            wrong_run_id="wrong",
            split="confirmation",
            standardized_dir=fixture["standardized"],
            candidate_manifest_path=candidate_path,
            protocol_path=fixture["protocol"],
            runs_dir=fixture["runs"],
        )
    null_metadata["history_assignment_sha256"] = assignment_files["null"][
        "sha256"
    ]
    write_json(null_metadata_path, null_metadata)

    holdout["analysis_selection_implementation"]["evaluator_digest"] = "0" * 64
    with pytest.raises(ValueError, match="evaluator implementation differs"):
        audit_motivation_v12_score_bundle(
            full_run_id="full",
            null_run_id="null",
            wrong_run_id="wrong",
            split="confirmation",
            standardized_dir=fixture["standardized"],
            candidate_manifest_path=candidate_path,
            protocol_path=fixture["protocol"],
            runs_dir=fixture["runs"],
        )
    holdout["analysis_selection_implementation"] = {
        "evaluator_digest": valid_evaluator_digest
    }

    metadata_path = fixture["runs"] / "null" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["holdout_integrity"]["integrity_lock_sha256"] = "f" * 64
    write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="holdout integrity mismatch"):
        audit_motivation_v12_score_bundle(
            full_run_id="full",
            null_run_id="null",
            wrong_run_id="wrong",
            split="confirmation",
            standardized_dir=fixture["standardized"],
            candidate_manifest_path=candidate_path,
            protocol_path=fixture["protocol"],
            runs_dir=fixture["runs"],
        )


def test_holdout_qrels_open_only_after_score_bundle_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    runs = tmp_path / "runs"
    standardized = tmp_path / "standardized"
    standardized.mkdir()
    locked_qrels = "a" * 64
    protocol_path = tmp_path / "protocol.yaml"
    write_json(
        protocol_path,
        {
            "protocol_id": "motivation_v1_2_first_round",
            "status": "test_frozen",
        },
    )

    def fake_score_audit(**kwargs: object) -> dict[str, object]:
        events.append("score_audit")
        return {
            "holdout_integrity": {
                "integrity_lock_sha256": "b" * 64,
                "qrels_opened": False,
            },
            "passed": True,
            "protocol": {
                "path": str(protocol_path),
                "sha256": sha256_file(protocol_path),
            },
            "qrels_read": False,
        }

    def fake_holdout_verify(*args: object, **kwargs: object) -> dict[str, object]:
        assert events == ["score_audit"]
        assert kwargs.get("open_qrels") is True
        events.append("qrels_lock_verify")
        return {
            "integrity_lock_sha256": "b" * 64,
            "verified_files": {
                "qrels_confirmation": {"sha256": locked_qrels}
            },
        }

    def fake_shared_evaluator(**kwargs: object) -> None:
        assert events == ["score_audit", "qrels_lock_verify"]
        events.append("shared_evaluator")
        (runs / str(kwargs["analysis_run_id"])).mkdir(parents=True)

    monkeypatch.setattr(
        v12_evidence, "audit_motivation_v12_score_bundle", fake_score_audit
    )
    monkeypatch.setattr(
        v12_evidence, "verify_published_holdout", fake_holdout_verify
    )
    monkeypatch.setattr(
        v12_evidence, "evaluate_history_response_runs", fake_shared_evaluator
    )
    monkeypatch.setattr(
        v12_evidence,
        "build_motivation_v12_evidence",
        lambda **kwargs: {"passed": True},
    )

    result = evaluate_motivation_v12_evidence(
        analysis_run_id="analysis",
        full_run_id="full",
        null_run_id="null",
        wrong_run_id="wrong",
        split="confirmation",
        standardized_dir=standardized,
        candidate_manifest_path=standardized / "candidate_manifest.json",
        activity_epsilon=0.01,
        utility_epsilon=0.0,
        expected_qrels_sha256=None,
        protocol_path=protocol_path,
        runs_dir=runs,
        dev_eval_log_path=tmp_path / "dev.jsonl",
    )
    assert result == {"passed": True}
    assert events == ["score_audit", "qrels_lock_verify", "shared_evaluator"]
    lock = json.loads(
        (runs / "analysis" / "qrels_hash_lock.json").read_text(encoding="utf-8")
    )
    assert lock["expected_qrels_sha256"] == locked_qrels
    assert lock["source"] == "published_holdout_integrity_lock"


def _convert_fixture_to_legacy_confirmation(fixture: dict[str, object]) -> None:
    standardized = fixture["standardized"]
    records_dev_path = standardized / "records_dev.jsonl"
    records_confirmation_path = standardized / "records_confirmation.jsonl"
    records_confirmation_path.write_bytes(records_dev_path.read_bytes())
    qrels_dev_path = standardized / "qrels_dev.jsonl"
    qrels_confirmation_path = standardized / "qrels_confirmation.jsonl"
    qrels_confirmation_path.write_bytes(qrels_dev_path.read_bytes())

    candidate_path = fixture["candidate_manifest"]
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    for entry in candidate["entries"]:
        entry["split"] = "confirmation"
    write_json(candidate_path, candidate)

    request_path = standardized / "request_manifest.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    for entry in request["entries"]:
        entry["split"] = "confirmation"
    write_json(request_path, request)

    protocol_path = fixture["protocol"]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    development = protocol["data"]["development_population"]
    development.update(
        {
            "candidate_manifest_sha256": sha256_file(candidate_path),
            "qrels_legacy_compatibility_sha256": sha256_file(
                qrels_confirmation_path
            ),
            "records_legacy_compatibility_sha256": sha256_file(
                records_confirmation_path
            ),
            "request_manifest_sha256": sha256_file(request_path),
        }
    )
    write_json(protocol_path, protocol)
    protocol_sha256 = sha256_file(protocol_path)
    for run_id in ("full", "null"):
        metadata_path = fixture["runs"] / run_id / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "candidate_manifest_sha256": sha256_file(candidate_path),
                "protocol_sha256": protocol_sha256,
                "request_manifest_sha256": sha256_file(request_path),
                "split": "confirmation",
            }
        )
        write_json(metadata_path, metadata)


def _write_bundle_fixture(tmp_path: Path, *, include_wrong: bool = False) -> dict[str, object]:
    standardized = tmp_path / "standardized"
    runs = tmp_path / "runs"
    standardized.mkdir(parents=True)
    write_json(
        standardized / "manifest.json",
        {"dataset_id": "kuaisearch", "dataset_version": "v1"},
    )
    definitions = [
        ("r_repeat", "Red Shoe", "a"),
        ("r_other", "Blue Bag", "b"),
        ("r_strict", " redshoe ", "x"),
        ("r_no_history", "Green Hat", None),
        ("r_no_positive", "Plain Coat", "x"),
    ]
    records = []
    for index, (request_id, query, history_item) in enumerate(definitions, start=1):
        history = (
            [
                {
                    "item_id": history_item,
                    "title": f"history {history_item}",
                    "event": "click",
                    "ts": index,
                }
            ]
            if history_item is not None
            else []
        )
        records.append(
            {
                "request_id": request_id,
                "user_id": f"u{index}",
                "session_id": f"s{index}",
                "ts": 100 + index,
                "query": query,
                "history": history,
                "candidates": [
                    {"item_id": "a", "title": "candidate a"},
                    {"item_id": "b", "title": "candidate b"},
                ],
                "masks": {"history_present": bool(history), "text_coverage": 1.0},
            }
        )
    records_path = standardized / "records_dev.jsonl"
    write_jsonl(records_path, records)
    candidate_manifest = standardized / "candidate_manifest.json"
    write_json(
        candidate_manifest,
        {
            "dataset_version": "v1",
            "entries": [
                {
                    "split": "dev",
                    "request_id": record["request_id"],
                    "candidate_item_ids": ["a", "b"],
                }
                for record in records
            ],
        },
    )
    materialize_request_manifest(
        [("dev", records_path)],
        standardized / "request_manifest.json",
        dataset_version="v1",
    )
    write_jsonl(
        standardized / "qrels_dev.jsonl",
        [
            {
                "request_id": record["request_id"],
                "clicked": [],
                "purchased": [],
                "relevance": (
                    {} if record["request_id"] == "r_no_positive" else {"a": 1}
                ),
            }
            for record in records
        ],
    )
    protocol_path = tmp_path / "protocol.yaml"
    write_json(
        protocol_path,
        {
            "data": {
                "dataset_id": "kuaisearch",
                "development_population": {
                    "candidate_manifest_sha256": sha256_file(candidate_manifest),
                    "dataset_version": "v1",
                    "manifest_sha256": sha256_file(
                        standardized / "manifest.json"
                    ),
                    "qrels_dev_sha256": sha256_file(
                        standardized / "qrels_dev.jsonl"
                    ),
                    "qrels_legacy_compatibility_sha256": sha256_file(
                        standardized / "qrels_dev.jsonl"
                    ),
                    "records_dev_sha256": sha256_file(records_path),
                    "records_legacy_compatibility_sha256": sha256_file(
                        records_path
                    ),
                    "request_manifest_sha256": sha256_file(
                        standardized / "request_manifest.json"
                    ),
                },
            },
            "protocol_id": "motivation_v1_2_first_round",
            "status": "fixture_frozen",
        },
    )

    favorable = {"a": 2.0, "b": 0.0}
    unfavorable = {"a": 0.0, "b": 2.0}
    full_scores = {
        "r_repeat": favorable,
        "r_other": unfavorable,
        "r_strict": favorable,
        "r_no_history": favorable,
        "r_no_positive": favorable,
    }
    null_scores = {
        "r_repeat": unfavorable,
        "r_other": favorable,
        "r_strict": unfavorable,
        "r_no_history": favorable,
        "r_no_positive": unfavorable,
    }
    condition_scores = {"full": full_scores, "null": null_scores}
    if include_wrong:
        condition_scores["wrong"] = null_scores
    candidate_hash = sha256_file(candidate_manifest)
    request_hash = sha256_file(standardized / "request_manifest.json")
    common = {
        "candidate_manifest_sha256": candidate_hash,
        "checkpoint_id": "checkpoint-fixed",
        "dataset_id": "kuaisearch",
        "dataset_version": "v1",
        "method_id": "q_fixture",
        "qrels_read": False,
        "protocol_sha256": sha256_file(protocol_path),
        "request_count": len(records),
        "request_manifest_sha256": request_hash,
        "score_non_degeneracy": {
            "max_request_range": 2.0,
            "mean_request_range": 2.0,
            "nonconstant_requests_at_1e_8": len(records),
            "threshold": 1.0e-8,
        },
        "score_rows": len(records) * 2,
        "scoring_signature": {
            "head": "fixed",
            "max_length": 32,
            "runtime_versions": {
                "python": "3.10.20",
                "torch": "2.6.0+cu124",
                "transformers": "5.12.1",
            },
        },
        "split": "dev",
    }
    for condition, score_map in condition_scores.items():
        run_dir = runs / condition
        score_rows = [
            {
                "request_id": request_id,
                "candidate_item_id": item_id,
                "score": score_map[request_id][item_id],
            }
            for request_id in [row["request_id"] for row in records]
            for item_id in ("a", "b")
        ]
        scores_path = run_dir / "scores.jsonl"
        write_jsonl(scores_path, score_rows)
        history_condition = "true" if condition == "full" else condition
        write_json(
            run_dir / "metadata.json",
            {
                **common,
                "history_assignment_sha256": f"assignment-{condition}",
                "history_condition": history_condition,
                "scores_sha256": sha256_file(scores_path),
            },
        )
    return {
        "candidate_manifest": candidate_manifest,
        "dev_log": tmp_path / "reports" / "dev_eval_log.jsonl",
        "runs": runs,
        "standardized": standardized,
        "qrels_sha256": sha256_file(standardized / "qrels_dev.jsonl"),
        "protocol": protocol_path,
    }
