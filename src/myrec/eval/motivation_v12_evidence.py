"""Shared-evaluator evidence aggregation for Motivation V1.2.

The pre-evaluation audit in this module is deliberately qrels-free.  Ranking
metrics and target-aware surfaces are produced only by
``history_response_evaluator``; the aggregation layer consumes those frozen
outputs and never implements a method-owned metric path.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from myrec.data.contracts import audit_standardized_file
from myrec.data.history_assignments import (
    verify_motivation_v12_history_assignments,
)
from myrec.data.kuaisearch_holdout import (
    V12_DATASET_VERSION,
    verify_published_holdout,
)
from myrec.eval.controlled_composition import (
    cluster_bootstrap_mean_ci,
    summarize_partition_contributions,
)
from myrec.eval.history_response import aggregate_history_response
from myrec.eval.history_response_evaluator import (
    COUNTERFACTUAL_IDENTITY_KEYS,
    _assert_counterfactual_identity,
    _assert_score_coverage,
    _load_candidates,
    _load_run,
    evaluate_history_response_runs,
)
from myrec.eval.target_aware_surfaces import ALL_REQUEST_PARTITION
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl, write_json


BOOTSTRAP_CLUSTER = "normalized_query"
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20260715
FROZEN_ACTIVITY_EPSILON = 0.01
FROZEN_UTILITY_EPSILON = 0.0
FROZEN_LABEL_MODE = "graded"
SCORE_NONDEGENERACY_EPSILON = 1.0e-8

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL_PATH = (
    REPOSITORY_ROOT / "experiments" / "motivation_v1_2" / "protocol.yaml"
)
_EVALUATOR_IMPLEMENTATION_PATHS = (
    "src/myrec/eval/controlled_composition.py",
    "src/myrec/eval/history_response.py",
    "src/myrec/eval/history_response_evaluator.py",
    "src/myrec/eval/motivation_v12_evidence.py",
    "src/myrec/eval/target_aware_surfaces.py",
)

SURFACE_ROLES = {
    "overall": "all",
    "recurrence": "target_repeat",
    "strict_transfer": "target_nonrepeat_no_candidate_overlap",
    "other_overlap": "target_nonrepeat_other_candidate_overlap",
    "nonrepeat_no_history": "target_nonrepeat_no_history",
    "no_observed_positive": "no_observed_positive",
}


def audit_motivation_v12_score_bundle(
    *,
    full_run_id: str,
    null_run_id: str,
    split: str,
    standardized_dir: str | Path,
    candidate_manifest_path: str | Path,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    wrong_run_id: str | None = None,
    runs_dir: str | Path = "runs",
) -> dict[str, Any]:
    """Audit a fixed-checkpoint score bundle without opening any qrels file."""

    if split not in {"dev", "confirmation"}:
        raise ValueError("Motivation V1.2 evidence supports dev or confirmation only")
    standardized_dir = Path(standardized_dir)
    candidate_manifest_path = Path(candidate_manifest_path)
    request_manifest_path = standardized_dir / "request_manifest.json"
    dataset_manifest_path = standardized_dir / "manifest.json"
    runs_dir = Path(runs_dir)

    protocol, protocol_audit = _load_frozen_protocol(protocol_path)
    dataset_manifest = _read_json(dataset_manifest_path)
    population_gate = _gate_population_before_qrels(
        protocol=protocol,
        protocol_audit=protocol_audit,
        dataset_manifest=dataset_manifest,
        dataset_manifest_path=dataset_manifest_path,
        standardized_dir=standardized_dir,
        candidate_manifest_path=candidate_manifest_path,
        request_manifest_path=request_manifest_path,
        split=split,
        wrong_run_id=wrong_run_id,
    )
    records_path = Path(population_gate["records_path"])
    records_audit = audit_standardized_file(records_path, split)
    candidates = _load_candidates(candidate_manifest_path, split)
    candidate_sha256 = sha256_file(candidate_manifest_path)
    request_sha256 = sha256_file(request_manifest_path)
    request_identity = _audit_request_identity(
        records_path=records_path,
        candidate_manifest_path=candidate_manifest_path,
        request_manifest_path=request_manifest_path,
        split=split,
        candidates=candidates,
    )

    bundle = {
        "true": _load_run(
            runs_dir, full_run_id, "true", candidate_sha256, request_sha256
        ),
        "null": _load_run(
            runs_dir, null_run_id, "null", candidate_sha256, request_sha256
        ),
    }
    if wrong_run_id is not None:
        bundle["wrong"] = _load_run(
            runs_dir, wrong_run_id, "wrong", candidate_sha256, request_sha256
        )
    _assert_counterfactual_identity(bundle)

    expected_score_rows = sum(len(item_ids) for item_ids in candidates.values())
    score_files: dict[str, Any] = {}
    method_ids: set[str] = set()
    for condition, values in bundle.items():
        metadata = values["metadata"]
        if metadata.get("qrels_read") is not False:
            raise ValueError(
                f"score run {values['run_id']} must declare qrels_read=false"
            )
        _assert_score_coverage(candidates, values["scores"])
        actual_rows = sum(len(items) for items in values["scores"].values())
        if metadata.get("request_count", len(candidates)) != len(candidates):
            raise ValueError(f"score run {values['run_id']} request_count mismatch")
        if metadata.get("score_rows", expected_score_rows) != expected_score_rows:
            raise ValueError(f"score run {values['run_id']} score_rows mismatch")
        if actual_rows != expected_score_rows:
            raise AssertionError("complete score coverage row count failed")
        request_ranges = [
            max(item_scores.values()) - min(item_scores.values())
            for item_scores in values["scores"].values()
        ]
        nonconstant_requests = sum(
            value > SCORE_NONDEGENERACY_EPSILON for value in request_ranges
        )
        if nonconstant_requests == 0:
            raise ValueError(
                f"score run {values['run_id']} is globally degenerate at the "
                "frozen 1e-8 threshold"
            )
        declared_nondegeneracy = metadata.get("score_non_degeneracy", {})
        if (
            float(declared_nondegeneracy.get("threshold", float("nan")))
            != SCORE_NONDEGENERACY_EPSILON
            or int(
                declared_nondegeneracy.get("nonconstant_requests_at_1e_8", -1)
            )
            != nonconstant_requests
        ):
            raise ValueError(
                f"score run {values['run_id']} non-degeneracy metadata mismatch"
            )
        declared_scores_sha256 = metadata.get("scores_sha256")
        if (
            declared_scores_sha256 is not None
            and declared_scores_sha256 != values["scores_sha256"]
        ):
            raise ValueError(f"score run {values['run_id']} scores_sha256 mismatch")
        if not metadata.get("method_id"):
            raise ValueError(f"score run {values['run_id']} is missing method_id")
        if _score_protocol_sha256(metadata, values["run_id"]) != protocol_audit[
            "sha256"
        ]:
            raise ValueError(
                f"score run {values['run_id']} protocol SHA does not match the "
                "current frozen protocol"
            )
        method_ids.add(str(metadata["method_id"]))
        score_path = runs_dir / values["run_id"] / "scores.jsonl"
        metadata_path = runs_dir / values["run_id"] / "metadata.json"
        score_files[_report_condition(condition)] = {
            "history_assignment_sha256": metadata["history_assignment_sha256"],
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "request_count": len(values["scores"]),
            "nonconstant_requests_at_1e_8": nonconstant_requests,
            "run_id": values["run_id"],
            "score_rows": actual_rows,
            "scores_path": str(score_path),
            "scores_sha256": values["scores_sha256"],
        }
    if len(method_ids) != 1:
        raise ValueError(f"counterfactual score runs have different method_id: {method_ids}")

    reference = bundle["true"]["metadata"]
    if str(dataset_manifest.get("dataset_id")) != str(reference["dataset_id"]):
        raise ValueError("score metadata and dataset manifest dataset_id differ")
    if str(dataset_manifest.get("dataset_version")) != str(
        reference["dataset_version"]
    ):
        raise ValueError("score metadata and dataset manifest version differ")
    manifest_versions = {
        request_identity["candidate_manifest_dataset_version"],
        request_identity["request_manifest_dataset_version"],
    }
    if manifest_versions != {str(reference["dataset_version"])}:
        raise ValueError(
            "score metadata and candidate/request manifest dataset versions differ"
        )
    if str(reference["split"]) != split:
        raise ValueError(f"score-bundle split={reference['split']} expected={split}")

    evaluator_implementation = _current_evaluator_implementation_identity()
    holdout_integrity = None
    analysis_selection_implementation = None
    history_assignments_audit = None
    if str(reference["dataset_version"]) == V12_DATASET_VERSION:
        holdout_integrity = verify_published_holdout(
            standardized_dir,
            open_qrels=False,
        )
        if holdout_integrity.get("protocol_sha256") != protocol_audit["sha256"]:
            raise ValueError(
                "registered holdout protocol SHA differs from the current frozen "
                "protocol"
            )
        analysis_selection_implementation = holdout_integrity.get(
            "analysis_selection_implementation"
        )
        if not isinstance(analysis_selection_implementation, dict):
            raise ValueError(
                "holdout analysis_selection_implementation must be an object"
            )
        expected_evaluator_digest = analysis_selection_implementation.get(
            "evaluator_digest"
        )
        if expected_evaluator_digest != evaluator_implementation["digest"]:
            raise ValueError(
                "current evaluator implementation differs from the holdout "
                "release lock"
            )
        assignment_manifest_values = [
            values["metadata"].get("history_assignment_manifest_path")
            for values in bundle.values()
        ]
        if not all(
            isinstance(value, str) and value.strip()
            for value in assignment_manifest_values
        ) or len(
            {str(Path(value).resolve()) for value in assignment_manifest_values}
        ) != 1:
            raise ValueError(
                "registered holdout score bundle must share one assignment manifest"
            )
        assignment_manifest_path = Path(
            assignment_manifest_values[0]
        ).resolve()
        history_assignments_audit = (
            verify_motivation_v12_history_assignments(
                assignment_manifest_path,
                standardized_dir=standardized_dir,
                release_lock_path=holdout_integrity[
                    "post_selection_recipe_checkpoint_lock_path"
                ],
            )
        )
        if history_assignments_audit.get("qrels_read") is not False:
            raise ValueError("history assignment audit must remain qrels-free")
        frozen = holdout_integrity["checkpoint_identities"].get(
            next(iter(method_ids))
        )
        if not isinstance(frozen, dict):
            raise ValueError("holdout release lock lacks the evaluated method")
        for condition, values in bundle.items():
            metadata = values["metadata"]
            assignment_file = history_assignments_audit["files"].get(condition)
            if not isinstance(assignment_file, dict):
                raise ValueError(
                    f"released history assignment is missing {condition}"
                )
            metadata_assignment_path = metadata.get("history_assignments_path")
            metadata_manifest_path = metadata.get(
                "history_assignment_manifest_path"
            )
            if (
                not isinstance(metadata_assignment_path, str)
                or Path(metadata_assignment_path).resolve()
                != Path(assignment_file["path"]).resolve()
                or metadata.get("history_assignment_sha256")
                != assignment_file["sha256"]
                or not isinstance(metadata_manifest_path, str)
                or Path(metadata_manifest_path).resolve()
                != Path(history_assignments_audit["manifest_path"]).resolve()
                or metadata.get("history_assignment_manifest_sha256")
                != history_assignments_audit["manifest_sha256"]
            ):
                raise ValueError(
                    f"score run {values['run_id']} is not bound to the released "
                    f"{condition} history assignment"
                )
            declared = metadata.get("holdout_integrity")
            expected_declared = {
                "checkpoint_identity_manifest_sha256": frozen[
                    "identity_manifest_sha256"
                ],
                "checkpoint_id": frozen["checkpoint_id"],
                "integrity_lock_sha256": holdout_integrity[
                    "integrity_lock_sha256"
                ],
                "manifest_sha256": holdout_integrity["manifest_sha256"],
                "post_selection_recipe_checkpoint_lock_sha256": holdout_integrity[
                    "post_selection_recipe_checkpoint_lock_sha256"
                ],
                "protocol_sha256": holdout_integrity["protocol_sha256"],
                "qrels_opened": False,
                "verified_before_model_load": True,
            }
            if declared != expected_declared:
                raise ValueError(
                    f"score run {values['run_id']} holdout integrity mismatch"
                )
            if metadata.get("checkpoint_id") != frozen["checkpoint_id"]:
                raise ValueError(
                    f"score run {values['run_id']} is not the released checkpoint"
                )
            if metadata.get("config_sha256") != frozen["config_sha256"]:
                raise ValueError(
                    f"score run {values['run_id']} is not the released config"
                )
            implementation = metadata.get("implementation_identity", {})
            if implementation.get("digest") != frozen["implementation_digest"]:
                raise ValueError(
                    f"score run {values['run_id']} is not the released implementation"
                )
            signature = metadata.get("scoring_signature", {})
            if signature.get("holdout_integrity_lock_sha256") != holdout_integrity[
                "integrity_lock_sha256"
            ] or signature.get(
                "holdout_release_lock_sha256"
            ) != holdout_integrity[
                "post_selection_recipe_checkpoint_lock_sha256"
            ]:
                raise ValueError(
                    f"score run {values['run_id']} scoring signature lacks holdout lock"
                )
    elif any(values["metadata"].get("holdout_integrity") for values in bundle.values()):
        raise ValueError("development/legacy score bundle claims holdout integrity")

    return {
        "schema_version": 1,
        "analysis_type": "motivation_v12_pre_qrels_score_bundle_audit",
        "passed": True,
        "qrels_read": False,
        "candidate_manifest": {
            "path": str(candidate_manifest_path),
            "requests": len(candidates),
            "score_rows": expected_score_rows,
            "sha256": candidate_sha256,
        },
        "counterfactual_identity": {
            key: reference[key] for key in COUNTERFACTUAL_IDENTITY_KEYS
        },
        "method_id": next(iter(method_ids)),
        "dataset_manifest": {
            "path": str(dataset_manifest_path),
            "sha256": sha256_file(dataset_manifest_path),
        },
        "evaluator_implementation": evaluator_implementation,
        "analysis_selection_implementation": analysis_selection_implementation,
        "history_assignments": history_assignments_audit,
        "holdout_integrity": (
            {
                "integrity_lock_sha256": holdout_integrity[
                    "integrity_lock_sha256"
                ],
                "manifest_sha256": holdout_integrity["manifest_sha256"],
                "post_selection_recipe_checkpoint_lock_sha256": holdout_integrity[
                    "post_selection_recipe_checkpoint_lock_sha256"
                ],
                "qrels_opened": False,
            }
            if holdout_integrity is not None
            else None
        ),
        "records": records_audit,
        "population_gate": population_gate,
        "protocol": protocol_audit,
        "request_identity": request_identity,
        "request_manifest": {
            "path": str(request_manifest_path),
            "sha256": request_sha256,
        },
        "score_files": score_files,
        "split": split,
    }


def evaluate_motivation_v12_evidence(
    *,
    analysis_run_id: str,
    full_run_id: str,
    null_run_id: str,
    split: str,
    standardized_dir: str | Path,
    candidate_manifest_path: str | Path,
    activity_epsilon: float,
    utility_epsilon: float,
    label_mode: str = "graded",
    expected_qrels_sha256: str | None = None,
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    wrong_run_id: str | None = None,
    runs_dir: str | Path = "runs",
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
) -> dict[str, Any]:
    """Audit, invoke the shared evaluator, then aggregate V1.2 evidence."""

    if label_mode != FROZEN_LABEL_MODE:
        raise ValueError("Motivation V1.2 label_mode is frozen at graded")
    if activity_epsilon != FROZEN_ACTIVITY_EPSILON:
        raise ValueError("Motivation V1.2 activity_epsilon is frozen at 0.01")
    if utility_epsilon != FROZEN_UTILITY_EPSILON:
        raise ValueError("Motivation V1.2 utility_epsilon is frozen at 0.0")

    audit = audit_motivation_v12_score_bundle(
        full_run_id=full_run_id,
        null_run_id=null_run_id,
        wrong_run_id=wrong_run_id,
        split=split,
        standardized_dir=standardized_dir,
        candidate_manifest_path=candidate_manifest_path,
        protocol_path=protocol_path,
        runs_dir=runs_dir,
    )
    if sha256_file(audit["protocol"]["path"]) != audit["protocol"]["sha256"]:
        raise ValueError("frozen protocol changed after the pre-qrels audit")
    qrels_path = Path(standardized_dir) / f"qrels_{split}.jsonl"
    holdout_qrels_audit = None
    if audit.get("holdout_integrity") is not None:
        # This transition happens only after complete finite score coverage and
        # counterfactual identity have passed above.
        holdout_qrels_audit = verify_published_holdout(
            standardized_dir,
            open_qrels=True,
        )
        observed_qrels_sha256 = holdout_qrels_audit["verified_files"][
            "qrels_confirmation"
        ]["sha256"]
        locked_qrels_sha256 = observed_qrels_sha256
        if (
            expected_qrels_sha256 is not None
            and expected_qrels_sha256 != locked_qrels_sha256
        ):
            raise ValueError("requested qrels SHA differs from the holdout integrity lock")
        expected_qrels_sha256 = locked_qrels_sha256
    else:
        locked_qrels_sha256 = audit["population_gate"]["expected_qrels_sha256"]
        if (
            expected_qrels_sha256 is not None
            and expected_qrels_sha256 != locked_qrels_sha256
        ):
            raise ValueError(
                "requested qrels SHA differs from the current frozen protocol"
            )
        expected_qrels_sha256 = locked_qrels_sha256
        observed_qrels_sha256 = sha256_file(qrels_path)
        if observed_qrels_sha256 != expected_qrels_sha256:
            raise ValueError(
                "frozen qrels hash mismatch before the shared evaluator label boundary"
            )
    # The shared evaluator owns the label boundary and appends the repository
    # evaluation ledger for every dev/confirmation call.
    evaluate_history_response_runs(
        analysis_run_id=analysis_run_id,
        true_run_id=full_run_id,
        null_run_id=null_run_id,
        wrong_run_id=wrong_run_id,
        split=split,
        label_mode=label_mode,
        candidate_manifest_path=candidate_manifest_path,
        standardized_dir=standardized_dir,
        activity_epsilon=activity_epsilon,
        utility_epsilon=utility_epsilon,
        runs_dir=runs_dir,
        dev_eval_log_path=dev_eval_log_path,
    )
    analysis_dir = Path(runs_dir) / analysis_run_id
    audit_path = analysis_dir / "pre_qrels_score_bundle_audit.json"
    write_json(audit_path, audit)
    write_json(
        analysis_dir / "qrels_hash_lock.json",
        {
            "expected_qrels_sha256": expected_qrels_sha256,
            "observed_qrels_sha256": observed_qrels_sha256,
            "qrels_path": str(qrels_path),
            "source": (
                "published_holdout_integrity_lock"
                if holdout_qrels_audit is not None
                else "predeclared_development_population_hash"
            ),
            "holdout_integrity_lock_sha256": (
                holdout_qrels_audit["integrity_lock_sha256"]
                if holdout_qrels_audit is not None
                else None
            ),
            "verified_before_shared_evaluator": True,
        },
    )
    return build_motivation_v12_evidence(
        analysis_run_id=analysis_run_id,
        standardized_dir=standardized_dir,
        runs_dir=runs_dir,
        score_audit_path=audit_path,
    )


def build_motivation_v12_evidence(
    *,
    analysis_run_id: str,
    standardized_dir: str | Path,
    runs_dir: str | Path = "runs",
    score_audit_path: str | Path | None = None,
    output_path: str | Path | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Consume shared-evaluator artifacts and write the V1.2 evidence summary."""

    _assert_frozen_bootstrap(bootstrap_samples, bootstrap_seed)
    analysis_dir = Path(runs_dir) / analysis_run_id
    standardized_dir = Path(standardized_dir)
    score_audit_path = Path(
        score_audit_path or analysis_dir / "pre_qrels_score_bundle_audit.json"
    )
    output_path = Path(output_path or analysis_dir / "motivation_v12_evidence.json")

    score_audit = _read_json(score_audit_path)
    if (
        score_audit.get("passed") is not True
        or score_audit.get("qrels_read") is not False
    ):
        raise ValueError("a passing pre-qrels score-bundle audit is required")
    _verify_audited_input_files(score_audit, Path(runs_dir))
    metrics_path = analysis_dir / "metrics.json"
    evaluator_metadata_path = analysis_dir / "metadata.json"
    per_request_path = analysis_dir / "per_request_history_response.jsonl"
    metrics = _read_json(metrics_path)
    evaluator_metadata = _read_json(evaluator_metadata_path)
    qrels_hash_lock_path = analysis_dir / "qrels_hash_lock.json"
    qrels_hash_lock = _read_json(qrels_hash_lock_path)
    if qrels_hash_lock.get("verified_before_shared_evaluator") is not True:
        raise ValueError("missing pre-evaluator qrels hash verification")
    if metrics.get("qrels_sha256") != qrels_hash_lock.get(
        "expected_qrels_sha256"
    ):
        raise ValueError("shared evaluator qrels hash differs from the frozen lock")
    if evaluator_metadata.get("qrels_read") is not True:
        raise ValueError("analysis is not an output of the qrels-reading shared evaluator")
    if evaluator_metadata.get("analysis_type") != "history_response_direction_gap":
        raise ValueError("unexpected shared evaluator analysis type")
    if metrics.get("analysis_run_id") != analysis_run_id:
        raise ValueError("shared evaluator analysis_run_id mismatch")
    if (
        metrics.get("candidate_manifest_sha256")
        != score_audit["candidate_manifest"]["sha256"]
    ):
        raise ValueError("evaluator/audit candidate manifest hash mismatch")
    if (
        metrics.get("request_manifest_sha256")
        != score_audit["request_manifest"]["sha256"]
    ):
        raise ValueError("evaluator/audit request manifest hash mismatch")
    if metrics.get("split") != score_audit["split"]:
        raise ValueError("evaluator/audit split mismatch")
    expected_runs = {
        "full": metrics.get("true_run_id"),
        "null": metrics.get("null_run_id"),
        "wrong_user": metrics.get("wrong_run_id"),
    }
    audited_runs = {
        condition: values["run_id"]
        for condition, values in score_audit["score_files"].items()
    }
    if {
        condition: run_id
        for condition, run_id in expected_runs.items()
        if run_id is not None
    } != audited_runs:
        raise ValueError("evaluator/audit counterfactual run IDs differ")

    rows = _rows_by_request(per_request_path)
    records_path = standardized_dir / f"records_{metrics['split']}.jsonl"
    records_audit = audit_standardized_file(records_path, str(metrics["split"]))
    clusters = _normalized_query_clusters(records_path)
    if set(rows) != set(clusters):
        raise ValueError("evaluator rows and label-free records have different coverage")

    surface_dir = analysis_dir / "target_aware_surfaces"
    surface_manifest_path = surface_dir / "manifest.json"
    surface_manifest = _read_json(surface_manifest_path)
    if surface_manifest.get("qrels_sha256") != metrics.get("qrels_sha256"):
        raise ValueError("target-surface/evaluator qrels hash mismatch")
    if surface_manifest.get("records_sha256") != records_audit["sha256"]:
        raise ValueError("target-surface records hash mismatch")
    surface_ids = _load_and_verify_surface_ids(surface_dir, surface_manifest)
    if surface_ids.get("all") != set(rows):
        raise ValueError("target-aware all surface does not cover evaluator rows")
    _assert_surface_partition(
        surface_ids["all"],
        {name: surface_ids[name] for name in ALL_REQUEST_PARTITION},
    )

    surface_results = {}
    for role, surface_name in SURFACE_ROLES.items():
        selected = [rows[request_id] for request_id in sorted(surface_ids[surface_name])]
        surface_results[role] = _summarize_surface(
            selected,
            clusters,
            surface_name=surface_name,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            utility_epsilon=float(metrics["utility_epsilon"]),
        )

    overall_mean = surface_results["overall"]["full_minus_null_ndcg@10"]["mean"]
    if not math.isclose(
        float(overall_mean),
        float(metrics["mean_true_minus_null_ndcg@10"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("overall surface mean differs from shared evaluator metrics")

    all_rows = [rows[request_id] for request_id in sorted(rows)]
    partition_rows = {
        name: [rows[request_id] for request_id in sorted(surface_ids[name])]
        for name in ALL_REQUEST_PARTITION
    }
    contributions = summarize_partition_contributions(
        all_rows=all_rows,
        partition_rows=partition_rows,
        metric="true_minus_null_ndcg@10",
    )
    if not math.isclose(
        float(contributions["all_mean"]),
        float(metrics["mean_true_minus_null_ndcg@10"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("population contribution identity differs from evaluator")

    report = {
        "schema_version": 1,
        "analysis_type": "motivation_v12_shared_evaluator_evidence",
        "analysis_run_id": analysis_run_id,
        "bootstrap": {
            "cluster": BOOTSTRAP_CLUSTER,
            "normalization": "unicode_casefold_then_remove_all_whitespace",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "counterfactual_runs": {
            "full": metrics["true_run_id"],
            "null": metrics["null_run_id"],
            "wrong_user": metrics.get("wrong_run_id"),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "integrity": {
            "pre_qrels_score_bundle_audit": {
                "path": str(score_audit_path),
                "sha256": sha256_file(score_audit_path),
                "passed": True,
                "qrels_read": False,
            },
            "shared_evaluator_qrels_read": True,
            "qrels_hash_lock": {
                "path": str(qrels_hash_lock_path),
                "sha256": sha256_file(qrels_hash_lock_path),
                "verified_before_shared_evaluator": True,
            },
            "metric_formulas_changed": False,
            "method_owned_metrics": False,
            "all_request_partition_reconstructs_overall": True,
        },
        "label_mode": metrics["label_mode"],
        "metric_source": "myrec.eval.history_response_evaluator",
        "population_weighted_contributions": contributions,
        "records": records_audit,
        "shared_evaluator_overall": {
            key: metrics.get(key)
            for key in (
                "mean_true_ndcg@10",
                "mean_true_ndcg@10_positive",
                "mean_true_minus_null_ndcg@10",
                "mean_true_minus_null_ndcg@10_positive",
                "mean_true_minus_wrong_ndcg@10",
                "num_positive_eligible_requests",
                "num_requests",
            )
            if key in metrics
        },
        "shared_evaluator_artifacts": {
            "metadata": {
                "path": str(evaluator_metadata_path),
                "sha256": sha256_file(evaluator_metadata_path),
            },
            "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
            "per_request": {
                "path": str(per_request_path),
                "sha256": sha256_file(per_request_path),
            },
            "target_aware_surfaces": {
                "path": str(surface_manifest_path),
                "sha256": sha256_file(surface_manifest_path),
            },
        },
        "split": metrics["split"],
        "surfaces": surface_results,
    }
    write_json(output_path, report)
    return report


def normalize_query_cluster(query: str) -> str:
    """Return the frozen V1.2 exact normalized-query cluster key."""

    return "".join(str(query).casefold().split())


def _current_evaluator_implementation_identity() -> dict[str, Any]:
    """Return the canonical digest sealed by the V1.2 release lock."""

    files = []
    for relative_path in sorted(_EVALUATOR_IMPLEMENTATION_PATHS):
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"missing load-bearing evaluator implementation: {path}"
            )
        files.append(
            {
                "path": Path(relative_path).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    return {
        "digest": sha256_text(
            json.dumps(files, sort_keys=True, separators=(",", ":"))
        ),
        "files": files,
    }


def _load_frozen_protocol(
    protocol_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(protocol_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing Motivation V1.2 protocol: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Motivation V1.2 protocol must be a YAML object")
    status = payload.get("status")
    if not isinstance(status, str) or not status.endswith("_frozen"):
        raise ValueError("Motivation V1.2 protocol is not frozen")
    if payload.get("protocol_id") != "motivation_v1_2_first_round":
        raise ValueError("unexpected Motivation V1.2 protocol_id")
    return payload, {
        "path": str(path),
        "protocol_id": payload["protocol_id"],
        "sha256": sha256_file(path),
        "status": status,
    }


def _gate_population_before_qrels(
    *,
    protocol: dict[str, Any],
    protocol_audit: dict[str, Any],
    dataset_manifest: dict[str, Any],
    dataset_manifest_path: Path,
    standardized_dir: Path,
    candidate_manifest_path: Path,
    request_manifest_path: Path,
    split: str,
    wrong_run_id: str | None,
) -> dict[str, Any]:
    """Allow only the two protocol populations without resolving any qrels."""

    data = protocol.get("data")
    if not isinstance(data, dict):
        raise ValueError("frozen protocol is missing data")
    development = data.get("development_population")
    if not isinstance(development, dict):
        raise ValueError("frozen protocol is missing development_population")
    if data.get("dataset_id") != "kuaisearch":
        raise ValueError("frozen protocol dataset_id must be kuaisearch")

    dataset_id = str(dataset_manifest.get("dataset_id", ""))
    dataset_version = str(dataset_manifest.get("dataset_version", ""))
    development_version = str(development.get("dataset_version", ""))
    if dataset_id == "kuaisearch" and dataset_version == development_version:
        if wrong_run_id is not None:
            raise ValueError(
                "development and legacy populations permit full/null only"
            )
        if split == "dev":
            role = "internal_dev"
            records_filename = "records_dev.jsonl"
            records_sha_field = "records_dev_sha256"
            qrels_sha_field = "qrels_dev_sha256"
        elif split == "confirmation":
            role = "legacy_compatibility"
            records_filename = "records_confirmation.jsonl"
            records_sha_field = "records_legacy_compatibility_sha256"
            qrels_sha_field = "qrels_legacy_compatibility_sha256"
        else:  # Defensive; the public audit checks the same split allowlist.
            raise ValueError("development population split is not allowed")

        expected_files = {
            "manifest": (
                dataset_manifest_path,
                "manifest_sha256",
            ),
            "candidate_manifest": (
                candidate_manifest_path,
                "candidate_manifest_sha256",
            ),
            "request_manifest": (
                request_manifest_path,
                "request_manifest_sha256",
            ),
            "records": (
                standardized_dir / records_filename,
                records_sha_field,
            ),
        }
        verified_files = {}
        for name, (path, protocol_field) in expected_files.items():
            expected_sha256 = _require_protocol_sha256(
                development.get(protocol_field),
                f"development_population.{protocol_field}",
            )
            if not path.is_file():
                raise FileNotFoundError(f"missing frozen {name}: {path}")
            observed_sha256 = sha256_file(path)
            if observed_sha256 != expected_sha256:
                raise ValueError(
                    f"{role} {name} hash differs from the frozen protocol"
                )
            verified_files[name] = {
                "path": str(path),
                "protocol_field": protocol_field,
                "sha256": observed_sha256,
            }
        expected_qrels_sha256 = _require_protocol_sha256(
            development.get(qrels_sha_field),
            f"development_population.{qrels_sha_field}",
        )
        return {
            "allowed": True,
            "conditions": ["full", "null"],
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "expected_qrels_sha256": expected_qrels_sha256,
            "population_role": role,
            "protocol_sha256": protocol_audit["sha256"],
            "qrels_opened": False,
            "qrels_sha256_protocol_field": qrels_sha_field,
            "records_path": str(standardized_dir / records_filename),
            "split": split,
            "verified_label_free_files": verified_files,
        }

    if dataset_id == "kuaisearch" and dataset_version == V12_DATASET_VERSION:
        if split != "confirmation" or wrong_run_id is None:
            raise ValueError(
                "the registered V1.2 holdout requires confirmation full/null/wrong"
            )
        return {
            "allowed": True,
            "conditions": ["full", "null", "wrong"],
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "expected_qrels_sha256": None,
            "population_role": "registered_new_holdout",
            "protocol_sha256": protocol_audit["sha256"],
            "qrels_opened": False,
            "qrels_sha256_protocol_field": None,
            "records_path": str(standardized_dir / "records_confirmation.jsonl"),
            "split": split,
            "verified_label_free_files": None,
        }

    raise ValueError(
        "dataset population is not allowlisted by the current frozen "
        "Motivation V1.2 protocol"
    )


def _require_protocol_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} is not a canonical SHA-256")
    return value


def _score_protocol_sha256(metadata: dict[str, Any], run_id: str) -> Any:
    """Read either scorer schema without weakening protocol binding.

    W0 publishes the frozen protocol SHA at the top level and inside its
    scoring signature.  The frozen Q-series scorer publishes it only inside
    ``scoring_signature``.  When both locations exist they must agree.
    """

    top_level = metadata.get("protocol_sha256")
    signature = metadata.get("scoring_signature")
    nested = signature.get("protocol_sha256") if isinstance(signature, dict) else None
    if top_level is not None and nested is not None and top_level != nested:
        raise ValueError(
            f"score run {run_id} declares conflicting protocol SHA values"
        )
    return top_level if top_level is not None else nested


def _audit_request_identity(
    *,
    records_path: Path,
    candidate_manifest_path: Path,
    request_manifest_path: Path,
    split: str,
    candidates: dict[str, list[str]],
) -> dict[str, Any]:
    candidate_manifest = _read_json(candidate_manifest_path)
    request_manifest = _read_json(request_manifest_path)
    records = {str(row["request_id"]): row for row in iter_jsonl(records_path)}
    if len(records) != sum(1 for _ in iter_jsonl(records_path)):
        raise ValueError("duplicate request_id in standardized records")
    entries: dict[str, dict[str, Any]] = {}
    for entry in request_manifest.get("entries", []):
        if entry.get("split") != split:
            continue
        request_id = str(entry["request_id"])
        if request_id in entries:
            raise ValueError(f"duplicate request manifest request_id={request_id}")
        entries[request_id] = entry
    if set(entries) != set(candidates) or set(records) != set(candidates):
        raise ValueError("records/candidate/request manifests have different coverage")
    for request_id, candidate_ids in candidates.items():
        record = records[request_id]
        record_candidate_ids = [
            str(candidate["item_id"]) for candidate in record["candidates"]
        ]
        if record_candidate_ids != candidate_ids:
            raise ValueError(
                f"record/candidate manifest identity mismatch request_id={request_id}"
            )
        entry = entries[request_id]
        expected_candidate_hash = sha256_text(
            json.dumps(candidate_ids, separators=(",", ":"))
        )
        if entry.get("candidate_item_ids_sha256") != expected_candidate_hash:
            raise ValueError(
                f"request manifest candidate hash mismatch request_id={request_id}"
            )
        if entry.get("query_sha256") != sha256_text(str(record["query"])):
            raise ValueError(f"request manifest query hash mismatch request_id={request_id}")
    return {
        "candidate_manifest_dataset_version": str(
            candidate_manifest.get("dataset_version", "")
        ),
        "candidate_order_exact": True,
        "query_hashes_exact": True,
        "request_manifest_dataset_version": str(
            request_manifest.get("dataset_version", "")
        ),
        "requests": len(candidates),
    }


def _load_and_verify_surface_ids(
    surface_dir: Path, manifest: dict[str, Any]
) -> dict[str, set[str]]:
    result = {}
    for name, values in manifest.get("files", {}).items():
        path = surface_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"missing target-aware surface: {path}")
        if sha256_file(path) != values.get("sha256"):
            raise ValueError(f"target-aware surface hash mismatch: {name}")
        request_ids = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if len(request_ids) != int(values.get("requests", -1)):
            raise ValueError(f"target-aware surface count mismatch: {name}")
        result[str(name)] = request_ids
    required = {"all", *ALL_REQUEST_PARTITION, *SURFACE_ROLES.values()}
    missing = required - set(result)
    if missing:
        raise ValueError(f"target-aware surface manifest is missing: {sorted(missing)}")
    return result


def _verify_audited_input_files(
    score_audit: dict[str, Any], runs_dir: Path
) -> None:
    for key in (
        "candidate_manifest",
        "dataset_manifest",
        "protocol",
        "request_manifest",
        "records",
    ):
        values = score_audit[key]
        if sha256_file(values["path"]) != values["sha256"]:
            raise ValueError(f"audited {key} changed after the pre-qrels audit")
    implementation = score_audit.get("evaluator_implementation")
    if not isinstance(implementation, dict):
        raise ValueError("pre-qrels audit lacks evaluator implementation identity")
    if implementation != _current_evaluator_implementation_identity():
        raise ValueError("evaluator implementation changed after the pre-qrels audit")
    history_assignments = score_audit.get("history_assignments")
    if history_assignments is not None:
        manifest_path = Path(history_assignments["manifest_path"])
        if sha256_file(manifest_path) != history_assignments["manifest_sha256"]:
            raise ValueError(
                "audited history assignment manifest changed after the pre-qrels audit"
            )
        for condition, values in history_assignments["files"].items():
            if sha256_file(values["path"]) != values["sha256"]:
                raise ValueError(
                    f"audited {condition} history assignment changed after the "
                    "pre-qrels audit"
                )
    for condition, values in score_audit["score_files"].items():
        run_dir = runs_dir / str(values["run_id"])
        if sha256_file(run_dir / "metadata.json") != values["metadata_sha256"]:
            raise ValueError(
                f"audited {condition} metadata changed after the pre-qrels audit"
            )
        if sha256_file(run_dir / "scores.jsonl") != values["scores_sha256"]:
            raise ValueError(
                f"audited {condition} scores changed after the pre-qrels audit"
            )


def _summarize_surface(
    rows: list[dict[str, Any]],
    cluster_by_request: dict[str, str],
    *,
    surface_name: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    utility_epsilon: float,
) -> dict[str, Any]:
    if not rows:
        return {
            "evaluator_surface": surface_name,
            "full_minus_null_ndcg@10": {"mean": None, "query_cluster_ci95": None},
            "full_minus_wrong_ndcg@10": None,
            "full_ndcg@10": None,
            "num_query_clusters": 0,
            "num_requests": 0,
        }
    aggregate = aggregate_history_response(rows, utility_epsilon=utility_epsilon)
    metrics = ["true_minus_null_ndcg@10"]
    has_wrong = all("true_minus_wrong_ndcg@10" in row for row in rows)
    if any("true_minus_wrong_ndcg@10" in row for row in rows) and not has_wrong:
        raise ValueError("wrong-user evaluator rows have incomplete surface coverage")
    if has_wrong:
        metrics.append("true_minus_wrong_ndcg@10")
    intervals = cluster_bootstrap_mean_ci(
        rows,
        cluster_by_request,
        metrics,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    wrong = None
    if has_wrong:
        wrong = {
            "mean": aggregate["mean_true_minus_wrong_ndcg@10"],
            "query_cluster_ci95": intervals["true_minus_wrong_ndcg@10"],
        }
    return {
        "evaluator_surface": surface_name,
        "full_minus_null_ndcg@10": {
            "mean": aggregate["mean_true_minus_null_ndcg@10"],
            "query_cluster_ci95": intervals["true_minus_null_ndcg@10"],
        },
        "full_minus_wrong_ndcg@10": wrong,
        "full_ndcg@10": aggregate["mean_true_ndcg@10"],
        "num_query_clusters": len(
            {cluster_by_request[str(row["request_id"])] for row in rows}
        ),
        "num_requests": len(rows),
    }


def _normalized_query_clusters(records_path: Path) -> dict[str, str]:
    result = {}
    for row in iter_jsonl(records_path):
        request_id = str(row["request_id"])
        if request_id in result:
            raise ValueError(f"duplicate record request_id={request_id}")
        result[request_id] = normalize_query_cluster(str(row.get("query", "")))
    if not result:
        raise ValueError(f"empty records file: {records_path}")
    return result


def _rows_by_request(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id in result:
            raise ValueError(f"duplicate evaluator request_id={request_id}")
        result[request_id] = row
    if not result:
        raise ValueError(f"empty evaluator per-request file: {path}")
    return result


def _assert_surface_partition(
    population: set[str], partition: dict[str, set[str]]
) -> None:
    union: set[str] = set()
    for name, request_ids in partition.items():
        overlap = union & request_ids
        if overlap:
            raise ValueError(
                f"target-aware all-request partition overlaps at {name}: "
                f"{sorted(overlap)[:5]}"
            )
        union.update(request_ids)
    if union != population:
        raise ValueError(
            "target-aware all-request partition does not cover the population"
        )


def _assert_frozen_bootstrap(samples: int, seed: int) -> None:
    if samples != BOOTSTRAP_SAMPLES or seed != BOOTSTRAP_SEED:
        raise ValueError(
            "Motivation V1.2 bootstrap is frozen at "
            f"samples={BOOTSTRAP_SAMPLES}, seed={BOOTSTRAP_SEED}"
        )


def _report_condition(condition: str) -> str:
    if condition == "true":
        return "full"
    if condition == "wrong":
        return "wrong_user"
    return condition


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value
