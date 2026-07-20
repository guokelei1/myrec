"""Shared evaluator for paired Motivation mechanism score probes.

The score-bundle audit in this module is deliberately qrels-free.  Graded
labels are opened only after both score runs have passed identity, provenance,
and complete finite coverage checks.  Scorers and intervention code must not
import this module.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.eval.controlled_composition import cluster_bootstrap_mean_ci
from myrec.eval.history_response import gain_ndcg_at_k
from myrec.eval.target_aware_surfaces import (
    ALL_REQUEST_PARTITION,
    build_target_aware_surface_memberships,
    materialize_target_aware_surfaces,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl, write_json, write_jsonl


BOOTSTRAP_CLUSTER = "normalized_query"
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20260715
SUPPORTED_SPLIT = "dev"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MECHANISM_PROBE_MANIFEST_PATH = "experiments/motivation/probe_manifest.yaml"
MECHANISM_PROBE_MANIFEST_SHA256 = (
    "adedf0e662b9d8529162b8abffedcf6b10962913f28580af6119d807cc5d929c"
)

_PAIR_INVARIANT_KEYS = (
    "dataset_id",
    "dataset_version",
    "split",
    "method_id",
    "checkpoint_id",
)
_FORBIDDEN_LABEL_FIELDS = {
    "clicked",
    "purchased",
    "relevance",
    "is_clicked",
    "is_purchased",
    "label",
    "labels",
    "target",
}
_FROZEN_HISTORY_CONDITION_IDS = {
    "true": "frozen_full",
    "null": "frozen_null",
    "wrong": "frozen_wrong_user",
}


@dataclass(frozen=True)
class _ScoreRun:
    run_id: str
    run_dir: Path
    metadata: dict[str, Any]
    scores: dict[str, dict[str, float]]
    scores_sha256: str
    metadata_sha256: str
    base_scoring_signature: Any


@dataclass(frozen=True)
class _AuditState:
    candidates: dict[str, list[str]]
    normalized_query_clusters: dict[str, str]
    treatment: _ScoreRun
    control: _ScoreRun
    report: dict[str, Any]
    candidate_manifest_path: Path
    request_manifest_path: Path
    records_path: Path


def audit_mechanism_score_bundle(
    *,
    treatment_run_id: str,
    control_run_id: str,
    standardized_dir: str | Path,
    candidate_manifest_path: str | Path | None = None,
    split: str = SUPPORTED_SPLIT,
    runs_dir: str | Path = "runs",
) -> dict[str, Any]:
    """Return a qrels-free audit report for one paired mechanism probe."""

    return _audit_mechanism_score_bundle(
        treatment_run_id=treatment_run_id,
        control_run_id=control_run_id,
        standardized_dir=standardized_dir,
        candidate_manifest_path=candidate_manifest_path,
        split=split,
        runs_dir=runs_dir,
    ).report


def evaluate_mechanism_probe(
    *,
    analysis_run_id: str,
    treatment_run_id: str,
    control_run_id: str,
    standardized_dir: str | Path,
    candidate_manifest_path: str | Path | None = None,
    split: str = SUPPORTED_SPLIT,
    runs_dir: str | Path = "runs",
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
) -> dict[str, Any]:
    """Audit and evaluate a treatment/control score pair on internal dev.

    ``pre_qrels_audit.json`` is materialized before this function opens
    ``qrels_dev.jsonl``.  Consequently any malformed score bundle fails on the
    qrels-free side of the boundary, even when the qrels file is itself bad.
    """

    _require_internal_dev(split)
    runs_dir = Path(runs_dir)
    analysis_dir = runs_dir / analysis_run_id
    if analysis_dir.exists():
        raise FileExistsError(f"analysis run already exists: {analysis_dir}")

    audit = _audit_mechanism_score_bundle(
        treatment_run_id=treatment_run_id,
        control_run_id=control_run_id,
        standardized_dir=standardized_dir,
        candidate_manifest_path=candidate_manifest_path,
        split=split,
        runs_dir=runs_dir,
    )

    # This write is the qrels boundary.  No qrels path has been constructed or
    # opened above this point.
    analysis_dir.mkdir(parents=True, exist_ok=False)
    pre_qrels_audit_path = analysis_dir / "pre_qrels_audit.json"
    write_json(pre_qrels_audit_path, audit.report)

    standardized_dir = Path(standardized_dir)
    qrels_path = standardized_dir / "qrels_dev.jsonl"
    gains = _load_graded_gains(qrels_path, audit.candidates)
    qrels_sha256 = sha256_file(qrels_path)

    memberships = build_target_aware_surface_memberships(
        audit.records_path,
        audit.candidates,
        gains,
    )
    materialize_target_aware_surfaces(
        audit.records_path,
        audit.candidates,
        gains,
        analysis_dir / "target_aware_surfaces",
        label_mode="graded",
        candidate_manifest_path=audit.candidate_manifest_path,
        qrels_path=qrels_path,
    )
    partition_by_request = _target_partition_by_request(memberships)
    treatment_condition_id = _resolved_condition_id(audit.treatment.metadata)
    control_condition_id = _resolved_condition_id(audit.control.metadata)
    treatment_raw_history_condition = audit.treatment.metadata.get(
        "history_condition"
    )
    control_raw_history_condition = audit.control.metadata.get("history_condition")

    per_request = []
    for request_id, item_ids in audit.candidates.items():
        row = _evaluate_request(
            request_id=request_id,
            item_ids=item_ids,
            gains=gains[request_id],
            treatment_scores=audit.treatment.scores[request_id],
            control_scores=audit.control.scores[request_id],
        )
        row["normalized_query_cluster"] = audit.normalized_query_clusters[request_id]
        row["target_aware_surface"] = partition_by_request[request_id]
        row["treatment_condition_id"] = treatment_condition_id
        row["control_condition_id"] = control_condition_id
        row["treatment_raw_history_condition"] = treatment_raw_history_condition
        row["control_raw_history_condition"] = control_raw_history_condition
        per_request.append(row)

    overall = _summarize_rows(per_request, audit.normalized_query_clusters)
    surface_metrics = {
        "all": overall,
        "observed_positive": _summarize_rows(
            [
                row
                for row in per_request
                if row["request_id"] in memberships["observed_positive"]
            ],
            audit.normalized_query_clusters,
        ),
        **{
            surface: _summarize_rows(
                [row for row in per_request if row["request_id"] in memberships[surface]],
                audit.normalized_query_clusters,
            )
            for surface in ALL_REQUEST_PARTITION
        },
    }
    population_weighted_contributions = _ndcg_partition_contributions(
        overall=overall,
        surfaces=surface_metrics,
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    invariants = audit.report["invariants"]
    metrics = {
        "schema_version": 1,
        "analysis_run_id": analysis_run_id,
        "analysis_type": "motivation_mechanism_paired_probe",
        "bootstrap": {
            "cluster": BOOTSTRAP_CLUSTER,
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "candidate_manifest_sha256": audit.report["candidate_manifest_sha256"],
        "checkpoint_id": invariants["checkpoint_id"],
        "control_condition_id": control_condition_id,
        "control_raw_history_condition": control_raw_history_condition,
        "control_run_id": control_run_id,
        "dataset_id": invariants["dataset_id"],
        "dataset_version": invariants["dataset_version"],
        "generated_at": generated_at,
        "label_mode": "graded",
        "mean_control_ndcg@10": overall["mean_control_ndcg@10"],
        "mean_target_margin_change": overall["mean_target_margin_change"],
        "mean_treatment_minus_control_ndcg@10": overall[
            "mean_treatment_minus_control_ndcg@10"
        ],
        "mean_treatment_ndcg@10": overall["mean_treatment_ndcg@10"],
        "method_id": invariants["method_id"],
        "num_margin_eligible_requests": overall["num_margin_eligible_requests"],
        "num_requests": overall["num_requests"],
        "population_weighted_ndcg_contributions": population_weighted_contributions,
        "mechanism_probe_manifest_admission": audit.report[
            "mechanism_probe_manifest_admission"
        ],
        "qrels_sha256": qrels_sha256,
        "query_cluster_ci95": overall["query_cluster_ci95"],
        "request_manifest_sha256": audit.report["request_manifest_sha256"],
        "split": SUPPORTED_SPLIT,
        "surfaces": surface_metrics,
        "target_margin_change": overall["target_margin_change"],
        "treatment_condition_id": treatment_condition_id,
        "treatment_raw_history_condition": treatment_raw_history_condition,
        "treatment_minus_control_ndcg@10": overall[
            "treatment_minus_control_ndcg@10"
        ],
        "treatment_run_id": treatment_run_id,
    }

    per_request_path = analysis_dir / "per_request.jsonl"
    metrics_path = analysis_dir / "metrics.json"
    write_jsonl(per_request_path, per_request)
    write_json(metrics_path, metrics)
    metadata = {
        "schema_version": 1,
        "analysis_run_id": analysis_run_id,
        "analysis_type": "motivation_mechanism_paired_probe",
        "base_scoring_signature": audit.treatment.base_scoring_signature,
        "bootstrap": metrics["bootstrap"],
        "candidate_manifest_path": str(audit.candidate_manifest_path),
        "candidate_manifest_sha256": audit.report["candidate_manifest_sha256"],
        "conditions": {
            "control": {
                "condition_id": control_condition_id,
                "raw_condition_id": audit.control.metadata.get("condition_id"),
                "raw_history_condition": control_raw_history_condition,
                "run_id": control_run_id,
            },
            "treatment": {
                "condition_id": treatment_condition_id,
                "raw_condition_id": audit.treatment.metadata.get("condition_id"),
                "raw_history_condition": treatment_raw_history_condition,
                "run_id": treatment_run_id,
            },
        },
        "invariants": invariants,
        "generated_at": generated_at,
        "label_mode": "graded",
        "margin_endpoint": {
            "competitor": (
                "highest-scoring candidate with gain strictly below the selected "
                "target, selected separately in each condition; candidate-order "
                "tie break"
            ),
            "contrast": "treatment_margin_minus_control_margin",
            "target": "highest graded gain; candidate-order tie break",
        },
        "mechanism_probe_manifest_admission": audit.report[
            "mechanism_probe_manifest_admission"
        ],
        "metrics_path": str(metrics_path),
        "per_request_path": str(per_request_path),
        "pre_qrels_audit_path": str(pre_qrels_audit_path),
        "pre_qrels_audit_sha256": sha256_file(pre_qrels_audit_path),
        "qrels_path": str(qrels_path),
        "qrels_read": True,
        "qrels_sha256": qrels_sha256,
        "records_path": str(audit.records_path),
        "request_manifest_path": str(audit.request_manifest_path),
        "request_manifest_sha256": audit.report["request_manifest_sha256"],
        "split": SUPPORTED_SPLIT,
        "target_aware_surfaces_manifest": str(
            analysis_dir / "target_aware_surfaces" / "manifest.json"
        ),
        "target_aware_surfaces_manifest_sha256": sha256_file(
            analysis_dir / "target_aware_surfaces" / "manifest.json"
        ),
    }
    write_json(analysis_dir / "metadata.json", metadata)
    _append_dev_eval_log(
        dev_eval_log_path,
        metrics=metrics,
        metrics_path=metrics_path,
    )
    return metrics


def _audit_mechanism_score_bundle(
    *,
    treatment_run_id: str,
    control_run_id: str,
    standardized_dir: str | Path,
    candidate_manifest_path: str | Path | None,
    split: str,
    runs_dir: str | Path,
) -> _AuditState:
    _require_internal_dev(split)
    standardized_dir = Path(standardized_dir)
    runs_dir = Path(runs_dir)
    candidate_manifest_path = Path(
        candidate_manifest_path
        if candidate_manifest_path is not None
        else standardized_dir / "candidate_manifest.json"
    )
    request_manifest_path = standardized_dir / "request_manifest.json"
    records_path = standardized_dir / "records_dev.jsonl"

    candidate_manifest_sha256 = sha256_file(candidate_manifest_path)
    request_manifest_sha256 = sha256_file(request_manifest_path)
    candidates, candidate_dataset_version = _load_candidates(
        candidate_manifest_path, SUPPORTED_SPLIT
    )
    normalized_query_clusters, records = _audit_label_free_records(
        records_path, candidates
    )
    _audit_request_manifest(request_manifest_path, candidates, records)
    request_dataset_version = _manifest_dataset_version(request_manifest_path)

    treatment = _load_score_run(
        runs_dir=runs_dir,
        run_id=treatment_run_id,
        candidate_manifest_sha256=candidate_manifest_sha256,
        request_manifest_sha256=request_manifest_sha256,
        candidates=candidates,
    )
    control = _load_score_run(
        runs_dir=runs_dir,
        run_id=control_run_id,
        candidate_manifest_sha256=candidate_manifest_sha256,
        request_manifest_sha256=request_manifest_sha256,
        candidates=candidates,
    )
    _assert_pair_invariants(treatment, control)
    mechanism_probe_manifest_admission = _audit_probe_manifest_admission(
        treatment,
        control,
    )

    dataset_version = str(treatment.metadata["dataset_version"])
    for source, declared in (
        ("candidate manifest", candidate_dataset_version),
        ("request manifest", request_dataset_version),
    ):
        if declared is not None and declared != dataset_version:
            raise ValueError(
                f"{source} dataset_version={declared!r} differs from score "
                f"dataset_version={dataset_version!r}"
            )
    dataset_manifest_path = standardized_dir / "manifest.json"
    if dataset_manifest_path.exists():
        dataset_manifest = _read_json_object(dataset_manifest_path)
        if (
            dataset_manifest.get("dataset_id") is not None
            and str(dataset_manifest["dataset_id"])
            != str(treatment.metadata["dataset_id"])
        ):
            raise ValueError("dataset manifest and score metadata dataset_id differ")
        if (
            dataset_manifest.get("dataset_version") is not None
            and str(dataset_manifest["dataset_version"]) != dataset_version
        ):
            raise ValueError("dataset manifest and score metadata dataset_version differ")

    expected_rows = sum(len(item_ids) for item_ids in candidates.values())
    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": 1,
        "analysis_type": "motivation_mechanism_pre_qrels_score_audit",
        "base_scoring_signature": treatment.base_scoring_signature,
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "checks": {
            "base_scoring_signature_equal": True,
            "candidate_manifest_hash_exact": True,
            "complete_finite_score_coverage": True,
            "dataset_version_split_method_checkpoint_equal": True,
            "duplicate_scores_absent": True,
            "label_free_records_exact": True,
            "qrels_read_false_attested": True,
            "request_manifest_hash_exact": True,
        },
        "generated_at": generated_at,
        "input_runs": {
            "control": _score_run_audit_row(control),
            "treatment": _score_run_audit_row(treatment),
        },
        "invariants": {
            **{key: treatment.metadata[key] for key in _PAIR_INVARIANT_KEYS},
            "base_scoring_signature": treatment.base_scoring_signature,
        },
        "num_requests": len(candidates),
        "mechanism_probe_manifest_admission": mechanism_probe_manifest_admission,
        "qrels_read": False,
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "request_manifest_path": str(request_manifest_path),
        "request_manifest_sha256": request_manifest_sha256,
        "score_rows_per_run": expected_rows,
        "split": SUPPORTED_SPLIT,
        "status": "passed",
    }
    return _AuditState(
        candidates=candidates,
        normalized_query_clusters=normalized_query_clusters,
        treatment=treatment,
        control=control,
        report=report,
        candidate_manifest_path=candidate_manifest_path,
        request_manifest_path=request_manifest_path,
        records_path=records_path,
    )


def _load_score_run(
    *,
    runs_dir: Path,
    run_id: str,
    candidate_manifest_sha256: str,
    request_manifest_sha256: str,
    candidates: dict[str, list[str]],
) -> _ScoreRun:
    run_dir = runs_dir / run_id
    metadata_path = run_dir / "metadata.json"
    scores_path = run_dir / "scores.jsonl"
    if not metadata_path.is_file() or not scores_path.is_file():
        raise FileNotFoundError(f"incomplete score run: {run_dir}")
    metadata = _read_json_object(metadata_path)
    required = {
        *_PAIR_INVARIANT_KEYS,
        "candidate_manifest_sha256",
        "request_manifest_sha256",
        "qrels_read",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"score run {run_id} is missing metadata: {missing}")
    empty_invariants = [
        key
        for key in _PAIR_INVARIANT_KEYS
        if metadata[key] is None
        or (isinstance(metadata[key], str) and not metadata[key].strip())
    ]
    if empty_invariants:
        raise ValueError(
            f"score run {run_id} has empty invariant metadata: {empty_invariants}"
        )
    if metadata["qrels_read"] is not False:
        raise ValueError(f"score run {run_id} must declare qrels_read=false")
    if metadata["split"] != SUPPORTED_SPLIT:
        raise ValueError(
            f"score run {run_id} split={metadata['split']!r}; only internal-dev dev is allowed"
        )
    if metadata["candidate_manifest_sha256"] != candidate_manifest_sha256:
        raise ValueError(f"score run {run_id} candidate manifest hash mismatch")
    if metadata["request_manifest_sha256"] != request_manifest_sha256:
        raise ValueError(f"score run {run_id} request manifest hash mismatch")

    base_scoring_signature = _base_scoring_signature(metadata, run_id)
    scores = _load_scores(scores_path)
    _assert_score_coverage(candidates, scores, run_id)
    expected_rows = sum(len(items) for items in candidates.values())
    if "request_count" in metadata and int(metadata["request_count"]) != len(candidates):
        raise ValueError(f"score run {run_id} request_count mismatch")
    if "score_rows" in metadata and int(metadata["score_rows"]) != expected_rows:
        raise ValueError(f"score run {run_id} score_rows mismatch")
    scores_sha256 = sha256_file(scores_path)
    if metadata.get("scores_sha256") not in (None, scores_sha256):
        raise ValueError(f"score run {run_id} scores_sha256 mismatch")
    return _ScoreRun(
        run_id=run_id,
        run_dir=run_dir,
        metadata=metadata,
        scores=scores,
        scores_sha256=scores_sha256,
        metadata_sha256=sha256_file(metadata_path),
        base_scoring_signature=base_scoring_signature,
    )


def _assert_pair_invariants(treatment: _ScoreRun, control: _ScoreRun) -> None:
    for key in _PAIR_INVARIANT_KEYS:
        if _canonical(treatment.metadata[key]) != _canonical(control.metadata[key]):
            raise ValueError(
                f"mechanism score pair invariant mismatch for {key}: "
                f"{treatment.metadata[key]!r} != {control.metadata[key]!r}"
            )
    if _canonical(treatment.base_scoring_signature) != _canonical(
        control.base_scoring_signature
    ):
        raise ValueError("mechanism score pair base_scoring_signature mismatch")


def _audit_probe_manifest_admission(
    treatment: _ScoreRun,
    control: _ScoreRun,
) -> dict[str, Any]:
    runs = {"treatment": treatment, "control": control}
    pair_contains_intervention = any(
        _is_mechanism_intervention(run.metadata) for run in runs.values()
    )
    actual_manifest_path = REPOSITORY_ROOT / MECHANISM_PROBE_MANIFEST_PATH
    if not actual_manifest_path.is_file():
        raise FileNotFoundError(
            f"missing frozen mechanism probe manifest: {actual_manifest_path}"
        )
    actual_sha256 = sha256_file(actual_manifest_path)
    if actual_sha256 != MECHANISM_PROBE_MANIFEST_SHA256:
        raise ValueError(
            "current mechanism probe manifest differs from its fixed expected SHA"
        )

    admissions = {}
    for role, run in runs.items():
        declaration = run.metadata.get("mechanism_probe_manifest")
        if declaration is None:
            if _is_frozen_legacy_reference(run.metadata):
                admissions[role] = {
                    "legacy_exemption": True,
                    "reason": (
                        "immutable first-round reference predates top-level "
                        "mechanism_probe_manifest provenance"
                    ),
                    "run_id": run.run_id,
                    "status": "legacy_exemption",
                }
                continue
            if pair_contains_intervention:
                raise ValueError(
                    f"mechanism intervention score run {run.run_id} is missing "
                    "mechanism_probe_manifest provenance"
                )
            admissions[role] = {
                "legacy_exemption": False,
                "reason": "pair contains no M1 mechanism intervention score run",
                "run_id": run.run_id,
                "status": "not_applicable_non_intervention_pair",
            }
            continue
        admissions[role] = _verify_probe_manifest_declaration(
            declaration,
            run_id=run.run_id,
        )
    return {
        "actual_path": str(actual_manifest_path),
        "actual_sha256": actual_sha256,
        "expected_path": MECHANISM_PROBE_MANIFEST_PATH,
        "expected_sha256": MECHANISM_PROBE_MANIFEST_SHA256,
        "pair_contains_mechanism_intervention": pair_contains_intervention,
        "runs": admissions,
    }


def _verify_probe_manifest_declaration(
    declaration: Any,
    *,
    run_id: str,
) -> dict[str, Any]:
    if not isinstance(declaration, dict):
        raise ValueError(
            f"score run {run_id} mechanism_probe_manifest must be an object"
        )
    expected = {
        "expected_sha256": MECHANISM_PROBE_MANIFEST_SHA256,
        "path": MECHANISM_PROBE_MANIFEST_PATH,
        "sha256": MECHANISM_PROBE_MANIFEST_SHA256,
        "verified": True,
    }
    for key, value in expected.items():
        matches = (
            declaration.get(key) is True
            if key == "verified"
            else declaration.get(key) == value
        )
        if not matches:
            raise ValueError(
                f"score run {run_id} mechanism_probe_manifest {key} mismatch"
            )
    return {
        "declared": dict(declaration),
        "legacy_exemption": False,
        "run_id": run_id,
        "status": "verified",
    }


def _is_mechanism_intervention(metadata: dict[str, Any]) -> bool:
    return "intervention" in metadata or metadata.get("evidence_mode") in {
        "mechanism_diagnostic",
        "smoke_non_result",
    }


def _is_frozen_legacy_reference(metadata: dict[str, Any]) -> bool:
    return (
        metadata.get("evidence_mode") == "first_round_pilot"
        and "intervention" not in metadata
        and "condition_id" not in metadata
        and metadata.get("history_condition") in {"true", "null", "wrong"}
    )


def _base_scoring_signature(metadata: dict[str, Any], run_id: str) -> Any:
    value = metadata.get("base_scoring_signature")
    if value is None:
        value = metadata.get("scoring_signature")
    if value is None:
        raise ValueError(
            f"score run {run_id} is missing base_scoring_signature/scoring_signature"
        )
    return value


def _load_candidates(path: Path, split: str) -> tuple[dict[str, list[str]], str | None]:
    manifest = _read_json_object(path)
    result: dict[str, list[str]] = {}
    for entry in manifest.get("entries", []):
        if entry.get("split") != split:
            continue
        request_id = str(entry["request_id"])
        if request_id in result:
            raise ValueError(f"duplicate candidate manifest request_id={request_id}")
        item_ids = [str(item_id) for item_id in entry["candidate_item_ids"]]
        if len(item_ids) < 2 or len(set(item_ids)) != len(item_ids):
            raise ValueError(f"invalid candidate slate for request_id={request_id}")
        result[request_id] = item_ids
    if not result:
        raise ValueError(f"no candidate entries for split={split}")
    declared_version = manifest.get("dataset_version")
    return result, str(declared_version) if declared_version is not None else None


def _manifest_dataset_version(path: Path) -> str | None:
    manifest = _read_json_object(path)
    value = manifest.get("dataset_version")
    return str(value) if value is not None else None


def _audit_label_free_records(
    path: Path, candidates: dict[str, list[str]]
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id in records:
            raise ValueError(f"duplicate records_dev request_id={request_id}")
        forbidden = sorted(_FORBIDDEN_LABEL_FIELDS & set(row))
        if forbidden:
            raise ValueError(
                f"records_dev contains forbidden label fields for request_id={request_id}: "
                f"{forbidden}"
            )
        records[request_id] = row
    if set(records) != set(candidates):
        raise ValueError("records_dev and candidate manifest have different request coverage")
    clusters = {}
    for request_id, item_ids in candidates.items():
        record_item_ids = [
            str(candidate["item_id"])
            for candidate in records[request_id].get("candidates", [])
        ]
        if record_item_ids != item_ids:
            raise ValueError(
                f"records_dev candidate order/identity mismatch for request_id={request_id}"
            )
        clusters[request_id] = _normalize_query_cluster(
            str(records[request_id].get("query", ""))
        )
    return clusters, records


def _audit_request_manifest(
    path: Path,
    candidates: dict[str, list[str]],
    records: dict[str, dict[str, Any]],
) -> None:
    manifest = _read_json_object(path)
    entries: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("entries", []):
        if entry.get("split") != SUPPORTED_SPLIT:
            continue
        request_id = str(entry["request_id"])
        if request_id in entries:
            raise ValueError(f"duplicate request manifest request_id={request_id}")
        entries[request_id] = entry
    if set(entries) != set(candidates):
        raise ValueError(
            "request manifest and candidate manifest have different dev coverage"
        )
    for request_id, item_ids in candidates.items():
        entry = entries[request_id]
        expected_candidate_hash = sha256_text(
            json.dumps(item_ids, separators=(",", ":"))
        )
        if entry.get("candidate_item_ids_sha256") != expected_candidate_hash:
            raise ValueError(
                f"request manifest candidate hash mismatch request_id={request_id}"
            )
        expected_query_hash = sha256_text(str(records[request_id].get("query", "")))
        if entry.get("query_sha256") != expected_query_hash:
            raise ValueError(
                f"request manifest query hash mismatch request_id={request_id}"
            )


def _load_scores(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in iter_jsonl(path):
        try:
            request_id = str(row["request_id"])
            item_id = str(row["candidate_item_id"])
        except KeyError as exc:
            raise ValueError(f"score row is missing identity field: {path}") from exc
        request_scores = result.setdefault(request_id, {})
        if item_id in request_scores:
            raise ValueError(
                f"duplicate score for request_id={request_id} item_id={item_id}"
            )
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid score for request_id={request_id} item_id={item_id}"
            ) from exc
        if not math.isfinite(score):
            raise ValueError(
                f"non-finite score for request_id={request_id} item_id={item_id}"
            )
        request_scores[item_id] = score
    if not result:
        raise ValueError(f"empty score file: {path}")
    return result


def _assert_score_coverage(
    candidates: dict[str, list[str]],
    scores: dict[str, dict[str, float]],
    run_id: str,
) -> None:
    if set(scores) != set(candidates):
        missing = sorted(set(candidates) - set(scores))[:5]
        extra = sorted(set(scores) - set(candidates))[:5]
        raise ValueError(
            f"score run {run_id} request coverage mismatch: missing={missing} extra={extra}"
        )
    for request_id, item_ids in candidates.items():
        if set(scores[request_id]) != set(item_ids):
            missing = sorted(set(item_ids) - set(scores[request_id]))[:5]
            extra = sorted(set(scores[request_id]) - set(item_ids))[:5]
            raise ValueError(
                f"score run {run_id} candidate coverage mismatch for "
                f"request_id={request_id}: missing={missing} extra={extra}"
            )


def _load_graded_gains(
    path: Path, candidates: dict[str, list[str]]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id in result:
            raise ValueError(f"duplicate qrels_dev request_id={request_id}")
        relevance = row.get("relevance", {})
        if not isinstance(relevance, dict):
            raise ValueError("graded qrels relevance must be an item-to-gain object")
        gains: dict[str, float] = {}
        for item_id, raw_gain in relevance.items():
            try:
                gain = float(raw_gain)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid graded gain for request_id={request_id} item_id={item_id}"
                ) from exc
            if not math.isfinite(gain) or gain < 0:
                raise ValueError(
                    f"invalid graded gain for request_id={request_id} item_id={item_id}"
                )
            gains[str(item_id)] = gain
        result[request_id] = gains
    if set(result) != set(candidates):
        raise ValueError("qrels_dev and candidate manifest have different request coverage")
    for request_id, gains in result.items():
        unknown = set(gains) - set(candidates[request_id])
        if unknown:
            raise ValueError(
                f"qrels_dev contains non-candidate items for request_id={request_id}: "
                f"{sorted(unknown)[:5]}"
            )
    return result


def _evaluate_request(
    *,
    request_id: str,
    item_ids: list[str],
    gains: dict[str, float],
    treatment_scores: dict[str, float],
    control_scores: dict[str, float],
) -> dict[str, Any]:
    gain_values = [float(gains.get(item_id, 0.0)) for item_id in item_ids]
    treatment_values = [treatment_scores[item_id] for item_id in item_ids]
    control_values = [control_scores[item_id] for item_id in item_ids]
    treatment_ndcg = gain_ndcg_at_k(
        request_id, item_ids, treatment_values, gain_values, 10
    )
    control_ndcg = gain_ndcg_at_k(
        request_id, item_ids, control_values, gain_values, 10
    )

    max_gain = max(gain_values)
    target_index = (
        next(
            (index for index, gain in enumerate(gain_values) if gain == max_gain),
            None,
        )
        if max_gain > 0
        else None
    )
    lower_gain_indexes = (
        [index for index, gain in enumerate(gain_values) if gain < max_gain]
        if max_gain > 0
        else []
    )
    treatment_competitor_index = _best_score_index(
        lower_gain_indexes, treatment_values
    )
    control_competitor_index = _best_score_index(lower_gain_indexes, control_values)
    margin_eligible = (
        target_index is not None
        and treatment_competitor_index is not None
        and control_competitor_index is not None
    )
    if margin_eligible:
        assert target_index is not None
        assert treatment_competitor_index is not None
        assert control_competitor_index is not None
        treatment_margin = (
            treatment_values[target_index]
            - treatment_values[treatment_competitor_index]
        )
        control_margin = (
            control_values[target_index] - control_values[control_competitor_index]
        )
        margin_change = treatment_margin - control_margin
        if not all(
            math.isfinite(value)
            for value in (treatment_margin, control_margin, margin_change)
        ):
            raise ValueError(
                f"non-finite derived target margin for request_id={request_id}"
            )
    else:
        treatment_margin = None
        control_margin = None
        margin_change = None

    return {
        "candidate_count": len(item_ids),
        "control_best_lower_gain_competitor_item_id": (
            item_ids[control_competitor_index]
            if control_competitor_index is not None
            else None
        ),
        "control_ndcg@10": control_ndcg,
        "control_target_margin": control_margin,
        "margin_eligible": margin_eligible,
        "positive_eligible": max_gain > 0,
        "request_id": request_id,
        "target_candidate_position": target_index,
        "target_gain": max_gain if max_gain > 0 else None,
        "target_item_id": item_ids[target_index] if max_gain > 0 else None,
        "target_margin_change": margin_change,
        "treatment_best_lower_gain_competitor_item_id": (
            item_ids[treatment_competitor_index]
            if treatment_competitor_index is not None
            else None
        ),
        "treatment_minus_control_ndcg@10": treatment_ndcg - control_ndcg,
        "treatment_ndcg@10": treatment_ndcg,
        "treatment_target_margin": treatment_margin,
    }


def _best_score_index(indexes: list[int], scores: list[float]) -> int | None:
    if not indexes:
        return None
    best = indexes[0]
    for index in indexes[1:]:
        # Strict comparison preserves candidate order as the score tie break.
        if scores[index] > scores[best]:
            best = index
    return best


def _summarize_rows(
    rows: list[dict[str, Any]], cluster_by_request: dict[str, str]
) -> dict[str, Any]:
    if not rows:
        return {
            "mean_control_ndcg@10": None,
            "mean_target_margin_change": None,
            "mean_treatment_minus_control_ndcg@10": None,
            "mean_treatment_ndcg@10": None,
            "num_margin_eligible_requests": 0,
            "num_query_clusters": 0,
            "num_requests": 0,
            "query_cluster_ci95": {
                "target_margin_change": None,
                "treatment_minus_control_ndcg@10": None,
            },
            "target_margin_change": {"mean": None, "query_cluster_ci95": None},
            "treatment_minus_control_ndcg@10": {
                "mean": None,
                "query_cluster_ci95": None,
            },
        }
    intervals = cluster_bootstrap_mean_ci(
        rows,
        cluster_by_request,
        ("treatment_minus_control_ndcg@10", "target_margin_change"),
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    treatment_mean = _mean_present(rows, "treatment_ndcg@10")
    control_mean = _mean_present(rows, "control_ndcg@10")
    ndcg_delta_mean = _mean_present(rows, "treatment_minus_control_ndcg@10")
    margin_mean = _mean_present(rows, "target_margin_change")
    return {
        "mean_control_ndcg@10": control_mean,
        "mean_target_margin_change": margin_mean,
        "mean_treatment_minus_control_ndcg@10": ndcg_delta_mean,
        "mean_treatment_ndcg@10": treatment_mean,
        "num_margin_eligible_requests": sum(bool(row["margin_eligible"]) for row in rows),
        "num_query_clusters": len(
            {cluster_by_request[str(row["request_id"])] for row in rows}
        ),
        "num_requests": len(rows),
        "query_cluster_ci95": intervals,
        "target_margin_change": {
            "mean": margin_mean,
            "query_cluster_ci95": intervals["target_margin_change"],
        },
        "treatment_minus_control_ndcg@10": {
            "mean": ndcg_delta_mean,
            "query_cluster_ci95": intervals[
                "treatment_minus_control_ndcg@10"
            ],
        },
    }


def _target_partition_by_request(
    memberships: dict[str, set[str]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for surface in ALL_REQUEST_PARTITION:
        for request_id in memberships[surface]:
            if request_id in result:
                raise AssertionError(
                    f"target-aware partition overlap for request_id={request_id}"
                )
            result[request_id] = surface
    if result.keys() != memberships["all"]:
        raise AssertionError("target-aware partition does not cover all requests")
    return result


def _ndcg_partition_contributions(
    *,
    overall: dict[str, Any],
    surfaces: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    total = int(overall["num_requests"])
    if total <= 0:
        raise ValueError("cannot decompose an empty mechanism population")
    result = {}
    reconstructed = 0.0
    for surface in ALL_REQUEST_PARTITION:
        values = surfaces[surface]
        mean = values["mean_treatment_minus_control_ndcg@10"]
        if mean is None and values["num_requests"]:
            raise AssertionError(f"missing NDCG contrast for nonempty surface={surface}")
        prevalence = int(values["num_requests"]) / total
        contribution = prevalence * float(mean or 0.0)
        reconstructed += contribution
        result[surface] = {
            "contribution": contribution,
            "mean": mean,
            "num_requests": values["num_requests"],
            "prevalence": prevalence,
        }
    overall_mean = float(overall["mean_treatment_minus_control_ndcg@10"])
    if not math.isclose(overall_mean, reconstructed, rel_tol=0.0, abs_tol=1.0e-12):
        raise AssertionError(
            "target-aware NDCG contribution identity failed: "
            f"{overall_mean} != {reconstructed}"
        )
    return {
        "metric": "treatment_minus_control_ndcg@10",
        "overall_mean": overall_mean,
        "reconstructed_mean": reconstructed,
        "surfaces": result,
    }


def _score_run_audit_row(run: _ScoreRun) -> dict[str, Any]:
    rows = sum(len(values) for values in run.scores.values())
    return {
        "condition_id": _resolved_condition_id(run.metadata),
        "metadata_path": str(run.run_dir / "metadata.json"),
        "metadata_sha256": run.metadata_sha256,
        "qrels_read": run.metadata["qrels_read"],
        "raw_condition_id": run.metadata.get("condition_id"),
        "raw_history_condition": run.metadata.get("history_condition"),
        "request_count": len(run.scores),
        "run_id": run.run_id,
        "score_rows": rows,
        "scores_path": str(run.run_dir / "scores.jsonl"),
        "scores_sha256": run.scores_sha256,
    }


def _append_dev_eval_log(
    path: str | Path, *, metrics: dict[str, Any], metrics_path: Path
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": 1,
        "analysis_type": metrics["analysis_type"],
        "control_condition_id": metrics["control_condition_id"],
        "control_raw_history_condition": metrics["control_raw_history_condition"],
        "control_ndcg@10": metrics["mean_control_ndcg@10"],
        "control_run_id": metrics["control_run_id"],
        "label_mode": "graded",
        "method_id": "shared_mechanism_evaluator",
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "ndcg@10": metrics["mean_treatment_ndcg@10"],
        "run_id": metrics["analysis_run_id"],
        "split": SUPPORTED_SPLIT,
        "target_margin_change": metrics["mean_target_margin_change"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "treatment_condition_id": metrics["treatment_condition_id"],
        "treatment_raw_history_condition": metrics[
            "treatment_raw_history_condition"
        ],
        "treatment_minus_control_ndcg@10": metrics[
            "mean_treatment_minus_control_ndcg@10"
        ],
        "treatment_run_id": metrics["treatment_run_id"],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _normalize_query_cluster(query: str) -> str:
    return "".join(str(query).casefold().split())


def _mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _resolved_condition_id(metadata: dict[str, Any]) -> Any:
    condition_id = metadata.get("condition_id")
    if condition_id is not None:
        return condition_id
    return _FROZEN_HISTORY_CONDITION_IDS.get(metadata.get("history_condition"))


def _require_internal_dev(split: str) -> None:
    if split != SUPPORTED_SPLIT:
        raise ValueError(
            "mechanism evaluation is restricted to label-free internal-dev "
            "split=dev; confirmation and test are locked"
        )
