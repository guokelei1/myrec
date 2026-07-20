"""Qrels-free statistical synthesis for completed mechanism analyses.

Only four immutable artifacts are read from each caller-supplied analysis
directory: ``metrics.json``, ``per_request.jsonl``, ``metadata.json``, and
``pre_qrels_audit.json``.  Paths embedded inside those artifacts are never
followed.  In particular, this module has no records, qrels, score, or source
test reader.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.utils.hashing import sha256_file, sha256_text


PROBE_MANIFEST_PATH = Path("experiments/motivation/probe_manifest.yaml")
PROBE_MANIFEST_SHA256 = (
    "adedf0e662b9d8529162b8abffedcf6b10962913f28580af6119d807cc5d929c"
)
PROBE_MANIFEST_ID = "motivation_mechanism_first_diagnosis_v1"
BOOTSTRAP_CLUSTER = "normalized_query"
BOOTSTRAP_SEED = 20_260_715
BOOTSTRAP_SAMPLES = 5_000
FDR_ALPHA = 0.05
STRICT_TRANSFER_SURFACE = "target_nonrepeat_no_candidate_overlap"
ANALYSIS_TYPE = "motivation_mechanism_paired_probe"
INPUT_FILENAMES = (
    "metrics.json",
    "per_request.jsonl",
    "metadata.json",
    "pre_qrels_audit.json",
)
NUMERIC_TOLERANCE = 1.0e-12
M0_FAMILY = "m0_recoverability"
M1_FAMILY = "m1_input_interventions"


@dataclass(frozen=True)
class EndpointSpec:
    endpoint_id: str
    manifest_name: str
    row_key: str
    metrics_mean_key: str


ENDPOINT_SPECS = (
    EndpointSpec(
        endpoint_id="strict_transfer_target_margin_delta",
        manifest_name="target_vs_best_lower_gain_competitor_margin_change",
        row_key="target_margin_change",
        metrics_mean_key="mean_target_margin_change",
    ),
    EndpointSpec(
        endpoint_id="strict_transfer_ndcg_delta",
        manifest_name="treatment_minus_control_graded_ndcg_at_10",
        row_key="treatment_minus_control_ndcg@10",
        metrics_mean_key="mean_treatment_minus_control_ndcg@10",
    ),
)
_ENDPOINT_BY_MANIFEST_NAME = {
    endpoint.manifest_name: endpoint for endpoint in ENDPOINT_SPECS
}
_M0_ENDPOINT_ALIASES = {
    "strict_transfer_target_vs_best_lower_gain_competitor_margin_change": (
        ENDPOINT_SPECS[0]
    ),
    "strict_transfer_treatment_minus_null_graded_ndcg_at_10": ENDPOINT_SPECS[1],
}


@dataclass(frozen=True)
class RegisteredCell:
    cell_id: str
    variant_id: str | None
    method_id: str | None
    treatment_condition_id: str
    control_condition_id: str


@dataclass(frozen=True)
class FamilyRegistration:
    family: str
    registration_kind: str
    models: tuple[str, ...]
    comparisons: tuple[tuple[str, str], ...]
    cells: tuple[RegisteredCell, ...]
    endpoints: tuple[EndpointSpec, ...]
    manifest_endpoint_names: tuple[str, ...]
    probe_identity: dict[str, Any]
    source_registration: dict[str, Any]


@dataclass(frozen=True)
class AnalysisArtifacts:
    analysis_dir: Path
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    pre_qrels_audit: dict[str, Any]
    per_request: tuple[dict[str, Any], ...]
    input_identity: dict[str, Any]


def synthesize_mechanism_statistics(
    *,
    family: str,
    analysis_dirs: Sequence[str | Path],
    output_path: str | Path,
    probe_manifest_path: str | Path = PROBE_MANIFEST_PATH,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate a complete registered family and write its machine JSON."""

    registration = load_family_registration(family, probe_manifest_path)
    if not analysis_dirs:
        raise ValueError("at least one --analysis-dir is required")
    resolved_dirs = [Path(path).resolve() for path in analysis_dirs]
    if len(resolved_dirs) != len(set(resolved_dirs)):
        raise ValueError("analysis directories must be unique")

    analysis_results = []
    input_identities = []
    for analysis_dir in resolved_dirs:
        artifacts = load_analysis_artifacts(analysis_dir)
        result = summarize_analysis(artifacts, registration)
        analysis_results.append(result)
        input_identities.append(artifacts.input_identity)
    ordered_analyses, family_identity_gate = _order_registered_analyses(
        analysis_results,
        registration,
    )
    hypotheses = []
    for analysis in ordered_analyses:
        for endpoint in registration.endpoints:
            endpoint_result = analysis["strict_transfer"]["endpoints"][
                endpoint.endpoint_id
            ]
            hypothesis_id = _hypothesis_id(
                analysis["method_id"],
                analysis["treatment_condition_id"],
                analysis["control_condition_id"],
                endpoint.endpoint_id,
                registered_cell_id=analysis["registered_cell_id"],
            )
            hypotheses.append(
                {
                    "analysis_run_id": analysis["analysis_run_id"],
                    "control_condition_id": analysis["control_condition_id"],
                    "endpoint_id": endpoint.endpoint_id,
                    "hypothesis_id": hypothesis_id,
                    "method_id": analysis["method_id"],
                    "registered_cell_id": analysis["registered_cell_id"],
                    "raw_p": endpoint_result["bootstrap"]["two_sided_p"],
                    "treatment_condition_id": analysis["treatment_condition_id"],
                }
            )
    fdr_rows = benjamini_hochberg(hypotheses, alpha=FDR_ALPHA)
    fdr_by_id = {row["hypothesis_id"]: row for row in fdr_rows}
    for analysis in ordered_analyses:
        for endpoint in registration.endpoints:
            hypothesis_id = _hypothesis_id(
                analysis["method_id"],
                analysis["treatment_condition_id"],
                analysis["control_condition_id"],
                endpoint.endpoint_id,
                registered_cell_id=analysis["registered_cell_id"],
            )
            row = fdr_by_id[hypothesis_id]
            analysis["strict_transfer"]["endpoints"][endpoint.endpoint_id][
                "fdr"
            ] = {
                "q_value": row["q_value"],
                "reject_at_0_05": row["reject_at_0_05"],
            }

    input_identities.sort(key=lambda row: row["analysis_dir"])
    registered_hypothesis_ids = [row["hypothesis_id"] for row in hypotheses]
    observed_hypothesis_ids = [row["hypothesis_id"] for row in fdr_rows]
    report = {
        "schema_version": 1,
        "analysis_type": "motivation_mechanism_stage_statistical_synthesis",
        "analyses": ordered_analyses,
        "bootstrap": {
            "ci": "sorted_draws[index_int_0.025B,index_min_B-1_int_0.975B]",
            "cluster": BOOTSTRAP_CLUSTER,
            "p_value": (
                "min(1,2*min((1+count(draw<=0))/(B+1),"
                "(1+count(draw>=0))/(B+1))); zero is included in both tails; "
                "point_estimate_zero maps to 1"
            ),
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "code_revision": _git_revision(),
        "command": list(command or sys.argv),
        "family": family,
        "family_identity_gate": family_identity_gate,
        "fdr": {
            "alpha": FDR_ALPHA,
            "family_size": len(fdr_rows),
            "method": "benjamini_hochberg",
            "missing_hypothesis_ids": [],
            "observed_hypothesis_ids": observed_hypothesis_ids,
            "registered_hypothesis_ids": registered_hypothesis_ids,
            "results": fdr_rows,
            "unexpected_hypothesis_ids": [],
        },
        "folds": {
            "count": 2,
            "rule": "int(sha256(normalized_query_cluster),16) mod 2",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "implementation_identity": statistical_synthesis_implementation_identity(),
        "input_analyses": input_identities,
        "input_set_sha256": _canonical_sha256(input_identities),
        "metrics_validation_tolerance": NUMERIC_TOLERANCE,
        "probe_manifest": registration.probe_identity,
        "registration": {
            "cells": [
                {
                    "cell_id": cell.cell_id,
                    "control_condition_id": cell.control_condition_id,
                    "method_id": cell.method_id,
                    "treatment_condition_id": cell.treatment_condition_id,
                    "variant_id": cell.variant_id,
                }
                for cell in registration.cells
            ],
            "comparisons": [list(value) for value in registration.comparisons],
            "endpoints": list(registration.manifest_endpoint_names),
            "models": list(registration.models),
            "registration_kind": registration.registration_kind,
            "source_registration": registration.source_registration,
        },
        "strict_transfer_surface": STRICT_TRANSFER_SURFACE,
    }
    _write_new_json(Path(output_path), report)
    return report


def load_family_registration(
    family: str,
    probe_manifest_path: str | Path = PROBE_MANIFEST_PATH,
) -> FamilyRegistration:
    """Load one complete family registration from the exact frozen manifest."""

    if not family or not str(family).strip():
        raise ValueError("family must be non-empty")
    root = Path(__file__).resolve().parents[3]
    expected_path = (root / PROBE_MANIFEST_PATH).resolve()
    supplied_path = Path(probe_manifest_path)
    supplied_path = (
        (Path.cwd() / supplied_path).resolve()
        if not supplied_path.is_absolute()
        else supplied_path.resolve()
    )
    if supplied_path != expected_path:
        raise ValueError(f"probe manifest path must be {PROBE_MANIFEST_PATH}")
    payload_bytes = _read_regular_file(supplied_path)
    observed_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if observed_sha256 != PROBE_MANIFEST_SHA256:
        raise ValueError("frozen mechanism probe manifest hash mismatch")
    import yaml

    payload = yaml.safe_load(payload_bytes.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("probe_manifest_id") != PROBE_MANIFEST_ID:
        raise ValueError("frozen mechanism probe manifest identity mismatch")
    randomization = payload.get("randomization")
    expected_randomization = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_cluster": BOOTSTRAP_CLUSTER,
        "two_fold_check": "sha256_normalized_query_mod_2",
        "within_family_fdr": "benjamini_hochberg_0_05",
    }
    if not isinstance(randomization, Mapping):
        raise ValueError("probe manifest randomization registration is missing")
    for key, expected in expected_randomization.items():
        if randomization.get(key) != expected:
            raise ValueError(f"probe manifest randomization drift: {key}")
    family_payload = payload.get(family)
    if not isinstance(family_payload, Mapping):
        raise ValueError(f"probe manifest has no family={family!r}")
    probe_identity = {
        "expected_sha256": PROBE_MANIFEST_SHA256,
        "path": PROBE_MANIFEST_PATH.as_posix(),
        "sha256": observed_sha256,
        "verified": True,
    }
    if family == M0_FAMILY:
        raw_conditions = family_payload.get("conditions")
        if not isinstance(raw_conditions, list) or not raw_conditions:
            raise ValueError("m0 conditions must be a non-empty list")
        conditions = [
            "null" if value is None else str(value)
            for value in raw_conditions
        ]
        if any(not value for value in conditions) or len(conditions) != len(
            set(conditions)
        ):
            raise ValueError("normalized M0 conditions are empty or duplicated")
        required_conditions = {
            "full",
            "null",
            "history_shuffle",
            "routing_query_shuffle",
        }
        if set(conditions) != required_conditions:
            raise ValueError("frozen M0 condition registration drift")
        negative_control = family_payload.get("separately_fitted_negative_control")
        if negative_control != "within_request_label_shuffle":
            raise ValueError("frozen M0 separately fitted negative control drift")
        endpoint_names = (
            family_payload.get("primary_endpoint"),
            family_payload.get("secondary_endpoint"),
        )
        if (
            not all(isinstance(value, str) and value for value in endpoint_names)
            or set(endpoint_names) != set(_M0_ENDPOINT_ALIASES)
        ):
            raise ValueError("frozen M0 strict endpoint registration drift")
        endpoints = tuple(_M0_ENDPOINT_ALIASES[str(name)] for name in endpoint_names)
        cells = (
            RegisteredCell(
                "real__full__vs__null", "real", None, "full", "null"
            ),
            RegisteredCell(
                "real__history_shuffle__vs__null",
                "real",
                None,
                "history_shuffle",
                "null",
            ),
            RegisteredCell(
                "real__routing_query_shuffle__vs__null",
                "real",
                None,
                "routing_query_shuffle",
                "null",
            ),
            RegisteredCell(
                "within_request_label_shuffle__full__vs__null",
                str(negative_control),
                None,
                "full",
                "null",
            ),
            RegisteredCell(
                "real__full__vs__history_shuffle",
                "real",
                None,
                "full",
                "history_shuffle",
            ),
            RegisteredCell(
                "real__full__vs__routing_query_shuffle",
                "real",
                None,
                "full",
                "routing_query_shuffle",
            ),
        )
        return FamilyRegistration(
            family=family,
            registration_kind="m0_condition_graph_with_separate_fitted_control",
            models=(),
            comparisons=tuple(
                dict.fromkeys(
                    (cell.treatment_condition_id, cell.control_condition_id)
                    for cell in cells
                )
            ),
            cells=cells,
            endpoints=endpoints,
            manifest_endpoint_names=tuple(str(value) for value in endpoint_names),
            probe_identity=probe_identity,
            source_registration={
                "conditions": conditions,
                "null_condition_yaml_value_normalized_from_none": (
                    any(value is None for value in raw_conditions)
                ),
                "separately_fitted_negative_control": negative_control,
            },
        )
    if family != M1_FAMILY:
        raise ValueError(
            "statistical synthesis supports only frozen M0 recoverability or M1 input interventions"
        )
    models = _nonempty_unique_strings(family_payload.get("models"), "models")
    raw_comparisons = family_payload.get("registered_comparisons")
    if not isinstance(raw_comparisons, list) or not raw_comparisons:
        raise ValueError(
            f"family={family!r} has no explicit registered_comparisons"
        )
    comparisons = []
    for index, value in enumerate(raw_comparisons):
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not all(isinstance(item, str) and item for item in value)
        ):
            raise ValueError(f"invalid registered comparison at index={index}")
        comparisons.append((value[0], value[1]))
    if len(comparisons) != len(set(comparisons)):
        raise ValueError("registered comparisons contain duplicates")
    endpoint_names = _nonempty_unique_strings(
        family_payload.get("primary_endpoints"),
        "primary_endpoints",
    )
    if len(endpoint_names) != 2 or set(endpoint_names) != set(
        _ENDPOINT_BY_MANIFEST_NAME
    ):
        raise ValueError(
            "statistical synthesis requires the two registered NDCG/margin endpoints"
        )
    endpoints = tuple(_ENDPOINT_BY_MANIFEST_NAME[name] for name in endpoint_names)
    cells = tuple(
        RegisteredCell(
            cell_id=f"{method_id}__{treatment}__vs__{control}",
            variant_id=None,
            method_id=method_id,
            treatment_condition_id=treatment,
            control_condition_id=control,
        )
        for method_id in models
        for treatment, control in comparisons
    )
    return FamilyRegistration(
        family=family,
        registration_kind="manifest_models_cross_registered_comparisons",
        models=tuple(models),
        comparisons=tuple(comparisons),
        cells=cells,
        endpoints=endpoints,
        manifest_endpoint_names=tuple(endpoint_names),
        probe_identity=probe_identity,
        source_registration={
            "models": list(models),
            "registered_comparisons": [list(value) for value in comparisons],
        },
    )


def load_analysis_artifacts(analysis_dir: str | Path) -> AnalysisArtifacts:
    """Read only the four fixed, non-symlink analysis artifacts."""

    analysis_dir = Path(analysis_dir).resolve()
    if not analysis_dir.is_dir():
        raise FileNotFoundError(f"analysis directory is missing: {analysis_dir}")
    payloads: dict[str, bytes] = {}
    file_identities = {}
    for filename in INPUT_FILENAMES:
        path = analysis_dir / filename
        payload = _read_regular_file(path)
        payloads[filename] = payload
        file_identities[filename] = {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    metrics = _parse_json_object(payloads["metrics.json"], "metrics.json")
    metadata = _parse_json_object(payloads["metadata.json"], "metadata.json")
    pre_qrels_audit = _parse_json_object(
        payloads["pre_qrels_audit.json"],
        "pre_qrels_audit.json",
    )
    per_request = _parse_jsonl_objects(
        payloads["per_request.jsonl"],
        "per_request.jsonl",
    )
    identity = {
        "analysis_dir": str(analysis_dir),
        "analysis_run_id": metrics.get("analysis_run_id"),
        "files": file_identities,
    }
    identity["combined_sha256"] = _canonical_sha256(identity)
    return AnalysisArtifacts(
        analysis_dir=analysis_dir,
        metrics=metrics,
        metadata=metadata,
        pre_qrels_audit=pre_qrels_audit,
        per_request=tuple(per_request),
        input_identity=identity,
    )


def summarize_analysis(
    artifacts: AnalysisArtifacts,
    registration: FamilyRegistration,
) -> dict[str, Any]:
    """Validate one completed evaluator output and derive strict-transfer stats."""

    metrics = artifacts.metrics
    metadata = artifacts.metadata
    audit = artifacts.pre_qrels_audit
    for source, payload in (("metrics", metrics), ("metadata", metadata)):
        if payload.get("analysis_type") != ANALYSIS_TYPE:
            raise ValueError(f"{source} analysis_type is not a mechanism paired probe")
        if payload.get("split") != "dev":
            raise ValueError(f"{source} must be internal-dev split")
        if payload.get("label_mode") != "graded":
            raise ValueError(f"{source} must use the graded endpoint")
    if audit.get("analysis_type") != "motivation_mechanism_pre_qrels_score_audit":
        raise ValueError("pre_qrels_audit analysis_type mismatch")
    if (
        audit.get("status") != "passed"
        or audit.get("qrels_read") is not False
        or audit.get("split") != "dev"
    ):
        raise ValueError("pre_qrels_audit did not pass on the label-free boundary")
    audit_checks = audit.get("checks")
    if (
        not isinstance(audit_checks, Mapping)
        or not audit_checks
        or any(value is not True for value in audit_checks.values())
    ):
        raise ValueError("pre_qrels_audit checks are incomplete or failed")
    analysis_run_id = _required_string(metrics, "analysis_run_id", "metrics")
    if metadata.get("analysis_run_id") != analysis_run_id:
        raise ValueError("metrics/metadata analysis_run_id mismatch")
    pre_audit_identity = artifacts.input_identity["files"]["pre_qrels_audit.json"]
    if metadata.get("pre_qrels_audit_sha256") != pre_audit_identity["sha256"]:
        raise ValueError("metadata pre_qrels_audit SHA mismatch")
    admissions = [
        metrics.get("mechanism_probe_manifest_admission"),
        metadata.get("mechanism_probe_manifest_admission"),
        audit.get("mechanism_probe_manifest_admission"),
    ]
    if admissions[0] != admissions[1] or admissions[0] != admissions[2]:
        raise ValueError("probe manifest admission differs across analysis artifacts")
    _validate_probe_admission(admissions[0])

    expected_bootstrap = {
        "cluster": BOOTSTRAP_CLUSTER,
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
    }
    if metrics.get("bootstrap") != expected_bootstrap or metadata.get(
        "bootstrap"
    ) != expected_bootstrap:
        raise ValueError("completed evaluator bootstrap registration drift")
    method_id = _required_string(metrics, "method_id", "metrics")
    checkpoint_id = _required_string(metrics, "checkpoint_id", "metrics")
    treatment = _required_string(metrics, "treatment_condition_id", "metrics")
    control = _required_string(metrics, "control_condition_id", "metrics")
    invariants = metadata.get("invariants")
    audit_invariants = audit.get("invariants")
    if not isinstance(invariants, Mapping) or not isinstance(audit_invariants, Mapping):
        raise ValueError("analysis invariants are missing")
    if invariants.get("method_id") != method_id or audit_invariants.get(
        "method_id"
    ) != method_id:
        raise ValueError("method identity differs across analysis artifacts")
    if invariants.get("checkpoint_id") != checkpoint_id or audit_invariants.get(
        "checkpoint_id"
    ) != checkpoint_id:
        raise ValueError("checkpoint identity differs across analysis artifacts")
    if registration.family == M1_FAMILY and method_id not in registration.models:
        raise ValueError(f"method is not registered for M1: {method_id}")
    if (treatment, control) not in registration.comparisons:
        raise ValueError(
            "analysis is not a registered family cell: "
            f"{method_id} {treatment} vs {control}"
        )
    base_signature = invariants.get("base_scoring_signature")
    if not isinstance(base_signature, Mapping) or base_signature != audit_invariants.get(
        "base_scoring_signature"
    ):
        raise ValueError("base scoring signature differs across analysis artifacts")
    if registration.family == M0_FAMILY and base_signature.get(
        "checkpoint_id"
    ) != checkpoint_id:
        raise ValueError("M0 base scoring signature/checkpoint identity mismatch")

    rows = _validate_per_request_rows(
        artifacts.per_request,
        treatment=treatment,
        control=control,
    )
    if int(metrics.get("num_requests", -1)) != len(rows) or int(
        audit.get("num_requests", -1)
    ) != len(rows):
        raise ValueError("overall request count differs across analysis artifacts")
    metadata_conditions = metadata.get("conditions")
    if not isinstance(metadata_conditions, Mapping):
        raise ValueError("metadata conditions are missing")
    for role, expected in (("treatment", treatment), ("control", control)):
        condition = metadata_conditions.get(role)
        if not isinstance(condition, Mapping) or condition.get("condition_id") != expected:
            raise ValueError(f"metadata {role} condition identity mismatch")
    score_run_identities = {
        role: _score_run_identity(
            metadata_conditions,
            audit,
            role=role,
            expected_condition=(treatment if role == "treatment" else control),
        )
        for role in ("treatment", "control")
    }
    for field in ("run_id", "metadata_sha256", "scores_sha256"):
        if (
            score_run_identities["treatment"][field]
            == score_run_identities["control"][field]
        ):
            raise ValueError(
                f"paired treatment/control reuse the same score identity: {field}"
            )
    strict_rows = [
        row
        for row in rows
        if row["target_aware_surface"] == STRICT_TRANSFER_SURFACE
    ]
    if not strict_rows:
        raise ValueError(f"analysis={analysis_run_id} has no strict-transfer requests")
    clusters = {str(row["normalized_query_cluster"]) for row in strict_rows}
    folds = summarize_two_folds(strict_rows, ENDPOINT_SPECS)
    draws = cluster_bootstrap_draws(
        strict_rows,
        ENDPOINT_SPECS,
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    surface_metrics = _strict_surface_metrics(metrics)
    if int(surface_metrics.get("num_requests", -1)) != len(strict_rows):
        raise ValueError("strict-transfer request count differs from metrics.json")
    if int(surface_metrics.get("num_query_clusters", -1)) != len(clusters):
        raise ValueError("strict-transfer cluster count differs from metrics.json")

    endpoint_results = {}
    for endpoint in registration.endpoints:
        values = _present_values(strict_rows, endpoint.row_key)
        if not values:
            raise ValueError(
                f"strict-transfer endpoint has no finite values: {endpoint.endpoint_id}"
            )
        endpoint_draws = draws[endpoint.endpoint_id]
        if len(endpoint_draws) != BOOTSTRAP_SAMPLES:
            raise ValueError(
                f"endpoint={endpoint.endpoint_id} has only "
                f"{len(endpoint_draws)}/{BOOTSTRAP_SAMPLES} valid bootstrap draws"
            )
        mean = sum(values) / len(values)
        if endpoint.row_key == "target_margin_change" and int(
            surface_metrics.get("num_margin_eligible_requests", -1)
        ) != len(values):
            raise ValueError(
                "strict-transfer margin-eligible count differs from metrics.json"
            )
        ci95 = percentile_ci(endpoint_draws)
        _validate_metrics_endpoint(
            surface_metrics,
            endpoint,
            observed_mean=mean,
            observed_ci95=ci95,
        )
        fold_means = {
            fold: folds[fold]["endpoints"][endpoint.endpoint_id]["mean"]
            for fold in ("0", "1")
        }
        endpoint_results[endpoint.endpoint_id] = {
            "bootstrap": {
                **two_sided_bootstrap_p(endpoint_draws, mean),
                "ci95": ci95,
                "samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
                "valid_draws": len(endpoint_draws),
            },
            "direction_consistent": direction_consistent(mean, fold_means),
            "fold_means": fold_means,
            "mean": mean,
            "metrics_json_validation": {
                "ci95_matches": True,
                "point_estimate_matches": True,
                "tolerance": NUMERIC_TOLERANCE,
            },
            "num_query_clusters": len(
                {
                    str(row["normalized_query_cluster"])
                    for row in strict_rows
                    if row.get(endpoint.row_key) is not None
                }
            ),
            "num_requests": len(values),
        }
    return {
        "analysis_dir": str(artifacts.analysis_dir),
        "analysis_run_id": analysis_run_id,
        "base_scoring_signature_sha256": _canonical_sha256(base_signature),
        "base_scoring_signature_without_checkpoint_sha256": _canonical_sha256(
            {
                key: value
                for key, value in base_signature.items()
                if key != "checkpoint_id"
            }
        ),
        "checkpoint_id": checkpoint_id,
        "control_condition_id": control,
        "folds": folds,
        "input_combined_sha256": artifacts.input_identity["combined_sha256"],
        "method_id": method_id,
        "registered_cell_id": None,
        "score_run_identities": score_run_identities,
        "strict_transfer": {
            "endpoints": endpoint_results,
            "num_query_clusters": len(clusters),
            "num_requests": len(strict_rows),
            "surface_id": STRICT_TRANSFER_SURFACE,
        },
        "treatment_condition_id": treatment,
    }


def _order_registered_analyses(
    analyses: Sequence[dict[str, Any]],
    registration: FamilyRegistration,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if registration.family == M0_FAMILY:
        return _order_m0_registered_analyses(analyses, registration)
    observed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for analysis in analyses:
        key = (
            str(analysis["method_id"]),
            str(analysis["treatment_condition_id"]),
            str(analysis["control_condition_id"]),
        )
        if key in observed:
            raise ValueError(f"duplicate registered analysis cell: {key}")
        observed[key] = analysis
    expected = [
        (
            str(cell.method_id),
            cell.treatment_condition_id,
            cell.control_condition_id,
        )
        for cell in registration.cells
    ]
    missing = [key for key in expected if key not in observed]
    unexpected = sorted(set(observed) - set(expected))
    if missing or unexpected:
        raise ValueError(
            "registered family analysis coverage mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    ordered = []
    for cell, key in zip(registration.cells, expected):
        analysis = observed[key]
        analysis["registered_cell_id"] = cell.cell_id
        analysis["probe_variant_id"] = cell.variant_id
        ordered.append(analysis)
    return ordered, {
        "checks": {
            "condition_pairs_exact": True,
            "manifest_method_matrix_exact": True,
            "no_duplicate_cells": True,
        },
        "kind": "manifest_method_matrix",
        "registered_cells": len(registration.cells),
    }


def _order_m0_registered_analyses(
    analyses: Sequence[dict[str, Any]],
    registration: FamilyRegistration,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(analyses) != len(registration.cells):
        raise ValueError(
            "registered M0 analysis coverage mismatch: "
            f"expected={len(registration.cells)} observed={len(analyses)}"
        )
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for analysis in analyses:
        pair = (
            str(analysis["treatment_condition_id"]),
            str(analysis["control_condition_id"]),
        )
        by_pair.setdefault(pair, []).append(analysis)
    expected_counts: dict[tuple[str, str], int] = {}
    for cell in registration.cells:
        pair = (cell.treatment_condition_id, cell.control_condition_id)
        expected_counts[pair] = expected_counts.get(pair, 0) + 1
    observed_counts = {pair: len(values) for pair, values in by_pair.items()}
    if observed_counts != expected_counts:
        raise ValueError(
            "registered M0 condition-pair coverage mismatch: "
            f"expected={expected_counts} observed={observed_counts}"
        )
    methods = {str(analysis["method_id"]) for analysis in analyses}
    if len(methods) != 1:
        raise ValueError(f"M0 method identity is not common across six cells: {methods}")

    full_null_pair = ("full", "null")
    full_null = by_pair[full_null_pair]
    real_only_pairs = (
        ("history_shuffle", "null"),
        ("routing_query_shuffle", "null"),
        ("full", "history_shuffle"),
        ("full", "routing_query_shuffle"),
    )
    real_only = [by_pair[pair][0] for pair in real_only_pairs]
    real_checkpoints = {str(value["checkpoint_id"]) for value in real_only}
    if len(real_checkpoints) != 1:
        raise ValueError(
            "M0 real comparison checkpoints are not identical: "
            f"{sorted(real_checkpoints)}"
        )
    real_checkpoint_id = next(iter(real_checkpoints))
    real_full_null = [
        value for value in full_null if value["checkpoint_id"] == real_checkpoint_id
    ]
    label_full_null = [
        value for value in full_null if value["checkpoint_id"] != real_checkpoint_id
    ]
    if len(real_full_null) != 1 or len(label_full_null) != 1:
        raise ValueError(
            "M0 full-vs-null real/label-shuffle checkpoint identity is ambiguous"
        )
    real_full_null_value = real_full_null[0]
    label_value = label_full_null[0]
    label_checkpoint_id = str(label_value["checkpoint_id"])
    if not label_checkpoint_id or label_checkpoint_id == real_checkpoint_id:
        raise ValueError("M0 label-shuffle checkpoint must differ from real checkpoint")

    real_by_pair = {
        full_null_pair: real_full_null_value,
        **{pair: by_pair[pair][0] for pair in real_only_pairs},
    }
    real_signature_sha256 = {
        str(value["base_scoring_signature_sha256"])
        for value in real_by_pair.values()
    }
    if len(real_signature_sha256) != 1:
        raise ValueError("M0 real comparisons do not share one base scoring signature")
    signature_without_checkpoint = {
        str(value["base_scoring_signature_without_checkpoint_sha256"])
        for value in analyses
    }
    if len(signature_without_checkpoint) != 1:
        raise ValueError(
            "M0 real and label-shuffle probes differ beyond checkpoint identity"
        )

    real_condition_runs = {
        "full": real_full_null_value["score_run_identities"]["treatment"],
        "null": real_full_null_value["score_run_identities"]["control"],
        "history_shuffle": by_pair[("history_shuffle", "null")][0][
            "score_run_identities"
        ]["treatment"],
        "routing_query_shuffle": by_pair[("routing_query_shuffle", "null")][0][
            "score_run_identities"
        ]["treatment"],
    }
    for field in ("run_id", "metadata_sha256", "scores_sha256"):
        values = [identity[field] for identity in real_condition_runs.values()]
        if len(values) != len(set(values)):
            raise ValueError(f"M0 real conditions reuse score identity field={field}")
    expected_graph = (
        (("history_shuffle", "null"), "treatment", "history_shuffle"),
        (("history_shuffle", "null"), "control", "null"),
        (("routing_query_shuffle", "null"), "treatment", "routing_query_shuffle"),
        (("routing_query_shuffle", "null"), "control", "null"),
        (("full", "history_shuffle"), "treatment", "full"),
        (("full", "history_shuffle"), "control", "history_shuffle"),
        (("full", "routing_query_shuffle"), "treatment", "full"),
        (("full", "routing_query_shuffle"), "control", "routing_query_shuffle"),
    )
    for pair, role, condition_id in expected_graph:
        observed_identity = by_pair[pair][0]["score_run_identities"][role]
        if observed_identity != real_condition_runs[condition_id]:
            raise ValueError(
                "M0 real score-run identity graph mismatch: "
                f"pair={pair} role={role} condition={condition_id}"
            )
    label_runs = label_value["score_run_identities"]
    real_run_ids = {value["run_id"] for value in real_condition_runs.values()}
    real_metadata_hashes = {
        value["metadata_sha256"] for value in real_condition_runs.values()
    }
    real_score_hashes = {value["scores_sha256"] for value in real_condition_runs.values()}
    for role, identity in label_runs.items():
        if (
            identity["run_id"] in real_run_ids
            or identity["metadata_sha256"] in real_metadata_hashes
            or identity["scores_sha256"] in real_score_hashes
        ):
            raise ValueError(
                f"M0 label-shuffle {role} score identity overlaps real controls"
            )

    assigned: dict[str, dict[str, Any]] = {}
    for cell in registration.cells:
        pair = (cell.treatment_condition_id, cell.control_condition_id)
        analysis = (
            label_value
            if cell.variant_id == "within_request_label_shuffle"
            else real_by_pair[pair]
        )
        if analysis["registered_cell_id"] is not None:
            raise ValueError("M0 analysis was assigned to more than one registered cell")
        analysis["registered_cell_id"] = cell.cell_id
        analysis["probe_variant_id"] = cell.variant_id
        assigned[cell.cell_id] = analysis
    ordered = [assigned[cell.cell_id] for cell in registration.cells]
    return ordered, {
        "checks": {
            "condition_pair_multiset_exact": True,
            "label_checkpoint_distinct": True,
            "label_score_identities_disjoint": True,
            "method_identity_common": True,
            "real_checkpoint_common": True,
            "real_condition_score_identity_graph_exact": True,
            "real_signature_common": True,
            "signature_equal_except_checkpoint": True,
        },
        "kind": "m0_real_vs_separately_fitted_negative_control_identity_gate",
        "label_shuffle": {
            "checkpoint_id": label_checkpoint_id,
            "method_id": next(iter(methods)),
            "score_run_identities": label_runs,
            "variant_id": "within_request_label_shuffle",
        },
        "real": {
            "checkpoint_id": real_checkpoint_id,
            "condition_score_run_identities": real_condition_runs,
            "method_id": next(iter(methods)),
            "variant_id": "real",
        },
        "registered_cells": len(registration.cells),
    }


def normalized_query_fold(normalized_query_cluster: str) -> int:
    """Frozen ``sha256(normalized_query_cluster) mod 2`` assignment."""

    cluster = str(normalized_query_cluster)
    if not cluster or cluster != "".join(cluster.casefold().split()):
        raise ValueError("normalized_query_cluster is empty or not normalized")
    return int(hashlib.sha256(cluster.encode("utf-8")).hexdigest(), 16) % 2


def summarize_two_folds(
    rows: Sequence[Mapping[str, Any]],
    endpoints: Sequence[EndpointSpec] = ENDPOINT_SPECS,
) -> dict[str, Any]:
    """Return hand-auditable request/cluster means for both fixed folds."""

    result = {}
    for fold in (0, 1):
        selected = [
            row
            for row in rows
            if normalized_query_fold(str(row["normalized_query_cluster"])) == fold
        ]
        endpoint_rows = {}
        for endpoint in endpoints:
            values = _present_values(selected, endpoint.row_key)
            endpoint_rows[endpoint.endpoint_id] = {
                "mean": sum(values) / len(values) if values else None,
                "num_query_clusters": len(
                    {
                        str(row["normalized_query_cluster"])
                        for row in selected
                        if row.get(endpoint.row_key) is not None
                    }
                ),
                "num_requests": len(values),
            }
        result[str(fold)] = {
            "endpoints": endpoint_rows,
            "num_query_clusters": len(
                {str(row["normalized_query_cluster"]) for row in selected}
            ),
            "num_requests": len(selected),
        }
    return result


def cluster_bootstrap_draws(
    rows: Sequence[Mapping[str, Any]],
    endpoints: Sequence[EndpointSpec] = ENDPOINT_SPECS,
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    """Replicate the evaluator's request-weighted query-cluster bootstrap."""

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    clusters: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        cluster = str(row.get("normalized_query_cluster") or "")
        normalized_query_fold(cluster)
        clusters.setdefault(cluster, []).append(row)
    keys = sorted(clusters)
    if not keys:
        raise ValueError("cluster bootstrap requires at least one row")
    draws = {endpoint.endpoint_id: [] for endpoint in endpoints}
    rng = random.Random(seed)
    for _ in range(samples):
        selected = [
            row
            for _index in range(len(keys))
            for row in clusters[keys[rng.randrange(len(keys))]]
        ]
        for endpoint in endpoints:
            values = _present_values(selected, endpoint.row_key)
            if values:
                draws[endpoint.endpoint_id].append(sum(values) / len(values))
    return draws


def percentile_ci(draws: Sequence[float]) -> list[float] | None:
    """Use the exact frozen evaluator percentile-index convention."""

    if not draws:
        return None
    values = sorted(float(value) for value in draws)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("bootstrap draws must be finite")
    return [
        values[int(0.025 * len(values))],
        values[min(len(values) - 1, int(0.975 * len(values)))],
    ]


def two_sided_bootstrap_p(
    draws: Sequence[float],
    point_estimate: float,
) -> dict[str, Any]:
    """Finite-sample corrected two-sided sign-tail bootstrap p-value."""

    values = [float(value) for value in draws]
    point_estimate = float(point_estimate)
    if not values or not math.isfinite(point_estimate) or any(
        not math.isfinite(value) for value in values
    ):
        raise ValueError("bootstrap p-value requires finite point and draws")
    lower_count = sum(value <= 0.0 for value in values)
    upper_count = sum(value >= 0.0 for value in values)
    denominator = len(values) + 1.0
    lower_tail = (1.0 + lower_count) / denominator
    upper_tail = (1.0 + upper_count) / denominator
    if point_estimate > 0.0:
        count = lower_count
        comparison = "draw<=0"
    elif point_estimate < 0.0:
        count = upper_count
        comparison = "draw>=0"
    else:
        count = len(values)
        comparison = "point_estimate_is_zero"
    corrected_one_sided = (1.0 + count) / denominator
    return {
        "inclusive_zero": True,
        "lower_inclusive_zero_draws": lower_count,
        "lower_tail_corrected": lower_tail,
        "opposite_direction_or_zero_comparison": comparison,
        "opposite_direction_or_zero_draws": count,
        "one_sided_corrected_tail": corrected_one_sided,
        "two_sided_p": (
            1.0
            if point_estimate == 0.0
            else min(1.0, 2.0 * min(lower_tail, upper_tail))
        ),
        "upper_inclusive_zero_draws": upper_count,
        "upper_tail_corrected": upper_tail,
    }


def direction_consistent(
    overall_mean: float | None,
    fold_means: Mapping[str, float | None],
) -> bool:
    """True only when both nonzero fold means share the overall sign."""

    if set(fold_means) != {"0", "1"} or overall_mean is None:
        return False
    values = [float(overall_mean)]
    for fold in ("0", "1"):
        value = fold_means[fold]
        if value is None:
            return False
        values.append(float(value))
    if any(not math.isfinite(value) or value == 0.0 for value in values):
        return False
    return all(value > 0.0 for value in values) or all(
        value < 0.0 for value in values
    )


def benjamini_hochberg(
    hypotheses: Sequence[Mapping[str, Any]],
    *,
    alpha: float = FDR_ALPHA,
) -> list[dict[str, Any]]:
    """Apply deterministic BH adjustment while retaining every hypothesis."""

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("BH alpha must lie in (0,1)")
    rows = [dict(row) for row in hypotheses]
    if not rows:
        raise ValueError("BH requires at least one hypothesis")
    identifiers = []
    for row in rows:
        hypothesis_id = str(row.get("hypothesis_id") or "")
        raw_p = float(row.get("raw_p"))
        if not hypothesis_id or not math.isfinite(raw_p) or not 0.0 <= raw_p <= 1.0:
            raise ValueError("BH hypotheses require unique IDs and p in [0,1]")
        identifiers.append(hypothesis_id)
        row["raw_p"] = raw_p
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("BH hypothesis IDs must be unique")
    order = sorted(range(len(rows)), key=lambda index: (rows[index]["raw_p"], rows[index]["hypothesis_id"]))
    adjusted = [1.0] * len(rows)
    running = 1.0
    family_size = len(rows)
    for position in range(family_size - 1, -1, -1):
        index = order[position]
        rank = position + 1
        candidate = rows[index]["raw_p"] * family_size / rank
        running = min(running, candidate)
        adjusted[index] = min(1.0, running)
    result = []
    for index, row in enumerate(rows):
        result.append(
            {
                **row,
                "q_value": adjusted[index],
                "reject_at_0_05": adjusted[index] <= float(alpha),
            }
        )
    return result


def statistical_synthesis_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = {
        "scripts/synthesize_mechanism_statistics.py": (
            root / "scripts/synthesize_mechanism_statistics.py"
        ),
        "src/myrec/mechanism/statistical_synthesis.py": Path(__file__).resolve(),
    }
    files = []
    for relative, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError(f"missing statistical synthesis code: {path}")
        files.append({"path": relative, "sha256": sha256_file(path)})
    return {"digest": _canonical_sha256(files), "files": files}


def _score_run_identity(
    metadata_conditions: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    role: str,
    expected_condition: str,
) -> dict[str, str]:
    condition = metadata_conditions.get(role)
    input_runs = audit.get("input_runs")
    if not isinstance(condition, Mapping) or not isinstance(input_runs, Mapping):
        raise ValueError(f"missing {role} score-run identity")
    audit_run = input_runs.get(role)
    if not isinstance(audit_run, Mapping):
        raise ValueError(f"pre_qrels_audit lacks {role} score-run identity")
    run_id = _required_string(condition, "run_id", f"metadata {role} condition")
    if (
        audit_run.get("run_id") != run_id
        or audit_run.get("condition_id") != expected_condition
        or audit_run.get("qrels_read") is not False
    ):
        raise ValueError(f"{role} score-run identity differs across artifacts")
    metadata_sha256 = _required_sha256(
        audit_run.get("metadata_sha256"),
        f"pre_qrels_audit {role} metadata_sha256",
    )
    scores_sha256 = _required_sha256(
        audit_run.get("scores_sha256"),
        f"pre_qrels_audit {role} scores_sha256",
    )
    return {
        "condition_id": expected_condition,
        "metadata_sha256": metadata_sha256,
        "run_id": run_id,
        "scores_sha256": scores_sha256,
    }


def _validate_per_request_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    treatment: str,
    control: str,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("per_request.jsonl is empty")
    result = []
    seen = set()
    for raw in rows:
        row = dict(raw)
        request_id = _required_string(row, "request_id", "per_request")
        if request_id in seen:
            raise ValueError(f"duplicate per-request row: {request_id}")
        seen.add(request_id)
        cluster = _required_string(row, "normalized_query_cluster", "per_request")
        normalized_query_fold(cluster)
        if row.get("treatment_condition_id") != treatment or row.get(
            "control_condition_id"
        ) != control:
            raise ValueError(f"per-request condition identity drift: {request_id}")
        _required_string(row, "target_aware_surface", "per_request")
        for endpoint in ENDPOINT_SPECS:
            value = row.get(endpoint.row_key)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(
                    f"non-finite endpoint={endpoint.row_key} request={request_id}"
                )
        if row.get("treatment_minus_control_ndcg@10") is None:
            raise ValueError(f"NDCG delta is missing for request={request_id}")
        margin_eligible = row.get("margin_eligible")
        if not isinstance(margin_eligible, bool):
            raise ValueError(f"margin_eligible is not boolean: {request_id}")
        if margin_eligible != (row.get("target_margin_change") is not None):
            raise ValueError(f"margin eligibility/value mismatch: {request_id}")
        result.append(row)
    return result


def _validate_metrics_endpoint(
    surface_metrics: Mapping[str, Any],
    endpoint: EndpointSpec,
    *,
    observed_mean: float,
    observed_ci95: list[float] | None,
) -> None:
    declared_mean = surface_metrics.get(endpoint.metrics_mean_key)
    declared_intervals = surface_metrics.get("query_cluster_ci95")
    declared_nested = surface_metrics.get(endpoint.row_key)
    if declared_mean is None or not isinstance(declared_intervals, Mapping):
        raise ValueError(f"metrics endpoint is missing: {endpoint.endpoint_id}")
    if not isinstance(declared_nested, Mapping):
        raise ValueError(f"metrics nested endpoint is missing: {endpoint.endpoint_id}")
    candidates = [
        ("mean", declared_mean, observed_mean),
        ("nested mean", declared_nested.get("mean"), observed_mean),
    ]
    for label, declared, observed in candidates:
        if declared is None or not math.isclose(
            float(declared),
            float(observed),
            rel_tol=NUMERIC_TOLERANCE,
            abs_tol=NUMERIC_TOLERANCE,
        ):
            raise ValueError(
                f"metrics {label} drift for endpoint={endpoint.endpoint_id}: "
                f"declared={declared} recomputed={observed}"
            )
    for label, declared in (
        ("query_cluster_ci95", declared_intervals.get(endpoint.row_key)),
        ("nested query_cluster_ci95", declared_nested.get("query_cluster_ci95")),
    ):
        if not _close_optional_interval(declared, observed_ci95):
            raise ValueError(
                f"metrics {label} drift for endpoint={endpoint.endpoint_id}: "
                f"declared={declared} recomputed={observed_ci95}"
            )


def _strict_surface_metrics(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    surfaces = metrics.get("surfaces")
    if not isinstance(surfaces, Mapping):
        raise ValueError("metrics.json lacks target-aware surfaces")
    surface = surfaces.get(STRICT_TRANSFER_SURFACE)
    if not isinstance(surface, Mapping):
        raise ValueError("metrics.json lacks the strict-transfer surface")
    return surface


def _validate_probe_admission(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("mechanism probe manifest admission is missing")
    expected = {
        "actual_sha256": PROBE_MANIFEST_SHA256,
        "expected_path": PROBE_MANIFEST_PATH.as_posix(),
        "expected_sha256": PROBE_MANIFEST_SHA256,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"mechanism probe manifest admission drift: {key}")


def _present_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"non-finite values for endpoint={key}")
    return values


def _close_optional_interval(left: Any, right: list[float] | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if not isinstance(left, list) or len(left) != 2 or len(right) != 2:
        return False
    return all(
        math.isclose(
            float(observed),
            float(expected),
            rel_tol=NUMERIC_TOLERANCE,
            abs_tol=NUMERIC_TOLERANCE,
        )
        for observed, expected in zip(left, right)
    )


def _hypothesis_id(
    method_id: str,
    treatment: str,
    control: str,
    endpoint_id: str,
    *,
    registered_cell_id: str,
) -> str:
    return "::".join(
        (
            str(registered_cell_id),
            str(method_id),
            str(treatment),
            str(control),
            str(endpoint_id),
        )
    )


def _required_string(value: Mapping[str, Any], key: str, source: str) -> str:
    observed = value.get(key)
    if not isinstance(observed, str) or not observed:
        raise ValueError(f"{source} requires non-empty {key}")
    return observed


def _required_sha256(value: Any, source: str) -> str:
    observed = str(value or "")
    if len(observed) != 64 or any(character not in "0123456789abcdef" for character in observed):
        raise ValueError(f"{source} must be a lowercase SHA256")
    return observed


def _nonempty_unique_strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{label} must be a non-empty string list")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicates")
    return list(value)


def _read_regular_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"required regular non-symlink file is missing: {path}")
    return path.read_bytes()


def _parse_json_object(payload: bytes, label: str) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _parse_jsonl_objects(payload: bytes, label: str) -> list[dict[str, Any]]:
    text = payload.decode("utf-8")
    if not text.strip():
        raise ValueError(f"{label} is empty")
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{label} contains blank line={line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{label} line={line_number} is not an object")
        rows.append(value)
    return rows


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"statistical synthesis output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f"statistical synthesis output already exists: {path}")
    os.replace(temporary, path)


def _git_revision() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: Any) -> str:
    return sha256_text(_canonical_json(value))


__all__ = [
    "ANALYSIS_TYPE",
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "ENDPOINT_SPECS",
    "FDR_ALPHA",
    "PROBE_MANIFEST_SHA256",
    "STRICT_TRANSFER_SURFACE",
    "benjamini_hochberg",
    "cluster_bootstrap_draws",
    "direction_consistent",
    "load_analysis_artifacts",
    "load_family_registration",
    "normalized_query_fold",
    "percentile_ci",
    "statistical_synthesis_implementation_identity",
    "summarize_analysis",
    "summarize_two_folds",
    "synthesize_mechanism_statistics",
    "two_sided_bootstrap_p",
]
