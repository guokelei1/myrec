"""Two-model planned-family synthesis for D2 selected-branch diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.mechanism.attention_edge_evaluator import _write_json
from myrec.mechanism.deep_dive_native_evaluator import benjamini_hochberg
from myrec.mechanism.selected_branch_evaluator import (
    CONTRAST_GROUPS,
    ENDPOINTS,
    SELECTED_BRANCH_FOLD_SCOPE,
    selected_branch_contrast_specs,
)
from myrec.utils.hashing import sha256_file


MODELS = ("q2_recranker_generalqwen", "q3_tallrec_generalqwen")
NEGATIVE_EXPECTED_GROUPS = {
    "same",
    "same_minus_cross",
    "same_minus_wrong_history",
    "direction_scale",
}


def synthesize_selected_branches(
    metrics_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Apply BH without shrinking any preregistered family."""

    if set(metrics_paths) - set(MODELS):
        raise ValueError("selected-branch synthesis received an unknown model")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"selected-branch synthesis output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = {}
    identities = {}
    specs = selected_branch_contrast_specs()
    for method_id, path in metrics_paths.items():
        path = Path(path)
        metrics = _read_json(path)
        if (
            metrics.get("analysis_type")
            != "transformer_deep_dive_d2_selected_branch"
            or metrics.get("status") != "completed"
            or metrics.get("method_id") != method_id
            or metrics.get("normalized_query_fold") != 1
        ):
            raise ValueError(f"invalid selected-branch metrics: {method_id}")
        if metrics.get("fold_scope") != SELECTED_BRANCH_FOLD_SCOPE:
            raise ValueError(
                f"selected-branch fold scope differs: {method_id}"
            )
        implementation_digest = str(metrics.get("implementation_digest") or "")
        if not implementation_digest:
            raise ValueError(
                f"selected-branch implementation digest is missing: {method_id}"
            )
        input_identity = _audit_selected_branch_metrics(
            metrics,
            method_id=method_id,
            specs=specs,
        )
        rows = metrics.get("family_rows")
        if not isinstance(rows, list) or len(rows) != 96:
            raise ValueError(f"selected-branch family row coverage drift: {method_id}")
        by_key = {
            (str(row["contrast_id"]), str(row["endpoint"])): dict(row)
            for row in rows
        }
        if len(by_key) != 96:
            raise ValueError("selected-branch metrics contain duplicate family rows")
        loaded[method_id] = {"metrics": metrics, "rows": by_key}
        identities[method_id] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "implementation_digest": implementation_digest,
            **input_identity,
        }
    implementation_digests = {
        row["implementation_digest"] for row in identities.values()
    }
    if len(implementation_digests) > 1:
        raise ValueError(
            "selected-branch metrics use different implementation digests"
        )

    families = {}
    all_rows = []
    for group, units_per_model in CONTRAST_GROUPS.items():
        contrast_ids = [
            contrast_id
            for contrast_id, spec in specs.items()
            if spec["group"] == group
        ]
        if len(contrast_ids) != units_per_model:
            raise AssertionError("selected-branch synthesis unit count drift")
        for endpoint in ENDPOINTS:
            rows = []
            for method_id in MODELS:
                evidence_role = (
                    loaded[method_id]["metrics"]["evidence_role"]
                    if method_id in loaded
                    else "missing_or_gate_stopped"
                )
                for contrast_id in contrast_ids:
                    if method_id in loaded:
                        source = loaded[method_id]["rows"][(contrast_id, endpoint)]
                        row = {
                            **source,
                            "method_id": method_id,
                            "evidence_role": evidence_role,
                            "missing": False,
                        }
                    else:
                        row = {
                            "contrast_id": contrast_id,
                            "group": group,
                            "endpoint": endpoint,
                            "method_id": method_id,
                            "evidence_role": evidence_role,
                            "two_sided_p": 1.0,
                            "mean": None,
                            "ci95": None,
                            "missing": True,
                        }
                    rows.append(row)
            planned_size = 2 * units_per_model
            if len(rows) != planned_size:
                raise AssertionError("selected-branch planned family size drift")
            q_values = benjamini_hochberg(
                [float(row["two_sided_p"]) for row in rows]
            )
            for row, q_value in zip(rows, q_values):
                row["bh_q"] = float(q_value)
                row["bh_significant"] = q_value < 0.05
                expected = "negative" if group in NEGATIVE_EXPECTED_GROUPS else None
                row["expected_sign"] = expected
                row["expected_sign_met"] = (
                    None
                    if expected is None or row["mean"] is None
                    else float(row["mean"]) < 0.0
                )
                row["registered_support"] = bool(
                    expected is not None
                    and row["expected_sign_met"] is True
                    and row["bh_significant"]
                    and row["evidence_role"]
                    == "registered_confirmatory_branch_localization"
                )
                if endpoint == "ndcg@10" and row["ci95"] is not None:
                    lower, upper = map(float, row["ci95"])
                    row["practically_equivalent_within_0.005"] = (
                        lower >= -0.005 and upper <= 0.005
                    )
                else:
                    row["practically_equivalent_within_0.005"] = None
            family_id = f"{group}__{endpoint}"
            families[family_id] = {
                "group": group,
                "endpoint": endpoint,
                "planned_family_size": planned_size,
                "observed_cells": sum(not row["missing"] for row in rows),
                "missing_cells_fixed_p1": sum(row["missing"] for row in rows),
                "rows": rows,
            }
            all_rows.extend(rows)
    result = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d2_selected_branch_synthesis",
        "analysis_run_id": analysis_run_id,
        "models": list(MODELS),
        "fold_scope": dict(SELECTED_BRANCH_FOLD_SCOPE),
        "input_metrics": identities,
        "multiple_testing": {
            "method": "benjamini_hochberg",
            "alpha": 0.05,
            "primary_and_ndcg_families_separate": True,
            "missing_gate_stopped_or_mechanical_cell_p": 1.0,
            "planned_family_size_never_shrinks": True,
        },
        "families": families,
        "rows": all_rows,
        "registered_support_rows": sum(
            row["registered_support"] for row in all_rows
        ),
        "command": list(command or []),
        "status": "completed",
    }
    _write_json(output_dir / "metrics.json", result)
    return result


def _audit_selected_branch_metrics(
    metrics: Mapping[str, Any],
    *,
    method_id: str,
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind evaluator tables and every pre/post-qrels evidence byte."""

    if (
        metrics.get("qrels_read") is not True
        or metrics.get("qrels_fold_opened") != 1
        or metrics.get("other_fold_qrels_opened") is not False
        or metrics.get("bootstrap")
        != {"cluster": "normalized_query", "samples": 5000, "seed": 20260715}
    ):
        raise ValueError(f"selected-branch evaluator boundary differs: {method_id}")
    strict_requests = metrics.get("strict_transfer_requests")
    wrong_requests = metrics.get("strict_transfer_wrong_user_eligible_requests")
    if (
        isinstance(strict_requests, bool)
        or not isinstance(strict_requests, int)
        or strict_requests <= 0
        or isinstance(wrong_requests, bool)
        or not isinstance(wrong_requests, int)
        or not 0 <= wrong_requests <= strict_requests
    ):
        raise ValueError(f"selected-branch evaluator population differs: {method_id}")
    family_policy = metrics.get("family_policy")
    expected_units = {
        group: 2 * units for group, units in CONTRAST_GROUPS.items()
    }
    if (
        not isinstance(family_policy, Mapping)
        or family_policy.get("BH_applied_only_in_two_model_synthesis") is not True
        or family_policy.get("per_endpoint_separate_families") is not True
        or family_policy.get("planned_two_model_units") != expected_units
    ):
        raise ValueError(f"selected-branch family policy differs: {method_id}")

    results = metrics.get("results")
    rows = metrics.get("family_rows")
    if (
        not isinstance(results, Mapping)
        or set(results) != set(specs)
        or not isinstance(rows, list)
        or len(rows) != len(specs) * len(ENDPOINTS)
    ):
        raise ValueError(f"selected-branch evaluator table coverage differs: {method_id}")
    by_key = {}
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"selected-branch family row is not an object: {method_id}")
        contrast_id = str(raw_row.get("contrast_id") or "")
        endpoint = str(raw_row.get("endpoint") or "")
        key = (contrast_id, endpoint)
        if (
            contrast_id not in specs
            or endpoint not in ENDPOINTS
            or key in by_key
            or raw_row.get("group") != specs[contrast_id]["group"]
        ):
            raise ValueError(f"selected-branch family row identity differs: {method_id}")
        p_value = _finite_float(raw_row.get("two_sided_p"), "two_sided_p")
        mean = _finite_float(raw_row.get("mean"), "mean")
        interval = raw_row.get("ci95")
        if (
            not 0.0 <= p_value <= 1.0
            or not isinstance(interval, list)
            or len(interval) != 2
        ):
            raise ValueError(f"selected-branch family row inference differs: {method_id}")
        lower = _finite_float(interval[0], "ci95 lower")
        upper = _finite_float(interval[1], "ci95 upper")
        if lower > upper:
            raise ValueError(f"selected-branch family interval is reversed: {method_id}")
        by_key[key] = {
            "two_sided_p": p_value,
            "mean": mean,
            "ci95": [lower, upper],
        }
    if set(by_key) != {
        (contrast_id, endpoint)
        for contrast_id in specs
        for endpoint in ENDPOINTS
    }:
        raise ValueError(f"selected-branch family row keys differ: {method_id}")

    for contrast_id, spec in specs.items():
        result = results[contrast_id]
        if not isinstance(result, Mapping):
            raise ValueError(f"selected-branch result is not an object: {method_id}")
        expected_surface = (
            "strict_transfer_and_frozen_wrong_user_eligible"
            if spec["group"] == "same_minus_wrong_history"
            else "strict_transfer"
        )
        if (
            result.get("group") != spec["group"]
            or result.get("node") != spec.get("node")
            or result.get("left_node") != spec.get("left_node")
            or result.get("right_node") != spec.get("right_node")
            or result.get("control") != spec.get("control")
            or result.get("eligible_surface") != expected_surface
            or not isinstance(result.get("endpoints"), Mapping)
            or set(result["endpoints"]) != set(ENDPOINTS)
        ):
            raise ValueError(f"selected-branch result schema differs: {method_id}")
        for endpoint in ENDPOINTS:
            inference = result["endpoints"][endpoint]
            if not isinstance(inference, Mapping):
                raise ValueError(
                    f"selected-branch endpoint inference differs: {method_id}"
                )
            expected = by_key[(contrast_id, endpoint)]
            observed = {
                "two_sided_p": _finite_float(
                    inference.get("two_sided_p"), "result two_sided_p"
                ),
                "mean": _finite_float(inference.get("mean"), "result mean"),
                "ci95": [
                    _finite_float(value, "result ci95")
                    for value in inference.get("ci95", [])
                ],
            }
            if observed != expected:
                raise ValueError(
                    f"selected-branch result/family row drift: {method_id}"
                )

    input_bundle = metrics.get("input_bundle")
    if not isinstance(input_bundle, Mapping) or set(input_bundle) != {
        "path",
        "metadata_sha256",
        "scores_sha256",
    }:
        raise ValueError(f"selected-branch input bundle identity differs: {method_id}")
    bundle_root = Path(str(input_bundle["path"]))
    _require_file_sha(
        bundle_root / "metadata.json",
        input_bundle["metadata_sha256"],
        f"{method_id} input metadata",
    )
    _require_file_sha(
        bundle_root / "scores.jsonl",
        input_bundle["scores_sha256"],
        f"{method_id} input scores",
    )

    pre_qrels_path = Path(str(metrics.get("pre_qrels_audit_path") or ""))
    _require_file_sha(
        pre_qrels_path,
        metrics.get("pre_qrels_audit_sha256"),
        f"{method_id} pre-qrels audit",
    )
    pre_qrels = _read_json(pre_qrels_path)
    expected_checks = {
        "fold1_request_candidate_coverage_complete_finite": True,
        "all_14_identity_controls_at_most_1e-5": True,
        "frozen_baseline_recompute_within_path_local_bf16_bound": True,
        "wrong_user_ineligible_scores_equal_frozen_null": True,
        "candidate_and_request_manifests_reconstructed": True,
        "minimal_selected_branch_contract_bound": True,
        "selected_branch_implementation_digest_bound": True,
    }
    if (
        pre_qrels.get("analysis_type")
        != "d2_selected_branch_fold1_pre_qrels_integrity"
        or pre_qrels.get("status") != "passed"
        or pre_qrels.get("method_id") != method_id
        or pre_qrels.get("selected_block") != metrics.get("selected_block")
        or pre_qrels.get("qrels_read") is not False
        or pre_qrels.get("checks") != expected_checks
        or pre_qrels.get("implementation_digest")
        != metrics.get("implementation_digest")
        or pre_qrels.get("bundle") != input_bundle
    ):
        raise ValueError(f"selected-branch pre-qrels audit differs: {method_id}")

    per_request_path = Path(str(metrics.get("per_request_contrasts_path") or ""))
    _require_file_sha(
        per_request_path,
        metrics.get("per_request_contrasts_sha256"),
        f"{method_id} per-request contrasts",
    )
    for field in (
        "qrels_fold_sha256",
        "qrels_split_manifest_sha256",
        "qrels_source_sha256",
    ):
        _require_sha256(metrics.get(field), f"{method_id} {field}")
    return {
        "input_bundle": dict(input_bundle),
        "pre_qrels_audit_path": str(pre_qrels_path),
        "pre_qrels_audit_sha256": sha256_file(pre_qrels_path),
        "per_request_contrasts_path": str(per_request_path),
        "per_request_contrasts_sha256": sha256_file(per_request_path),
        "qrels_fold_sha256": str(metrics["qrels_fold_sha256"]),
        "qrels_split_manifest_sha256": str(
            metrics["qrels_split_manifest_sha256"]
        ),
        "qrels_source_sha256": str(metrics["qrels_source_sha256"]),
        "evaluator_tables_cross_checked": True,
    }


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"selected-branch {label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"selected-branch {label} is non-finite")
    return result


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"selected-branch {label} is not SHA-256")
    return text


def _require_file_sha(path: Path, expected: Any, label: str) -> None:
    expected_sha = _require_sha256(expected, label)
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError(f"selected-branch {label} bytes differ")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value
