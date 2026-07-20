"""Outcome-independent DID synthesis for the registered Q2 matched control.

This module is intentionally downstream of the shared evaluator.  It never
constructs or opens a qrels path: it joins the two already-admitted paired
``per_request.jsonl`` files and adds the two-fold, query-cluster bootstrap, and
two-endpoint BH family frozen before matched-control outcomes existed.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from myrec.mechanism.statistical_synthesis import (
    benjamini_hochberg,
    cluster_bootstrap_draws,
    direction_consistent,
    normalized_query_fold,
    percentile_ci,
    two_sided_bootstrap_p,
)
from myrec.utils.hashing import sha256_file


REGISTRATION_PATH = Path(
    "experiments/motivation/m3_matched_did_synthesis_registration.yaml"
)
REGISTRATION_SHA256 = (
    "20f87426639d3dbe4ea2a8b32008fc11a389e756e50057e2737d1a6334f2e7db"
)
PROBE_MANIFEST_PATH = Path("experiments/motivation/probe_manifest.yaml")
PROBE_MANIFEST_SHA256 = (
    "adedf0e662b9d8529162b8abffedcf6b10962913f28580af6119d807cc5d929c"
)
ANALYSIS_RUN_ID = "20260717_kuaisearch_mech_m3_q2_matched_did_analysis"
TOP_LEVEL_RUN_ID = "20260717_kuaisearch_mech_m3_q2_matched_control_analysis"
ORIGINAL_PAIR_RUN_ID = (
    "20260717_kuaisearch_mech_m3_q2_matched_control_analysis_"
    "original_mixture_full_vs_null"
)
BALANCED_PAIR_RUN_ID = (
    "20260717_kuaisearch_mech_m3_q2_matched_control_analysis_"
    "surface_balanced_full_vs_null"
)
METHOD_ID = "q2_recranker_generalqwen"
ROLE = "diagnostic_control_not_paper_method"
ANALYSIS_TYPE = "q2_matched_training_control_joined_did_synthesis"
STRICT_SURFACE = "target_nonrepeat_no_candidate_overlap"
EXPECTED_REQUESTS = 8000
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20260715
FDR_ALPHA = 0.05
ENDPOINTS = (
    (
        "strict_transfer_ndcg_history_response_did",
        "ndcg_did",
        "treatment_minus_control_ndcg@10",
        "primary",
    ),
    (
        "strict_transfer_target_margin_change_did",
        "margin_did",
        "target_margin_change",
        "secondary_coherence",
    ),
)
SURFACES = (
    "all",
    "observed_positive",
    "target_repeat",
    "target_nonrepeat_other_candidate_overlap",
    STRICT_SURFACE,
    "target_nonrepeat_no_history",
    "no_observed_positive",
)


@dataclass(frozen=True)
class _EndpointSpec:
    endpoint_id: str
    row_key: str


_BOOTSTRAP_ENDPOINTS = tuple(
    _EndpointSpec(endpoint_id=endpoint_id, row_key=did_key)
    for endpoint_id, did_key, _pair_key, _role in ENDPOINTS
)


def synthesize_q2_matched_control_did(
    *,
    analysis_run_id: str = ANALYSIS_RUN_ID,
    top_level_run_id: str = TOP_LEVEL_RUN_ID,
    original_pair_run_id: str = ORIGINAL_PAIR_RUN_ID,
    balanced_pair_run_id: str = BALANCED_PAIR_RUN_ID,
    runs_dir: str | Path = "runs",
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    registration_path: str | Path = REGISTRATION_PATH,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Join the two frozen pair analyses and persist the exact DID family."""

    expected_ids = {
        "analysis_run_id": ANALYSIS_RUN_ID,
        "top_level_run_id": TOP_LEVEL_RUN_ID,
        "original_pair_run_id": ORIGINAL_PAIR_RUN_ID,
        "balanced_pair_run_id": BALANCED_PAIR_RUN_ID,
    }
    observed_ids = {
        "analysis_run_id": analysis_run_id,
        "top_level_run_id": top_level_run_id,
        "original_pair_run_id": original_pair_run_id,
        "balanced_pair_run_id": balanced_pair_run_id,
    }
    if observed_ids != expected_ids:
        raise ValueError("matched DID run identities differ from preregistration")
    recorded_command = list(sys.argv if command is None else command)
    if not recorded_command:
        raise ValueError("matched DID synthesis command must be non-empty")

    root = Path(__file__).resolve().parents[3]
    registration = _load_registration(root, registration_path)
    runs_dir = Path(runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = root / runs_dir
    analysis_dir = runs_dir / analysis_run_id
    if analysis_dir.exists():
        raise FileExistsError(f"matched DID analysis already exists: {analysis_dir}")

    top = _load_analysis(runs_dir, top_level_run_id, require_per_request=False)
    original = _load_analysis(runs_dir, original_pair_run_id, require_per_request=True)
    balanced = _load_analysis(runs_dir, balanced_pair_run_id, require_per_request=True)
    _validate_source_analyses(top, original, balanced)
    joined = join_pair_rows(original["rows"], balanced["rows"])
    if len(joined) != EXPECTED_REQUESTS:
        raise ValueError("matched DID join must contain exactly 8000 requests")
    surface_summaries = summarize_surfaces(joined)
    _validate_producer_point_estimates(top["metrics"], surface_summaries)
    inference = summarize_strict_did(joined)
    implementation = matched_control_synthesis_implementation_identity()
    input_artifacts = {
        "top_level": _input_identity(top),
        "original_mixture": _input_identity(original),
        "surface_balanced": _input_identity(balanced),
    }
    qrels_sha256 = str(original["metrics"]["qrels_sha256"])
    generated_at = _utc_now()
    metrics = {
        "schema_version": 1,
        "analysis_run_id": analysis_run_id,
        "analysis_type": ANALYSIS_TYPE,
        "bootstrap": {
            "cluster": "normalized_query",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "command": recorded_command,
        "endpoints": inference["endpoints"],
        "fdr": inference["fdr"],
        "folds": {
            "count": 2,
            "rule": "int(sha256(normalized_query_cluster),16) mod 2",
        },
        "generated_at": generated_at,
        "input_artifacts": input_artifacts,
        "joined_request_count": len(joined),
        "matched_control_synthesis_implementation_identity": implementation,
        "method_id": METHOD_ID,
        "parent_probe_manifest": registration["parent_probe_manifest"],
        "primary_surface": STRICT_SURFACE,
        "qrels_read": False,
        "qrels_sha256": qrels_sha256,
        "registration": {
            "id": registration["registration_id"],
            "path": REGISTRATION_PATH.as_posix(),
            "sha256": REGISTRATION_SHA256,
            "status": registration["status"],
        },
        "role": ROLE,
        "source_test_opened": False,
        "split": "dev",
        "status": "completed",
        "strict_transfer_request_count": inference["num_requests"],
        "surface_summaries": surface_summaries,
    }

    analysis_dir.mkdir(parents=True, exist_ok=False)
    joined_path = analysis_dir / "joined_per_request.jsonl"
    metrics_path = analysis_dir / "metrics.json"
    _write_jsonl_atomic(joined_path, joined)
    _write_json_atomic(metrics_path, metrics)
    ledger_path = Path(dev_eval_log_path)
    if not ledger_path.is_absolute():
        ledger_path = root / ledger_path
    _append_ledger_exact_once(
        ledger_path,
        {
            "schema_version": 1,
            "analysis_type": ANALYSIS_TYPE,
            "command": recorded_command,
            "derived_label_input": True,
            "matched_control_synthesis_implementation_digest": implementation[
                "digest"
            ],
            "method_id": "shared_mechanism_statistical_synthesis",
            "metrics_path": _relative_to_root(metrics_path, root),
            "metrics_sha256": sha256_file(metrics_path),
            "qrels_read": False,
            "qrels_sha256": qrels_sha256,
            "registration_sha256": REGISTRATION_SHA256,
            "request_count": len(joined),
            "role": ROLE,
            "run_id": analysis_run_id,
            "source_test_opened": False,
            "split": "dev",
            "status": "completed",
            "subject_method_id": METHOD_ID,
            "timestamp": generated_at,
        },
    )
    metadata = {
        "schema_version": 1,
        "analysis_run_id": analysis_run_id,
        "analysis_type": ANALYSIS_TYPE,
        "command": recorded_command,
        "generated_at": generated_at,
        "input_artifacts": input_artifacts,
        "joined_per_request_path": _relative_to_root(joined_path, root),
        "joined_per_request_sha256": sha256_file(joined_path),
        "matched_control_synthesis_implementation_identity": implementation,
        "metrics_path": _relative_to_root(metrics_path, root),
        "metrics_sha256": sha256_file(metrics_path),
        "qrels_read": False,
        "qrels_sha256": qrels_sha256,
        "registration_path": REGISTRATION_PATH.as_posix(),
        "registration_sha256": REGISTRATION_SHA256,
        "request_count": len(joined),
        "role": ROLE,
        "source_test_opened": False,
        "split": "dev",
        "status": "completed",
    }
    # Metadata is the final completion marker after the durable ledger append.
    _write_json_atomic(analysis_dir / "metadata.json", metadata)
    return metrics


def join_pair_rows(
    original_rows: Sequence[Mapping[str, Any]],
    balanced_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join full-minus-null rows by request and calculate both registered DIDs."""

    original = _index_rows(original_rows, "original_mixture")
    balanced = _index_rows(balanced_rows, "surface_balanced")
    if set(original) != set(balanced):
        raise ValueError("matched DID pair request sets differ")
    joined = []
    for request_id in sorted(original):
        left = original[request_id]
        right = balanced[request_id]
        for key in ("normalized_query_cluster", "target_aware_surface"):
            if left.get(key) != right.get(key):
                raise ValueError(f"matched DID request {key} differs: {request_id}")
        ndcg_left = _finite_required(left.get("treatment_minus_control_ndcg@10"))
        ndcg_right = _finite_required(right.get("treatment_minus_control_ndcg@10"))
        margin_left = _finite_optional(left.get("target_margin_change"))
        margin_right = _finite_optional(right.get("target_margin_change"))
        if (margin_left is None) != (margin_right is None):
            raise ValueError(f"matched DID margin eligibility differs: {request_id}")
        joined.append(
            {
                "balanced_ndcg_history_response": ndcg_right,
                "balanced_target_margin_change": margin_right,
                "margin_did": (
                    None
                    if margin_left is None
                    else float(margin_right) - float(margin_left)
                ),
                "ndcg_did": ndcg_right - ndcg_left,
                "normalized_query_cluster": str(left["normalized_query_cluster"]),
                "original_ndcg_history_response": ndcg_left,
                "original_target_margin_change": margin_left,
                "request_id": request_id,
                "target_aware_surface": str(left["target_aware_surface"]),
            }
        )
    return joined


def summarize_strict_did(
    joined: Sequence[Mapping[str, Any]],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return the exact two-endpoint strict-transfer inferential family."""

    strict = [row for row in joined if row.get("target_aware_surface") == STRICT_SURFACE]
    if not strict:
        raise ValueError("matched DID strict-transfer surface is empty")
    draws = cluster_bootstrap_draws(
        strict,
        _BOOTSTRAP_ENDPOINTS,
        samples=samples,
        seed=seed,
    )
    endpoint_rows: dict[str, dict[str, Any]] = {}
    hypotheses = []
    for endpoint_id, did_key, pair_key, role in ENDPOINTS:
        values = [_finite_required(row[did_key]) for row in strict if row.get(did_key) is not None]
        if not values:
            raise ValueError(f"matched DID endpoint has no eligible requests: {endpoint_id}")
        mean = sum(values) / len(values)
        folds = {}
        fold_means: dict[str, float | None] = {}
        for fold in (0, 1):
            selected = [
                row
                for row in strict
                if normalized_query_fold(str(row["normalized_query_cluster"])) == fold
                and row.get(did_key) is not None
            ]
            fold_values = [_finite_required(row[did_key]) for row in selected]
            fold_mean = sum(fold_values) / len(fold_values) if fold_values else None
            fold_means[str(fold)] = fold_mean
            folds[str(fold)] = {
                "mean": fold_mean,
                "num_query_clusters": len(
                    {str(row["normalized_query_cluster"]) for row in selected}
                ),
                "num_requests": len(fold_values),
            }
        p_value = two_sided_bootstrap_p(draws[endpoint_id], mean)
        endpoint_rows[endpoint_id] = {
            "bootstrap_draws": len(draws[endpoint_id]),
            "direction_consistent_in_both_folds": direction_consistent(
                mean, fold_means
            ),
            "folds": folds,
            "mean": mean,
            "num_query_clusters": len(
                {str(row["normalized_query_cluster"]) for row in strict if row.get(did_key) is not None}
            ),
            "num_requests": len(values),
            "pair_row_key": pair_key,
            "query_cluster_ci95": percentile_ci(draws[endpoint_id]),
            "role": role,
            "two_sided_bootstrap_p": p_value,
        }
        hypotheses.append(
            {"hypothesis_id": endpoint_id, "raw_p": p_value["two_sided_p"]}
        )
    fdr_rows = benjamini_hochberg(hypotheses, alpha=FDR_ALPHA)
    fdr_by_id = {str(row["hypothesis_id"]): row for row in fdr_rows}
    for endpoint_id in endpoint_rows:
        endpoint_rows[endpoint_id]["bh_fdr"] = fdr_by_id[endpoint_id]
    return {
        "endpoints": endpoint_rows,
        "fdr": {
            "alpha": FDR_ALPHA,
            "family_id": "m3_q2_matched_strict_transfer_did",
            "family_size": 2,
            "method": "benjamini_hochberg",
            "results": fdr_rows,
        },
        "num_requests": len(strict),
    }


def summarize_surfaces(joined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reconstruct all producer point estimates; inference remains strict-only."""

    result = {}
    for surface in SURFACES:
        if surface == "all":
            rows = list(joined)
        elif surface == "observed_positive":
            rows = [
                row
                for row in joined
                if row.get("target_aware_surface") != "no_observed_positive"
            ]
        else:
            rows = [row for row in joined if row.get("target_aware_surface") == surface]
        ndcg = [_finite_required(row["ndcg_did"]) for row in rows]
        margins = [
            _finite_required(row["margin_did"])
            for row in rows
            if row.get("margin_did") is not None
        ]
        result[surface] = {
            "balanced_minus_original_history_response_ndcg@10": (
                sum(ndcg) / len(ndcg) if ndcg else None
            ),
            "balanced_minus_original_target_margin_change": (
                sum(margins) / len(margins) if margins else None
            ),
            "num_margin_eligible_requests": len(margins),
            "num_requests": len(rows),
        }
    return result


def matched_control_synthesis_implementation_identity() -> dict[str, Any]:
    """Bind the producer to its CLI, registration, and generic statistics."""

    root = Path(__file__).resolve().parents[3]
    paths = {
        REGISTRATION_PATH.as_posix(): root / REGISTRATION_PATH,
        "scripts/synthesize_q2_matched_control.py": root
        / "scripts/synthesize_q2_matched_control.py",
        "src/myrec/mechanism/matched_control_synthesis.py": Path(__file__).resolve(),
        "src/myrec/mechanism/statistical_synthesis.py": Path(__file__).with_name(
            "statistical_synthesis.py"
        ),
    }
    files = []
    for relative, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"path": relative, "sha256": sha256_file(path)})
    payload = json.dumps(
        files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    import hashlib

    return {"digest": hashlib.sha256(payload).hexdigest(), "files": files}


def _load_registration(root: Path, supplied: str | Path) -> dict[str, Any]:
    path = Path(supplied)
    if not path.is_absolute():
        path = root / path
    expected = root / REGISTRATION_PATH
    if path.resolve() != expected.resolve():
        raise ValueError("matched DID registration path differs from frozen path")
    if sha256_file(path) != REGISTRATION_SHA256:
        raise ValueError("matched DID registration hash mismatch")
    probe = root / PROBE_MANIFEST_PATH
    if sha256_file(probe) != PROBE_MANIFEST_SHA256:
        raise ValueError("parent probe manifest hash mismatch")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("matched DID registration must be a mapping")
    expected_top = {
        "registration_id": "motivation_m3_q2_matched_did_synthesis_v1",
        "status": "frozen_before_matched_control_outcomes",
    }
    for key, expected_value in expected_top.items():
        if value.get(key) != expected_value:
            raise ValueError(f"matched DID registration drift: {key}")
    parent = value.get("parent_probe_manifest")
    if not isinstance(parent, Mapping) or parent.get("sha256") != PROBE_MANIFEST_SHA256:
        raise ValueError("matched DID registration parent probe mismatch")
    return value


def _load_analysis(
    runs_dir: Path, run_id: str, *, require_per_request: bool
) -> dict[str, Any]:
    run_dir = runs_dir / run_id
    metrics_path = run_dir / "metrics.json"
    metadata_path = run_dir / "metadata.json"
    metrics = _read_json(metrics_path)
    metadata = _read_json(metadata_path)
    result: dict[str, Any] = {
        "metrics": metrics,
        "metrics_path": metrics_path,
        "metadata": metadata,
        "metadata_path": metadata_path,
        "run_id": run_id,
    }
    if require_per_request:
        per_request_path = run_dir / "per_request.jsonl"
        result["per_request_path"] = per_request_path
        result["rows"] = _read_jsonl(per_request_path)
    return result


def _validate_source_analyses(
    top: Mapping[str, Any],
    original: Mapping[str, Any],
    balanced: Mapping[str, Any],
) -> None:
    pair_expected = {
        ORIGINAL_PAIR_RUN_ID: ("original_mixture__full", "original_mixture__null"),
        BALANCED_PAIR_RUN_ID: (
            "surface_balanced__full",
            "surface_balanced__null",
        ),
    }
    for source in (original, balanced):
        metrics = source["metrics"]
        metadata = source["metadata"]
        run_id = str(source["run_id"])
        treatment, control = pair_expected[run_id]
        expected = {
            "analysis_run_id": run_id,
            "analysis_type": "motivation_mechanism_paired_probe",
            "control_condition_id": control,
            "label_mode": "graded",
            "method_id": METHOD_ID,
            "num_requests": EXPECTED_REQUESTS,
            "split": "dev",
            "treatment_condition_id": treatment,
        }
        for key, value in expected.items():
            if metrics.get(key) != value:
                raise ValueError(f"matched DID pair source mismatch {run_id}: {key}")
        if metadata.get("qrels_read") is not True or metadata.get("split") != "dev":
            raise ValueError(f"matched DID pair completion mismatch: {run_id}")
        if len(source["rows"]) != EXPECTED_REQUESTS:
            raise ValueError(f"matched DID pair row coverage mismatch: {run_id}")
    equality_keys = (
        "candidate_manifest_sha256",
        "dataset_id",
        "dataset_version",
        "label_mode",
        "method_id",
        "num_requests",
        "qrels_sha256",
        "request_manifest_sha256",
        "split",
    )
    for key in equality_keys:
        if original["metrics"].get(key) != balanced["metrics"].get(key):
            raise ValueError(f"matched DID pair invariant differs: {key}")
    top_metrics = top["metrics"]
    if top_metrics.get("analysis_run_id") != TOP_LEVEL_RUN_ID or top_metrics.get(
        "analysis_type"
    ) != "q2_matched_training_control_cross_checkpoint":
        raise ValueError("matched DID top-level source identity mismatch")
    if top["metadata"].get("status") != "completed" or top_metrics.get(
        "status"
    ) != "completed":
        raise ValueError("matched DID top-level source is incomplete")
    if top_metrics.get("original_mixture_metrics") != original["metrics"]:
        raise ValueError("matched DID embedded original metrics differ")
    if top_metrics.get("surface_balanced_metrics") != balanced["metrics"]:
        raise ValueError("matched DID embedded balanced metrics differ")
    if top_metrics.get("qrels_sha256") != original["metrics"].get("qrels_sha256"):
        raise ValueError("matched DID top-level qrels identity differs")


def _validate_producer_point_estimates(
    top_metrics: Mapping[str, Any], summaries: Mapping[str, Any]
) -> None:
    producer = top_metrics.get("surfaces")
    if not isinstance(producer, Mapping) or set(producer) != set(SURFACES):
        raise ValueError("matched DID producer surface set differs")
    for surface in SURFACES:
        expected = producer[surface]
        observed = summaries[surface]
        for key in (
            "balanced_minus_original_history_response_ndcg@10",
            "balanced_minus_original_target_margin_change",
            "num_requests",
        ):
            if not _same_number_or_none(expected.get(key), observed.get(key)):
                raise ValueError(
                    f"matched DID recomputed producer point differs: {surface}.{key}"
                )
    if not _same_number_or_none(
        top_metrics.get("balanced_minus_original_history_response_ndcg@10"),
        summaries["all"]["balanced_minus_original_history_response_ndcg@10"],
    ):
        raise ValueError("matched DID top-level NDCG point differs")
    if not _same_number_or_none(
        top_metrics.get("balanced_minus_original_target_margin_change"),
        summaries["all"]["balanced_minus_original_target_margin_change"],
    ):
        raise ValueError("matched DID top-level margin point differs")


def _input_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "metadata_path": str(source["metadata_path"]),
        "metadata_sha256": sha256_file(source["metadata_path"]),
        "metrics_path": str(source["metrics_path"]),
        "metrics_sha256": sha256_file(source["metrics_path"]),
        "run_id": source["run_id"],
    }
    if "per_request_path" in source:
        result.update(
            {
                "per_request_path": str(source["per_request_path"]),
                "per_request_sha256": sha256_file(source["per_request_path"]),
            }
        )
    return result


def _index_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        request_id = str(row.get("request_id") or "")
        if not request_id or request_id in result:
            raise ValueError(f"matched DID {label} has empty/duplicate request_id")
        cluster = str(row.get("normalized_query_cluster") or "")
        normalized_query_fold(cluster)
        surface = str(row.get("target_aware_surface") or "")
        if surface not in SURFACES[2:]:
            raise ValueError(f"matched DID {label} has invalid target surface")
        result[request_id] = row
    return result


def _finite_required(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("matched DID value must be finite") from exc
    if not math.isfinite(result):
        raise ValueError("matched DID value must be finite")
    return result


def _finite_optional(value: Any) -> float | None:
    return None if value is None else _finite_required(value)


def _same_number_or_none(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12)
    except (TypeError, ValueError):
        return left == right


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL row: {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row: {path}:{line_number}")
        rows.append(value)
    return rows


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_ledger_exact_once(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        for existing in _read_jsonl(path):
            if existing.get("run_id") == row.get("run_id"):
                raise ValueError("matched DID ledger run_id already exists")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "ANALYSIS_RUN_ID",
    "ANALYSIS_TYPE",
    "BALANCED_PAIR_RUN_ID",
    "ORIGINAL_PAIR_RUN_ID",
    "REGISTRATION_SHA256",
    "STRICT_SURFACE",
    "TOP_LEVEL_RUN_ID",
    "join_pair_rows",
    "matched_control_synthesis_implementation_identity",
    "summarize_strict_did",
    "summarize_surfaces",
    "synthesize_q2_matched_control_did",
]
