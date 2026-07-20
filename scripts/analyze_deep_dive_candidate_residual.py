#!/usr/bin/env python3
"""Decompose history-conditioned candidate deltas into common and residual parts.

This qrels-blind D1 diagnostic expands the 482 requests selected by the frozen
candidate-row control to their complete candidate slates.  At every hidden
state it separates the full-minus-null candidate displacement into a
request-common component and a candidate-relative residual.  It also measures
their alignment with the frozen train-only category/brand probe subspaces.

The output is descriptive geometry.  It is not a causal layer selector and it
does not authorize a method or architecture change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FROZEN_SAMPLE_MANIFEST_SHA256 = (
    "84cdf68a0fabefcab055806bb690adf96f2a36ad2921c2d10c5d0aae8310aa61"
)
FROZEN_SAMPLE_ROWS_SHA256 = (
    "258f9303b15d0778d8ca7fe91883f694424f25bc18271cb76f4a9da2941eb985"
)
SAMPLE_DIR = Path(
    "artifacts/motivation_transformer_deep_dive/frozen_controls/"
    "fixed_candidate_rows_v1"
)
MODELS = {
    "q2": {
        "method_id": "q2_recranker_generalqwen",
        "full": Path("runs/20260718_kuaisearch_mech_d1_q2_dev_full_all29"),
        "null": Path("runs/20260718_kuaisearch_mech_d1_q2_dev_null_all29"),
        "probe": Path("runs/20260718_kuaisearch_mech_d1_q2_probe_all29_v2"),
        "evaluation": Path("runs/20260718_kuaisearch_mech_d1_q2_eval_all29_v2"),
    },
    "q3": {
        "method_id": "q3_tallrec_generalqwen",
        "full": Path("runs/20260718_kuaisearch_mech_d1_q3_dev_full_all29"),
        "null": Path("runs/20260718_kuaisearch_mech_d1_q3_dev_null_all29"),
        "probe": Path("runs/20260718_kuaisearch_mech_d1_q3_probe_all29_v2"),
        "evaluation": Path("runs/20260718_kuaisearch_mech_d1_q3_eval_all29_v2"),
    },
}
STATES = tuple(range(29))
TASKS = ("brand", "category")
CONTROLS = ("real_labels", "random_labels")
REGIONS = {
    "blocks_00_06": tuple(range(1, 8)),
    "blocks_07_13": tuple(range(8, 15)),
    "blocks_14_20": tuple(range(15, 22)),
    "blocks_21_27": tuple(range(22, 29)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="runs/20260718_kuaisearch_mech_d1_candidate_residual_v1",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_manifest_path = root / SAMPLE_DIR / "manifest.json"
    sample_rows_path = root / SAMPLE_DIR / "candidate_rows.jsonl"
    if _sha256_file(sample_manifest_path) != FROZEN_SAMPLE_MANIFEST_SHA256:
        raise ValueError("frozen candidate-row manifest hash drift")
    if _sha256_file(sample_rows_path) != FROZEN_SAMPLE_ROWS_SHA256:
        raise ValueError("frozen candidate-row data hash drift")
    sample_manifest = _read_json(sample_manifest_path)
    if (
        sample_manifest.get("qrels_read") is not False
        or sample_manifest.get("source_test_opened") is not False
        or sample_manifest.get("selected_candidate_rows") != 512
    ):
        raise ValueError("frozen request-anchor safety boundary failed")
    sample_rows = list(_iter_jsonl(sample_rows_path))
    ordered_requests = tuple(
        dict.fromkeys(str(row["request_id"]) for row in sample_rows)
    )
    if len(sample_rows) != 512 or len(ordered_requests) != 482:
        raise ValueError("frozen request-anchor population drift")

    state_rows: list[dict[str, Any]] = []
    source_audit: dict[str, Any] = {}
    for model_key, spec in MODELS.items():
        bases, probe_audit = _load_probe_bases(root / spec["probe"], spec)
        request_metrics, representation_audit = _load_request_metrics(
            root, model_key, spec, ordered_requests, bases
        )
        source_audit[model_key] = {
            **representation_audit,
            **probe_audit,
        }
        state_rows.extend(_summarize_model(model_key, request_metrics))

    region_rows = _build_region_rows(state_rows)
    result = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d1_candidate_residual_geometry",
        "status": "completed",
        "descriptive_only": True,
        "confirmatory_family_membership": False,
        "causal_layer_selector": False,
        "interpretation_boundary": (
            "A request-common activation displacement is rank-invariant only under "
            "a shared linear readout; the decomposition diagnoses geometry and does "
            "not prove downstream score cancellation, causality, or an erasure layer."
        ),
        "request_anchor": {
            "selection": (
                "all candidates belonging to the 482 requests present in the "
                "qrels-blind frozen 512 candidate-row sample"
            ),
            "sample_manifest_path": SAMPLE_DIR.joinpath("manifest.json").as_posix(),
            "sample_manifest_sha256": FROZEN_SAMPLE_MANIFEST_SHA256,
            "sample_rows_path": SAMPLE_DIR.joinpath("candidate_rows.jsonl").as_posix(),
            "sample_rows_sha256": FROZEN_SAMPLE_ROWS_SHA256,
            "anchor_candidate_rows": len(sample_rows),
            "requests": len(ordered_requests),
            "qrels_read": False,
        },
        "decomposition_definition": (
            "For each request/state, delta_i=h_full_i-h_null_i; common=mean_i(delta_i); "
            "residual_i=delta_i-common. Energy uses equal-candidate means within a "
            "request and equal-request means in reported summaries."
        ),
        "probe_projection_definition": (
            "Raw-coordinate centered train-only probe coefficient row spaces; "
            "projection energy is reported separately for common and residual deltas."
        ),
        "hidden_state_indices": list(STATES),
        "tasks": list(TASKS),
        "controls": list(CONTROLS),
        "sources": source_audit,
        "state_rows": state_rows,
        "fixed_region_rows": region_rows,
        "qrels_read": False,
        "dev_confirmation_test_qrels_read": False,
        "source_test_opened": False,
        "command": " ".join(os.sys.argv),
    }
    output_path = output_dir / "metrics.json"
    _write_json_atomic(output_path, result)
    print(
        json.dumps(
            {
                "status": "completed",
                "state_rows": len(state_rows),
                "region_rows": len(region_rows),
                "output": str(output_path),
                "sha256": _sha256_file(output_path),
            },
            sort_keys=True,
        )
    )


def _load_probe_bases(
    probe_dir: Path, spec: Mapping[str, Any]
) -> tuple[dict[tuple[str, str, int], np.ndarray], dict[str, Any]]:
    metadata_path = probe_dir / "metadata.json"
    weights_path = probe_dir / "probe_weights.npz"
    metadata = _read_json(metadata_path)
    if metadata.get("method_id") != spec["method_id"]:
        raise ValueError("probe method identity differs")
    if metadata.get("weights_sha256") != _sha256_file(weights_path):
        raise ValueError("probe weights hash differs")
    if metadata.get("dev_qrels_read") is not False:
        raise ValueError("probe crossed the dev qrels boundary")
    bases: dict[tuple[str, str, int], np.ndarray] = {}
    with np.load(weights_path, allow_pickle=False) as weights:
        for task in TASKS:
            for control in CONTROLS:
                for state in STATES:
                    prefix = (
                        f"candidate_readout__{task}__state_{state}__{control}"
                    )
                    scale = np.asarray(weights[f"{prefix}__scale"], dtype=np.float64)
                    coefficient = np.asarray(
                        weights[f"{prefix}__coefficient"], dtype=np.float64
                    )
                    bases[(task, control, state)] = _probe_basis(scale, coefficient)
    return bases, {
        "probe_metadata_sha256": _sha256_file(metadata_path),
        "probe_weights_sha256": _sha256_file(weights_path),
    }


def _load_request_metrics(
    root: Path,
    model_key: str,
    spec: Mapping[str, Any],
    ordered_requests: Sequence[str],
    bases: Mapping[tuple[str, str, int], np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full_dir = root / spec["full"]
    null_dir = root / spec["null"]
    full_meta = _read_json(full_dir / "metadata.json")
    null_meta = _read_json(null_dir / "metadata.json")
    full_index = _read_json(full_dir / "index.json")
    null_index = _read_json(null_dir / "index.json")
    for condition, metadata in (("full", full_meta), ("null", null_meta)):
        expected = {
            "status": "completed",
            "result_eligible": True,
            "method_id": spec["method_id"],
            "condition_id": condition,
            "qrels_read": False,
            "source_test_opened": False,
            "hidden_state_indices": list(STATES),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"{model_key} {condition} metadata differs: {key}")
    invariants = (
        "method_id",
        "checkpoint_id",
        "config_sha256",
        "records_sha256",
        "candidate_manifest_sha256",
        "request_manifest_sha256",
        "dataset_manifest_sha256",
        "deep_dive_manifest_sha256",
        "request_positions",
        "candidate_positions",
        "hidden_state_indices",
    )
    for key in invariants:
        if full_meta.get(key) != null_meta.get(key):
            raise ValueError(f"{model_key} full/null invariant differs: {key}")
    for index in (full_index, null_index):
        if index.get("request_count") != 8000 or index.get("candidate_count") != 160753:
            raise ValueError(f"{model_key} representation population differs")
    if len(full_index.get("shards", [])) != len(null_index.get("shards", [])):
        raise ValueError(f"{model_key} full/null shard count differs")

    evaluation_dir = root / spec["evaluation"]
    evaluation_metrics = _read_json(evaluation_dir / "metrics.json")
    pre_qrels_path = evaluation_dir / "pre_qrels_audit.json"
    if evaluation_metrics.get("status") != "completed":
        raise ValueError(f"{model_key} D1 evaluation is incomplete")
    if evaluation_metrics.get("pre_qrels_audit_sha256") != _sha256_file(pre_qrels_path):
        raise ValueError(f"{model_key} D1 pre-qrels audit hash differs")
    pre_qrels = _read_json(pre_qrels_path)
    for condition, directory in (("full", full_dir), ("null", null_dir)):
        declared = pre_qrels["bundles"][condition]
        if declared.get("metadata_sha256") != _sha256_file(directory / "metadata.json"):
            raise ValueError(f"{model_key} evaluated metadata hash differs: {condition}")
        if declared.get("index_sha256") != _sha256_file(directory / "index.json"):
            raise ValueError(f"{model_key} evaluated index hash differs: {condition}")

    selected = set(ordered_requests)
    seen: set[str] = set()
    metrics: list[dict[str, Any]] = []
    verified_shards = 0
    selected_candidate_rows = 0
    for full_shard, null_shard in zip(full_index["shards"], null_index["shards"]):
        if full_shard["path"] != null_shard["path"]:
            raise ValueError(f"{model_key} full/null shard partition differs")
        full_path = full_dir / "shards" / full_shard["path"]
        null_path = null_dir / "shards" / null_shard["path"]
        with np.load(full_path, allow_pickle=False) as full:
            request_ids = [str(value) for value in full["request_ids"].tolist()]
        if not selected.intersection(request_ids):
            continue
        if _sha256_file(full_path) != full_shard["sha256"]:
            raise ValueError(f"{model_key} selected full shard hash differs")
        if _sha256_file(null_path) != null_shard["sha256"]:
            raise ValueError(f"{model_key} selected null shard hash differs")
        verified_shards += 1
        with np.load(full_path, allow_pickle=False) as full, np.load(
            null_path, allow_pickle=False
        ) as null:
            for key in (
                "request_ids",
                "normalized_queries",
                "candidate_offsets",
                "candidate_ids",
                "hidden_state_indices",
                "request_positions",
            ):
                if not np.array_equal(full[key], null[key]):
                    raise ValueError(f"{model_key} selected shard alignment differs: {key}")
            request_ids = [str(value) for value in full["request_ids"].tolist()]
            queries = [str(value) for value in full["normalized_queries"].tolist()]
            offsets = np.asarray(full["candidate_offsets"], dtype=np.int64)
            for local, request_id in enumerate(request_ids):
                if request_id not in selected:
                    continue
                if request_id in seen:
                    raise ValueError(f"{model_key} duplicate selected request")
                seen.add(request_id)
                start, end = int(offsets[local]), int(offsets[local + 1])
                delta = (
                    np.asarray(full["candidate_activations"][start:end], dtype=np.float32)
                    - np.asarray(null["candidate_activations"][start:end], dtype=np.float32)
                )
                history_delta = (
                    np.asarray(full["request_activations"][local, 1], dtype=np.float32)
                    - np.asarray(null["request_activations"][local, 1], dtype=np.float32)
                )
                if len(delta) < 2 or delta.shape[1:] != (len(STATES), 1024):
                    raise ValueError(f"{model_key} selected slate shape differs")
                if history_delta.shape != (len(STATES), 1024):
                    raise ValueError(f"{model_key} history delta shape differs")
                if not np.isfinite(delta).all() or not np.isfinite(history_delta).all():
                    raise ValueError(f"{model_key} selected delta is non-finite")
                row = _request_metrics(delta, history_delta, bases)
                row.update(
                    {
                        "request_id": request_id,
                        "normalized_query": queries[local],
                        "candidate_count": len(delta),
                    }
                )
                metrics.append(row)
                selected_candidate_rows += len(delta)
    if seen != selected or len(metrics) != len(ordered_requests):
        raise ValueError(f"{model_key} selected request coverage is incomplete")
    metrics.sort(key=lambda row: ordered_requests.index(str(row["request_id"])))
    return metrics, {
        "method_id": spec["method_id"],
        "checkpoint_id": full_meta["checkpoint_id"],
        "full_bundle": spec["full"].as_posix(),
        "null_bundle": spec["null"].as_posix(),
        "full_metadata_sha256": _sha256_file(full_dir / "metadata.json"),
        "null_metadata_sha256": _sha256_file(null_dir / "metadata.json"),
        "full_index_sha256": _sha256_file(full_dir / "index.json"),
        "null_index_sha256": _sha256_file(null_dir / "index.json"),
        "selected_shards_verified": verified_shards,
        "selected_requests": len(metrics),
        "selected_candidate_rows": selected_candidate_rows,
        "qrels_read": False,
        "source_test_opened": False,
    }


def _request_metrics(
    delta: np.ndarray,
    history_delta: np.ndarray,
    bases: Mapping[tuple[str, str, int], np.ndarray],
) -> dict[str, Any]:
    common, residual, total_mse, common_mse, residual_mse = (
        _decompose_candidate_deltas(delta)
    )
    identity_error = np.abs(total_mse - common_mse - residual_mse)
    tolerance = 2.0e-6 * np.maximum(total_mse, 1.0)
    if np.any(identity_error > tolerance):
        raise ValueError("candidate common/residual energy identity failed")
    common_fraction = _safe_ratio(common_mse, total_mse)
    residual_to_common_rms = np.sqrt(_safe_ratio(residual_mse, common_mse))
    history_mse = np.mean(np.square(history_delta, dtype=np.float64), axis=1)
    common_history_cosine = np.asarray(
        [_cosine(common[state], history_delta[state]) for state in STATES],
        dtype=np.float64,
    )
    common_history_rms_ratio = np.sqrt(_safe_ratio(common_mse, history_mse))
    projection: dict[str, np.ndarray] = {}
    for task in TASKS:
        for control in CONTROLS:
            common_values = []
            residual_values = []
            for state in STATES:
                basis = bases[(task, control, state)]
                common_values.append(_projection_energy_fraction(common[state], basis))
                residual_values.append(
                    _projection_energy_fraction(residual[:, state], basis)
                )
            projection[f"common_{task}_{control}"] = np.asarray(
                common_values, dtype=np.float64
            )
            projection[f"residual_{task}_{control}"] = np.asarray(
                residual_values, dtype=np.float64
            )
    return {
        "total_mse": total_mse,
        "common_mse": common_mse,
        "residual_mse": residual_mse,
        "common_fraction": common_fraction,
        "residual_to_common_rms": residual_to_common_rms,
        "history_mse": history_mse,
        "common_history_cosine": common_history_cosine,
        "common_history_rms_ratio": common_history_rms_ratio,
        "projection": projection,
        "maximum_energy_identity_error": float(np.max(identity_error)),
    }


def _decompose_candidate_deltas(
    delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return common/residual tensors and per-state mean squared energies."""

    values = np.asarray(delta, dtype=np.float64)
    if values.ndim != 3 or len(values) < 2:
        raise ValueError("candidate deltas must have shape [candidate,state,hidden]")
    common = values.mean(axis=0)
    residual = values - common[None, :, :]
    total_mse = np.mean(values**2, axis=(0, 2))
    common_mse = np.mean(common**2, axis=1)
    residual_mse = np.mean(residual**2, axis=(0, 2))
    return common, residual, total_mse, common_mse, residual_mse


def _projection_energy_fraction(values: np.ndarray, basis: np.ndarray) -> float:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or basis.ndim != 2 or matrix.shape[1] != basis.shape[1]:
        raise ValueError("projection inputs have incompatible shapes")
    denominator = float(np.sum(matrix**2))
    if denominator <= 1.0e-20 or len(basis) == 0:
        return 0.0
    return float(np.sum((matrix @ basis.T) ** 2) / denominator)


def _probe_basis(scale: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    raw = np.asarray(coefficient, dtype=np.float64) / np.asarray(
        scale, dtype=np.float64
    )[None, :]
    centered = raw - raw.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if not len(singular_values):
        return np.empty((0, raw.shape[1]), dtype=np.float64)
    threshold = max(float(singular_values[0]) * 1.0e-10, 1.0e-12)
    rank = int(np.sum(singular_values > threshold))
    basis = vh[:rank]
    if rank and np.max(np.abs(basis @ basis.T - np.eye(rank))) > 1.0e-9:
        raise ValueError("probe basis is not orthonormal")
    return basis


def _summarize_model(
    model_key: str, request_metrics: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for state in STATES:
        for fold_name in ("all", "0", "1"):
            selected = [
                row
                for row in request_metrics
                if fold_name == "all"
                or _fold(str(row["normalized_query"])) == int(fold_name)
            ]
            row: dict[str, Any] = {
                "model_key": model_key,
                "hidden_state_index": state,
                "normalized_query_fold": fold_name,
                "requests": len(selected),
                "candidate_rows": int(
                    sum(int(value["candidate_count"]) for value in selected)
                ),
                "mean_candidate_count": _mean(
                    [float(value["candidate_count"]) for value in selected]
                ),
                "mean_total_rms": math.sqrt(
                    _mean([float(value["total_mse"][state]) for value in selected])
                ),
                "mean_common_rms": math.sqrt(
                    _mean([float(value["common_mse"][state]) for value in selected])
                ),
                "mean_candidate_relative_residual_rms": math.sqrt(
                    _mean([float(value["residual_mse"][state]) for value in selected])
                ),
                "mean_common_energy_fraction": _mean_finite(
                    [float(value["common_fraction"][state]) for value in selected]
                ),
                "median_common_energy_fraction": _median_finite(
                    [float(value["common_fraction"][state]) for value in selected]
                ),
                "mean_residual_to_common_rms_ratio": _mean_finite(
                    [
                        float(value["residual_to_common_rms"][state])
                        for value in selected
                    ]
                ),
                "mean_common_history_cosine": _mean_finite(
                    [
                        float(value["common_history_cosine"][state])
                        for value in selected
                    ]
                ),
                "mean_absolute_common_history_cosine": _mean_finite(
                    [
                        abs(float(value["common_history_cosine"][state]))
                        for value in selected
                    ]
                ),
                "mean_common_history_rms_ratio": _mean_finite(
                    [
                        float(value["common_history_rms_ratio"][state])
                        for value in selected
                    ]
                ),
                "maximum_energy_identity_error": max(
                    float(value["maximum_energy_identity_error"])
                    for value in selected
                ),
            }
            for component in ("common", "residual"):
                for task in TASKS:
                    values = {}
                    for control in CONTROLS:
                        key = f"{component}_{task}_{control}"
                        values[control] = _mean(
                            [
                                float(value["projection"][key][state])
                                for value in selected
                            ]
                        )
                        row[f"mean_{key}_projection_fraction"] = values[control]
                    row[f"mean_{component}_{task}_real_minus_random_projection"] = (
                        values["real_labels"] - values["random_labels"]
                    )
            rows.append(row)
    return rows


def _build_region_rows(
    state_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["model_key"]),
            str(row["normalized_query_fold"]),
            int(row["hidden_state_index"]),
        ): row
        for row in state_rows
    }
    metric_names = (
        "mean_total_rms",
        "mean_common_rms",
        "mean_candidate_relative_residual_rms",
        "mean_common_energy_fraction",
        "mean_residual_to_common_rms_ratio",
        "mean_common_history_cosine",
        "mean_common_history_rms_ratio",
        "mean_common_brand_real_minus_random_projection",
        "mean_residual_brand_real_minus_random_projection",
        "mean_common_category_real_minus_random_projection",
        "mean_residual_category_real_minus_random_projection",
    )
    rows = []
    for model_key in MODELS:
        for fold_name in ("all", "0", "1"):
            for region, states in REGIONS.items():
                selected = [lookup[(model_key, fold_name, state)] for state in states]
                row: dict[str, Any] = {
                    "model_key": model_key,
                    "normalized_query_fold": fold_name,
                    "region": region,
                    "hidden_state_indices": list(states),
                }
                for metric in metric_names:
                    row[metric] = _mean_finite(
                        [float(value[metric]) for value in selected]
                    )
                rows.append(row)
    return rows


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full_like(np.asarray(numerator, dtype=np.float64), np.nan)
    valid = np.asarray(denominator, dtype=np.float64) > 1.0e-20
    result[valid] = np.asarray(numerator, dtype=np.float64)[valid] / np.asarray(
        denominator, dtype=np.float64
    )[valid]
    return result


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    if denominator <= 1.0e-20:
        return float("nan")
    return float(np.clip(np.dot(left64, right64) / denominator, -1.0, 1.0))


def _fold(normalized_query: str) -> int:
    digest = hashlib.sha256(normalized_query.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(math.fsum(values) / len(values))


def _mean_finite(values: Sequence[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return None if not finite else _mean(finite)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take the median of an empty sequence")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _median_finite(values: Sequence[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return None if not finite else _median(finite)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
