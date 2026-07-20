"""Qrels-blind full/null all-layer trajectory summary for breadth model Q0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.mechanism.deep_dive_representation_analysis import audit_all_state_bundle
from myrec.mechanism.deep_dive_representation_evaluator import BLOCK_REGIONS
from myrec.mechanism.deep_dive_representation_runtime import (
    ALL_HIDDEN_STATE_INDICES,
    REQUEST_POSITIONS,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


Q0_METHOD_ID = "q0_qwen3_reranker_06b"
GEOMETRY_METRICS = (
    "delta_l2_per_sqrt_hidden",
    "full_null_cosine",
    "full_rms",
    "null_rms",
    "rms_ratio",
)


def evaluate_q0_all_layer_trajectory(
    standardized_dir: str | Path,
    full_bundle_dir: str | Path,
    null_bundle_dir: str | Path,
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Audit full Q0 coverage and summarize 29-state geometry without qrels."""

    standardized_dir = Path(standardized_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Q0 trajectory output is not empty: {output_dir}")
    records_path = standardized_dir / "records_dev.jsonl"
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("Q0 trajectory requires all 8000 dev requests")
    bundles = {
        condition: audit_all_state_bundle(
            path,
            expected_records=records,
            expected_role="dev_representation",
            expected_condition=condition,
            allowed_method_ids=(Q0_METHOD_ID,),
        )
        for condition, path in (("full", full_bundle_dir), ("null", null_bundle_dir))
    }
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
        if bundles["full"].metadata.get(key) != bundles["null"].metadata.get(key):
            raise ValueError(f"Q0 full/null bundle invariant differs: {key}")
    if bundles["full"].metadata.get("method_id") != Q0_METHOD_ID:
        raise ValueError("Q0 trajectory received another model")
    implementation_digest = _common_implementation_digest(bundles)

    request_arrays = {
        position: {metric: [] for metric in GEOMETRY_METRICS}
        for position in (*REQUEST_POSITIONS, "candidate_readout")
    }
    candidate_weighted = {metric: [] for metric in GEOMETRY_METRICS}
    full_index = bundles["full"].index["shards"]
    null_index = bundles["null"].index["shards"]
    if len(full_index) != len(null_index):
        raise ValueError("Q0 full/null shard count differs")
    for full_shard, null_shard in zip(full_index, null_index):
        if (
            full_shard["start_request_ordinal"] != null_shard["start_request_ordinal"]
            or full_shard["request_count"] != null_shard["request_count"]
            or full_shard["candidate_count"] != null_shard["candidate_count"]
        ):
            raise ValueError("Q0 full/null shard boundaries differ")
        with np.load(
            bundles["full"].root / "shards" / full_shard["path"], allow_pickle=False
        ) as full_payload, np.load(
            bundles["null"].root / "shards" / null_shard["path"], allow_pickle=False
        ) as null_payload:
            if not np.array_equal(full_payload["request_ids"], null_payload["request_ids"]):
                raise ValueError("Q0 full/null shard request identities differ")
            if not np.array_equal(full_payload["candidate_ids"], null_payload["candidate_ids"]):
                raise ValueError("Q0 full/null shard candidate identities differ")
            if not np.array_equal(
                full_payload["candidate_offsets"], null_payload["candidate_offsets"]
            ):
                raise ValueError("Q0 full/null candidate offsets differ")
            full_request = np.asarray(full_payload["request_activations"], dtype=np.float32)
            null_request = np.asarray(null_payload["request_activations"], dtype=np.float32)
            for position_ordinal, position in enumerate(REQUEST_POSITIONS):
                geometry = trajectory_geometry(
                    full_request[:, position_ordinal], null_request[:, position_ordinal]
                )
                for metric in GEOMETRY_METRICS:
                    request_arrays[position][metric].append(geometry[metric])
            full_candidate = np.asarray(full_payload["candidate_activations"], dtype=np.float32)
            null_candidate = np.asarray(null_payload["candidate_activations"], dtype=np.float32)
            geometry = trajectory_geometry(full_candidate, null_candidate)
            offsets = np.asarray(full_payload["candidate_offsets"], dtype=np.int64)
            for metric in GEOMETRY_METRICS:
                candidate_weighted[metric].append(geometry[metric])
                per_request = np.stack(
                    [
                        geometry[metric][int(offsets[row]) : int(offsets[row + 1])].mean(axis=0)
                        for row in range(len(offsets) - 1)
                    ]
                )
                request_arrays["candidate_readout"][metric].append(per_request)

    request_values = {
        position: {
            metric: np.concatenate(chunks, axis=0)
            for metric, chunks in metrics.items()
        }
        for position, metrics in request_arrays.items()
    }
    candidate_values = {
        metric: np.concatenate(chunks, axis=0)
        for metric, chunks in candidate_weighted.items()
    }
    rows = []
    for position, metrics in request_values.items():
        for metric, matrix in metrics.items():
            rows.extend(
                trajectory_summary_rows(
                    matrix,
                    position=position,
                    metric=metric,
                    weighting="request",
                )
            )
    for metric, matrix in candidate_values.items():
        rows.extend(
            trajectory_summary_rows(
                matrix,
                position="candidate_readout",
                metric=metric,
                weighting="candidate",
            )
        )
    region_rows = []
    for row in rows:
        if row["weighting"] != "request":
            continue
        for region, states in BLOCK_REGIONS.items():
            if row["hidden_state_index"] == min(states):
                selected = [
                    value
                    for value in rows
                    if value["position"] == row["position"]
                    and value["metric"] == row["metric"]
                    and value["weighting"] == "request"
                    and value["hidden_state_index"] in states
                ]
                region_rows.append(
                    {
                        "position": row["position"],
                        "metric": row["metric"],
                        "weighting": "request",
                        "region": region,
                        "mean_over_state_point_means": float(
                            np.mean([value["mean"] for value in selected])
                        ),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=False)
    per_request_path = output_dir / "per_request_geometry.npz"
    np.savez(
        per_request_path,
        request_ids=np.asarray([record.request_id for record in records], dtype=np.str_),
        **{
            f"{position}__{metric}": matrix
            for position, metrics in request_values.items()
            for metric, matrix in metrics.items()
        },
    )
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "d6_q0_all_layer_trajectory_integrity",
        "analysis_run_id": analysis_run_id,
        "status": "passed",
        "qrels_read": False,
        "all_8000_requests_complete": True,
        "all_candidates_complete_finite": True,
        "all_29_states_present": True,
        "full_null_share_one_implementation_digest": True,
        "implementation_digest": implementation_digest,
        "bundles": {
            condition: {
                "path": str(bundle.root),
                "metadata_sha256": sha256_file(bundle.root / "metadata.json"),
                "index_sha256": sha256_file(bundle.root / "index.json"),
            }
            for condition, bundle in bundles.items()
        },
    }
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json(pre_qrels_path, pre_qrels)
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d6_q0_all_layer_trajectory",
        "analysis_run_id": analysis_run_id,
        "method_id": Q0_METHOD_ID,
        "evidence_mode": "registered_descriptive_breadth",
        "request_count": len(records),
        "candidate_count": bundles["full"].candidate_count,
        "implementation_digest": implementation_digest,
        "hidden_state_indices": list(ALL_HIDDEN_STATE_INDICES),
        "geometry_rows": rows,
        "region_rows": region_rows,
        "per_request_geometry_path": str(per_request_path),
        "per_request_geometry_sha256": sha256_file(per_request_path),
        "pre_qrels_audit_path": str(pre_qrels_path),
        "pre_qrels_audit_sha256": sha256_file(pre_qrels_path),
        "qrels_read": False,
        "source_test_opened": False,
        "command": list(command or []),
        "status": "completed",
    }
    _write_json(output_dir / "metrics.json", metrics)
    return metrics


def _common_implementation_digest(bundles):
    metadata_rows = [bundle.metadata for bundle in bundles.values()]
    digests = {
        str(metadata.get("implementation_identity", {}).get("digest") or "")
        for metadata in metadata_rows
    }
    if len(digests) != 1 or not next(iter(digests), ""):
        raise ValueError(
            "Q0 trajectory full/null bundles use different implementation digests"
        )
    digest = next(iter(digests))
    if any(
        metadata.get("run_contract", {}).get("implementation_digest") != digest
        for metadata in metadata_rows
    ):
        raise ValueError("Q0 trajectory implementation differs from run contract")
    return digest


def trajectory_geometry(full: np.ndarray, null: np.ndarray) -> dict[str, np.ndarray]:
    """Return hand-auditable row/state geometry for aligned hidden tensors."""

    full = np.asarray(full, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    if full.shape != null.shape or full.ndim != 3 or full.shape[1] != 29:
        raise ValueError("trajectory geometry expects aligned [row,29,hidden] tensors")
    hidden = full.shape[-1]
    full_norm = np.linalg.norm(full, axis=-1)
    null_norm = np.linalg.norm(null, axis=-1)
    denominator = np.maximum(full_norm * null_norm, 1.0e-12)
    full_rms = full_norm / np.sqrt(hidden)
    null_rms = null_norm / np.sqrt(hidden)
    return {
        "delta_l2_per_sqrt_hidden": np.linalg.norm(full - null, axis=-1)
        / np.sqrt(hidden),
        "full_null_cosine": np.sum(full * null, axis=-1) / denominator,
        "full_rms": full_rms,
        "null_rms": null_rms,
        "rms_ratio": full_rms / np.maximum(null_rms, 1.0e-12),
    }


def trajectory_summary_rows(
    matrix: np.ndarray, *, position: str, metric: str, weighting: str
) -> list[dict[str, Any]]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 29 or not np.isfinite(matrix).all():
        raise ValueError("trajectory summary matrix is invalid")
    return [
        {
            "position": position,
            "metric": metric,
            "weighting": weighting,
            "hidden_state_index": state,
            "rows": int(matrix.shape[0]),
            "mean": float(matrix[:, state].mean()),
            "median": float(np.median(matrix[:, state])),
            "q25": float(np.quantile(matrix[:, state], 0.25)),
            "q75": float(np.quantile(matrix[:, state], 0.75)),
        }
        for state in ALL_HIDDEN_STATE_INDICES
    ]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
