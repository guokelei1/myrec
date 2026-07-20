"""Qrels-blind evaluator for complete Q1 KV-cache trajectory bundles."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.mechanism.deep_dive_representation_evaluator import BLOCK_REGIONS
from myrec.mechanism.q0_trajectory_evaluator import (
    GEOMETRY_METRICS,
    trajectory_summary_rows,
)
from myrec.mechanism.q1_kv_trajectory import Q1_METHOD_ID
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


def evaluate_q1_kv_trajectory(
    standardized_dir: str | Path,
    bundle_dir: str | Path,
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Summarize request-, candidate-, and token-weighted Q1 trajectories."""

    standardized_dir = Path(standardized_dir)
    bundle_dir = Path(bundle_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Q1 trajectory output is not empty: {output_dir}")
    records = [
        sanitize_record_for_model(row)
        for row in iter_jsonl(standardized_dir / "records_dev.jsonl")
    ]
    if len(records) != 8000:
        raise ValueError("Q1 trajectory evaluator requires all 8000 requests")
    metadata = _read_json(bundle_dir / "metadata.json")
    index = _read_json(bundle_dir / "index.json")
    implementation_digest = _implementation_digest(metadata)
    if (
        metadata.get("status") != "completed"
        or metadata.get("result_eligible") is not True
        or metadata.get("method_id") != Q1_METHOD_ID
        or metadata.get("qrels_read") is not False
        or metadata.get("source_test_opened") is not False
        or metadata.get("complete_finite_trajectory_coverage") is not True
        or metadata.get("native_score_identity_passed") is not True
        or float(metadata.get("maximum_native_score_identity_delta", math.inf)) > 1.0e-5
        or tuple(metadata.get("geometry_metrics", ())) != GEOMETRY_METRICS
        or int(index.get("request_count", -1)) != len(records)
        or int(index.get("candidate_count", -1)) != 160753
        or metadata.get("index_sha256") != sha256_file(bundle_dir / "index.json")
        or index.get("run_contract_sha256") != metadata.get("run_contract_sha256")
    ):
        raise ValueError("Q1 trajectory bundle metadata failed integrity")

    request_values = {
        position: {metric: [] for metric in GEOMETRY_METRICS}
        for position in (
            "query_end",
            "history_summary_end",
            "prompt_readout",
            "response_mean",
            "response_first_token",
            "response_continuation",
        )
    }
    candidate_values = {
        position: {metric: [] for metric in GEOMETRY_METRICS}
        for position in (
            "response_mean",
            "response_first_token",
            "response_continuation",
        )
    }
    token_weighted_values = {metric: [] for metric in GEOMETRY_METRICS}
    observed_requests = []
    observed_candidates = []
    response_tokens = 0
    for shard in index["shards"]:
        path = bundle_dir / "shards" / str(shard["path"])
        if sha256_file(path) != shard["sha256"]:
            raise ValueError("Q1 trajectory shard hash mismatch")
        with np.load(path, allow_pickle=False) as payload:
            request_ids = [str(value) for value in payload["request_ids"].tolist()]
            candidate_ids = [str(value) for value in payload["candidate_ids"].tolist()]
            offsets = np.asarray(payload["candidate_offsets"], dtype=np.int64)
            lengths = np.asarray(payload["target_lengths"], dtype=np.int64)
            request_geometry = np.asarray(payload["request_geometry"], dtype=np.float64)
            prompt_geometry = np.asarray(
                payload["prompt_readout_geometry"], dtype=np.float64
            )
            response = np.asarray(payload["response_geometry"], dtype=np.float64)
            first = np.asarray(payload["first_token_geometry"], dtype=np.float64)
            continuation = np.asarray(
                payload["continuation_geometry"], dtype=np.float64
            )
            if payload["geometry_metrics"].tolist() != list(GEOMETRY_METRICS):
                raise ValueError("Q1 trajectory metric order drift")
            expected_shapes = {
                "request": (len(request_ids), 2, 29, len(GEOMETRY_METRICS)),
                "prompt": (len(request_ids), 29, len(GEOMETRY_METRICS)),
                "response": (len(candidate_ids), 29, len(GEOMETRY_METRICS)),
            }
            if (
                request_geometry.shape != expected_shapes["request"]
                or prompt_geometry.shape != expected_shapes["prompt"]
                or response.shape != expected_shapes["response"]
                or first.shape != expected_shapes["response"]
                or continuation.shape != expected_shapes["response"]
                or offsets.shape != (len(request_ids) + 1,)
                or int(offsets[-1]) != len(candidate_ids)
                or lengths.shape != (len(candidate_ids),)
                or np.any(lengths < 2)
            ):
                raise ValueError("Q1 trajectory shard array shape drift")
            expected = records[len(observed_requests) : len(observed_requests) + len(request_ids)]
            if request_ids != [record.request_id for record in expected]:
                raise ValueError("Q1 trajectory request identity/order drift")
            expected_items = [
                str(candidate["item_id"])
                for record in expected
                for candidate in record.candidates
            ]
            if candidate_ids != expected_items:
                raise ValueError("Q1 trajectory candidate identity/order drift")
            for metric_index, metric in enumerate(GEOMETRY_METRICS):
                request_values["query_end"][metric].append(
                    request_geometry[:, 0, :, metric_index]
                )
                request_values["history_summary_end"][metric].append(
                    request_geometry[:, 1, :, metric_index]
                )
                request_values["prompt_readout"][metric].append(
                    prompt_geometry[:, :, metric_index]
                )
                for position, values in (
                    ("response_mean", response[:, :, metric_index]),
                    ("response_first_token", first[:, :, metric_index]),
                    ("response_continuation", continuation[:, :, metric_index]),
                ):
                    candidate_values[position][metric].append(values)
                    request_values[position][metric].append(
                        np.stack(
                            [
                                values[int(offsets[row]) : int(offsets[row + 1])].mean(axis=0)
                                for row in range(len(request_ids))
                            ]
                        )
                    )
                token_weighted_values[metric].append(response[:, :, metric_index])
            observed_requests.extend(request_ids)
            observed_candidates.extend(candidate_ids)
            response_tokens += int(lengths.sum())
    if observed_requests != [record.request_id for record in records]:
        raise ValueError("Q1 trajectory request coverage is incomplete")
    if response_tokens != int(index["response_tokens"]):
        raise ValueError("Q1 trajectory response-token count drift")

    request_matrices = {
        position: {
            metric: np.concatenate(chunks, axis=0)
            for metric, chunks in values.items()
        }
        for position, values in request_values.items()
    }
    candidate_matrices = {
        position: {
            metric: np.concatenate(chunks, axis=0)
            for metric, chunks in values.items()
        }
        for position, values in candidate_values.items()
    }
    rows = []
    for position, values in request_matrices.items():
        for metric, matrix in values.items():
            rows.extend(
                trajectory_summary_rows(
                    matrix,
                    position=position,
                    metric=metric,
                    weighting="request",
                )
            )
    for position, values in candidate_matrices.items():
        for metric, matrix in values.items():
            rows.extend(
                trajectory_summary_rows(
                    matrix,
                    position=position,
                    metric=metric,
                    weighting="candidate",
                )
            )
    # Each response row is already a within-candidate token mean.  The exact
    # global token weighting is recovered from the stored target lengths.
    all_lengths = []
    all_response = {metric: [] for metric in GEOMETRY_METRICS}
    for shard in index["shards"]:
        with np.load(bundle_dir / "shards" / shard["path"], allow_pickle=False) as payload:
            all_lengths.append(np.asarray(payload["target_lengths"], dtype=np.float64))
            values = np.asarray(payload["response_geometry"], dtype=np.float64)
            for metric_index, metric in enumerate(GEOMETRY_METRICS):
                all_response[metric].append(values[:, :, metric_index])
    lengths = np.concatenate(all_lengths)
    for metric in GEOMETRY_METRICS:
        matrix = np.concatenate(all_response[metric], axis=0)
        weighted_mean = np.average(matrix, axis=0, weights=lengths)
        rows.extend(
            [
                {
                    "position": "response_mean",
                    "metric": metric,
                    "weighting": "token",
                    "hidden_state_index": state,
                    "rows": int(response_tokens),
                    "mean": float(weighted_mean[state]),
                    "median": None,
                    "q25": None,
                    "q75": None,
                }
                for state in range(29)
            ]
        )

    region_rows = []
    for position in request_matrices:
        for metric in GEOMETRY_METRICS:
            selected_rows = [
                row
                for row in rows
                if row["position"] == position
                and row["metric"] == metric
                and row["weighting"] == "request"
            ]
            for region, states in BLOCK_REGIONS.items():
                region_rows.append(
                    {
                        "position": position,
                        "metric": metric,
                        "weighting": "request",
                        "region": region,
                        "mean_over_state_point_means": float(
                            np.mean(
                                [
                                    row["mean"]
                                    for row in selected_rows
                                    if row["hidden_state_index"] in states
                                ]
                            )
                        ),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=False)
    per_request_path = output_dir / "per_request_geometry.npz"
    np.savez(
        per_request_path,
        request_ids=np.asarray(observed_requests, dtype=np.str_),
        **{
            f"{position}__{metric}": matrix.astype(np.float32)
            for position, values in request_matrices.items()
            for metric, matrix in values.items()
        },
    )
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "d6_q1_kv_trajectory_integrity",
        "analysis_run_id": analysis_run_id,
        "status": "passed",
        "qrels_read": False,
        "all_requests_candidates_response_tokens_complete": True,
        "native_score_identity_at_most_1e-5": True,
        "prefix_and_continuation_call_accounting_complete": True,
        "index_hash_and_run_contract_bound": True,
        "implementation_digest_bound": True,
        "implementation_digest": implementation_digest,
        "bundle_metadata_sha256": sha256_file(bundle_dir / "metadata.json"),
        "bundle_index_sha256": sha256_file(bundle_dir / "index.json"),
    }
    pre_qrels_path = output_dir / "pre_qrels_audit.json"
    _write_json(pre_qrels_path, pre_qrels)
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d6_q1_kv_trajectory",
        "analysis_run_id": analysis_run_id,
        "method_id": Q1_METHOD_ID,
        "evidence_mode": "registered_descriptive_breadth",
        "implementation_digest": implementation_digest,
        "request_count": len(records),
        "candidate_count": len(observed_candidates),
        "response_token_count": response_tokens,
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


def _implementation_digest(metadata: Mapping[str, Any]) -> str:
    digest = str(metadata.get("implementation_identity", {}).get("digest") or "")
    if not digest:
        raise ValueError("Q1 trajectory implementation digest is missing")
    if metadata.get("run_contract", {}).get("implementation_digest") != digest:
        raise ValueError("Q1 trajectory implementation digest differs from run contract")
    return digest


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
