"""Qrels-blind aggregation of all registered D3 attention heads and KV groups."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.mechanism.attention_observation_runtime import (
    FIXED_BLOCKS,
    SAMPLE_MANIFEST,
    SAMPLE_MANIFEST_SHA256,
    SUPPORTED_METHODS,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


def evaluate_attention_observations(
    bundle_dirs: Mapping[str, Mapping[int, str | Path]],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Audit six bundles and aggregate every head without outcome selection."""

    if set(bundle_dirs) != set(SUPPORTED_METHODS) or any(
        set(map(int, blocks)) != set(FIXED_BLOCKS) for blocks in bundle_dirs.values()
    ):
        raise ValueError("attention observation evaluator requires Q2/Q3 x 13/20/27")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"attention observation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_manifest = json.loads(Path(SAMPLE_MANIFEST).read_text(encoding="utf-8"))
    if sha256_file(SAMPLE_MANIFEST) != SAMPLE_MANIFEST_SHA256:
        raise ValueError("attention observation sample manifest differs")
    samples = list(iter_jsonl(Path(sample_manifest["path"])))
    bundles = {
        method_id: {
            block: _audit_bundle(bundle_dirs[method_id][block], method_id, block, samples)
            for block in FIXED_BLOCKS
        }
        for method_id in SUPPORTED_METHODS
    }
    implementation_digest = _common_implementation_digest(bundles)
    identities = {
        method_id: {
            str(block): {
                "path": str(bundle[0]),
                "metadata_sha256": sha256_file(bundle[0] / "metadata.json"),
                "observations_sha256": sha256_file(bundle[0] / "observations.jsonl"),
            }
            for block, bundle in blocks.items()
        }
        for method_id, blocks in bundles.items()
    }
    pre_qrels = {
        "schema_version": 1,
        "analysis_type": "d3_attention_head_observation_integrity",
        "analysis_run_id": analysis_run_id,
        "status": "passed",
        "qrels_read": False,
        "checks": {
            "all_six_fixed_bundles_present": True,
            "all_512_frozen_rows_complete_finite": True,
            "all_16_query_heads_and_8_kv_groups_present": True,
            "no_op_native_score_identity_at_most_1e-5": True,
            "manual_attention_reconstruction_within_low_precision_bound": True,
            "all_six_bundles_share_one_implementation_digest": True,
            "sample_qrels_and_model_score_blind": True,
        },
        "implementation_digest": implementation_digest,
        "bundles": identities,
    }
    pre_qrels_path = output_dir / "integrity.json"
    _write_json(pre_qrels_path, pre_qrels)
    results = {}
    for method_id, blocks in bundles.items():
        results[method_id] = {}
        for block, (_root, metadata, rows) in blocks.items():
            path_names = sorted(rows[0]["paths"])
            path_results = {}
            for path_name in path_names:
                path_rows = [row["paths"][path_name] for row in rows]
                observations = {}
                for scope in ("history_summary", "native_readout"):
                    observations[scope] = {}
                    for span in ("query", "history", "candidate"):
                        observations[scope][span] = {}
                        for metric in (
                            "attention_mass",
                            "o_proj_contribution_norm",
                            "o_proj_contribution_cosine_to_total_head",
                        ):
                            query_values = np.asarray(
                                [
                                    row["observations"][scope][span][metric]["query_head"]
                                    for row in path_rows
                                ],
                                dtype=np.float64,
                            )
                            gqa_values = np.asarray(
                                [
                                    row["observations"][scope][span][metric]["gqa_group"]
                                    for row in path_rows
                                ],
                                dtype=np.float64,
                            )
                            observations[scope][span][metric] = {
                                "query_head": _column_summary(query_values, 16),
                                "gqa_group": _column_summary(gqa_values, 8),
                            }
                geometry = {}
                position_count = len(path_rows[0]["qk_geometry"]["q"]["pre_norm"]["full_norm"])
                position_labels = ["query_end", "history_summary_end"] + [
                    f"native_readout_{index}" for index in range(position_count - 2)
                ]
                for kind, heads in (("q", 16), ("k", 8)):
                    geometry[kind] = {}
                    for stage in ("pre_norm", "post_norm", "post_rope"):
                        geometry[kind][stage] = {}
                        for metric in (
                            "full_norm", "null_norm", "full_null_delta_norm", "full_null_cosine"
                        ):
                            values = np.asarray(
                                [row["qk_geometry"][kind][stage][metric] for row in path_rows],
                                dtype=np.float64,
                            )
                            if values.shape != (512, position_count, heads):
                                raise ValueError("attention QK geometry shape differs")
                            geometry[kind][stage][metric] = {
                                label: _column_summary(values[:, index, :], heads)
                                for index, label in enumerate(position_labels)
                            }
                path_results[path_name] = {
                    "rows": len(path_rows),
                    "observations": observations,
                    "qk_geometry": geometry,
                    "manual_attention_error": _scalar_summary(
                        np.asarray([row["manual_attention_error"] for row in path_rows])
                    ),
                    "manual_attention_low_precision_ratio": _scalar_summary(
                        np.asarray(
                            [
                                row["manual_attention_low_precision_ratio"]
                                for row in path_rows
                            ]
                        )
                    ),
                }
            results[method_id][str(block)] = {
                "block_zero_based": block,
                "paths": path_results,
                "maximum_score_identity_delta": metadata["maximum_score_identity_delta"],
                "maximum_manual_attention_error": metadata["maximum_manual_attention_error"],
                "maximum_manual_attention_low_precision_ratio": metadata[
                    "maximum_manual_attention_low_precision_ratio"
                ],
            }
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d3_attention_head_observation",
        "analysis_run_id": analysis_run_id,
        "sample_rows": 512,
        "query_heads": 16,
        "kv_heads": 8,
        "gqa_heads_per_kv": 2,
        "blocks": list(FIXED_BLOCKS),
        "descriptive_only": True,
        "selection": "all heads and all KV groups; no best-head selection",
        "implementation_digest": implementation_digest,
        "qrels_read": False,
        "source_test_opened": False,
        "integrity_path": str(pre_qrels_path),
        "integrity_sha256": sha256_file(pre_qrels_path),
        "results": results,
        "command": list(command or []),
        "status": "completed",
    }
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics)
    _append_jsonl(
        Path(dev_eval_log_path),
        {
            "analysis_type": metrics["analysis_type"],
            "run_id": analysis_run_id,
            "method_ids": list(SUPPORTED_METHODS),
            "split": "dev_qrels_blind_fixed_candidate_sample",
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    return metrics


def _audit_bundle(root, method_id, block, samples):
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        "analysis_stage": "transformer_deep_dive_d3_attention_head_observation",
        "method_id": method_id,
        "block_zero_based": block,
        "status": "completed",
        "result_eligible": True,
        "qrels_read": False,
        "source_test_opened": False,
        "observation_rows": 512,
        "complete_finite_observation_coverage": True,
        "manual_attention_reconstruction_dtype": "float32",
        "query_heads": 16,
        "kv_heads": 8,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"attention observation metadata differs: {key}")
    if float(metadata.get("maximum_score_identity_delta", math.inf)) > 1.0e-5:
        raise ValueError("attention observation score identity failed")
    if float(
        metadata.get("maximum_manual_attention_low_precision_ratio", math.inf)
    ) > 1.0:
        raise ValueError("attention observation manual reconstruction failed")
    path = root / "observations.jsonl"
    if metadata.get("observations_sha256") != sha256_file(path):
        raise ValueError("attention observation score hash differs")
    rows = list(iter_jsonl(path))
    if len(rows) != 512:
        raise ValueError("attention observation row count differs")
    for index, (row, sample) in enumerate(zip(rows, samples)):
        if (
            row.get("row_index") != index
            or row.get("request_id") != sample.get("request_id")
            or row.get("candidate_item_id") != sample.get("candidate_item_id")
            or row.get("selection_sha256") != sample.get("selection_sha256")
            or not _all_finite(row)
        ):
            raise ValueError("attention observation row identity/finite coverage differs")
    return root, metadata, rows


def _common_implementation_digest(bundles):
    metadata_rows = [
        metadata
        for blocks in bundles.values()
        for _root, metadata, _rows in blocks.values()
    ]
    digests = {
        str(metadata.get("implementation_identity", {}).get("digest") or "")
        for metadata in metadata_rows
    }
    if len(digests) != 1 or not next(iter(digests), ""):
        raise ValueError(
            "attention observation bundles use different implementation digests"
        )
    digest = next(iter(digests))
    if any(
        metadata.get("run_contract", {}).get("implementation_digest") != digest
        for metadata in metadata_rows
    ):
        raise ValueError("attention observation implementation differs from run contract")
    return digest


def _column_summary(values: np.ndarray, columns: int) -> dict[str, Any]:
    if values.ndim != 2 or values.shape[1] != columns or not np.isfinite(values).all():
        raise ValueError("attention observation summary matrix differs")
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "rows": int(values.shape[0]),
    }


def _scalar_summary(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("attention observation scalar summary differs")
    return {
        "mean": float(values.mean()),
        "maximum": float(values.max()),
        "median": float(np.median(values)),
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
