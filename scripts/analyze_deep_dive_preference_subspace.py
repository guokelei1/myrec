#!/usr/bin/env python3
"""Measure full-null displacement inside frozen preference-probe subspaces.

The analysis uses the qrels-blind 512 candidate-row sample frozen before D1
outcomes.  It reports all 29 states, both anchor models, all registered
positions, both preference tasks, and real/random-label probe subspaces.  It is
descriptive geometry, not a new inference family or a causal layer selector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

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
POSITIONS = ("query_end", "history_summary_end", "candidate_readout")
REQUEST_POSITIONS = ("query_end", "history_summary_end")
TASKS = ("brand", "category")
CONTROLS = ("real_labels", "random_labels")
STATES = tuple(range(29))
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
        default="runs/20260718_kuaisearch_mech_d1_preference_subspace_v1",
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
        raise ValueError("frozen candidate-row safety boundary failed")
    sample_rows = list(_iter_jsonl(sample_rows_path))
    _audit_sample_rows(sample_rows)
    ordered_requests = tuple(dict.fromkeys(str(row["request_id"]) for row in sample_rows))
    if len(ordered_requests) != 482:
        raise ValueError("frozen sample request count drift")

    all_state_rows: list[dict[str, Any]] = []
    probe_transport_rows: list[dict[str, Any]] = []
    cross_position_transport_rows: list[dict[str, Any]] = []
    source_audit: dict[str, Any] = {}
    for model_key, spec in MODELS.items():
        deltas, queries, model_sources = _load_sample_deltas(
            root, model_key, spec, sample_rows, ordered_requests
        )
        cross_position_transport_rows.extend(
            _build_cross_position_transport_rows(
                model_key, deltas, queries, sample_rows, ordered_requests
            )
        )
        source_audit[model_key] = model_sources
        probe_dir = root / spec["probe"]
        probe_metadata_path = probe_dir / "metadata.json"
        probe_metadata = _read_json(probe_metadata_path)
        weights_path = probe_dir / "probe_weights.npz"
        if probe_metadata.get("method_id") != spec["method_id"]:
            raise ValueError(f"probe method identity differs for {model_key}")
        if probe_metadata.get("weights_sha256") != _sha256_file(weights_path):
            raise ValueError(f"probe weights hash differs for {model_key}")
        if probe_metadata.get("dev_qrels_read") is not False:
            raise ValueError(f"probe crossed dev qrels boundary for {model_key}")
        source_audit[model_key]["probe_metadata_sha256"] = _sha256_file(
            probe_metadata_path
        )
        source_audit[model_key]["probe_weights_sha256"] = _sha256_file(weights_path)

        with np.load(weights_path, allow_pickle=False) as weights:
            probe_transport_rows.extend(_build_probe_transport_rows(weights, model_key))
            for position in POSITIONS:
                matrix = deltas[position]
                request_ids = (
                    ordered_requests
                    if position in REQUEST_POSITIONS
                    else tuple(str(row["request_id"]) for row in sample_rows)
                )
                normalized_queries = (
                    [queries[request_id] for request_id in ordered_requests]
                    if position in REQUEST_POSITIONS
                    else [queries[str(row["request_id"])] for row in sample_rows]
                )
                for task in TASKS:
                    for state in STATES:
                        for control in CONTROLS:
                            scale, coefficient = _read_probe(
                                weights, position, task, state, control
                            )
                            metrics = _row_metrics(matrix[:, state], scale, coefficient)
                            request_metrics = _aggregate_by_request(
                                request_ids, normalized_queries, metrics
                            )
                            for fold_name in ("all", "0", "1"):
                                selected = [
                                    value
                                    for value in request_metrics
                                    if fold_name == "all"
                                    or _fold(value["normalized_query"]) == int(fold_name)
                                ]
                                all_state_rows.append(
                                    _summarize(
                                        model_key,
                                        position,
                                        task,
                                        state,
                                        control,
                                        fold_name,
                                        selected,
                                        metrics["subspace_rank"],
                                        matrix.shape[-1],
                                    )
                                )

    excess_rows = _build_excess_rows(all_state_rows)
    region_rows = _build_region_rows(excess_rows)
    probe_transport_region_rows = _build_probe_transport_region_rows(
        probe_transport_rows
    )
    cross_position_transport_region_rows = _build_cross_position_transport_region_rows(
        cross_position_transport_rows
    )
    result = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d1_preference_subspace_geometry",
        "status": "completed",
        "descriptive_only": True,
        "confirmatory_family_membership": False,
        "causal_layer_selector": False,
        "interpretation_boundary": (
            "Projection into a train-only linear-probe discriminative row space "
            "measures geometric alignment, not preference causality or native-readout use. "
            "The fixed 512-row sample is qrels-blind and cannot support population NDCG."
        ),
        "projection_definition": (
            "delta_std=(full-null)/probe_scale; centered probe coefficients define "
            "an orthonormal discriminative row-space basis; energy_fraction="
            "||Proj(delta_std)||^2/||delta_std||^2"
        ),
        "sample": {
            "manifest_path": SAMPLE_DIR.joinpath("manifest.json").as_posix(),
            "manifest_sha256": FROZEN_SAMPLE_MANIFEST_SHA256,
            "rows_path": SAMPLE_DIR.joinpath("candidate_rows.jsonl").as_posix(),
            "rows_sha256": FROZEN_SAMPLE_ROWS_SHA256,
            "candidate_rows": len(sample_rows),
            "requests": len(ordered_requests),
            "selection": sample_manifest["selection"],
            "qrels_read": False,
        },
        "sources": source_audit,
        "hidden_state_indices": list(STATES),
        "positions": list(POSITIONS),
        "tasks": list(TASKS),
        "controls": list(CONTROLS),
        "state_rows": all_state_rows,
        "real_minus_random_rows": excess_rows,
        "fixed_region_rows": region_rows,
        "probe_subspace_adjacent_rows": probe_transport_rows,
        "probe_subspace_fixed_region_rows": probe_transport_region_rows,
        "history_to_candidate_delta_transport_rows": cross_position_transport_rows,
        "history_to_candidate_delta_fixed_region_rows": (
            cross_position_transport_region_rows
        ),
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
                "status": result["status"],
                "state_rows": len(all_state_rows),
                "excess_rows": len(excess_rows),
                "region_rows": len(region_rows),
                "probe_transport_rows": len(probe_transport_rows),
                "probe_transport_region_rows": len(probe_transport_region_rows),
                "cross_position_transport_rows": len(cross_position_transport_rows),
                "cross_position_transport_region_rows": len(
                    cross_position_transport_region_rows
                ),
                "output": str(output_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _load_sample_deltas(
    root: Path,
    model_key: str,
    spec: Mapping[str, Any],
    sample_rows: Sequence[Mapping[str, Any]],
    ordered_requests: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
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
    if full_index.get("request_count") != 8000 or full_index.get("candidate_count") != 160753:
        raise ValueError(f"{model_key} full index population differs")
    if null_index.get("request_count") != 8000 or null_index.get("candidate_count") != 160753:
        raise ValueError(f"{model_key} null index population differs")
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
            raise ValueError(f"{model_key} evaluated metadata hash differs for {condition}")
        if declared.get("index_sha256") != _sha256_file(directory / "index.json"):
            raise ValueError(f"{model_key} evaluated index hash differs for {condition}")

    selected_by_request: defaultdict[str, list[tuple[int, str, int]]] = defaultdict(list)
    for row_index, row in enumerate(sample_rows):
        selected_by_request[str(row["request_id"])].append(
            (int(row["candidate_ordinal"]), str(row["candidate_item_id"]), row_index)
        )
    selected_requests = set(ordered_requests)
    request_index = {request_id: ordinal for ordinal, request_id in enumerate(ordered_requests)}
    request_delta = np.empty(
        (len(ordered_requests), len(REQUEST_POSITIONS), len(STATES), 1024),
        dtype=np.float32,
    )
    candidate_delta = np.empty((len(sample_rows), len(STATES), 1024), dtype=np.float32)
    queries: dict[str, str] = {}
    seen_requests: set[str] = set()
    seen_candidates: set[int] = set()
    verified_shards = 0
    for full_shard, null_shard in zip(full_index["shards"], null_index["shards"]):
        if full_shard["path"] != null_shard["path"]:
            raise ValueError(f"{model_key} full/null shard partition differs")
        full_path = full_dir / "shards" / full_shard["path"]
        null_path = null_dir / "shards" / null_shard["path"]
        with np.load(full_path, allow_pickle=False) as full:
            shard_request_ids = [str(value) for value in full["request_ids"].tolist()]
        if not selected_requests.intersection(shard_request_ids):
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
            normalized_queries = [str(value) for value in full["normalized_queries"].tolist()]
            offsets = np.asarray(full["candidate_offsets"], dtype=np.int64)
            candidate_ids = [str(value) for value in full["candidate_ids"].tolist()]
            for local, request_id in enumerate(request_ids):
                if request_id not in selected_requests:
                    continue
                if request_id in seen_requests:
                    raise ValueError(f"{model_key} duplicate selected request")
                seen_requests.add(request_id)
                queries[request_id] = normalized_queries[local]
                request_delta[request_index[request_id]] = (
                    np.asarray(full["request_activations"][local], dtype=np.float32)
                    - np.asarray(null["request_activations"][local], dtype=np.float32)
                )
                width = int(offsets[local + 1] - offsets[local])
                for candidate_ordinal, candidate_item_id, row_index in selected_by_request[
                    request_id
                ]:
                    if not 0 <= candidate_ordinal < width:
                        raise ValueError(f"{model_key} selected candidate ordinal is invalid")
                    absolute = int(offsets[local]) + candidate_ordinal
                    if candidate_ids[absolute] != candidate_item_id:
                        raise ValueError(f"{model_key} selected candidate identity differs")
                    candidate_delta[row_index] = (
                        np.asarray(full["candidate_activations"][absolute], dtype=np.float32)
                        - np.asarray(null["candidate_activations"][absolute], dtype=np.float32)
                    )
                    seen_candidates.add(row_index)
    if seen_requests != selected_requests or len(seen_candidates) != len(sample_rows):
        raise ValueError(f"{model_key} frozen sample coverage is incomplete")
    if not np.isfinite(request_delta).all() or not np.isfinite(candidate_delta).all():
        raise ValueError(f"{model_key} sample delta contains non-finite values")
    return (
        {
            "query_end": request_delta[:, 0],
            "history_summary_end": request_delta[:, 1],
            "candidate_readout": candidate_delta,
        },
        queries,
        {
            "method_id": spec["method_id"],
            "checkpoint_id": full_meta["checkpoint_id"],
            "full_bundle": spec["full"].as_posix(),
            "null_bundle": spec["null"].as_posix(),
            "full_metadata_sha256": _sha256_file(full_dir / "metadata.json"),
            "null_metadata_sha256": _sha256_file(null_dir / "metadata.json"),
            "full_index_sha256": _sha256_file(full_dir / "index.json"),
            "null_index_sha256": _sha256_file(null_dir / "index.json"),
            "selected_shards_verified": verified_shards,
            "requests": len(ordered_requests),
            "candidate_rows": len(sample_rows),
            "qrels_read": False,
            "source_test_opened": False,
        },
    )


def _read_probe(
    weights: Any, position: str, task: str, state: int, control: str
) -> tuple[np.ndarray, np.ndarray]:
    key = f"{position}__{task}__state_{state}__{control}"
    scale = np.asarray(weights[f"{key}__scale"], dtype=np.float64)
    coefficient = np.asarray(weights[f"{key}__coefficient"], dtype=np.float64)
    if scale.shape != (1024,) or coefficient.ndim != 2 or coefficient.shape[1] != 1024:
        raise ValueError(f"probe tensor shape differs: {key}")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError(f"probe scale is invalid: {key}")
    return scale, coefficient


def _probe_basis(scale: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    """Return the discriminative probe row space in raw residual coordinates."""

    raw_coefficient = np.asarray(coefficient, dtype=np.float64) / np.asarray(
        scale, dtype=np.float64
    )[None, :]
    centered = raw_coefficient - raw_coefficient.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    threshold = max(float(singular_values[0]) * 1.0e-10, 1.0e-12)
    rank = int(np.sum(singular_values > threshold))
    basis = vh[:rank]
    if rank > 0 and np.max(np.abs(basis @ basis.T - np.eye(rank))) > 1.0e-9:
        raise ValueError("probe raw-coordinate basis is not orthonormal")
    return basis


def _subspace_similarity(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("probe subspaces have incompatible shapes")
    if len(left) == 0 or len(right) == 0:
        return {
            "left_rank": len(left),
            "right_rank": len(right),
            "mean_squared_canonical_cosine": None,
            "minimum_canonical_cosine": None,
            "maximum_canonical_cosine": None,
        }
    canonical = np.clip(np.linalg.svd(left @ right.T, compute_uv=False), 0.0, 1.0)
    return {
        "left_rank": len(left),
        "right_rank": len(right),
        "mean_squared_canonical_cosine": float(np.mean(canonical**2)),
        "minimum_canonical_cosine": float(np.min(canonical)),
        "maximum_canonical_cosine": float(np.max(canonical)),
    }


def _build_probe_transport_rows(weights: Any, model_key: str) -> list[dict[str, Any]]:
    rows = []
    for position in POSITIONS:
        for task in TASKS:
            for control in CONTROLS:
                bases = []
                for state in STATES:
                    scale, coefficient = _read_probe(
                        weights, position, task, state, control
                    )
                    bases.append(_probe_basis(scale, coefficient))
                for to_state in range(1, len(STATES)):
                    rows.append(
                        {
                            "model_key": model_key,
                            "position": position,
                            "task": task,
                            "label_control": control,
                            "from_hidden_state_index": to_state - 1,
                            "to_hidden_state_index": to_state,
                            **_subspace_similarity(
                                bases[to_state - 1], bases[to_state]
                            ),
                        }
                    )
    return rows


def _build_probe_transport_region_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup: defaultdict[tuple[str, str, str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        lookup[
            (
                str(row["model_key"]),
                str(row["position"]),
                str(row["task"]),
                str(row["label_control"]),
            )
        ].append(row)
    result = []
    for model_key in MODELS:
        for position in POSITIONS:
            for task in TASKS:
                for region, states in REGIONS.items():
                    summaries = {}
                    for control in CONTROLS:
                        selected = [
                            row
                            for row in lookup[(model_key, position, task, control)]
                            if int(row["to_hidden_state_index"]) in states
                            and row["mean_squared_canonical_cosine"] is not None
                        ]
                        summaries[control] = _mean(
                            [
                                float(row["mean_squared_canonical_cosine"])
                                for row in selected
                            ]
                        )
                    result.append(
                        {
                            "model_key": model_key,
                            "position": position,
                            "task": task,
                            "region": region,
                            "to_hidden_state_indices": list(states),
                            "real_mean_squared_canonical_cosine": summaries[
                                "real_labels"
                            ],
                            "random_mean_squared_canonical_cosine": summaries[
                                "random_labels"
                            ],
                            "real_minus_random_mean_squared_canonical_cosine": (
                                summaries["real_labels"]
                                - summaries["random_labels"]
                            ),
                        }
                    )
    return result


def _build_cross_position_transport_rows(
    model_key: str,
    deltas: Mapping[str, np.ndarray],
    queries: Mapping[str, str],
    sample_rows: Sequence[Mapping[str, Any]],
    ordered_requests: Sequence[str],
) -> list[dict[str, Any]]:
    request_index = {
        request_id: ordinal for ordinal, request_id in enumerate(ordered_requests)
    }
    sample_request_ids = [str(row["request_id"]) for row in sample_rows]
    history = deltas["history_summary_end"][
        [request_index[request_id] for request_id in sample_request_ids]
    ]
    candidate = deltas["candidate_readout"]
    if history.shape != candidate.shape:
        raise ValueError("history/candidate sampled delta shapes differ")
    normalized_queries = [queries[request_id] for request_id in sample_request_ids]
    rows = []
    hidden_size = history.shape[-1]
    for state in STATES:
        h = np.asarray(history[:, state], dtype=np.float64)
        c = np.asarray(candidate[:, state], dtype=np.float64)
        h_squared = np.einsum("ij,ij->i", h, h)
        c_squared = np.einsum("ij,ij->i", c, c)
        dot = np.einsum("ij,ij->i", h, c)
        valid = (h_squared > 1.0e-20) & (c_squared > 1.0e-20)
        cosine = np.full(len(h), np.nan, dtype=np.float64)
        cosine[valid] = np.clip(
            dot[valid] / np.sqrt(h_squared[valid] * c_squared[valid]), -1.0, 1.0
        )
        scale = np.full(len(h), np.nan, dtype=np.float64)
        scale[h_squared > 1.0e-20] = dot[h_squared > 1.0e-20] / h_squared[
            h_squared > 1.0e-20
        ]
        ratio = np.full(len(h), np.nan, dtype=np.float64)
        ratio[h_squared > 1.0e-20] = np.sqrt(
            c_squared[h_squared > 1.0e-20] / h_squared[h_squared > 1.0e-20]
        )
        residual = np.full(len(h), np.nan, dtype=np.float64)
        valid_history = h_squared > 1.0e-20
        residual[valid_history] = np.linalg.norm(
            c[valid_history] - scale[valid_history, None] * h[valid_history], axis=1
        ) / math.sqrt(hidden_size)
        request_metrics = _aggregate_transport_by_request(
            sample_request_ids,
            normalized_queries,
            valid,
            cosine,
            scale,
            ratio,
            residual,
        )
        for fold_name in ("all", "0", "1"):
            selected = [
                row
                for row in request_metrics
                if fold_name == "all"
                or _fold(str(row["normalized_query"])) == int(fold_name)
            ]
            valid_rows = [row for row in selected if row["cosine"] is not None]
            rows.append(
                {
                    "model_key": model_key,
                    "hidden_state_index": state,
                    "normalized_query_fold": fold_name,
                    "requests": len(selected),
                    "valid_delta_pairs": len(valid_rows),
                    "mean_cosine": _optional_mean(
                        [row["cosine"] for row in valid_rows]
                    ),
                    "mean_absolute_cosine": _optional_mean(
                        [abs(float(row["cosine"])) for row in valid_rows]
                    ),
                    "mean_cosine_squared": _optional_mean(
                        [float(row["cosine"]) ** 2 for row in valid_rows]
                    ),
                    "mean_candidate_over_history_rms": _optional_mean(
                        [row["candidate_over_history_rms"] for row in valid_rows]
                    ),
                    "mean_signed_candidate_projection_scale": _optional_mean(
                        [row["signed_candidate_projection_scale"] for row in valid_rows]
                    ),
                    "mean_orthogonal_residual_rms": _optional_mean(
                        [row["orthogonal_residual_rms"] for row in valid_rows]
                    ),
                }
            )
    return rows


def _aggregate_transport_by_request(
    request_ids: Sequence[str],
    normalized_queries: Sequence[str],
    valid: np.ndarray,
    cosine: np.ndarray,
    scale: np.ndarray,
    ratio: np.ndarray,
    residual: np.ndarray,
) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[int]] = defaultdict(list)
    query_by_request: dict[str, str] = {}
    for index, (request_id, query) in enumerate(zip(request_ids, normalized_queries, strict=True)):
        groups[request_id].append(index)
        previous = query_by_request.setdefault(request_id, query)
        if previous != query:
            raise ValueError("normalized query differs within transport request")
    result = []
    for request_id, indices in groups.items():
        selected = [index for index in indices if bool(valid[index])]
        result.append(
            {
                "request_id": request_id,
                "normalized_query": query_by_request[request_id],
                "cosine": _optional_mean([float(cosine[index]) for index in selected]),
                "signed_candidate_projection_scale": _optional_mean(
                    [float(scale[index]) for index in selected]
                ),
                "candidate_over_history_rms": _optional_mean(
                    [float(ratio[index]) for index in selected]
                ),
                "orthogonal_residual_rms": _optional_mean(
                    [float(residual[index]) for index in selected]
                ),
            }
        )
    return result


def _build_cross_position_transport_region_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["model_key"]),
            int(row["hidden_state_index"]),
            str(row["normalized_query_fold"]),
        ): row
        for row in rows
    }
    result = []
    for model_key in MODELS:
        for region, states in REGIONS.items():
            for fold_name in ("all", "0", "1"):
                selected = [lookup[(model_key, state, fold_name)] for state in states]
                result.append(
                    {
                        "model_key": model_key,
                        "region": region,
                        "hidden_state_indices": list(states),
                        "normalized_query_fold": fold_name,
                        "mean_cosine": _optional_mean(
                            [row["mean_cosine"] for row in selected]
                        ),
                        "mean_absolute_cosine": _optional_mean(
                            [row["mean_absolute_cosine"] for row in selected]
                        ),
                        "mean_cosine_squared": _optional_mean(
                            [row["mean_cosine_squared"] for row in selected]
                        ),
                        "mean_candidate_over_history_rms": _optional_mean(
                            [
                                row["mean_candidate_over_history_rms"]
                                for row in selected
                            ]
                        ),
                        "mean_signed_candidate_projection_scale": _optional_mean(
                            [
                                row["mean_signed_candidate_projection_scale"]
                                for row in selected
                            ]
                        ),
                        "mean_orthogonal_residual_rms": _optional_mean(
                            [row["mean_orthogonal_residual_rms"] for row in selected]
                        ),
                    }
                )
    return result


def _row_metrics(
    delta: np.ndarray, scale: np.ndarray, coefficient: np.ndarray
) -> dict[str, Any]:
    values = np.asarray(delta, dtype=np.float64) / scale[None, :]
    centered = coefficient - coefficient.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    threshold = max(float(singular_values[0]) * 1.0e-10, 1.0e-12)
    rank = int(np.sum(singular_values > threshold))
    basis = vh[:rank]
    if rank > 0 and np.max(np.abs(basis @ basis.T - np.eye(rank))) > 1.0e-9:
        raise ValueError("probe row-space basis is not orthonormal")
    total_squared = np.einsum("ij,ij->i", values, values)
    projected = values @ basis.T if rank > 0 else np.zeros((len(values), 0))
    projected_squared = np.einsum("ij,ij->i", projected, projected)
    logit_delta = values @ centered.T
    nonzero = total_squared > 1.0e-20
    fraction = np.full(len(values), np.nan, dtype=np.float64)
    fraction[nonzero] = (
        np.clip(projected_squared[nonzero] / total_squared[nonzero], 0.0, 1.0)
        if rank > 0
        else 0.0
    )
    return {
        "subspace_rank": rank,
        "nonzero": nonzero,
        "energy_fraction": fraction,
        "standardized_delta_rms": np.sqrt(total_squared / values.shape[1]),
        "discriminative_logit_delta_rms": np.sqrt(np.mean(logit_delta**2, axis=1)),
    }


def _aggregate_by_request(
    request_ids: Sequence[str],
    normalized_queries: Sequence[str],
    metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[int]] = defaultdict(list)
    query_by_request: dict[str, str] = {}
    for index, (request_id, query) in enumerate(zip(request_ids, normalized_queries, strict=True)):
        groups[request_id].append(index)
        previous = query_by_request.setdefault(request_id, query)
        if previous != query:
            raise ValueError("normalized query differs within sampled request")
    result = []
    for request_id, indices in groups.items():
        fraction = np.asarray(metrics["energy_fraction"])[indices]
        result.append(
            {
                "request_id": request_id,
                "normalized_query": query_by_request[request_id],
                "nonzero": bool(np.asarray(metrics["nonzero"])[indices].any()),
                "energy_fraction": float(np.nanmean(fraction))
                if np.isfinite(fraction).any()
                else None,
                "standardized_delta_rms": float(
                    np.mean(np.asarray(metrics["standardized_delta_rms"])[indices])
                ),
                "discriminative_logit_delta_rms": float(
                    np.mean(
                        np.asarray(metrics["discriminative_logit_delta_rms"])[indices]
                    )
                ),
            }
        )
    return result


def _summarize(
    model_key: str,
    position: str,
    task: str,
    state: int,
    control: str,
    fold_name: str,
    rows: Sequence[Mapping[str, Any]],
    rank: int,
    hidden_size: int,
) -> dict[str, Any]:
    fractions = [float(row["energy_fraction"]) for row in rows if row["energy_fraction"] is not None]
    nonzero = [row for row in rows if row["nonzero"]]
    isotropic = rank / hidden_size
    return {
        "model_key": model_key,
        "position": position,
        "task": task,
        "hidden_state_index": state,
        "label_control": control,
        "normalized_query_fold": fold_name,
        "requests": len(rows),
        "nonzero_delta_requests": len(nonzero),
        "subspace_rank": rank,
        "isotropic_rank_over_hidden_baseline": isotropic,
        "mean_energy_fraction": _mean(fractions) if fractions else None,
        "median_energy_fraction": float(median(fractions)) if fractions else None,
        "mean_fraction_over_isotropic_baseline": (
            _mean(fractions) / isotropic if fractions and isotropic > 0.0 else None
        ),
        "mean_standardized_delta_rms": _mean(
            [float(row["standardized_delta_rms"]) for row in rows]
        ),
        "mean_discriminative_logit_delta_rms": _mean(
            [float(row["discriminative_logit_delta_rms"]) for row in rows]
        ),
    }


def _build_excess_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            row["model_key"],
            row["position"],
            row["task"],
            row["hidden_state_index"],
            row["normalized_query_fold"],
            row["label_control"],
        ): row
        for row in rows
    }
    result = []
    for model_key in MODELS:
        for position in POSITIONS:
            for task in TASKS:
                for state in STATES:
                    for fold_name in ("all", "0", "1"):
                        base_key = (model_key, position, task, state, fold_name)
                        real = lookup[(*base_key, "real_labels")]
                        random = lookup[(*base_key, "random_labels")]
                        real_fraction = real["mean_energy_fraction"]
                        random_fraction = random["mean_energy_fraction"]
                        result.append(
                            {
                                "model_key": model_key,
                                "position": position,
                                "task": task,
                                "hidden_state_index": state,
                                "normalized_query_fold": fold_name,
                                "requests": real["requests"],
                                "nonzero_delta_requests": real["nonzero_delta_requests"],
                                "real_mean_energy_fraction": real_fraction,
                                "random_mean_energy_fraction": random_fraction,
                                "real_minus_random_energy_fraction": (
                                    float(real_fraction) - float(random_fraction)
                                    if real_fraction is not None and random_fraction is not None
                                    else None
                                ),
                                "real_minus_random_logit_delta_rms": (
                                    float(real["mean_discriminative_logit_delta_rms"])
                                    - float(random["mean_discriminative_logit_delta_rms"])
                                ),
                            }
                        )
    return result


def _build_region_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            row["model_key"],
            row["position"],
            row["task"],
            row["hidden_state_index"],
            row["normalized_query_fold"],
        ): row
        for row in rows
    }
    result = []
    for model_key in MODELS:
        for position in POSITIONS:
            for task in TASKS:
                for region, states in REGIONS.items():
                    for fold_name in ("all", "0", "1"):
                        selected = [
                            lookup[(model_key, position, task, state, fold_name)]
                            for state in states
                        ]
                        values = [
                            row["real_minus_random_energy_fraction"]
                            for row in selected
                            if row["real_minus_random_energy_fraction"] is not None
                        ]
                        result.append(
                            {
                                "model_key": model_key,
                                "position": position,
                                "task": task,
                                "region": region,
                                "hidden_state_indices": list(states),
                                "normalized_query_fold": fold_name,
                                "mean_real_minus_random_energy_fraction": (
                                    _mean([float(value) for value in values])
                                    if values
                                    else None
                                ),
                            }
                        )
    return result


def _audit_sample_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 512:
        raise ValueError("frozen sample row count differs")
    seen: set[tuple[str, int]] = set()
    previous_selection = ""
    for row in rows:
        key = (str(row["request_id"]), int(row["candidate_ordinal"]))
        if key in seen:
            raise ValueError("frozen sample contains duplicate candidate row")
        seen.add(key)
        selection = str(row["selection_sha256"])
        if len(selection) != 64 or selection < previous_selection:
            raise ValueError("frozen sample selection order differs")
        previous_selection = selection


def _fold(normalized_query: str) -> int:
    if not normalized_query:
        raise ValueError("normalized query is empty")
    return int(hashlib.sha256(normalized_query.encode("utf-8")).hexdigest(), 16) % 2


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(values) / len(values))


def _optional_mean(values: Sequence[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return _mean(finite) if finite else None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"expected JSON object row: {path}")
                yield value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
