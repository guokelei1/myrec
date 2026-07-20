"""Integrity-gated aggregation for fixed-grid SwiGLU formation observations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.mechanism.attention_pattern_analysis import BLOCKS, MODELS
from myrec.mechanism.mlp_feature_formation import MLP_FEATURE_STAGES
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


STAGE_METRICS = (
    "full_rms",
    "null_rms",
    "delta_rms",
    "full_null_cosine",
    "sign_flip_fraction",
    "full_hoyer_sparsity",
    "null_hoyer_sparsity",
)
DECOMPOSITION_TERMS = (
    "gate_change_times_null_up",
    "null_gate_times_up_change",
    "gate_up_interaction",
)
DECOMPOSITION_TERM_METRICS = ("rms", "cosine_to_product_delta")
DECOMPOSITION_SCALARS = (
    "actual_product_delta_rms",
    "algebraic_product_delta_rms",
    "maximum_recomposition_abs_error",
    "maximum_actual_product_quantization_abs_error",
)


def evaluate_mlp_feature_bundles(
    bundles: Mapping[str, Mapping[int, str | Path]],
    *,
    expected_rows: int = 512,
) -> dict[str, Any]:
    if set(bundles) != set(MODELS) or any(
        set(model_bundles) != set(BLOCKS) for model_bundles in bundles.values()
    ):
        raise ValueError("MLP formation fixed-grid bundle coverage differs")
    cells = []
    sources = {}
    implementation_digests = set()
    reference_identities = None
    maximum_identity = 0.0
    maximum_product_ratio = 0.0
    maximum_delta_error = 0.0
    maximum_quantization_error = 0.0
    for model_id in MODELS:
        sources[model_id] = {}
        expected_paths = ("prompt",) if model_id == MODELS[0] else ("yes", "no")
        for block in BLOCKS:
            root = Path(bundles[model_id][block])
            metadata_path = root / "metadata.json"
            rows_path = root / "rows.jsonl"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            digest = metadata.get("implementation_identity", {}).get("digest")
            if (
                metadata.get("status") != "completed"
                or metadata.get("analysis_stage")
                != "transformer_deep_dive_d4_mlp_feature_formation_extension"
                or metadata.get("method_id") != model_id
                or metadata.get("block_zero_based") != block
                or metadata.get("result_eligible") is not True
                or metadata.get("confirmatory_family_member") is not False
                or metadata.get("layer_or_group_selection_authorized") is not False
                or metadata.get("qrels_read") is not False
                or metadata.get("source_test_opened") is not False
                or metadata.get("complete_finite_observation_coverage") is not True
                or metadata.get("observation_rows") != expected_rows
                or metadata.get("rows_path") != str(rows_path)
                or metadata.get("rows_sha256") != sha256_file(rows_path)
                or metadata.get("run_contract", {}).get("target_rows") != expected_rows
                or metadata.get("run_contract", {}).get("implementation_digest") != digest
                or not isinstance(digest, str)
                or float(metadata.get("maximum_score_identity_delta", math.inf)) > 1.0e-5
                or float(
                    metadata.get(
                        "maximum_product_recomposition_low_precision_ratio",
                        math.inf,
                    )
                )
                > 1.0
            ):
                raise ValueError("MLP formation completed bundle boundary differs")
            mechanical_values = (
                float(metadata["maximum_score_identity_delta"]),
                float(metadata["maximum_product_recomposition_low_precision_ratio"]),
                float(metadata["maximum_delta_recomposition_abs_error"]),
                float(metadata["maximum_actual_product_quantization_abs_error"]),
            )
            if not all(math.isfinite(value) and value >= 0 for value in mechanical_values):
                raise ValueError("MLP formation completed mechanical metric differs")
            implementation_digests.add(digest)
            rows = list(iter_jsonl(rows_path))
            identities = _validate_rows(
                rows,
                model_id=model_id,
                block=block,
                expected_rows=expected_rows,
                expected_paths=expected_paths,
            )
            if reference_identities is None:
                reference_identities = identities
            elif identities != reference_identities:
                raise ValueError("MLP formation row identity/order differs across grid")
            cells.append(
                {
                    "method_id": model_id,
                    "block_zero_based": block,
                    "paths": {
                        path_name: _aggregate_path(rows, path_name)
                        for path_name in expected_paths
                    },
                }
            )
            maximum_identity = max(
                maximum_identity,
                float(metadata["maximum_score_identity_delta"]),
            )
            maximum_product_ratio = max(
                maximum_product_ratio,
                float(metadata["maximum_product_recomposition_low_precision_ratio"]),
            )
            maximum_delta_error = max(
                maximum_delta_error,
                float(metadata["maximum_delta_recomposition_abs_error"]),
            )
            maximum_quantization_error = max(
                maximum_quantization_error,
                float(metadata["maximum_actual_product_quantization_abs_error"]),
            )
            sources[model_id][str(block)] = {
                "root": str(root),
                "metadata_sha256": sha256_file(metadata_path),
                "rows_sha256": sha256_file(rows_path),
            }
    if len(implementation_digests) != 1:
        raise ValueError("MLP formation implementation digest differs across grid")
    return {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d4_mlp_feature_formation_extension",
        "status": "completed",
        "descriptive_only": True,
        "confirmatory_family_member": False,
        "qrels_read": False,
        "source_test_opened": False,
        "models": list(MODELS),
        "blocks": list(BLOCKS),
        "rows_per_cell": expected_rows,
        "groups": 16,
        "feature_stages": list(MLP_FEATURE_STAGES),
        "implementation_digest": next(iter(implementation_digests)),
        "sources": sources,
        "cells": cells,
        "maximum_score_identity_delta": maximum_identity,
        "maximum_product_recomposition_low_precision_ratio": maximum_product_ratio,
        "maximum_delta_recomposition_abs_error": maximum_delta_error,
        "maximum_actual_product_quantization_abs_error": maximum_quantization_error,
        "interpretation_boundary": (
            "Complete fixed-grid descriptive formation geometry for gate preactivation, "
            "SiLU gate, up projection and their product. Exact algebraic terms are not "
            "causal contributions; no layer, group, neuron or architecture is selected."
        ),
    }


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    block: int,
    expected_rows: int,
    expected_paths: Sequence[str],
) -> list[tuple[str, str, int, str]]:
    if len(rows) != expected_rows:
        raise ValueError("MLP formation row count differs")
    identities = []
    for ordinal, row in enumerate(rows):
        identity = (
            str(row.get("request_id") or ""),
            str(row.get("candidate_item_id") or ""),
            int(row.get("candidate_ordinal", -1)),
            str(row.get("selection_sha256") or ""),
        )
        if (
            row.get("ordinal") != ordinal
            or not identity[0]
            or not identity[1]
            or identity[2] < 0
            or not identity[3]
            or set(row.get("paths", {})) != set(expected_paths)
            or not math.isfinite(
                float(row.get("maximum_score_identity_delta", math.inf))
            )
            or float(row.get("maximum_score_identity_delta", math.inf)) > 1.0e-5
        ):
            raise ValueError("MLP formation row identity/schema differs")
        for path_name in expected_paths:
            summary = row["paths"][path_name].get("summary", {})
            positions = summary.get("positions")
            expected_position_count = 3 if model_id == MODELS[0] else 4
            if (
                summary.get("groups") != 16
                or not isinstance(positions, list)
                or len(positions) != expected_position_count
                or not math.isfinite(
                    float(
                        summary.get(
                            "maximum_product_identity_low_precision_ratio", math.inf
                        )
                    )
                )
                or float(summary.get("maximum_product_identity_low_precision_ratio", math.inf)) > 1.0
            ):
                raise ValueError("MLP formation row summary boundary differs")
            for position_index, position in enumerate(positions):
                groups = position.get("groups")
                if position.get("position_index") != position_index or not isinstance(groups, list) or len(groups) != 16:
                    raise ValueError("MLP formation position/group coverage differs")
                for group_id, group in enumerate(groups):
                    _validate_group(group, group_id)
        identities.append(identity)
    if len(set(identities)) != expected_rows:
        raise ValueError("MLP formation row identity is duplicated")
    return identities


def _validate_group(group: Mapping[str, Any], group_id: int) -> None:
    if (
        group.get("group_id") != group_id
        or int(group.get("dimensions", 0)) <= 0
        or set(group.get("stages", {})) != set(MLP_FEATURE_STAGES)
    ):
        raise ValueError("MLP formation group schema differs")
    for stage in MLP_FEATURE_STAGES:
        values = group["stages"][stage]
        if set(values) != set(STAGE_METRICS):
            raise ValueError("MLP formation stage metrics differ")
        _finite_scalars(values.values())
    decomposition = group.get("product_delta_decomposition", {})
    if set(decomposition) != set(DECOMPOSITION_TERMS) | set(DECOMPOSITION_SCALARS):
        raise ValueError("MLP formation decomposition metrics differ")
    for term in DECOMPOSITION_TERMS:
        if set(decomposition[term]) != set(DECOMPOSITION_TERM_METRICS):
            raise ValueError("MLP formation decomposition term differs")
        _finite_scalars(decomposition[term].values())
    _finite_scalars(decomposition[name] for name in DECOMPOSITION_SCALARS)


def _aggregate_path(rows: Sequence[Mapping[str, Any]], path_name: str) -> dict[str, Any]:
    first_positions = rows[0]["paths"][path_name]["summary"]["positions"]
    labels = ("query_end", "history_summary_end") + tuple(
        f"native_readout_{index}" for index in range(len(first_positions) - 2)
    )
    result = {}
    for position_index, label in enumerate(labels):
        groups = []
        for group_id in range(16):
            stage_result = {}
            for stage in MLP_FEATURE_STAGES:
                stage_result[stage] = {
                    metric: _summary(
                        [
                            row["paths"][path_name]["summary"]["positions"][position_index]["groups"][group_id]["stages"][stage][metric]
                            for row in rows
                        ]
                    )
                    for metric in STAGE_METRICS
                }
            decomposition_result = {
                term: {
                    metric: _summary(
                        [
                            row["paths"][path_name]["summary"]["positions"][position_index]["groups"][group_id]["product_delta_decomposition"][term][metric]
                            for row in rows
                        ]
                    )
                    for metric in DECOMPOSITION_TERM_METRICS
                }
                for term in DECOMPOSITION_TERMS
            }
            decomposition_result.update(
                {
                    metric: _summary(
                        [
                            row["paths"][path_name]["summary"]["positions"][position_index]["groups"][group_id]["product_delta_decomposition"][metric]
                            for row in rows
                        ]
                    )
                    for metric in DECOMPOSITION_SCALARS
                }
            )
            groups.append(
                {
                    "group_id": group_id,
                    "stages": stage_result,
                    "product_delta_decomposition": decomposition_result,
                }
            )
        result[label] = {"groups": groups}
    return {"positions": result}


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("MLP formation aggregate values differ")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": float(array.std()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "rows": int(array.size),
    }


def _finite_scalars(values) -> None:
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("MLP formation scalar metric is non-finite")
