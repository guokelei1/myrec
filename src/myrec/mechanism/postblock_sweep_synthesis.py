"""Registered cross-model D2 all-layer and adjacent-transition synthesis."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.mechanism.deep_dive_native_evaluator import (
    benjamini_hochberg,
    cluster_mean_inference,
)
from myrec.mechanism.postblock_sweep_evaluator import ENDPOINTS, POSTBLOCK_BLOCKS
from myrec.mechanism.postblock_sweep_runtime import SUPPORTED_METHODS
from myrec.utils.hashing import sha256_file


ALL_LAYER_FAMILY_SIZE = 30
ADJACENT_FAMILY_SIZE = 28


def synthesize_postblock_sweeps(
    model_inputs: Mapping[str, Mapping[str, str | Path] | None],
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Combine frozen folds; gate-stopped models retain p=1 planned cells."""

    if set(model_inputs) != set(SUPPORTED_METHODS):
        raise ValueError("D2 synthesis requires explicit Q2 and Q3 entries")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"D2 synthesis output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = {}
    source_identities = {}
    for method_id in SUPPORTED_METHODS:
        inputs = model_inputs[method_id]
        if inputs is None:
            loaded[method_id] = None
            source_identities[method_id] = {
                "status": "gate_stopped_or_mechanical_missing",
                "planned_cells_use_p": 1.0,
            }
            continue
        if set(inputs) != {"fold0_selection", "fold1_confirmation"}:
            raise ValueError("each present D2 model needs selection and confirmation")
        selection_path = Path(inputs["fold0_selection"])
        confirmation_path = Path(inputs["fold1_confirmation"])
        selection = _read_json(selection_path)
        confirmation = _read_json(confirmation_path)
        if (
            selection.get("analysis_type")
            != "transformer_deep_dive_d2_postblock_fold0_selection"
            or confirmation.get("analysis_type")
            != "transformer_deep_dive_d2_postblock_fold1_confirmation"
            or selection.get("method_id") != method_id
            or confirmation.get("method_id") != method_id
            or confirmation.get("selection", {}).get("sha256")
            != sha256_file(selection_path)
        ):
            raise ValueError(f"D2 fold result binding differs for {method_id}")
        implementation_digest = str(selection.get("implementation_digest") or "")
        if not implementation_digest or confirmation.get(
            "implementation_digest"
        ) != implementation_digest:
            raise ValueError(f"D2 fold implementation binding differs for {method_id}")
        fold0_npz = Path(selection["per_request_effects_path"])
        fold1_npz = Path(confirmation["per_request_effects_path"])
        if (
            selection.get("per_request_effects_sha256") != sha256_file(fold0_npz)
            or confirmation.get("per_request_effects_sha256") != sha256_file(fold1_npz)
        ):
            raise ValueError(f"D2 per-request array hash differs for {method_id}")
        loaded[method_id] = {
            "selection": selection,
            "confirmation": confirmation,
            "fold0": np.load(fold0_npz, allow_pickle=False),
            "fold1": np.load(fold1_npz, allow_pickle=False),
        }
        source_identities[method_id] = {
            "fold0_selection_path": str(selection_path),
            "fold0_selection_sha256": sha256_file(selection_path),
            "fold1_confirmation_path": str(confirmation_path),
            "fold1_confirmation_sha256": sha256_file(confirmation_path),
            "implementation_digest": implementation_digest,
        }

    all_layer = {endpoint: [] for endpoint in ENDPOINTS}
    adjacent = {endpoint: [] for endpoint in ENDPOINTS}
    for method_id in SUPPORTED_METHODS:
        model = loaded[method_id]
        for endpoint in ENDPOINTS:
            if model is None:
                all_layer[endpoint].extend(
                    _missing_row(method_id, block=block) for block in POSTBLOCK_BLOCKS
                )
                adjacent[endpoint].extend(
                    _missing_row(method_id, block=block) for block in POSTBLOCK_BLOCKS[1:]
                )
                continue
            arrays, clusters, strict = _merge_model_folds(model)
            block_values = {
                block: arrays[f"block_{block}__{endpoint}"] for block in POSTBLOCK_BLOCKS
            }
            for block in POSTBLOCK_BLOCKS:
                mask = strict & np.isfinite(block_values[block])
                all_layer[endpoint].append(
                    {
                        "method_id": method_id,
                        "block_zero_based": block,
                        **cluster_mean_inference(block_values[block][mask], clusters[mask]),
                        "missing_cell": False,
                    }
                )
            for block in POSTBLOCK_BLOCKS[1:]:
                values = block_values[block] - block_values[block - 1]
                mask = strict & np.isfinite(values)
                adjacent[endpoint].append(
                    {
                        "method_id": method_id,
                        "block_zero_based": block,
                        "contrast": f"block_{block}_minus_block_{block - 1}",
                        **cluster_mean_inference(values[mask], clusters[mask]),
                        "missing_cell": False,
                    }
                )
    for endpoint in ENDPOINTS:
        if len(all_layer[endpoint]) != ALL_LAYER_FAMILY_SIZE:
            raise AssertionError("D2 all-layer family size drifted")
        if len(adjacent[endpoint]) != ADJACENT_FAMILY_SIZE:
            raise AssertionError("D2 adjacent family size drifted")
        for rows in (all_layer[endpoint], adjacent[endpoint]):
            q_values = benjamini_hochberg([float(row["two_sided_p"]) for row in rows])
            for row, q_value in zip(rows, q_values):
                row["bh_q"] = float(q_value)

    localization = {}
    margin_adjacent = adjacent["target_margin"]
    for method_id in SUPPORTED_METHODS:
        model = loaded[method_id]
        if model is None:
            localization[method_id] = {
                "status": "not_run_due_to_gate_or_mechanical_stop",
                "resolved": False,
            }
            continue
        selection = model["selection"]
        confirmation = model["confirmation"]
        selected = selection.get("selected_block")
        if selected is None:
            localization[method_id] = {
                "status": "fold0_no_negative_transition",
                "resolved": False,
            }
            continue
        selected = int(selected)
        family_row = next(
            row for row in margin_adjacent
            if row["method_id"] == method_id and row["block_zero_based"] == selected
        )
        fold_gate = confirmation["fixed_transition_confirmation"]
        passed = (
            fold_gate.get("same_negative_sign_both_folds") is True
            and float(family_row["mean"]) < 0
            and float(family_row["bh_q"]) < 0.05
        )
        localization[method_id] = {
            "status": "resolved" if passed else "unresolved",
            "resolved": passed,
            "selected_block": selected,
            "selected_predecessor": selected - 1,
            "both_folds_negative": fold_gate.get("same_negative_sign_both_folds"),
            "combined_mean": family_row["mean"],
            "combined_two_sided_p": family_row["two_sided_p"],
            "combined_bh_q": family_row["bh_q"],
        }
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d2_postblock_synthesis",
        "analysis_run_id": analysis_run_id,
        "primary_surface": "target_nonrepeat_no_candidate_overlap",
        "endpoints": list(ENDPOINTS),
        "multiple_testing": {
            "method": "benjamini_hochberg",
            "all_layer_family_size_per_endpoint": ALL_LAYER_FAMILY_SIZE,
            "adjacent_family_size_per_endpoint": ADJACENT_FAMILY_SIZE,
            "missing_gate_stopped_or_mechanical_cell_p": 1.0,
            "planned_family_size_never_shrinks": True,
        },
        "all_layer": all_layer,
        "adjacent_transition": adjacent,
        "localization": localization,
        "sources": source_identities,
        "qrels_read": False,
        "source_evaluators_read_fold_specific_qrels": True,
        "command": list(command or []),
        "status": "completed",
    }
    metrics_path = output_dir / "metrics.json"
    _write_json_atomic(metrics_path, metrics)
    _append_jsonl(
        Path(dev_eval_log_path),
        {
            "analysis_type": metrics["analysis_type"],
            "run_id": analysis_run_id,
            "method_ids": list(SUPPORTED_METHODS),
            "split": "dev_fold0_plus_fold1",
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
        },
    )
    return metrics


def _merge_model_folds(model):
    fold0 = model["fold0"]
    fold1 = model["fold1"]
    keys = [f"block_{block}__{endpoint}" for block in POSTBLOCK_BLOCKS for endpoint in ENDPOINTS]
    arrays = {key: np.concatenate([fold0[key], fold1[key]]) for key in keys}
    clusters = np.concatenate([fold0["normalized_queries"], fold1["normalized_queries"]])
    strict = np.concatenate([fold0["strict_mask"], fold1["strict_mask"]]).astype(bool)
    if len(set(map(str, np.concatenate([fold0["request_ids"], fold1["request_ids"]])))) != 8000:
        raise ValueError("merged D2 folds do not cover 8000 unique requests")
    return arrays, clusters, strict


def _missing_row(method_id: str, *, block: int) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "block_zero_based": block,
        "requests": 0,
        "normalized_query_clusters": 0,
        "mean": None,
        "ci95": [None, None],
        "two_sided_p": 1.0,
        "bootstrap_samples": 0,
        "missing_cell": True,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
