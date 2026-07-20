"""Train-only synthesis for exact Q2/Q3 step-501 optimizer replays."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.mechanism.attention_edge_runtime import (
    DEEP_DIVE_MANIFEST_PATH,
    _load_manifest,
)
from myrec.mechanism.gradient_diagnostic import CONTROLS, SURFACES
from myrec.mechanism.q2_optimizer_replay_runtime import OBJECTIVES, OBJECTIVE_PAIRS
from myrec.mechanism.q3_optimizer_replay_runtime import MODES
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


TASKS = 36


def evaluate_optimizer_replays(
    q2_bundle: str | Path,
    q3_bundle: str | Path,
    output_dir: str | Path,
    analysis_run_id: str,
    *,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate six frozen replay blocks per control/surface cell."""

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"optimizer replay output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(DEEP_DIVE_MANIFEST_PATH)
    replay_bindings = manifest["optimizer_replay"]
    q2_root, q2_metadata, q2_rows = _audit_q2(
        q2_bundle, replay_bindings["q2_step500"], manifest["_sha256"]
    )
    q3_root, q3_metadata, q3_rows = _audit_q3(
        q3_bundle, replay_bindings["q3_step500"], manifest["_sha256"]
    )
    implementation_digests = {
        "q2": _implementation_digest(q2_metadata),
        "q3": _implementation_digest(q3_metadata),
    }
    integrity = {
        "schema_version": 1,
        "analysis_type": "d7_optimizer_replay_integrity",
        "analysis_run_id": analysis_run_id,
        "status": "passed",
        "qrels_access": "train_only",
        "checks": {
            "both_step500_bindings_passed": True,
            "all_36_replay_blocks_complete_per_model": True,
            "real_adamw_steps_executed_then_exact_theta_restore": True,
            "scheduler_step501_executed_and_audited": True,
            "Q2_combined_raw_gradient_identity_passed": True,
            "Q3_inactive_factor_exactly_unchanged": True,
            "both_step500_bindings_match_frozen_manifest_fields": True,
            "model_specific_implementation_digests_nonempty": True,
        },
        "implementation_digests": implementation_digests,
        "bundles": {
            "q2": {
                "path": str(q2_root),
                "metadata_sha256": sha256_file(q2_root / "metadata.json"),
                "replays_sha256": sha256_file(q2_root / "replays.jsonl"),
            },
            "q3": {
                "path": str(q3_root),
                "metadata_sha256": sha256_file(q3_root / "metadata.json"),
                "replays_sha256": sha256_file(q3_root / "replays.jsonl"),
            },
        },
    }
    integrity_path = output_dir / "integrity.json"
    _write_json(integrity_path, integrity)
    q2_results = {}
    q3_results = {}
    for control in CONTROLS:
        q2_results[control] = {}
        q3_results[control] = {}
        for surface in SURFACES:
            q2_cell = [row for row in q2_rows if row["control"] == control and row["surface"] == surface]
            q3_cell = [row for row in q3_rows if row["control"] == control and row["surface"] == surface]
            if len(q2_cell) != 6 or len(q3_cell) != 6:
                raise ValueError("optimizer replay cell does not contain six blocks")
            q2_results[control][surface] = _q2_cell(q2_cell)
            q3_results[control][surface] = _q3_cell(q3_cell)
    metrics = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d7_exact_optimizer_replay",
        "analysis_run_id": analysis_run_id,
        "step": 501,
        "replay_blocks_per_control_surface": 6,
        "requests_per_block": 16,
        "controls": list(CONTROLS),
        "implementation_digests": implementation_digests,
        "surfaces": list(SURFACES),
        "q2": {
            "method_id": q2_metadata["method_id"],
            "checkpoint_id": q2_metadata["step500_binding"]["checkpoint_id"],
            "objectives": list(OBJECTIVES),
            "results": q2_results,
        },
        "q3": {
            "method_id": q3_metadata["method_id"],
            "checkpoint_id": q3_metadata["step500_binding"]["checkpoint_id"],
            "coordinate_modes": list(MODES),
            "results": q3_results,
        },
        "interpretation_boundary": "one-step train-only update geometry; not a dev/test performance result",
        "dev_confirmation_test_qrels_read": False,
        "source_test_opened": False,
        "integrity_path": str(integrity_path),
        "integrity_sha256": sha256_file(integrity_path),
        "command": list(command or []),
        "status": "completed",
    }
    metrics_path = output_dir / "metrics.json"
    _write_json(metrics_path, metrics)
    return metrics


def _q2_cell(rows):
    objective_results = {}
    stages = (
        "raw_gradient", "clipped_gradient", "adam_preconditioned_direction",
        "moment_delta", "weight_decay_delta", "total_delta",
    )
    for objective in OBJECTIVES:
        objective_results[objective] = {}
        for stage in stages:
            summaries = [row["objectives"][objective][stage] for row in rows]
            families = sorted(summaries[0]["family_share"])
            objective_results[objective][stage] = {
                "norm": _summary([value["norm"] for value in summaries]),
                "family_share": {
                    family: _summary([value["family_share"][family] for value in summaries])
                    for family in families
                },
            }
        objective_results[objective]["actual_step_vs_algebra_relative_l2_error"] = _summary(
            [row["objectives"][objective]["actual_step_vs_algebra_identity"]["relative_l2_error"] for row in rows]
        )
    stage_cosines = {}
    for stage in stages:
        stage_cosines[stage] = {
            f"{left}_vs_{right}": _summary(
                [row["stage_objective_cosines"][stage][f"{left}_vs_{right}"] for row in rows]
            )
            for left, right in OBJECTIVE_PAIRS
        }
    return {
        "blocks": len(rows),
        "objectives": objective_results,
        "stage_objective_cosines": stage_cosines,
        "combined_raw_gradient_relative_l2_error": _summary(
            [row["combined_raw_gradient_identity"]["relative_l2_error"] for row in rows]
        ),
    }


def _q3_cell(rows):
    modes = {}
    for mode in MODES:
        modes[mode] = {
            stage: {
                "norm": _summary([row["modes"][mode][stage]["norm"] for row in rows]),
                "family_share": {
                    family: _summary(
                        [row["modes"][mode][stage]["family_share"].get(family, 0.0) for row in rows]
                    )
                    for family in sorted(
                        set().union(
                            *(row["modes"][mode][stage]["family_share"] for row in rows)
                        )
                    )
                },
            }
            for stage in (
                "raw_gradient", "clipped_gradient", "adam_preconditioned_direction",
                "moment_delta", "weight_decay_delta", "total_delta",
            )
        }
        modes[mode]["actual_step_vs_algebra_relative_l2_error"] = _summary(
            [row["modes"][mode]["actual_step_vs_algebra_identity"]["relative_l2_error"] for row in rows]
        )
    path_results = {}
    for block in range(28):
        path_results[str(block)] = {}
        for projection in ("q", "v"):
            paths = [
                next(
                    value for value in row["lora_paths"]
                    if value["block_zero_based"] == block and value["projection"] == projection
                )
                for row in rows
            ]
            path_results[str(block)][projection] = {
                metric: _summary([path[metric] for path in paths])
                for metric in (
                    "a_only_replay_function_norm",
                    "b_only_replay_function_norm",
                    "joint_function_norm",
                    "joint_a_component_norm",
                    "joint_b_component_norm",
                    "joint_second_order_interaction_norm",
                    "function_recomposition_max_abs_error",
                    "step500_effective_weight_norm",
                    "post_step501_effective_weight_norm",
                    "step501_effective_delta_norm",
                )
            }
    return {"blocks": len(rows), "modes": modes, "lora_paths": path_results}


def _audit_q2(root, expected_binding, manifest_sha256):
    root = Path(root)
    metadata = _metadata(
        root,
        "transformer_deep_dive_d7_q2_step501_optimizer_replay",
        "q2_recranker_generalqwen",
        expected_binding,
        manifest_sha256,
    )
    if metadata.get("completed_replay_blocks") != TASKS or metadata.get("optimizer_steps_performed_then_exactly_restored") != TASKS * len(OBJECTIVES):
        raise ValueError("Q2 optimizer replay step coverage differs")
    rows = _rows(root, metadata)
    for row in rows:
        if row.get("combined_raw_gradient_identity", {}).get("passed") is not True:
            raise ValueError("Q2 combined gradient identity failed")
    return root, metadata, rows


def _audit_q3(root, expected_binding, manifest_sha256):
    root = Path(root)
    metadata = _metadata(
        root,
        "transformer_deep_dive_d7_q3_step501_lora_replay",
        "q3_tallrec_generalqwen",
        expected_binding,
        manifest_sha256,
    )
    if metadata.get("completed_replay_blocks") != TASKS or metadata.get("optimizer_steps_performed_then_exactly_restored") != TASKS * len(MODES):
        raise ValueError("Q3 optimizer replay step coverage differs")
    rows = _rows(root, metadata)
    for row in rows:
        if row.get("inactive_factor_none_verified") is not True or any(
            float(row["modes"][mode]["maximum_inactive_parameter_delta"]) != 0.0
            for mode in MODES
        ):
            raise ValueError("Q3 inactive factor contract failed")
    return root, metadata, rows


def _metadata(root, stage, method_id, expected_binding, manifest_sha256):
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    expected = {
        "analysis_stage": stage,
        "status": "completed",
        "result_eligible": True,
        "dev_confirmation_test_qrels_read": False,
        "source_test_opened": False,
        "method_id": method_id,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"optimizer replay metadata differs: {key}")
    _audit_step500_binding(
        metadata.get("step500_binding", {}),
        expected_binding,
        method_id=method_id,
        manifest_sha256=manifest_sha256,
    )
    _implementation_digest(metadata)
    return metadata


def _audit_step500_binding(observed, expected, *, method_id, manifest_sha256):
    exact_fields = (
        "checkpoint_id",
        "optimizer_steps",
        "scheduler_last_epoch",
        "scheduler_step_count",
        "current_lr",
        "optimizer_parameter_count",
        "parameter_order_digest",
    )
    if (
        observed.get("status") != "passed"
        or observed.get("method_id") != method_id
        or observed.get("deep_dive_manifest_sha256") != manifest_sha256
        or observed.get("all_moments_finite") is not True
        or observed.get("rng_state_complete") is not True
        or observed.get("bf16_scaler_empty") is not True
        or any(observed.get(key) != expected.get(key) for key in exact_fields)
    ):
        raise ValueError("optimizer replay step500 binding differs from manifest")
    observed_hashes = observed.get("observed_hashes", {})
    hash_fields = ["trainer_state_sha256", "progress_sha256"]
    hash_fields.append(
        "model_weights_sha256"
        if method_id == "q2_recranker_generalqwen"
        else "adapter_weights_sha256"
    )
    if any(observed_hashes.get(key) != expected.get(key) for key in hash_fields):
        raise ValueError("optimizer replay step500 hash binding differs from manifest")


def _implementation_digest(metadata):
    digest = str(metadata.get("implementation_identity", {}).get("digest") or "")
    if not digest:
        raise ValueError("optimizer replay implementation digest is missing")
    if metadata.get("run_contract", {}).get("implementation_digest") != digest:
        raise ValueError("optimizer replay implementation differs from run contract")
    return digest


def _rows(root, metadata):
    path = root / "replays.jsonl"
    if metadata.get("replays_sha256") != sha256_file(path):
        raise ValueError("optimizer replay rows hash differs")
    rows = list(iter_jsonl(path))
    if len(rows) != TASKS or not _all_finite(rows):
        raise ValueError("optimizer replay row/finite coverage differs")
    return rows


def _summary(values) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("optimizer replay summary values differ")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
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
