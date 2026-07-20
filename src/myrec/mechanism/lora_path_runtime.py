"""Qrels-free D7 Q3 LoRA factor and gauge-invariant function-path analysis."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_ranker import (
    CHECKPOINT_DIRNAME,
    TRAINING_METADATA,
    _checkpoint_identity,
    _git_revision,
    _load_model_and_tokenizer,
    _runtime_metadata,
    _validate_run_id,
    _validate_scoring_checkpoint_provenance,
    load_v12_ranker_config,
)
from myrec.mechanism.attention_edge_runtime import (
    DEEP_DIVE_MANIFEST_PATH,
    _canonical_sha256,
    _load_manifest,
    _read_json,
    _write_json,
)
from myrec.mechanism.gradient_diagnostic import _load_state_model
from myrec.mechanism.optimizer_replay_math import (
    lora_parameter_identity,
    lora_singular_values,
    parameter_order_digest,
    parameter_order_rows,
)
from myrec.utils.hashing import sha256_file


Q3_METHOD_ID = "q3_tallrec_generalqwen"
Q3_EXPECTED_PARAMETER_ORDER_DIGEST = (
    "cae9b185ec9486a366247a8222598fc645aefe76aede1374123a72f6d583368f"
)
Q3_STEP500_ROOT = Path(
    "artifacts/motivation_v1_2/resume_canary/q3_step500_seed20260714/checkpoint_latest"
)
LORA_STATES = ("base_initialization", "step_500", "frozen_final_checkpoint")


def analyze_q3_lora_path(
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    device: str = "cpu",
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = DEEP_DIVE_MANIFEST_PATH,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Analyze all 28 x q/v LoRA paths at base, step-500, and final."""

    _validate_run_id(run_id)
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Q3 LoRA path output is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    manifest = _load_manifest(manifest_path)
    config = load_v12_ranker_config(config_path)
    if config["method_id"] != Q3_METHOD_ID:
        raise ValueError("LoRA path analysis is Q3-only")
    frozen_model = manifest["frozen_inputs"]["models"][Q3_METHOD_ID]
    if config["_config_sha256"] != frozen_model["config_sha256"]:
        raise ValueError("Q3 LoRA config differs from frozen manifest")
    final_training_metadata_path = checkpoint_root / TRAINING_METADATA
    final_training_metadata = _read_json(final_training_metadata_path)
    _validate_scoring_checkpoint_provenance(final_training_metadata, config, allow_smoke=False)
    final_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    final_checkpoint_id, final_files = _checkpoint_identity(final_model_dir, Q3_METHOD_ID)
    if final_checkpoint_id != frozen_model["checkpoint_id"]:
        raise ValueError("Q3 LoRA final checkpoint differs from frozen manifest")
    replay = manifest["optimizer_replay"]["q3_step500"]
    step_model_dir = Q3_STEP500_ROOT / "model"
    step_trainer_path = Q3_STEP500_ROOT / "trainer_state.pt"
    step_progress_path = Q3_STEP500_ROOT / "progress.json"
    step_checkpoint_id, step_files = _checkpoint_identity(step_model_dir, Q3_METHOD_ID)
    if (
        step_checkpoint_id != replay["checkpoint_id"]
        or sha256_file(step_trainer_path) != replay["trainer_state_sha256"]
        or sha256_file(step_progress_path) != replay["progress_sha256"]
        or sha256_file(step_model_dir / "adapter_model.safetensors")
        != replay["adapter_weights_sha256"]
    ):
        raise ValueError("Q3 step-500 replay binding differs from frozen manifest")
    implementation = q3_lora_path_implementation_identity()
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d7_q3_lora_path",
        "run_id": run_id,
        "method_id": Q3_METHOD_ID,
        "states": list(LORA_STATES),
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "final_checkpoint_id": final_checkpoint_id,
        "final_checkpoint_files": final_files,
        "step500_checkpoint_id": step_checkpoint_id,
        "step500_checkpoint_files": step_files,
        "step500_trainer_state_sha256": sha256_file(step_trainer_path),
        "step500_progress_sha256": sha256_file(step_progress_path),
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "expected_parameter_order_digest": Q3_EXPECTED_PARAMETER_ORDER_DIGEST,
        "qrels_read": False,
        "source_test_opened": False,
        "implementation_identity": implementation,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
        "status": "running",
    }
    _write_json(run_dir / "metadata.json", metadata)
    try:
        import torch
        import transformers

        state_factors: dict[str, dict[str, dict[str, Any]]] = {}
        state_results: dict[str, Any] = {}
        maximum_gauge_error = 0.0
        maximum_base_b_abs = 0.0
        for state in LORA_STATES:
            if state == "step_500":
                _tokenizer, model = _load_model_and_tokenizer(
                    config,
                    device=str(device),
                    training=True,
                    checkpoint_model_dir=step_model_dir,
                )
            else:
                _tokenizer, model = _load_state_model(
                    config,
                    state=state,
                    device=str(device),
                    checkpoint_model_dir=final_model_dir,
                    torch_module=torch,
                )
            named = [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ]
            digest = parameter_order_digest(named)
            if digest != Q3_EXPECTED_PARAMETER_ORDER_DIGEST or len(named) != 112:
                raise ValueError(f"Q3 LoRA parameter order differs at {state}")
            factors = _collect_factors(named)
            if len(factors) != 56:
                raise ValueError("Q3 LoRA factor pair coverage is not 28 x q/v")
            rows = []
            for key in sorted(factors):
                a = factors[key]["A"]
                b = factors[key]["B"]
                gauge_error = _orthogonal_gauge_error(a, b, key)
                maximum_gauge_error = max(maximum_gauge_error, gauge_error)
                if state == "base_initialization":
                    maximum_base_b_abs = max(
                        maximum_base_b_abs, float(b.abs().max().item())
                    )
                rows.append(
                    {
                        "path": key,
                        "block_zero_based": factors[key]["block_zero_based"],
                        "projection": factors[key]["projection"],
                        "a_norm": float(a.double().norm().item()),
                        "b_norm": float(b.double().norm().item()),
                        "delta_w_norm": _low_rank_frobenius_norm(a, b, scaling=2.0),
                        "singular_geometry": lora_singular_values(a, b, scaling=2.0),
                        "orthogonal_gauge_max_abs_error": gauge_error,
                    }
                )
            state_factors[state] = factors
            state_results[state] = {
                "parameter_order_digest": digest,
                "parameter_order_rows": parameter_order_rows(named),
                "paths": rows,
            }
            del model
        comparisons = []
        for left, right in (
            ("base_initialization", "step_500"),
            ("step_500", "frozen_final_checkpoint"),
            ("base_initialization", "frozen_final_checkpoint"),
        ):
            for key in sorted(state_factors[left]):
                left_pair = state_factors[left][key]
                right_pair = state_factors[right][key]
                comparisons.append(
                    {
                        "left_state": left,
                        "right_state": right,
                        "path": key,
                        "block_zero_based": left_pair["block_zero_based"],
                        "projection": left_pair["projection"],
                        "delta_w_cosine": _low_rank_function_cosine(
                            left_pair["A"],
                            left_pair["B"],
                            right_pair["A"],
                            right_pair["B"],
                        ),
                    }
                )
        if maximum_base_b_abs != 0.0:
            raise ValueError("Q3 base LoRA B is not exactly zero")
        if maximum_gauge_error > 1.0e-5:
            raise ValueError("Q3 LoRA orthogonal gauge identity failed")
        report = {
            "schema_version": 1,
            "analysis_type": "transformer_deep_dive_d7_q3_lora_path",
            "states": state_results,
            "comparisons": comparisons,
            "mechanical_controls": {
                "base_b_exact_zero": True,
                "maximum_base_b_abs": maximum_base_b_abs,
                "orthogonal_gauge_identity_passed": True,
                "maximum_orthogonal_gauge_error": maximum_gauge_error,
                "svd_near_zero_rule": "sigma <= max(sigma1*1e-6,1e-8)",
                "svd_degenerate_gap_rule": "relative_gap < 1e-4",
            },
            "qrels_read": False,
            "source_test_opened": False,
            "status": "completed",
        }
        _write_json(run_dir / "lora_path_analysis.json", report)
        metadata.update(
            {
                **_runtime_metadata(Q3_METHOD_ID, torch, transformers),
                "status": "completed",
                "elapsed_seconds": time.monotonic() - started,
                "analysis_path": str(run_dir / "lora_path_analysis.json"),
                "analysis_sha256": sha256_file(run_dir / "lora_path_analysis.json"),
                "parameter_paths": 56,
                "orthogonal_gauge_identity_passed": True,
                "base_b_exact_zero": True,
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        return metadata
    except Exception as exc:
        metadata.update(
            {
                "status": "mechanical_failure",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        raise


def q3_lora_path_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/lora_path_runtime.py",
        "src/myrec/mechanism/optimizer_replay_math.py",
        "scripts/analyze_deep_dive_q3_lora_path.py",
    )
    files = [
        {
            "path": relative,
            "sha256": sha256_file(root / relative),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in paths
    ]
    return {"files": files, "digest": _canonical_sha256(files)}


def _collect_factors(
    named: Sequence[tuple[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, parameter in named:
        identity = lora_parameter_identity(name)
        key = f"block_{identity['block_zero_based']:02d}.{identity['projection']}_proj"
        row = result.setdefault(
            key,
            {
                "block_zero_based": identity["block_zero_based"],
                "projection": identity["projection"],
            },
        )
        factor = identity["factor"]
        if factor in row:
            raise ValueError(f"duplicate Q3 LoRA factor: {key} {factor}")
        row[factor] = parameter.detach().double().cpu().contiguous()
    if any(set(row) != {"block_zero_based", "projection", "A", "B"} for row in result.values()):
        raise ValueError("Q3 LoRA factor pairs are incomplete")
    return result


def _low_rank_frobenius_norm(a: Any, b: Any, *, scaling: float) -> float:
    gram_a = a @ a.T
    gram_b = b.T @ b
    squared = float((gram_a * gram_b.T).sum().item()) * float(scaling) ** 2
    return math.sqrt(max(0.0, squared))


def _low_rank_function_cosine(a1: Any, b1: Any, a2: Any, b2: Any) -> float | None:
    left = _low_rank_frobenius_norm(a1, b1, scaling=2.0)
    right = _low_rank_frobenius_norm(a2, b2, scaling=2.0)
    if left == 0.0 or right == 0.0:
        return None
    dot = 4.0 * float(((a1 @ a2.T) * (b1.T @ b2)).sum().item())
    return max(-1.0, min(1.0, dot / (left * right)))


def _orthogonal_gauge_error(a: Any, b: Any, key: str) -> float:
    import torch

    seed = int.from_bytes(key.encode("utf-8"), "little") % (2**31)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(a.shape[0], a.shape[0], generator=generator, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(raw)
    transformed_a = rotation @ a
    transformed_b = b @ rotation.T
    return float((transformed_b @ transformed_a - b @ a).abs().max().item())
