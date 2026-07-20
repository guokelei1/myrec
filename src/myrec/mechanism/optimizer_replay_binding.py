"""Immutable step-500 checkpoint binding for registered D7 optimizer replay."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Mapping

from myrec.baselines.motivation_v12_ranker import _checkpoint_identity
from myrec.mechanism.attention_edge_runtime import _load_manifest
from myrec.utils.hashing import sha256_file


STEP500_ROOTS = {
    "q2_recranker_generalqwen": Path(
        "artifacts/motivation_v1_2/resume_canary/q2_step500_seed20260714"
    ),
    "q3_tallrec_generalqwen": Path(
        "artifacts/motivation_v1_2/resume_canary/q3_step500_seed20260714"
    ),
}
_MANIFEST_KEYS = {
    "q2_recranker_generalqwen": "q2_step500",
    "q3_tallrec_generalqwen": "q3_step500",
}
TRAINING_CUDA_RNG_INDEX = {
    "q2_recranker_generalqwen": 2,
    "q3_tallrec_generalqwen": 3,
}


def restore_bound_rng_state(
    torch: Any,
    rng: Mapping[str, Any],
    *,
    method_id: str,
    device: str,
) -> dict[str, Any]:
    """Restore the training device RNG onto one replay-visible CUDA device."""

    if method_id not in TRAINING_CUDA_RNG_INDEX:
        raise ValueError(f"unregistered optimizer replay RNG method: {method_id}")
    if set(rng) != {"python", "torch_cpu", "torch_cuda"}:
        raise ValueError("step-500 RNG state coverage drifted")
    cuda_states = list(rng["torch_cuda"])
    training_index = TRAINING_CUDA_RNG_INDEX[method_id]
    if not 0 <= training_index < len(cuda_states):
        raise ValueError("training CUDA RNG index is outside frozen snapshot")
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch_cpu"])
    if not str(device).startswith("cuda"):
        raise ValueError("registered optimizer replay RNG restore requires CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("registered optimizer replay CUDA is unavailable")
    selected = cuda_states[training_index]
    target = torch.device(device)
    torch.cuda.set_rng_state(selected, device=target)
    observed = torch.cuda.get_rng_state(device=target)
    if not bool(torch.equal(observed.cpu(), selected.cpu())):
        raise ValueError("optimizer replay CUDA RNG restore identity failed")
    return {
        "policy": "training_logical_cuda_rng_to_replay_device",
        "method_id": method_id,
        "saved_cuda_rng_state_count": len(cuda_states),
        "training_logical_cuda_index": training_index,
        "replay_device": str(device),
        "selected_cuda_rng_sha256": _tensor_sha256(selected),
        "restore_identity_exact": True,
    }


def load_bound_step500_state(
    method_id: str,
    *,
    manifest_path: str | Path = "experiments/motivation/transformer_deep_dive_manifest.yaml",
    root: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash-audit and load one frozen step-500 trainer state on CPU."""

    if method_id not in STEP500_ROOTS:
        raise ValueError(f"unregistered optimizer replay method: {method_id}")
    manifest = _load_manifest(manifest_path)
    binding = dict(manifest["optimizer_replay"][_MANIFEST_KEYS[method_id]])
    root = Path(root or STEP500_ROOTS[method_id])
    checkpoint = root / "checkpoint_latest"
    model_dir = checkpoint / "model"
    trainer_path = checkpoint / "trainer_state.pt"
    progress_path = checkpoint / "progress.json"
    weight_path = model_dir / (
        "model.safetensors"
        if method_id == "q2_recranker_generalqwen"
        else "adapter_model.safetensors"
    )
    weight_hash_key = (
        "model_weights_sha256"
        if method_id == "q2_recranker_generalqwen"
        else "adapter_weights_sha256"
    )
    for path in (trainer_path, progress_path, weight_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    observed_hashes = {
        "trainer_state_sha256": sha256_file(trainer_path),
        "progress_sha256": sha256_file(progress_path),
        weight_hash_key: sha256_file(weight_path),
    }
    for key, value in observed_hashes.items():
        if value != str(binding[key]):
            raise ValueError(f"step-500 binding hash mismatch: {key}")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))

    import torch

    state = torch.load(trainer_path, map_location="cpu", weights_only=False)
    checkpoint_id, checkpoint_files = _checkpoint_identity(model_dir, method_id)
    audit = audit_loaded_step500_state(
        state,
        progress,
        binding,
        checkpoint_id=checkpoint_id,
    )
    audit.update(
        {
            "method_id": method_id,
            "root": str(root),
            "trainer_state_path": str(trainer_path),
            "progress_path": str(progress_path),
            "model_dir": str(model_dir),
            "checkpoint_files": checkpoint_files,
            "observed_hashes": observed_hashes,
            "deep_dive_manifest_sha256": manifest["_sha256"],
        }
    )
    return state, audit


def audit_loaded_step500_state(
    state: Mapping[str, Any],
    progress: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    checkpoint_id: str,
) -> dict[str, Any]:
    """Validate trainer/optimizer/scheduler structure without model access."""

    required = {
        "checkpoint_id",
        "config_sha256",
        "optimizer",
        "progress",
        "rng",
        "scaler",
        "scheduler",
        "training_contract",
    }
    if set(state) != required:
        raise ValueError("step-500 trainer-state key set drifted")
    if state["checkpoint_id"] != checkpoint_id or checkpoint_id != binding["checkpoint_id"]:
        raise ValueError("step-500 checkpoint identity drifted")
    if dict(state["progress"]) != dict(progress):
        raise ValueError("step-500 trainer and progress JSON differ")
    if int(progress.get("optimizer_steps", -1)) != int(binding["optimizer_steps"]):
        raise ValueError("step-500 optimizer progress differs")
    optimizer = state["optimizer"]
    groups = optimizer.get("param_groups")
    states = optimizer.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(states, dict):
        raise ValueError("step-500 optimizer must contain one dense parameter group")
    group = groups[0]
    indices = list(group.get("params", []))
    expected_count = int(binding["optimizer_parameter_count"])
    if indices != list(range(expected_count)) or set(states) != set(indices):
        raise ValueError("step-500 optimizer parameter index coverage drifted")
    current_lr = float(binding["current_lr"])
    if not math.isclose(float(group["lr"]), current_lr, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("step-500 optimizer current LR drifted")
    if bool(group.get("amsgrad")) or bool(group.get("maximize")):
        raise ValueError("step-500 optimizer is not registered standard AdamW")
    tensor_shapes = []
    for index in indices:
        parameter_state = states[index]
        if set(parameter_state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("step-500 AdamW state key set drifted")
        step = parameter_state["step"]
        step = int(step.item() if hasattr(step, "item") else step)
        if step != int(binding["optimizer_steps"]):
            raise ValueError("step-500 per-parameter Adam step drifted")
        exp_avg = parameter_state["exp_avg"]
        exp_avg_sq = parameter_state["exp_avg_sq"]
        if exp_avg.shape != exp_avg_sq.shape or not exp_avg.is_floating_point():
            raise ValueError("step-500 Adam moment tensors are invalid")
        if not bool(__import__("torch").isfinite(exp_avg).all().item()) or not bool(
            __import__("torch").isfinite(exp_avg_sq).all().item()
        ):
            raise FloatingPointError("step-500 Adam moments are non-finite")
        tensor_shapes.append(list(exp_avg.shape))
    scheduler = state["scheduler"]
    if int(scheduler.get("last_epoch", -1)) != int(binding["scheduler_last_epoch"]):
        raise ValueError("step-500 scheduler last_epoch drifted")
    if int(scheduler.get("_step_count", -1)) != int(binding["scheduler_step_count"]):
        raise ValueError("step-500 scheduler step count drifted")
    last_lr = list(scheduler.get("_last_lr", []))
    if len(last_lr) != 1 or not math.isclose(
        float(last_lr[0]), current_lr, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("step-500 scheduler current LR drifted")
    if state["scaler"] != {}:
        raise ValueError("registered BF16 step-500 GradScaler state must be empty")
    if set(state["rng"]) != {"python", "torch_cpu", "torch_cuda"}:
        raise ValueError("step-500 RNG state coverage drifted")
    return {
        "checkpoint_id": checkpoint_id,
        "optimizer_steps": int(progress["optimizer_steps"]),
        "optimizer_parameter_count": expected_count,
        "parameter_order_digest": str(binding["parameter_order_digest"]),
        "scheduler_last_epoch": int(scheduler["last_epoch"]),
        "scheduler_step_count": int(scheduler["_step_count"]),
        "current_lr": current_lr,
        "betas": [float(value) for value in group["betas"]],
        "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "moment_tensor_shapes_sha256": _canonical_sha256(tensor_shapes),
        "all_moments_finite": True,
        "rng_state_complete": True,
        "bf16_scaler_empty": True,
        "status": "passed",
    }


def _canonical_sha256(value: Any) -> str:
    import hashlib

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _tensor_sha256(value: Any) -> str:
    import hashlib

    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
