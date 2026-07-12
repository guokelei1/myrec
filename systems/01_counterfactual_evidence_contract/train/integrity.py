"""Fail-closed integrity checks shared by every C01 executable."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from train.data import assert_candidate_manifest, sha256_file


CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CANDIDATE_ROOT.parents[1]
LOCK_PATH = CANDIDATE_ROOT / "notes" / "proposal_lock.json"
CONFIG_PATH = CANDIDATE_ROOT / "configs" / "screening.yaml"
LOCKED_FILE_ORDER = (
    "README.md",
    "environment.txt",
    "notes/proposal.md",
    "notes/mechanism_fingerprint.md",
    "notes/nearest_neighbors.md",
    "notes/gate_protocol.md",
    "configs/screening.yaml",
)


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("C01 config must be a mapping")
    return config


def verify_proposal_lock(config: dict[str, Any]) -> dict[str, Any]:
    """Verify that every pre-outcome design artifact remains byte frozen."""

    with LOCK_PATH.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    if lock.get("status") != "locked_before_any_c01_outcome":
        raise ValueError("proposal lock is not in its frozen pre-outcome state")
    if lock.get("candidate_id") != config.get("candidate_id"):
        raise ValueError("candidate id differs from proposal lock")
    if lock.get("seed") != config.get("seed"):
        raise ValueError("seed differs from proposal lock")
    if lock.get("run_id") != config.get("run_id"):
        raise ValueError("run id differs from proposal lock")

    recorded = lock["design_files"]
    sum_lines: list[str] = []
    for relative in LOCKED_FILE_ORDER:
        actual = sha256_file(CANDIDATE_ROOT / relative)
        expected = recorded.get(relative)
        if actual != expected:
            raise ValueError(f"locked design changed: {relative}: {actual} != {expected}")
        repository_relative = (
            Path("systems") / CANDIDATE_ROOT.name / relative
        ).as_posix()
        sum_lines.append(f"{actual}  {repository_relative}\n")
    candidate_hash = hashlib.sha256("".join(sum_lines).encode("utf-8")).hexdigest()
    if candidate_hash != lock["candidate_hash"]:
        raise ValueError(
            f"candidate lock hash mismatch: {candidate_hash} != {lock['candidate_hash']}"
        )
    assert_candidate_manifest(
        REPO_ROOT / config["paths"]["candidate_manifest"],
        config["paths"]["candidate_manifest_sha256"],
    )
    return lock


def assert_source_isolation() -> dict[str, Any]:
    """Reject source that names forbidden held-out or sibling-candidate paths."""

    forbidden = (
        "qrels_" + "dev.jsonl",
        "qrels_" + "test.jsonl",
        "records_" + "test.jsonl",
        "systems/" + "02_",
        "systems/" + "03_",
        "systems/" + "04_",
        "doc/" + "design_prompts",
    )
    checked = 0
    for path in sorted(CANDIDATE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        checked += 1
        hits = [fragment for fragment in forbidden if fragment in source]
        if hits:
            raise ValueError(f"forbidden source-path reference in {path}: {hits}")
    return {"python_files_checked": checked, "forbidden_references": 0}


def assert_gpu_binding(expected_physical_gpu: int = 0) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(expected_physical_gpu):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must be exactly {expected_physical_gpu}, got {visible!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C01 requires exactly one visible CUDA device")
    return {
        "cuda_visible_devices": visible,
        "logical_device": 0,
        "physical_device": expected_physical_gpu,
        "device_name": torch.cuda.get_device_name(0),
        "torch_cuda": torch.version.cuda,
    }


def set_determinism(seed: int) -> None:
    # Must be present before the first cuBLAS operation when deterministic
    # algorithms are enforced. Device discovery itself is safe before this.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def environment_record() -> dict[str, Any]:
    return {
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "numpy": np.__version__,
        "python_runtime": os.sys.version.split()[0],
        "torch": torch.__version__,
    }
