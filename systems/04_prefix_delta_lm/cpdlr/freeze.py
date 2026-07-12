"""Create the pre-outcome C04 proposal lock."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import transformers

from .io import assert_candidate_manifest, load_yaml, sha256_file, write_json


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def freeze_proposal(
    config_path: str | Path, candidate_root: str | Path
) -> dict[str, Any]:
    config = load_yaml(config_path)
    candidate_hash = assert_candidate_manifest(
        config["paths"]["candidate_manifest"], config["candidate_manifest_sha256"]
    )
    root = Path(candidate_root)
    locked_files = {}
    exclusions = {
        root / "notes" / "proposal_lock.json",
        root / "notes" / "final_report.md",
        root / "notes" / "screening_audit.json",
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path in exclusions:
            continue
        if (
            "__pycache__" in path.parts
            or ".pytest_cache" in path.parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        locked_files[str(path)] = sha256_file(path)
    status = _git("status", "--short")
    dev_log_path = Path("reports/dev_eval_log.jsonl")
    prior_calls = 0
    if dev_log_path.exists():
        with dev_log_path.open("r", encoding="utf-8") as handle:
            prior_calls = sum(
                1
                for line in handle
                if "20260710_kuaisearch_c04_" in line
            )
    if prior_calls:
        raise ValueError(f"C04 dev evaluator outcome exists before lock: {prior_calls}")
    lock = {
        "budget": config["budget"],
        "candidate_id": config["candidate_id"],
        "candidate_manifest_sha256": candidate_hash,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "declarations": {
            "c04_dev_outcome_not_read": True,
            "other_candidate_designs_not_read_or_copied": True,
            "proposal_locked_before_any_c04_gpu_model_outcome": True,
            "qrels_not_read": True,
            "test_not_read": True,
        },
        "dev_evaluator_calls_before_lock": prior_calls,
        "environment": {
            "conda_env": "myrec-c04",
            "conda_prefix": os.environ.get("CONDA_PREFIX", "unknown"),
            "cuda_build": torch.version.cuda,
            "numpy": __import__("numpy").__version__,
            "platform": platform.platform(),
            "pytest": __import__("pytest").__version__,
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(status),
            "status_sha256": __import__("hashlib").sha256(status.encode()).hexdigest(),
        },
        "gpu": {
            "code_device": "cuda:0",
            "physical_gpu": 3,
            "required_binding": "CUDA_VISIBLE_DEVICES=3",
        },
        "locked_at": datetime.now().astimezone().isoformat(),
        "locked_files": locked_files,
        "seed": int(config["seed"]),
    }
    output_path = root / "notes" / "proposal_lock.json"
    write_json(output_path, lock)
    return lock
