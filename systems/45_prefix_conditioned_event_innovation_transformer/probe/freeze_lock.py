"""Freeze the C45 proposal before any trained-model outcome."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    config_path = SYSTEM_ROOT / "configs/design_gate.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    lock_path = REPO_ROOT / config["paths"]["proposal_lock"]
    if lock_path.exists():
        raise FileExistsError(lock_path)
    relative_files = [
        "README.md",
        "environment.txt",
        "configs/design_gate.yaml",
        "model/__init__.py",
        "model/pceit.py",
        "probe/__init__.py",
        "probe/synthetic.py",
        "probe/audit_generator.py",
        "probe/freeze_lock.py",
        "probe/run_design_gate.py",
        "tests/test_model.py",
        "tests/test_protocol.py",
        "notes/proposal.md",
        "notes/mechanism_fingerprint.md",
        "notes/nearest_neighbors.md",
        "notes/design_gate_protocol.md",
    ]
    files = {name: sha256(SYSTEM_ROOT / name) for name in relative_files}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    value = {
        "candidate_id": "c45",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "locked_before_any_c45_trained_model_outcome",
        "git_commit": commit,
        "git_dirty": dirty,
        "environment": "/data/gkl/conda_envs/myrec-c37",
        "files": files,
        "declarations": {
            "trained_c45_model_outcome_observed": False,
            "repository_data_read": False,
            "repository_labels_read": False,
            "dev_test_qrels_read": False,
            "shared_evaluator_calls": 0,
        },
    }
    atomic_json(lock_path, value)
    print(json.dumps({"candidate_id": "c45", "status": value["status"], "lock": str(lock_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
