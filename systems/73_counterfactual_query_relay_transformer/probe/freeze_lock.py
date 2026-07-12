"""Freeze every C73 design-gate source before trained-model outcome."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

from probe.locking import atomic_json, load_config, sha256_file  # noqa: E402


LOCKED_FILES = (
    "README.md",
    "environment.txt",
    "configs/design_gate.yaml",
    "model/__init__.py",
    "model/query_relay.py",
    "probe/__init__.py",
    "probe/synthetic.py",
    "probe/locking.py",
    "probe/freeze_lock.py",
    "probe/run_design_gate.py",
    "tests/test_model.py",
    "tests/test_protocol.py",
    "notes/proposal.md",
    "notes/mechanism_fingerprint.md",
    "notes/nearest_neighbors.md",
    "notes/design_gate_protocol.md",
    "notes/preimplementation_review.md",
)


def main() -> None:
    config = load_config()
    lock_path = REPO_ROOT / config["paths"]["proposal_lock"]
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
        "candidate_id": "c73",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "locked_before_any_c73_trained_model_outcome",
        "git_commit": commit,
        "git_dirty": dirty,
        "files": {
            relative: sha256_file(SYSTEM_ROOT / relative)
            for relative in LOCKED_FILES
        },
        "declarations": {
            "trained_c73_model_outcome_observed": False,
            "repository_data_read": False,
            "repository_labels_read": False,
            "dev_test_qrels_read": False,
            "shared_evaluator_calls": 0,
        },
    }
    atomic_json(lock_path, value)
    print(json.dumps({"lock": str(lock_path), "status": value["status"]}))


if __name__ == "__main__":
    main()
