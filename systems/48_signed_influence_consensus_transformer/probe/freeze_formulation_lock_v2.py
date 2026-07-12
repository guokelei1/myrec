"""Supplemental lock for C48's negative-stride mechanical repair."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from freeze_formulation_lock import (
    REPO_ROOT,
    SYSTEM_ROOT,
    load_config,
    verify_formulation_lock,
    write_once,
)

C47_ROOT = REPO_ROOT / "systems/47_posterior_supported_ridge_transformer"
if str(C47_ROOT) not in sys.path:
    sys.path.insert(0, str(C47_ROOT))
from probe.locking import sha256_file  # noqa: E402


FILES = (
    "notes/formulation_execution_abort.md",
    "notes/formulation_lock.json",
    "probe/freeze_formulation_lock_v2.py",
    "probe/run_formulation_gate_v2.py",
    "tests/test_runtime_stride.py",
)


def freeze_v2(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    _, v1_hash = verify_formulation_lock(config)
    target = SYSTEM_ROOT / "notes/formulation_lock_v2.json"
    output_root = REPO_ROOT / config["paths"]["artifact_root"]
    promoted = REPO_ROOT / config["paths"]["promoted_report"]
    if target.exists() or output_root.exists() or promoted.exists():
        raise RuntimeError("C48 output exists before v2 mechanical lock")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "systems/48_signed_influence_consensus_transformer/tests/test_runtime_stride.py",
    ]
    test = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    if test.returncode != 0 or "1 passed" not in test.stdout:
        raise RuntimeError(f"C48 v2 regression failed: {test.stdout}\n{test.stderr}")
    value = {
        "candidate_id": "c48",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "locked_v2_contiguous_array_boundary_only",
        "formulation_lock_sha256": v1_hash,
        "files_sha256": {
            str((SYSTEM_ROOT / relative).relative_to(REPO_ROOT)): sha256_file(SYSTEM_ROOT / relative)
            for relative in FILES
        },
        "declarations": {
            "scientific_settings_changed": False,
            "labels_metrics_outputs_opened_before_v2": False,
            "fresh_reserve_opened": False,
            "dev_test_qrels_opened": False,
            "repair": "copy negative-stride NumPy views to contiguous arrays at Torch boundary",
            "regression_tests_passed": 1,
        },
    }
    write_once(target, value)
    return value


def verify_formulation_lock_v2(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    _, v1_hash = verify_formulation_lock(config)
    target = SYSTEM_ROOT / "notes/formulation_lock_v2.json"
    value = json.loads(target.read_text(encoding="utf-8"))
    if value.get("status") != "locked_v2_contiguous_array_boundary_only":
        raise RuntimeError("C48 v2 lock status differs")
    if value.get("formulation_lock_sha256") != v1_hash:
        raise RuntimeError("C48 v2 lock does not bind v1")
    for relative, expected in value["files_sha256"].items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"C48 v2 locked input changed: {relative}")
    return value, sha256_file(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = freeze_v2(args.config)
    print(json.dumps({"candidate_id": "c48", "status": result["status"]}, sort_keys=True))
