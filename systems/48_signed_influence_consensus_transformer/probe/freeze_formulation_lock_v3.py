"""Supplemental lock for C48's length-one stride repair."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from freeze_formulation_lock import REPO_ROOT, SYSTEM_ROOT, load_config, write_once
from freeze_formulation_lock_v2 import verify_formulation_lock_v2

C47_ROOT = REPO_ROOT / "systems/47_posterior_supported_ridge_transformer"
if str(C47_ROOT) not in sys.path:
    sys.path.insert(0, str(C47_ROOT))
from probe.locking import sha256_file  # noqa: E402


FILES = (
    "notes/formulation_lock.json",
    "notes/formulation_lock_v2.json",
    "notes/formulation_v2_execution_abort.md",
    "probe/freeze_formulation_lock_v3.py",
    "probe/run_formulation_gate_v3.py",
    "tests/test_runtime_stride_v3.py",
)


def freeze_v3(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    _, v2_hash = verify_formulation_lock_v2(config)
    target = SYSTEM_ROOT / "notes/formulation_lock_v3.json"
    output_root = REPO_ROOT / config["paths"]["artifact_root"]
    promoted = REPO_ROOT / config["paths"]["promoted_report"]
    if target.exists() or output_root.exists() or promoted.exists():
        raise RuntimeError("C48 output exists before v3 mechanical lock")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "systems/48_signed_influence_consensus_transformer/tests/test_runtime_stride_v3.py",
    ]
    test = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    if test.returncode != 0 or "1 passed" not in test.stdout:
        raise RuntimeError(f"C48 v3 regression failed: {test.stdout}\n{test.stderr}")
    value = {
        "candidate_id": "c48",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "locked_v3_unconditional_C_order_copy_only",
        "formulation_lock_v2_sha256": v2_hash,
        "files_sha256": {
            str((SYSTEM_ROOT / relative).relative_to(REPO_ROOT)): sha256_file(SYSTEM_ROOT / relative)
            for relative in FILES
        },
        "declarations": {
            "scientific_settings_changed": False,
            "labels_metrics_outputs_opened_before_v3": False,
            "fresh_reserve_opened": False,
            "dev_test_qrels_opened": False,
            "repair": "unconditional C-order copy handles length-one negative strides",
            "regression_tests_passed": 1,
        },
    }
    write_once(target, value)
    return value


def verify_formulation_lock_v3(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    _, v2_hash = verify_formulation_lock_v2(config)
    target = SYSTEM_ROOT / "notes/formulation_lock_v3.json"
    value = json.loads(target.read_text(encoding="utf-8"))
    if value.get("status") != "locked_v3_unconditional_C_order_copy_only":
        raise RuntimeError("C48 v3 lock status differs")
    if value.get("formulation_lock_v2_sha256") != v2_hash:
        raise RuntimeError("C48 v3 lock does not bind v2")
    for relative, expected in value["files_sha256"].items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"C48 v3 locked input changed: {relative}")
    return value, sha256_file(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = freeze_v3(args.config)
    print(json.dumps({"candidate_id": "c48", "status": result["status"]}, sort_keys=True))
