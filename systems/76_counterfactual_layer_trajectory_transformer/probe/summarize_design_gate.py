#!/usr/bin/env python
"""Promote the three-seed C76 design-gate decision."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from probe.freeze_lock import verify_execution_lock  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    _, lock_hash = verify_execution_lock()
    reports = []
    report_files = []
    for seed in (20265701, 20265702, 20265703):
        path = ROOT / "runs/design_gate" / f"seed_{seed}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["execution_lock_sha256"] != lock_hash:
            raise RuntimeError("C76 seed execution lock differs")
        reports.append(value)
        report_files.append({"path": str(path.relative_to(REPO)), "sha256": sha256(path)})
    all_passed = all(value["passed"] for value in reports)
    decision = (
        "authorize_fresh_real_fit_probe"
        if all_passed
        else "close_c76_before_repository_data"
    )
    output = {
        "candidate_id": "c76",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "data_free_design_gate",
        "decision": decision,
        "passed": all_passed,
        "execution_lock_sha256": lock_hash,
        "seeds": {
            str(value["seed"]): {
                "passed": value["passed"],
                "checks": value["checks"],
                "primary": value["evaluation"]["counterfactual_trajectory"],
                "controls": value["control_advantage"],
                "training": value["training"],
                "g0": value["g0"],
            }
            for value in reports
        },
        "seed_reports": report_files,
        "access": {"repository_data": False, "labels": False, "dev_test_qrels": False},
    }
    path = REPO / "reports/pps_c76_design_gate.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "passed": all_passed, "report": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
