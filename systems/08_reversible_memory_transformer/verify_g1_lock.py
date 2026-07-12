"""Independent pre/post verifier for the C08 G1 execution manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from g1_protocol import config_dict


ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(phase: str) -> dict[str, object]:
    lock_path = ROOT / "G1_EXECUTION_LOCK.json"
    pre_path = ROOT / "PRE_OUTCOME_LOCK.json"
    errors: list[str] = []
    if not lock_path.is_file() or not pre_path.is_file():
        return {"passed": False, "errors": ["missing lock file"]}
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    pre = json.loads(pre_path.read_text(encoding="utf-8"))

    for relative, expected in pre["files_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected:
            errors.append(f"PRE_OUTCOME_LOCK mismatch: {relative}")

    aggregate_lines: list[str] = []
    for relative in sorted(lock["files_sha256"]):
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            errors.append(f"non-local manifest path: {relative}")
            continue
        path = ROOT / relative
        expected = lock["files_sha256"][relative]
        if not path.is_file() or digest(path) != expected:
            errors.append(f"execution manifest mismatch: {relative}")
        aggregate_lines.append(f"{expected}  {relative}\n")
    aggregate = hashlib.sha256("".join(aggregate_lines).encode("utf-8")).hexdigest()
    if aggregate != lock["aggregate_sha256"]:
        errors.append("aggregate hash mismatch")
    if lock["protocol_constants"] != config_dict():
        errors.append("executable constants mismatch")
    if digest(pre_path) != lock["files_sha256"].get("PRE_OUTCOME_LOCK.json"):
        errors.append("PRE_OUTCOME_LOCK is not bound by execution lock")

    run_dir = ROOT / "runs" / f"g1_{lock['aggregate_sha256'][:16]}"
    if phase == "pre":
        for relative in pre["required_absent_before_execution"]:
            if "*" not in relative and (ROOT / relative).exists():
                errors.append(f"pre-execution outcome already exists: {relative}")
        if run_dir.exists():
            errors.append("locked one-shot run directory already exists")
    elif phase == "post":
        if not (run_dir / "RUN_COMPLETE.json").is_file():
            errors.append("post-execution RUN_COMPLETE missing")
    else:
        errors.append(f"unknown phase: {phase}")

    return {
        "phase": phase,
        "passed": not errors,
        "errors": errors,
        "execution_lock_sha256": digest(lock_path),
        "aggregate_sha256": lock["aggregate_sha256"],
        "manifest_file_count": len(lock["files_sha256"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pre", "post"), required=True)
    args = parser.parse_args()
    result = verify(args.phase)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
