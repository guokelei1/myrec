"""C25 proposal and execution lock verification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from train.structure import REPO_ROOT, SYSTEM_ROOT, read_json, sha256_file


def verify_proposal_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = Path(config["paths"]["proposal_lock"])
    if not path.is_file():
        raise PermissionError("C25 proposal lock missing")
    lock = read_json(path)
    if lock.get("lock_id") != "c25_anchored_mobius_interaction_v1" or lock.get(
        "status"
    ) != "locked_before_any_c25_label_or_ranking_outcome":
        raise ValueError("unexpected C25 proposal lock")
    failures: list[str] = []
    lines: list[str] = []
    for relative, expected in sorted(lock["files_sha256"].items()):
        candidate_path = SYSTEM_ROOT / relative
        if not candidate_path.is_file() or sha256_file(candidate_path) != expected:
            failures.append(f"candidate:{relative}")
        lines.append(f"{expected}  {relative}\n")
    if hashlib.sha256("".join(lines).encode()).hexdigest() != lock["aggregate_sha256"]:
        failures.append("aggregate")
    for relative, expected in sorted(lock["external_inputs_sha256"].items()):
        if sha256_file(REPO_ROOT / relative) != expected:
            failures.append(f"external:{relative}")
    if sha256_file(config["paths"]["selection"]) != lock["selection_sha256"]:
        failures.append("selection")
    if failures:
        raise RuntimeError(f"C25 proposal lock mismatch: {failures}")
    declarations = lock["declarations"]
    if declarations.get("c25_any_labels_opened") is not False or declarations.get(
        "c25_ranking_outcome_observed"
    ) is not False:
        raise ValueError("C25 lock is not pre-outcome")
    return lock, sha256_file(path)


def verify_execution_lock(
    config: Mapping[str, Any], proposal_hash: str
) -> tuple[dict[str, Any], str]:
    path = Path(config["paths"]["execution_lock"])
    if not path.is_file():
        raise PermissionError("C25 execution lock missing")
    lock = read_json(path)
    if lock.get("status") != "locked_after_G0_before_training_or_internal_A_access":
        raise ValueError("unexpected C25 execution lock")
    if lock.get("proposal_lock_sha256") != proposal_hash:
        raise ValueError("C25 execution lock proposal mismatch")
    g0_path = Path(config["paths"]["artifact_root"]) / "g0_report.json"
    if sha256_file(g0_path) != lock["g0_report_sha256"]:
        raise RuntimeError("C25 G0 report changed")
    report = read_json(g0_path)
    for name, metadata in report["outputs"].items():
        expected = lock["g0_output_sha256"].get(name)
        if expected != metadata["sha256"] or sha256_file(metadata["path"]) != expected:
            raise RuntimeError(f"C25 G0 output changed: {name}")
    return lock, sha256_file(path)
