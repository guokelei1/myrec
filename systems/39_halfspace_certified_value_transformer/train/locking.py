"""Verify C39 proposal and post-G0 execution locks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from train.selection import read_json, sha256_file


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]


def verify_proposal_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = Path(config["paths"]["proposal_lock"])
    if not path.is_file():
        raise PermissionError("C39 proposal lock missing")
    lock = read_json(path)
    if lock.get("lock_id") != "c39_halfspace_certified_value_transformer_v1":
        raise ValueError("unexpected C39 proposal lock")
    if lock.get("status") != "locked_before_fit_labels_training_or_internal_A_scores":
        raise ValueError("C39 proposal lock stage differs")
    failures = []
    lines = []
    for relative, expected in sorted(lock["files_sha256"].items()):
        source = SYSTEM_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            failures.append(f"candidate:{relative}")
        lines.append(f"{expected}  {relative}\n")
    if hashlib.sha256("".join(lines).encode()).hexdigest() != lock["aggregate_sha256"]:
        failures.append("aggregate")
    for relative, expected in sorted(lock["external_inputs_sha256"].items()):
        source = REPO_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            failures.append(f"external:{relative}")
    if failures:
        raise RuntimeError(f"C39 proposal lock mismatch: {failures}")
    declarations = lock["declarations"]
    if declarations.get("internal_A_labels_scores_opened") is not False:
        raise ValueError("C39 proposal is not pre-A")
    if declarations.get("dev_test_records_labels_qrels_opened") is not False:
        raise ValueError("C39 proposal dev/test declaration differs")
    return lock, sha256_file(path)


def verify_execution_lock(
    config: Mapping[str, Any],
    proposal_hash: str,
) -> tuple[dict[str, Any], str]:
    path = Path(config["paths"]["execution_lock"])
    if not path.is_file():
        raise PermissionError("C39 execution lock missing")
    lock = read_json(path)
    if lock.get("lock_id") != "c39_halfspace_certified_value_execution_v1":
        raise ValueError("unexpected C39 execution lock")
    if lock.get("status") != "locked_after_G0_before_training_or_internal_A_scores":
        raise ValueError("C39 execution lock stage differs")
    if lock.get("proposal_lock_sha256") != proposal_hash:
        raise ValueError("C39 execution lock proposal differs")
    failures = []
    for relative, expected in sorted(lock["external_inputs_sha256"].items()):
        source = REPO_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"C39 execution inputs changed: {failures}")
    return lock, sha256_file(path)
