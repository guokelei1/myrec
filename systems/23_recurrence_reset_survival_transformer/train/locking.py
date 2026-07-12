"""Fail-closed verification of the pre-label C23 proposal lock."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from train.structure import REPO_ROOT, SYSTEM_ROOT, read_json, sha256_file


def verify_proposal_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = Path(config["paths"]["proposal_lock"])
    if not path.is_file():
        raise PermissionError("C23 proposal lock is missing")
    lock = read_json(path)
    if lock.get("candidate_id") != "c23" or lock.get("lock_id") != (
        "c23_recurrence_reset_train_gate_v1"
    ):
        raise ValueError("unexpected C23 proposal lock")
    if lock.get("status") != "locked_before_any_c23_label_or_ranking_outcome":
        raise ValueError("C23 lock status is not pre-outcome")
    failures: list[str] = []
    aggregate_lines: list[str] = []
    for relative, expected in sorted(lock.get("files_sha256", {}).items()):
        candidate = SYSTEM_ROOT / relative
        if not candidate.is_file() or sha256_file(candidate) != expected:
            failures.append(f"candidate:{relative}")
        aggregate_lines.append(f"{expected}  {relative}\n")
    aggregate = hashlib.sha256("".join(aggregate_lines).encode("utf-8")).hexdigest()
    if aggregate != lock.get("aggregate_sha256"):
        failures.append("candidate:aggregate")
    for relative, expected in sorted(lock.get("external_inputs_sha256", {}).items()):
        candidate = REPO_ROOT / relative
        if not candidate.is_file() or sha256_file(candidate) != expected:
            failures.append(f"external:{relative}")
    selection_path = Path(config["paths"]["selection"])
    if sha256_file(selection_path) != config["paths"]["selection_sha256"]:
        failures.append("selection")
    if lock.get("selection_sha256") != config["paths"]["selection_sha256"]:
        failures.append("lock-selection")
    if failures:
        raise RuntimeError(f"C23 proposal lock mismatch: {failures}")
    declarations = lock.get("declarations", {})
    required_false = (
        "c23_fit_labels_opened",
        "c23_internal_A_labels_opened",
        "c23_ranking_outcome_observed",
        "delayed_B_or_escrow_opened",
        "dev_test_qrels_or_metrics_read",
    )
    if any(declarations.get(name) is not False for name in required_false):
        raise ValueError("C23 proposal lock lacks pre-outcome declarations")
    return lock, sha256_file(path)


def verify_execution_lock(
    config: Mapping[str, Any], proposal_lock_sha256: str
) -> tuple[dict[str, Any], str]:
    path = Path(config["paths"]["execution_lock"])
    if not path.is_file():
        raise PermissionError("C23 execution lock is missing")
    lock = read_json(path)
    if lock.get("candidate_id") != "c23" or lock.get("lock_id") != (
        "c23_recurrence_reset_execution_v1"
    ):
        raise ValueError("unexpected C23 execution lock")
    if lock.get("status") != "locked_after_G0_before_training_or_internal_A_access":
        raise ValueError("C23 execution lock status differs")
    if lock.get("proposal_lock_sha256") != proposal_lock_sha256:
        raise ValueError("C23 execution lock names another proposal lock")
    g0_path = Path(config["paths"]["artifact_root"]) / "g0_report.json"
    if sha256_file(g0_path) != lock.get("g0_report_sha256"):
        raise RuntimeError("C23 G0 report changed after execution lock")
    g0 = read_json(g0_path)
    for name, metadata in g0.get("outputs", {}).items():
        expected = lock.get("g0_output_sha256", {}).get(name)
        if expected != metadata.get("sha256") or sha256_file(metadata["path"]) != expected:
            raise RuntimeError(f"C23 G0 output changed: {name}")
    declarations = lock.get("declarations", {})
    required_false = (
        "training_started",
        "internal_A_labels_opened",
        "ranking_outcome_observed",
        "delayed_B_or_escrow_opened",
        "dev_test_qrels_or_metrics_read",
    )
    if any(declarations.get(name) is not False for name in required_false):
        raise ValueError("C23 execution lock lacks pre-training declarations")
    return lock, sha256_file(path)
