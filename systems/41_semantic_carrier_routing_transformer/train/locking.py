"""Verify C41 proposal and post-G0 execution locks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from train.store import read_json, sha256_file


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]


def verify_proposal_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = Path(config["paths"]["proposal_lock"])
    if not path.is_file():
        raise PermissionError("C41 proposal lock missing")
    lock = read_json(path)
    if lock.get("lock_id") != "c41_semantic_carrier_routing_v1":
        raise ValueError("unexpected C41 proposal lock")
    if lock.get("status") != "locked_before_features_fit_labels_training_or_A_scores":
        raise ValueError("C41 proposal stage differs")
    failures = []
    for relative, expected in lock["candidate_files_sha256"].items():
        source = SYSTEM_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            failures.append(f"candidate:{relative}")
    for relative, expected in lock["external_inputs_sha256"].items():
        source = REPO_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            failures.append(f"external:{relative}")
    if failures:
        raise RuntimeError(f"C41 proposal lock mismatch: {failures}")
    declarations = lock["declarations"]
    if declarations["c41_fit_label_artifact_opened"] or declarations["internal_A_features_scores_labels_opened"]:
        raise ValueError("C41 proposal declarations differ")
    if declarations["dev_test_opened"]:
        raise ValueError("C41 dev/test declaration differs")
    return lock, sha256_file(path)


def verify_execution_lock(
    config: Mapping[str, Any], proposal_sha256: str
) -> tuple[dict[str, Any], str]:
    path = Path(config["paths"]["execution_lock"])
    if not path.is_file():
        raise PermissionError("C41 execution lock missing")
    lock = read_json(path)
    if lock.get("lock_id") != "c41_semantic_carrier_execution_v1":
        raise ValueError("unexpected C41 execution lock")
    if lock.get("proposal_lock_sha256") != proposal_sha256:
        raise ValueError("C41 execution proposal differs")
    failures = []
    for relative, expected in lock["execution_inputs_sha256"].items():
        source = REPO_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"C41 execution inputs changed: {failures}")
    return lock, sha256_file(path)
