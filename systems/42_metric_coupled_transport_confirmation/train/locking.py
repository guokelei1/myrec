"""Verify C42 proposal and execution locks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from train.store import read_json, sha256_file


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]


def verify_proposal_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = Path(config["paths"]["proposal_lock"])
    lock = read_json(path)
    if lock.get("lock_id") != "c42_metric_coupled_confirmation_v1":
        raise ValueError("unexpected C42 proposal lock")
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
        raise RuntimeError(f"C42 proposal mismatch: {failures}")
    return lock, sha256_file(path)


def verify_execution_lock(
    config: Mapping[str, Any], proposal_sha256: str
) -> tuple[dict[str, Any], str]:
    path = Path(config["paths"]["execution_lock"])
    lock = read_json(path)
    if lock.get("lock_id") != "c42_metric_coupled_execution_v1":
        raise ValueError("unexpected C42 execution lock")
    if lock.get("proposal_lock_sha256") != proposal_sha256:
        raise ValueError("C42 execution proposal differs")
    failures = []
    for relative, expected in lock["execution_inputs_sha256"].items():
        source = REPO_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"C42 execution mismatch: {failures}")
    return lock, sha256_file(path)
