"""Verify the immutable C43 proposal and post-G0 execution locks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from train.structure import REPO_ROOT, SYSTEM_ROOT, read_json, sha256_file


def verify_proposal_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = Path(config["paths"]["proposal_lock"])
    if not path.is_file():
        raise PermissionError("C43 proposal lock missing")
    lock = read_json(path)
    if lock.get("lock_id") != "c43_cross_domain_metric_coupled_gate_v1":
        raise ValueError("unexpected C43 proposal lock")
    if lock.get("status") != "locked_before_c43_feature_label_score_or_outcome":
        raise ValueError("C43 proposal lock stage differs")

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
        source = REPO_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            failures.append(f"external:{relative}")
    if sha256_file(config["paths"]["selection"]) != lock["selection_sha256"]:
        failures.append("selection")
    if failures:
        raise RuntimeError(f"C43 proposal lock mismatch: {failures}")

    declarations = lock["declarations"]
    if declarations.get("c43_internal_A_features_labels_scores_opened") is not False:
        raise ValueError("C43 proposal lock is not pre-A")
    if declarations.get("c37_delayed_B_and_escrow_promoted_to_c43_A_unopened") is not True:
        raise ValueError("C43 source A boundary differs")
    if declarations.get("c43_code_dev_test_qrels_metrics_read") is not False:
        raise ValueError("C43 method-boundary declaration differs")
    if declarations.get("optimizer_steps") != 0:
        raise ValueError("C43 proposal is not pre-training")
    return lock, sha256_file(path)


def verify_execution_lock(
    config: Mapping[str, Any], proposal_hash: str
) -> tuple[dict[str, Any], str]:
    path = Path(config["paths"]["execution_lock"])
    if not path.is_file():
        raise PermissionError("C43 execution lock missing")
    lock = read_json(path)
    if lock.get("lock_id") != "c43_cross_domain_metric_coupled_execution_v1":
        raise ValueError("unexpected C43 execution lock")
    if lock.get("status") != "locked_after_G0_before_training_or_A_score":
        raise ValueError("C43 execution lock stage differs")
    if lock.get("proposal_lock_sha256") != proposal_hash:
        raise ValueError("C43 execution lock proposal mismatch")
    report_path = Path(config["paths"]["artifact_root"]) / "g0_report.json"
    if sha256_file(report_path) != lock["g0_report_sha256"]:
        raise RuntimeError("C43 G0 report changed")
    report = read_json(report_path)
    for name, metadata in report["outputs"].items():
        expected = lock["g0_outputs_sha256"].get(name)
        if expected != metadata["sha256"] or sha256_file(metadata["path"]) != expected:
            raise RuntimeError(f"C43 G0 output changed: {name}")
    return lock, sha256_file(path)
