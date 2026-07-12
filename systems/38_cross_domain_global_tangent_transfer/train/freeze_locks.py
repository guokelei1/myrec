"""Create immutable C38 proposal and execution locks."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from train.selection import read_json, sha256_file, write_json  # noqa: E402


EXCLUDED_CANDIDATE_FILES = {
    "notes/proposal_lock.json",
    "notes/execution_lock.json",
    "notes/train_gate_outcome.md",
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("C38 config must be an object")
    return value


def candidate_hashes() -> dict[str, str]:
    output = {}
    for path in sorted(SYSTEM_ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = str(path.relative_to(SYSTEM_ROOT))
        if relative in EXCLUDED_CANDIDATE_FILES:
            continue
        output[relative] = sha256_file(path)
    return output


def relative_repo(path: str | Path) -> str:
    resolved = Path(path).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError(f"C38 lock input is outside the repository: {path}")
    return str(resolved.relative_to(REPO_ROOT.resolve()))


def external_hashes(paths: list[str]) -> dict[str, str]:
    output = {}
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(path)
        output[relative_repo(path)] = sha256_file(path)
    return dict(sorted(output.items()))


def freeze_proposal(config: dict[str, Any]) -> dict[str, Any]:
    target = Path(config["paths"]["proposal_lock"])
    if target.exists():
        raise FileExistsError(target)
    c0 = read_json(config["paths"]["c0_report"])
    if c0.get("overall_status") != "passed":
        raise PermissionError("Amazon-C4 C0 has not passed")
    c1 = read_json(config["paths"]["c1_report"])
    if c1.get("overall_status") != "passed":
        raise PermissionError("Amazon-C4 C1 has not passed")
    selection = read_json(config["paths"]["selection"])
    if selection["label_access"]["records_train_labels_opened"] is not False:
        raise PermissionError("C38 train labels opened before proposal lock")
    files = candidate_hashes()
    lines = [f"{value}  {name}\n" for name, value in sorted(files.items())]
    value = {
        "aggregate_sha256": hashlib.sha256("".join(lines).encode()).hexdigest(),
        "candidate_id": "c38",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "declarations": {
            "fit_labels_opened": False,
            "internal_A_labels_scores_opened": False,
            "delayed_B_features_labels_scores_opened": False,
            "escrow_opened": False,
            "dev_test_records_labels_qrels_opened": False,
            "ranking_outcome_observed": False,
            "cross_dataset_confirmatory_not_novelty_claim": True,
        },
        "external_inputs_sha256": external_hashes(
            list(config["locking"]["proposal_external_inputs"])
        ),
        "files_sha256": files,
        "git_commit_at_lock": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "lock_id": "c38_cross_domain_global_tangent_transfer_v1",
        "status": "locked_before_fit_labels_training_or_internal_A_scores",
    }
    write_json(target, value)
    return value


def freeze_execution(config: dict[str, Any]) -> dict[str, Any]:
    target = Path(config["paths"]["execution_lock"])
    if target.exists():
        raise FileExistsError(target)
    proposal_path = Path(config["paths"]["proposal_lock"])
    if not proposal_path.is_file():
        raise PermissionError("C38 proposal lock missing")
    g0 = read_json(config["paths"]["g0_report"])
    if g0.get("status") != "passed":
        raise PermissionError("C38 G0 has not passed")
    if g0.get("internal_A_labels_scores_opened") is not False:
        raise PermissionError("C38 A opened before execution lock")
    value = {
        "candidate_id": "c38",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "declarations": {
            "fit_labels_opened_after_proposal_lock": True,
            "internal_A_label_free_features_opened": True,
            "internal_A_labels_scores_opened": False,
            "delayed_B_features_labels_scores_opened": False,
            "escrow_dev_test_opened": False,
            "training_started": False,
        },
        "external_inputs_sha256": external_hashes(
            list(config["locking"]["execution_external_inputs"])
        ),
        "g0_report_sha256": sha256_file(config["paths"]["g0_report"]),
        "lock_id": "c38_cross_domain_global_tangent_execution_v1",
        "proposal_lock_sha256": sha256_file(proposal_path),
        "status": "locked_after_G0_before_training_or_internal_A_scores",
    }
    write_json(target, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("proposal", "execution"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    value = freeze_proposal(config) if args.stage == "proposal" else freeze_execution(config)
    print(value["lock_id"])


if __name__ == "__main__":
    main()
