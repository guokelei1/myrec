"""Create immutable C41 proposal and execution locks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from train.store import read_json, sha256_file, write_json  # noqa: E402


EXCLUDED = {
    "notes/proposal_lock.json",
    "notes/execution_lock.json",
    "notes/train_gate_outcome.md",
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("C41 config must be an object")
    return value


def relative_repo(path: str | Path) -> str:
    resolved = Path(path).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError(f"C41 lock input outside repository: {path}")
    return str(resolved.relative_to(REPO_ROOT.resolve()))


def hashes(paths: Iterable[str | Path]) -> dict[str, str]:
    output = {}
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(path)
        output[relative_repo(path)] = sha256_file(path)
    return dict(sorted(output.items()))


def candidate_hashes() -> dict[str, str]:
    output = {}
    for path in sorted(SYSTEM_ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = str(path.relative_to(SYSTEM_ROOT))
        if relative in EXCLUDED:
            continue
        output[relative] = sha256_file(path)
    return output


def proposal_inputs(config: dict[str, Any]) -> list[str]:
    paths = config["paths"]
    output = [
        paths["standardized_manifest"],
        paths["records_train"],
        paths["records_train_blind"],
        paths["candidate_manifest"],
        paths["c0_report"],
        paths["c1_report"],
        paths["design_gate_report"],
        paths["c40_report"],
        paths["c38_report"],
        paths["c38_selection"],
        paths["c39_selection"],
        paths["selection"],
        paths["shared_metric_source"],
        "systems/38_cross_domain_global_tangent_transfer/model/global_tangent.py",
        "systems/38_cross_domain_global_tangent_transfer/train/features.py",
        "systems/38_cross_domain_global_tangent_transfer/train/selection.py",
    ]
    snapshot = Path(paths["bge_snapshot"])
    output.extend(
        str(snapshot / name)
        for name in (
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        )
    )
    root = Path(paths["c38_checkpoint_root"])
    output.extend(
        str(root / f"seed_{seed}_query_attended_unprojected.pt")
        for seed in config["training"]["c38_control_seeds"]
    )
    return output


def freeze_proposal(config: dict[str, Any]) -> dict[str, Any]:
    target = Path(config["paths"]["proposal_lock"])
    if target.exists():
        raise FileExistsError(target)
    c0 = read_json(config["paths"]["c0_report"])
    c1 = read_json(config["paths"]["c1_report"])
    design = read_json(config["paths"]["design_gate_report"])
    if c0.get("overall_status") != "passed" or c1.get("overall_status") != "passed":
        raise PermissionError("C41 C0/C1 not passed")
    if design.get("status") != "passed_design_gate":
        raise PermissionError("C41 design gate not passed")
    if sha256_file(config["paths"]["design_gate_report"]) != config["paths"]["design_gate_report_sha256"]:
        raise RuntimeError("C41 design report changed")
    selection = read_json(config["paths"]["selection"])
    isolation = selection["outcome_isolation"]
    if any(
        isolation[key] != 0
        for key in (
            "internal_A_overlap_c38_internal_A",
            "internal_A_overlap_c39_internal_A",
            "internal_A_overlap_c38_feature_materialized",
            "delayed_B_overlap_any_prior_feature_materialized",
        )
    ):
        raise PermissionError("C41 cohort isolation failed")
    if selection["label_access"]["records_train_labels_opened"]:
        raise PermissionError("C41 labels opened before proposal")
    value = {
        "candidate_id": "c41",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_id": "c41_semantic_carrier_routing_v1",
        "status": "locked_before_features_fit_labels_training_or_A_scores",
        "candidate_files_sha256": candidate_hashes(),
        "external_inputs_sha256": hashes(proposal_inputs(config)),
        "declarations": {
            "design_gate_passed": True,
            "novelty_status": "boundary_only",
            "source_fit_labels_previously_training_authorized": True,
            "c41_fit_label_artifact_opened": False,
            "internal_A_features_scores_labels_opened": False,
            "delayed_B_features_scores_labels_opened": False,
            "dev_test_opened": False,
            "ranking_outcome_observed": False,
        },
        "git_commit_at_lock": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
    }
    write_json(target, value)
    return value


def freeze_execution(config: dict[str, Any]) -> dict[str, Any]:
    target = Path(config["paths"]["execution_lock"])
    if target.exists():
        raise FileExistsError(target)
    proposal = Path(config["paths"]["proposal_lock"])
    if not proposal.is_file():
        raise PermissionError("C41 proposal missing")
    g0 = read_json(config["paths"]["g0_report"])
    if g0.get("status") != "passed":
        raise PermissionError("C41 G0 not passed")
    if g0.get("internal_A_labels_scores_opened") is not False:
        raise PermissionError("C41 A opened before execution lock")
    feature_root = Path(config["paths"]["feature_root"])
    inputs = sorted(path for path in feature_root.iterdir() if path.is_file())
    inputs.extend([Path(config["paths"]["fit_labels"]), Path(config["paths"]["g0_report"])])
    value = {
        "candidate_id": "c41",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_id": "c41_semantic_carrier_execution_v1",
        "status": "locked_after_G0_before_training_or_A_scores",
        "proposal_lock_sha256": sha256_file(proposal),
        "execution_inputs_sha256": hashes(str(path) for path in inputs),
        "declarations": {
            "fit_labels_opened_after_proposal": True,
            "internal_A_label_free_features_opened": True,
            "internal_A_labels_scores_opened": False,
            "delayed_B_features_scores_labels_opened": False,
            "dev_test_opened": False,
            "training_started": False,
        },
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
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
