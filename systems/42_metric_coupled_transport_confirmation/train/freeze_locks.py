"""Create C42 proposal and execution locks."""

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
    "notes/confirmation_outcome.md",
}


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("C42 config must be an object")
    return value


def relative_repo(path: str | Path) -> str:
    resolved = Path(path).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError(f"C42 input outside repository: {path}")
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
        if relative not in EXCLUDED:
            output[relative] = sha256_file(path)
    return output


def checkpoint_paths(config: dict[str, Any]) -> list[Path]:
    output = []
    c41_root = Path(config["paths"]["c41_checkpoint_root"])
    for seed in config["checkpoints"]["c41_seeds"]:
        for mode in ("semantic_routing", "single_wide_routing", "asymmetric_routing", "coupled_content"):
            output.append(c41_root / f"seed_{seed}_{mode}.pt")
    c38_root = Path(config["paths"]["c38_checkpoint_root"])
    output.extend(
        c38_root / f"seed_{seed}_query_attended_unprojected.pt"
        for seed in config["checkpoints"]["c38_seeds"]
    )
    return output


def proposal_inputs(config: dict[str, Any]) -> list[str | Path]:
    paths = config["paths"]
    output: list[str | Path] = [
        paths["standardized_manifest"],
        paths["records_train"],
        paths["records_train_blind"],
        paths["candidate_manifest"],
        paths["c0_report"],
        paths["c1_report"],
        paths["trigger_report"],
        paths["c41_report"],
        paths["c38_report"],
        paths["c38_selection"],
        paths["selection"],
        paths["shared_metric_source"],
        "systems/41_semantic_carrier_routing_transformer/model/semantic_routing.py",
        "systems/38_cross_domain_global_tangent_transfer/model/global_tangent.py",
        "systems/38_cross_domain_global_tangent_transfer/train/features.py",
        "systems/38_cross_domain_global_tangent_transfer/train/selection.py",
    ]
    output.extend(checkpoint_paths(config))
    snapshot = Path(paths["bge_snapshot"])
    output.extend(
        snapshot / name
        for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt")
    )
    return output


def freeze_proposal(config: dict[str, Any]) -> dict[str, Any]:
    target = Path(config["paths"]["proposal_lock"])
    if target.exists():
        raise FileExistsError(target)
    trigger = read_json(config["paths"]["trigger_report"])
    selection = read_json(config["paths"]["selection"])
    c0 = read_json(config["paths"]["c0_report"])
    c1 = read_json(config["paths"]["c1_report"])
    if c0.get("overall_status") != "passed" or c1.get("overall_status") != "passed":
        raise PermissionError("C42 C0/C1 not passed")
    if trigger.get("status") != "passed_c42_trigger" or not all(trigger["checks"].values()):
        raise PermissionError("C42 trigger not passed")
    if sha256_file(config["paths"]["trigger_report"]) != config["paths"]["trigger_report_sha256"]:
        raise RuntimeError("C42 trigger changed")
    isolation = selection["outcome_isolation"]
    if isolation["internal_A_overlap_any_prior_feature_materialized"] != 0:
        raise PermissionError("C42 A was previously materialized")
    if selection["label_access"]["records_train_labels_opened"]:
        raise PermissionError("C42 labels opened before proposal")
    if sha256_file(
        REPO_ROOT / "systems/41_semantic_carrier_routing_transformer/model/semantic_routing.py"
    ) != config["integrity"]["c41_model_sha256"]:
        raise RuntimeError("C41 model source changed")
    for seed in config["checkpoints"]["c41_seeds"]:
        for mode in ("semantic_routing", "single_wide_routing", "asymmetric_routing", "coupled_content"):
            path = Path(config["paths"]["c41_checkpoint_root"]) / f"seed_{seed}_{mode}.pt"
            if sha256_file(path) != config["integrity"]["c41_checkpoint_sha256"][str(seed)][mode]:
                raise RuntimeError(f"C41 checkpoint changed before lock: {seed}/{mode}")
    value = {
        "candidate_id": "c42",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_id": "c42_metric_coupled_confirmation_v1",
        "status": "locked_before_features_scores_or_A_labels",
        "candidate_files_sha256": candidate_hashes(),
        "external_inputs_sha256": hashes(proposal_inputs(config)),
        "declarations": {
            "optimizer_steps": 0,
            "internal_A_features_scores_labels_opened": False,
            "dev_test_opened": False,
            "weights_preserved": True,
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
    g0 = read_json(config["paths"]["g0_report"])
    if not proposal.is_file() or g0.get("status") != "passed":
        raise PermissionError("C42 proposal/G0 missing")
    if g0["internal_A_scores_labels_opened"]:
        raise PermissionError("C42 A opened before execution lock")
    feature_root = Path(config["paths"]["feature_root"])
    inputs = sorted(path for path in feature_root.iterdir() if path.is_file())
    inputs.append(Path(config["paths"]["g0_report"]))
    value = {
        "candidate_id": "c42",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lock_id": "c42_metric_coupled_execution_v1",
        "status": "locked_after_G0_before_scores_or_A_labels",
        "proposal_lock_sha256": sha256_file(proposal),
        "execution_inputs_sha256": hashes(inputs),
        "declarations": {
            "optimizer_steps": 0,
            "internal_A_features_opened": True,
            "internal_A_scores_labels_opened": False,
            "dev_test_opened": False,
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
