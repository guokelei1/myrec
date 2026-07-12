"""Freeze and verify C55 proposal and fit-internal execution."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
FROZEN_FILES = (
    "README.md", "environment.txt", "configs/signal_gate.yaml",
    "notes/proposal.md", "notes/nearest_neighbors.md",
    "probe/__init__.py", "probe/locking.py", "probe/run_signal_gate.py",
    "tests/test_residual_target.py",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if value.get("candidate_id") != "c55":
        raise ValueError("not C55 config")
    return value


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def proposal_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    paths, integrity = config["paths"], config["integrity"]
    external = {
        paths["c54_config"]: integrity["c54_config_sha256"],
        paths["c54_proposal_lock"]: integrity["c54_proposal_lock_sha256"],
        paths["c54_execution_lock"]: integrity["c54_execution_lock_sha256"],
        paths["c54_model"]: integrity["c54_model_sha256"],
        paths["c54_runner"]: integrity["c54_runner_sha256"],
        paths["c47_selection"]: integrity["c47_selection_sha256"],
        paths["c34_selection"]: integrity["c34_selection_sha256"],
        paths["c38_config"]: integrity["c38_config_sha256"],
        paths["c38_selection"]: integrity["c38_selection_sha256"],
    }
    c54_proposal = json.loads((REPO_ROOT / paths["c54_proposal_lock"]).read_text(encoding="utf-8"))
    external.update(c54_proposal["external_inputs_sha256"])
    for name, expected in external.items():
        if sha256_file(REPO_ROOT / name) != expected:
            raise RuntimeError(f"C55 external changed: {name}")
    files = {name: sha256_file(SYSTEM_ROOT / name) for name in FROZEN_FILES}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    value: dict[str, Any] = {
        "candidate_id": "c55", "lock_id": "c55_proposal_v1",
        "status": "locked_before_c55_split_or_fit_labels",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_lock": commit, "files_sha256": files,
        "external_inputs_sha256": external,
        "declarations": {
            "C55_fit_labels_not_yet_read": True,
            "C53_A_reserve_dev_test_qrels_closed": True,
            "signal_falsifier_not_novelty_claim": True,
        },
    }
    value["aggregate_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


def verify_proposal(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = REPO_ROOT / config["paths"]["proposal_lock"]
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "locked_before_c55_split_or_fit_labels":
        raise RuntimeError("C55 proposal status differs")
    for name, expected in value["files_sha256"].items():
        if sha256_file(SYSTEM_ROOT / name) != expected:
            raise RuntimeError(f"C55 frozen source changed: {name}")
    for name, expected in value["external_inputs_sha256"].items():
        if sha256_file(REPO_ROOT / name) != expected:
            raise RuntimeError(f"C55 frozen external changed: {name}")
    return value, sha256_file(path)


def execution_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    _, proposal_hash = verify_proposal(config)
    selection_path = REPO_ROOT / config["paths"]["selection"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "label_free_split_frozen" or selection.get("fit_labels_read") is not False:
        raise RuntimeError("C55 label-free split invalid")
    if not all(selection["checks"].values()):
        raise RuntimeError("C55 selection checks failed")
    value: dict[str, Any] = {
        "candidate_id": "c55", "lock_id": "c55_execution_v1",
        "status": "locked_before_c55_fit_labels_or_training",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposal_lock_sha256": proposal_hash,
        "selection": {"path": str(selection_path.relative_to(REPO_ROOT)), "sha256": sha256_file(selection_path)},
        "declarations": {
            "C55_fit_labels_not_yet_read": True,
            "C53_A_reserve_dev_test_qrels_closed": True,
        },
    }
    value["aggregate_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


def verify_execution(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    verify_proposal(config)
    path = REPO_ROOT / config["paths"]["execution_lock"]
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "locked_before_c55_fit_labels_or_training":
        raise RuntimeError("C55 execution status differs")
    selection_path = REPO_ROOT / value["selection"]["path"]
    if sha256_file(selection_path) != value["selection"]["sha256"]:
        raise RuntimeError("C55 selection changed")
    return value, sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("proposal", "execution", "verify-proposal", "verify-execution"))
    args = parser.parse_args(); config = load_config(args.config)
    if args.stage == "proposal":
        value = proposal_payload(config); write_once(REPO_ROOT / config["paths"]["proposal_lock"], value)
    elif args.stage == "execution":
        value = execution_payload(config); write_once(REPO_ROOT / config["paths"]["execution_lock"], value)
    elif args.stage == "verify-proposal":
        value = verify_proposal(config)[0]
    else:
        value = verify_execution(config)[0]
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
