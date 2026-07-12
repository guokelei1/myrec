"""Freeze the C44 data-free design before observing the synthetic outcome."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
EXCLUDED = {"notes/design_lock.json", "notes/design_gate_outcome.md"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: str | Path, value: dict) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(target)


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


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or value.get("candidate_id") != "c44":
        raise ValueError("unexpected C44 config")
    return value


def freeze(config_path: str | Path) -> dict:
    config = load_config(config_path)
    if Path(config["paths"]["report"]).exists():
        raise PermissionError("C44 outcome exists before design lock")
    forbidden = ("data/", "records_", "qrels", "candidate_labels", "runs/")
    for name, raw in config["paths"].items():
        if name in {"artifact_root", "report"}:
            continue
        if any(token in str(raw).lower() for token in forbidden):
            raise PermissionError(f"C44 data-free config has forbidden path: {name}")
    c43_path = Path(config["paths"]["c43_report"])
    if sha256_file(c43_path) != config["integrity"]["c43_report_sha256"]:
        raise RuntimeError("C43 trigger report changed")
    c43 = json.loads(c43_path.read_text(encoding="utf-8"))
    if c43.get("status") != "failed_A1_terminal":
        raise PermissionError("C43 terminal trigger differs")
    if c43["A1"]["checks"].get("true_over_wrong_ci") is not False:
        raise PermissionError("C43 specificity failure trigger differs")
    files = candidate_hashes()
    lines = [f"{value}  {name}\n" for name, value in sorted(files.items())]
    external = {
        str(c43_path): sha256_file(c43_path),
        str(config["paths"]["shared_metric_source"]): sha256_file(
            config["paths"]["shared_metric_source"]
        ),
    }
    value = {
        "candidate_id": "c44",
        "lock_id": "c44_partial_evidence_logit_flow_design_v1",
        "status": "locked_before_data_free_outcome",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_lock": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "files_sha256": files,
        "aggregate_sha256": hashlib.sha256("".join(lines).encode()).hexdigest(),
        "external_inputs_sha256": external,
        "declarations": {
            "repository_dataset_read": False,
            "train_labels_read": False,
            "dev_test_qrels_read": False,
            "synthetic_outcome_observed": False,
            "optimizer_steps": 0,
        },
    }
    write_once(config["paths"]["design_lock"], value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    value = freeze(args.config)
    print(value["lock_id"])


if __name__ == "__main__":
    main()
