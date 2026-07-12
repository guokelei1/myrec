"""Freeze the weights-preserving C30 continuation before canonical scoring."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT / "train"))

from c30_protocol import (  # noqa: E402
    candidate_hashes,
    load_config,
    read_json,
    sha256_file,
    write_json_once,
)


def relative_repo(path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["paths"]["artifact_root"])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("C30 score output exists before lock")
    paths = config["paths"]
    registered = [
        paths["c29_config"],
        paths["c29_proposal_lock"],
        paths["c29_execution_lock"],
        paths["c29_g0_report"],
        paths["c29_train_report"],
        paths["c29_selection"],
        paths["train_candidate_labels"],
        paths["shared_metric_source"],
    ]
    for row in config["source_seeds"].values():
        registered.extend((row["report"], row["scores"], row["checkpoint"]))
        for name in ("report", "scores", "checkpoint"):
            if sha256_file(row[name]) != row[f"{name}_sha256"]:
                raise RuntimeError(f"C30 registered seed input changed: {name}")
    for name in (
        "c29_config",
        "c29_proposal_lock",
        "c29_execution_lock",
        "c29_g0_report",
        "c29_train_report",
        "c29_selection",
    ):
        if sha256_file(paths[name]) != paths[f"{name}_sha256"]:
            raise RuntimeError(f"C30 registered source changed: {name}")
    report = read_json(paths["c29_train_report"])
    failed = [name for name, passed in report["A0"]["checks"].items() if not passed]
    if report.get("status") != "failed_A0_terminal" or failed != ["candidate_permutation"]:
        raise PermissionError("C30 source is not the one-failure C29 terminal state")
    if report.get("internal_A_labels_opened") is not False:
        raise PermissionError("C30 source A labels were opened")
    packed = Path(paths["packed_train_root"])
    registered.extend(
        packed / name
        for name in ("candidate_offsets.npy", "candidate_item_ids.npy", "request_ids.jsonl")
    )
    external = {
        relative_repo(path): sha256_file(path) for path in sorted({Path(p) for p in registered})
    }
    files = candidate_hashes()
    lines = [f"{value}  {name}\n" for name, value in sorted(files.items())]
    value = {
        "candidate_id": "c30",
        "lock_id": "c30_canonical_order_continuation_v1",
        "status": "locked_before_c30_canonical_score_or_A_label",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_lock": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "aggregate_sha256": hashlib.sha256("".join(lines).encode()).hexdigest(),
        "files_sha256": files,
        "external_inputs_sha256": external,
        "declarations": {
            "c29_internal_A_scores_opened": True,
            "c29_internal_A_labels_opened": False,
            "c30_canonical_scores_opened": False,
            "c30_internal_A_labels_opened": False,
            "weights_changed": False,
            "optimizer_steps_authorized": 0,
            "threshold_changed": False,
            "delayed_B_escrow_dev_test_opened": False,
        },
    }
    write_json_once(paths["continuation_lock"], value)
    print(value["lock_id"])


if __name__ == "__main__":
    main()
