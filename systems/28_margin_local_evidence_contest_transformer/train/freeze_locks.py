"""Create immutable C28 proposal and post-G0 execution locks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from train.structure import load_config, read_json, sha256_file, write_json_once  # noqa: E402


EXCLUDED_CANDIDATE_FILES = {
    "notes/proposal_lock.json",
    "notes/execution_lock.json",
    "notes/train_gate_outcome.md",
}


def relative_repo(path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))


def candidate_hashes() -> dict[str, str]:
    output: dict[str, str] = {}
    for path in sorted(SYSTEM_ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = str(path.relative_to(SYSTEM_ROOT))
        if relative in EXCLUDED_CANDIDATE_FILES:
            continue
        output[relative] = sha256_file(path)
    return output


def external_paths(config: dict[str, Any]) -> list[Path]:
    paths = config["paths"]
    selected = [
        paths["selection"],
        paths["c26_selection"],
        paths["c27_selection"],
        paths["c27_g0_report"],
        paths["c27_train_report"],
        paths["d2_config"],
        paths["candidate_manifest"],
        paths["packed_manifest"],
        paths["query_token_manifest"],
        paths["raw_item_embeddings"],
        paths["corpus"],
        paths["item_id2idx"],
        paths["calibration_checkpoint"],
        paths["internal_train_popularity"],
        paths["train_candidate_labels"],
        paths["shared_metric_source"],
        "src/myrec/analysis/finetuned_query_tower.py",
    ]
    packed = Path(paths["packed_train_root"])
    selected.extend(
        packed / name
        for name in (
            "request_ids.jsonl",
            "candidate_offsets.npy",
            "candidate_embedding_indices.npy",
            "candidate_item_ids.npy",
            "history_offsets.npy",
            "history_embedding_indices.npy",
            "history_event_weights.npy",
            "timestamps.npy",
        )
    )
    query_root = Path(paths["query_tokens"])
    selected.extend(query_root / name for name in ("train_input_ids.npy", "train_attention_mask.npy"))
    c27_root = Path(paths["c27_artifact_root"])
    selected.extend(
        c27_root / name
        for name in ("fit_request_indices.npy", "fit_label_offsets.npy", "fit_labels.npy")
    )
    selected.extend(path for path in Path(paths["bge_snapshot"]).rglob("*") if path.is_file())
    result = sorted({Path(path).resolve() for path in selected})
    outside = [path for path in result if REPO_ROOT.resolve() not in path.parents]
    if outside:
        raise ValueError(f"C28 external input outside repository: {outside}")
    return result


def freeze_proposal(config: dict[str, Any]) -> dict[str, Any]:
    target = Path(config["paths"]["proposal_lock"])
    files = candidate_hashes()
    lines = [f"{value}  {name}\n" for name, value in sorted(files.items())]
    external = {relative_repo(path): sha256_file(path) for path in external_paths(config)}
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    value = {
        "aggregate_sha256": hashlib.sha256("".join(lines).encode()).hexdigest(),
        "candidate_id": "c28",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "declarations": {
            "c27_fit_labels_previously_opened": True,
            "c27_escrow_features_or_labels_opened": False,
            "c28_delayed_B_labels_opened": False,
            "c28_internal_A_labels_opened": False,
            "c28_ranking_outcome_observed": False,
            "dev_test_qrels_or_metrics_read": False,
            "escrow_opened": False,
        },
        "external_inputs_sha256": external,
        "files_sha256": files,
        "git_commit_at_lock": commit,
        "lock_id": "c28_margin_local_evidence_contest_v1",
        "selection_path": relative_repo(config["paths"]["selection"]),
        "selection_sha256": config["paths"]["selection_sha256"],
        "status": "locked_before_any_c28_internal_or_delayed_label_or_outcome",
    }
    write_json_once(target, value)
    return value


def freeze_execution(config: dict[str, Any]) -> dict[str, Any]:
    proposal = Path(config["paths"]["proposal_lock"])
    report_path = Path(config["paths"]["artifact_root"]) / "g0_report.json"
    report = read_json(report_path)
    if report.get("status") != "passed" or report.get("internal_A_labels_opened") is not False:
        raise PermissionError("C28 G0 is not a closed-label pass")
    value = {
        "candidate_id": "c28",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "declarations": {
            "delayed_B_labels_opened": False,
            "escrow_dev_test_opened": False,
            "fit_labels_reused_only_from_frozen_c27_compact": True,
            "internal_A_labels_opened": False,
            "ranking_outcome_observed": False,
            "training_started": False,
        },
        "g0_outputs_sha256": {
            name: metadata["sha256"] for name, metadata in report["outputs"].items()
        },
        "g0_report_path": relative_repo(report_path),
        "g0_report_sha256": sha256_file(report_path),
        "lock_id": "c28_margin_local_evidence_contest_execution_v1",
        "proposal_lock_sha256": sha256_file(proposal),
        "selection_sha256": config["paths"]["selection_sha256"],
        "status": "locked_after_G0_before_training_or_internal_A_access",
    }
    write_json_once(config["paths"]["execution_lock"], value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("proposal", "execution"), required=True)
    args = parser.parse_args()
    config = load_config(args.config, require_selection=True)
    value = freeze_proposal(config) if args.stage == "proposal" else freeze_execution(config)
    print(value["lock_id"])


if __name__ == "__main__":
    main()
