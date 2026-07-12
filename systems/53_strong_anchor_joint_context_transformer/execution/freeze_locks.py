"""Freeze and verify C53 proposal/execution locks."""

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
    "README.md", "environment.txt", "configs/foundation_gate.yaml",
    "model/__init__.py", "model/joint_context.py", "notes/proposal.md",
    "notes/nearest_neighbors.md", "execution/__init__.py",
    "execution/freeze_locks.py", "execution/run_gate.py", "tests/test_model.py",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if value.get("candidate_id") != "c53":
        raise ValueError("not C53 config")
    return value


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def manifest_inputs(path: Path, expected_manifest_hash: str) -> dict[str, str]:
    """Bind a feature manifest and every concrete file it names."""
    if sha256_file(path) != expected_manifest_hash:
        raise RuntimeError(f"C53 feature manifest changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    output = {str(path.relative_to(REPO_ROOT)): expected_manifest_hash}
    for row in value["files"].values():
        source = Path(row["path"])
        if not source.is_absolute():
            source = REPO_ROOT / source
        source = source.resolve()
        try:
            relative = source.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(f"C53 manifest input leaves repository: {source}") from exc
        if sha256_file(source) != row["sha256"]:
            raise RuntimeError(f"C53 manifest member changed: {source}")
        output[str(relative)] = row["sha256"]
    return output


def proposal_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    paths, integrity = config["paths"], config["integrity"]
    external = {
        paths["c47_selection"]: integrity["c47_selection_sha256"],
        paths["c47_amazon_adapter_selection"]: integrity["c47_amazon_adapter_selection_sha256"],
        paths["c38_config"]: integrity["c38_config_sha256"],
        paths["c38_selection"]: integrity["c38_selection_sha256"],
        paths["amazon_records_train"]: integrity["amazon_records_train_sha256"],
        paths["d2_config"]: integrity["d2_config_sha256"],
        paths["d2_final_config"]: integrity["d2_final_config_sha256"],
        paths["d2_checkpoint"]: integrity["d2_checkpoint_sha256"],
        paths["kuai_popularity"]: integrity["kuai_popularity_sha256"],
        paths["kuai_candidate_labels"]: integrity["kuai_candidate_labels_sha256"],
        str(Path(paths["kuai_query_tokens"]) / "train_input_ids.npy"): integrity["kuai_train_input_ids_sha256"],
        str(Path(paths["kuai_query_tokens"]) / "train_attention_mask.npy"): integrity["kuai_train_attention_mask_sha256"],
        paths["kuai_item_embeddings"]: integrity["kuai_item_embeddings_sha256"],
        str(Path(paths["kuai_packed_root"]) / "request_ids.jsonl"): integrity["kuai_request_ids_sha256"],
        str(Path(paths["kuai_packed_root"]) / "query_indices.npy"): integrity["kuai_query_indices_sha256"],
        str(Path(paths["kuai_packed_root"]) / "candidate_offsets.npy"): integrity["kuai_candidate_offsets_sha256"],
        str(Path(paths["kuai_packed_root"]) / "candidate_embedding_indices.npy"): integrity["kuai_candidate_indices_sha256"],
        str(Path(paths["kuai_packed_root"]) / "candidate_item_ids.npy"): integrity["kuai_candidate_item_ids_sha256"],
        str(Path(paths["kuai_packed_root"]) / "history_offsets.npy"): integrity["kuai_history_offsets_sha256"],
        str(Path(paths["kuai_packed_root"]) / "history_embedding_indices.npy"): integrity["kuai_history_indices_sha256"],
    }
    external.update(manifest_inputs(
        REPO_ROOT / paths["c38_feature_root"] / "feature_manifest.json",
        integrity["c38_feature_manifest_sha256"],
    ))
    external.update(manifest_inputs(
        REPO_ROOT / paths["c38_feature_root"] / "embedding_manifest.json",
        integrity["c38_embedding_manifest_sha256"],
    ))
    external.update(manifest_inputs(
        REPO_ROOT / paths["c47_amazon_feature_root"] / "feature_manifest.json",
        integrity["c47_amazon_feature_manifest_sha256"],
    ))
    external.update(manifest_inputs(
        REPO_ROOT / paths["c47_amazon_feature_root"] / "embedding_manifest.json",
        integrity["c47_amazon_embedding_manifest_sha256"],
    ))
    for name, expected in external.items():
        if sha256_file(REPO_ROOT / name) != expected:
            raise RuntimeError(f"C53 external changed: {name}")
    files = {name: sha256_file(SYSTEM_ROOT / name) for name in FROZEN_FILES}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True).stdout.strip()
    value: dict[str, Any] = {
        "candidate_id": "c53", "lock_id": "c53_proposal_v1",
        "status": "locked_before_c53_feature_or_outcome",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_lock": commit, "files_sha256": files,
        "external_inputs_sha256": external,
        "declarations": {
            "known_foundation_not_novelty_claim": True,
            "fit_labels_not_yet_opened_by_c53": True,
            "exposed_A_labels_not_used_by_c53": True,
            "fresh_reserve_dev_test_qrels_closed": True,
        },
    }
    value["aggregate_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return value


def verify_proposal(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = REPO_ROOT / config["paths"]["proposal_lock"]
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "locked_before_c53_feature_or_outcome":
        raise RuntimeError("C53 proposal lock status differs")
    for name, expected in value["files_sha256"].items():
        if sha256_file(SYSTEM_ROOT / name) != expected:
            raise RuntimeError(f"C53 frozen source changed: {name}")
    for name, expected in value["external_inputs_sha256"].items():
        if sha256_file(REPO_ROOT / name) != expected:
            raise RuntimeError(f"C53 frozen external changed: {name}")
    return value, sha256_file(path)


def execution_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    _, proposal_hash = verify_proposal(config)
    report_path = REPO_ROOT / config["paths"]["materialization_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("A_labels_read") is not False:
        raise RuntimeError("C53 materialization did not preserve A barrier")
    for row in report["outputs"].values():
        if sha256_file(REPO_ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError("C53 materialized output changed")
    value: dict[str, Any] = {
        "candidate_id": "c53", "lock_id": "c53_execution_v1",
        "status": "locked_before_c53_training_or_A_score",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposal_lock_sha256": proposal_hash,
        "materialization_report": {"path": str(report_path.relative_to(REPO_ROOT)), "sha256": sha256_file(report_path)},
        "materialized_outputs_sha256": {name: row["sha256"] for name, row in report["outputs"].items()},
        "declarations": {
            "fit_labels_materialized": True, "A_labels_used": False,
            "fresh_reserve_dev_test_qrels_closed": True,
        },
    }
    value["aggregate_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return value


def verify_execution(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    verify_proposal(config)
    path = REPO_ROOT / config["paths"]["execution_lock"]
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "locked_before_c53_training_or_A_score":
        raise RuntimeError("C53 execution lock status differs")
    report_path = REPO_ROOT / value["materialization_report"]["path"]
    if sha256_file(report_path) != value["materialization_report"]["sha256"]:
        raise RuntimeError("C53 materialization report changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for name, expected in value["materialized_outputs_sha256"].items():
        if sha256_file(REPO_ROOT / report["outputs"][name]["path"]) != expected:
            raise RuntimeError(f"C53 materialized input changed: {name}")
    return value, sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("proposal", "execution", "verify-proposal", "verify-execution"))
    args = parser.parse_args()
    config = load_config(args.config)
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
