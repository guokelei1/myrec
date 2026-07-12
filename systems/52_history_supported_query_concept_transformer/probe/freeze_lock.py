"""Freeze or verify the C52 exposed formulation execution lock."""

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
    "README.md",
    "environment.txt",
    "configs/formulation_gate.yaml",
    "model/__init__.py",
    "model/concept_attention.py",
    "notes/proposal.md",
    "notes/reduction_audit.md",
    "notes/nearest_neighbors.md",
    "probe/__init__.py",
    "probe/freeze_lock.py",
    "probe/run_formulation_gate.py",
    "tests/test_operator.py",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if value.get("candidate_id") != "c52":
        raise ValueError("not a C52 config")
    return value


def payload(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_config(config_path)
    paths, integrity = config["paths"], config["integrity"]
    external = {
        paths["c47_selection"]: integrity["c47_selection_sha256"],
        paths["c47_report"]: integrity["c47_report_sha256"],
        paths["c47_kuai_scores"]: integrity["c47_kuai_scores_sha256"],
        paths["c47_amazon_scores"]: integrity["c47_amazon_scores_sha256"],
        paths["kuai_query_token_manifest"]: integrity["kuai_query_token_manifest_sha256"],
        paths["kuai_corpus"]: integrity["kuai_corpus_sha256"],
        paths["kuai_item_id2idx"]: integrity["kuai_item_id2idx_sha256"],
        str(Path(paths["kuai_bge_snapshot"]) / "model.safetensors"): integrity["kuai_model_sha256"],
        str(Path(paths["amazon_bge_snapshot"]) / "model.safetensors"): integrity["amazon_model_sha256"],
        str(Path(paths["kuai_bge_snapshot"]) / "tokenizer.json"): integrity["kuai_tokenizer_sha256"],
        str(Path(paths["kuai_bge_snapshot"]) / "config.json"): integrity["kuai_model_config_sha256"],
        str(Path(paths["amazon_bge_snapshot"]) / "tokenizer.json"): integrity["amazon_tokenizer_sha256"],
        str(Path(paths["amazon_bge_snapshot"]) / "config.json"): integrity["amazon_model_config_sha256"],
        paths["amazon_records_train"]: integrity["amazon_records_train_sha256"],
        str(Path(paths["kuai_query_tokens"]) / "train_input_ids.npy"): integrity["kuai_train_input_ids_sha256"],
        str(Path(paths["kuai_query_tokens"]) / "train_attention_mask.npy"): integrity["kuai_train_attention_mask_sha256"],
        paths["kuai_item_embeddings"]: integrity["kuai_item_embeddings_sha256"],
        paths["kuai_query_embeddings"]: integrity["kuai_query_embeddings_sha256"],
        str(Path(paths["kuai_packed_root"]) / "request_ids.jsonl"): integrity["kuai_request_ids_sha256"],
        str(Path(paths["kuai_packed_root"]) / "query_indices.npy"): integrity["kuai_query_indices_sha256"],
        str(Path(paths["kuai_packed_root"]) / "candidate_offsets.npy"): integrity["kuai_candidate_offsets_sha256"],
        str(Path(paths["kuai_packed_root"]) / "candidate_embedding_indices.npy"): integrity["kuai_candidate_indices_sha256"],
        str(Path(paths["kuai_packed_root"]) / "candidate_item_ids.npy"): integrity["kuai_candidate_item_ids_sha256"],
        str(Path(paths["kuai_packed_root"]) / "history_offsets.npy"): integrity["kuai_history_offsets_sha256"],
        str(Path(paths["kuai_packed_root"]) / "history_embedding_indices.npy"): integrity["kuai_history_indices_sha256"],
        paths["amazon_adapter_selection"]: integrity["amazon_adapter_selection_sha256"],
        str(Path(paths["amazon_feature_root"]) / "feature_manifest.json"): integrity["amazon_feature_manifest_sha256"],
        str(Path(paths["amazon_feature_root"]) / "embedding_manifest.json"): integrity["amazon_embedding_manifest_sha256"],
        str(Path(paths["amazon_feature_root"]) / "items.jsonl"): integrity["amazon_items_sha256"],
        str(Path(paths["amazon_feature_root"]) / "requests.jsonl"): integrity["amazon_requests_sha256"],
        str(Path(paths["amazon_feature_root"]) / "item_embeddings.npy"): integrity["amazon_item_embeddings_sha256"],
        str(Path(paths["amazon_feature_root"]) / "query_embeddings.npy"): integrity["amazon_query_embeddings_sha256"],
        str(Path(paths["amazon_feature_root"]) / "candidate_offsets.npy"): integrity["amazon_candidate_offsets_sha256"],
        str(Path(paths["amazon_feature_root"]) / "candidate_item_positions.npy"): integrity["amazon_candidate_positions_sha256"],
        str(Path(paths["amazon_feature_root"]) / "true_history_offsets.npy"): integrity["amazon_true_history_offsets_sha256"],
        str(Path(paths["amazon_feature_root"]) / "true_history_item_positions.npy"): integrity["amazon_true_history_positions_sha256"],
        str(Path(paths["amazon_feature_root"]) / "wrong_history_offsets.npy"): integrity["amazon_wrong_history_offsets_sha256"],
        str(Path(paths["amazon_feature_root"]) / "wrong_history_item_positions.npy"): integrity["amazon_wrong_history_positions_sha256"],
    }
    for path, expected in external.items():
        if sha256_file(REPO_ROOT / path) != expected:
            raise RuntimeError(f"C52 external source changed: {path}")
    files = {name: sha256_file(SYSTEM_ROOT / name) for name in FROZEN_FILES}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True,
        capture_output=True,
    ).stdout.strip()
    value: dict[str, Any] = {
        "candidate_id": "c52",
        "lock_id": config["gate_id"],
        "status": "locked_before_any_c52_token_feature_or_score",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_lock": commit,
        "files_sha256": files,
        "external_inputs_sha256": external,
        "declarations": {
            "c47_A_labels_already_exposed": True,
            "c47_A_used_only_after_this_lock": True,
            "kuai_reserve_opened": False,
            "amazon_reserve_opened": False,
            "dev_test_qrels_or_metrics_read": False,
            "training_authorized": False,
        },
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["aggregate_sha256"] = hashlib.sha256(canonical).hexdigest()
    return value


def freeze(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    target = REPO_ROOT / config["paths"]["execution_lock"]
    if target.exists():
        raise FileExistsError(target)
    value = payload(config_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def verify(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    target = REPO_ROOT / config["paths"]["execution_lock"]
    value = json.loads(target.read_text(encoding="utf-8"))
    if value.get("candidate_id") != "c52" or value.get("status") != "locked_before_any_c52_token_feature_or_score":
        raise RuntimeError("C52 execution lock state differs")
    for name, expected in value["files_sha256"].items():
        if sha256_file(SYSTEM_ROOT / name) != expected:
            raise RuntimeError(f"C52 frozen source changed: {name}")
    for name, expected in value["external_inputs_sha256"].items():
        if sha256_file(REPO_ROOT / name) != expected:
            raise RuntimeError(f"C52 frozen external changed: {name}")
    return value, sha256_file(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    value = verify(config)[0] if args.verify else freeze(args.config)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
