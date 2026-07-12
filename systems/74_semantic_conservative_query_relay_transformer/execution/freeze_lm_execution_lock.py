"""Freeze C74 pretrained-LM training, scoring, and data after passed G0."""

from __future__ import annotations

import json
from pathlib import Path
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

from execution.lm_locking import (  # noqa: E402
    atomic_json,
    load_config,
    sha256_file,
    timestamp,
    verify_g0_lock,
)


SOURCES = (
    "systems/74_semantic_conservative_query_relay_transformer/configs/kuai_lm_probe.yaml",
    "systems/74_semantic_conservative_query_relay_transformer/model/adaptive_semantic_relay.py",
    "systems/74_semantic_conservative_query_relay_transformer/train/data_bridge.py",
    "systems/74_semantic_conservative_query_relay_transformer/train/gate_metrics.py",
    "systems/74_semantic_conservative_query_relay_transformer/execution/lm_locking.py",
    "systems/74_semantic_conservative_query_relay_transformer/execution/freeze_lm_execution_lock.py",
    "systems/74_semantic_conservative_query_relay_transformer/execution/run_lm_probe.py",
    "src/myrec/eval/metrics.py",
)


def data_paths(config: dict) -> tuple[str, ...]:
    artifact = config["paths"]["c26_artifact_root"]
    packed = config["paths"]["packed_train_root"]
    snapshot = config["paths"]["bge_snapshot"]
    return (
        config["paths"]["c26_selection"],
        f"{artifact}/feature_request_indices.npy",
        f"{artifact}/feature_candidate_offsets.npy",
        f"{artifact}/base_scores.npy",
        f"{artifact}/item_embedding_indices.npy",
        f"{artifact}/item_token_ids.npy",
        f"{artifact}/item_attention_mask.npy",
        f"{artifact}/item_content_mask.npy",
        f"{artifact}/query_token_ids.npy",
        f"{artifact}/query_attention_mask.npy",
        f"{artifact}/query_content_mask.npy",
        f"{artifact}/fit_request_indices.npy",
        f"{artifact}/fit_label_offsets.npy",
        f"{artifact}/fit_labels.npy",
        f"{packed}/request_ids.jsonl",
        f"{packed}/candidate_offsets.npy",
        f"{packed}/candidate_embedding_indices.npy",
        f"{packed}/candidate_item_ids.npy",
        f"{packed}/history_offsets.npy",
        f"{packed}/history_embedding_indices.npy",
        f"{packed}/history_event_weights.npy",
        f"{snapshot}/config.json",
        f"{snapshot}/model.safetensors",
        f"{config['paths']['artifact_root']}/split_manifest.json",
        config["paths"]["lm_probe_g0"],
        config["paths"]["lm_probe_g0_lock"],
        config["paths"]["design_gate_report"],
        config["paths"]["design_proposal_lock"],
        "systems/64_end_to_end_lm_representation_probe/train/data.py",
    )


def main() -> None:
    config = load_config()
    _, g0_lock_hash = verify_g0_lock(config)
    g0_path = REPO_ROOT / config["paths"]["lm_probe_g0"]
    g0 = json.loads(g0_path.read_text(encoding="utf-8"))
    if g0["status"] != "passed" or g0["g0_lock_sha256"] != g0_lock_hash:
        raise PermissionError("C74 execution lock requires passed LM G0")
    target = REPO_ROOT / config["paths"]["lm_probe_execution_lock"]
    value = {
        "candidate_id": "c74",
        "created_at": timestamp(),
        "decision": "authorize_three_registered_exposed_fit_pretrained_lm_seeds",
        "g0_lock_sha256": g0_lock_hash,
        "g0_report_sha256": sha256_file(g0_path),
        "source_sha256": {
            relative: sha256_file(REPO_ROOT / relative) for relative in SOURCES
        },
        "data_sha256": {
            relative: sha256_file(REPO_ROOT / relative)
            for relative in data_paths(config)
        },
        "outcome_boundary": {
            "fit_train_labels_authorized": True,
            "validation_labels_authorized_before_A0": False,
            "fresh_features_scores_labels_opened": False,
            "dev_test_qrels_opened": False,
        },
    }
    atomic_json(target, value)
    print(json.dumps({"path": str(target), "sha256": sha256_file(target)}))


if __name__ == "__main__":
    main()
