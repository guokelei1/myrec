"""Freeze C66 training/evaluation sources and consumed data after G0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

from execution.locking import (  # noqa: E402
    atomic_json,
    load_config,
    sha256_file,
    timestamp,
    verify_g0_lock,
)


SOURCES = (
    "systems/66_canonical_counterfactual_residual_state_transformer/configs/train_gate.yaml",
    "systems/66_canonical_counterfactual_residual_state_transformer/model/__init__.py",
    "systems/66_canonical_counterfactual_residual_state_transformer/model/canonical_residual.py",
    "systems/66_canonical_counterfactual_residual_state_transformer/train/__init__.py",
    "systems/66_canonical_counterfactual_residual_state_transformer/train/data_bridge.py",
    "systems/66_canonical_counterfactual_residual_state_transformer/train/gate_metrics.py",
    "systems/66_canonical_counterfactual_residual_state_transformer/execution/locking.py",
    "systems/66_canonical_counterfactual_residual_state_transformer/execution/run_gate.py",
    "systems/66_canonical_counterfactual_residual_state_transformer/tests/test_model.py",
    "systems/65_counterfactual_residual_state_transformer/model/counterfactual_residual.py",
    "systems/64_end_to_end_lm_representation_probe/model/adaptive_joint_ranker.py",
    "systems/64_end_to_end_lm_representation_probe/train/data.py",
    "src/myrec/eval/metrics.py",
)


def data_paths(config: dict) -> tuple[str, ...]:
    artifact = config["paths"]["c26_artifact_root"]
    packed = config["paths"]["packed_train_root"]
    snapshot = config["paths"]["bge_snapshot"]
    return (
        config["paths"]["c26_selection"],
        config["paths"]["c64_split_manifest"],
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
        f"{config['paths']['artifact_root']}/g0_report.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=SYSTEM_ROOT / "configs/train_gate.yaml"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    _, g0_lock_hash = verify_g0_lock(config)
    g0_path = REPO_ROOT / config["paths"]["artifact_root"] / "g0_report.json"
    g0 = json.loads(g0_path.read_text(encoding="utf-8"))
    if g0["status"] != "passed" or g0["g0_lock_sha256"] != g0_lock_hash:
        raise PermissionError("C66 execution lock requires passed G0")
    target = REPO_ROOT / config["paths"]["execution_lock"]
    value = {
        "candidate_id": "c66",
        "created_at": timestamp(),
        "decision": "authorize_three_registered_exposed_fit_seeds",
        "g0_lock_sha256": g0_lock_hash,
        "g0_report_sha256": sha256_file(g0_path),
        "source_sha256": {
            source: sha256_file(REPO_ROOT / source) for source in SOURCES
        },
        "data_sha256": {
            source: sha256_file(REPO_ROOT / source)
            for source in data_paths(config)
        },
        "outcome_boundary": {
            "fit_train_labels_authorized": True,
            "validation_labels_authorized_before_A0": False,
            "fresh_features_scores_labels_opened": False,
            "dev_test_qrels_opened": False,
        },
    }
    atomic_json(target, value)
    print(target.relative_to(REPO_ROOT))
    print(sha256_file(target))


if __name__ == "__main__":
    main()
