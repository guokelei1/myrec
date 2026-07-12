from __future__ import annotations

import argparse
from pathlib import Path
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))
from execution.locking import (  # noqa: E402
    atomic_json, load_config, sha256_file, timestamp, verify_proposal_lock,
)


SOURCES = (
    "systems/69_semantic_null_behavior_relation_probe/configs/signal_gate.yaml",
    "systems/69_semantic_null_behavior_relation_probe/model/__init__.py",
    "systems/69_semantic_null_behavior_relation_probe/model/behavior_relation.py",
    "systems/69_semantic_null_behavior_relation_probe/execution/__init__.py",
    "systems/69_semantic_null_behavior_relation_probe/execution/locking.py",
    "systems/69_semantic_null_behavior_relation_probe/execution/runtime.py",
    "systems/69_semantic_null_behavior_relation_probe/execution/freeze_proposal_lock.py",
    "systems/69_semantic_null_behavior_relation_probe/execution/freeze_execution_lock.py",
    "systems/69_semantic_null_behavior_relation_probe/execution/run_seed.py",
    "systems/69_semantic_null_behavior_relation_probe/execution/aggregate_gate.py",
    "systems/69_semantic_null_behavior_relation_probe/notes/preoutcome_import_amendment.md",
    "systems/69_semantic_null_behavior_relation_probe/tests/__init__.py",
    "systems/69_semantic_null_behavior_relation_probe/tests/test_model.py",
    "systems/47_posterior_supported_ridge_transformer/configs/signal_gate_v1.yaml",
    "systems/47_posterior_supported_ridge_transformer/probe/run_signal_gate.py",
    "systems/47_posterior_supported_ridge_transformer/probe/data.py",
    "systems/38_cross_domain_global_tangent_transfer/configs/train_gate.yaml",
    "systems/38_cross_domain_global_tangent_transfer/train/store.py",
    "systems/38_cross_domain_global_tangent_transfer/train/gate_metrics.py",
    "src/myrec/eval/metrics.py",
    "artifacts/c47_posterior_supported_ridge_transformer/signal_gate_v1/selection.json",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/signal_gate.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    _, proposal_hash = verify_proposal_lock(config)
    target = REPO_ROOT / config["paths"]["execution_lock"]
    value = {
        "candidate_id": "c69",
        "created_at": timestamp(),
        "decision": "authorize_six_fixed_fit_and_label_free_score_runs",
        "proposal_lock_sha256": proposal_hash,
        "source_sha256": {relative: sha256_file(REPO_ROOT / relative) for relative in SOURCES},
        "outcome_boundary": {
            "c47_A_labels_open_only_after_all_scores": True,
            "fresh_reserve_opened": False,
            "dev_test_qrels_opened": False,
        },
    }
    atomic_json(target, value)
    print(target.relative_to(REPO_ROOT))
    print(sha256_file(target))


if __name__ == "__main__":
    main()
