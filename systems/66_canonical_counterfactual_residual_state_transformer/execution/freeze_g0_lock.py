"""Freeze C66 canonical mechanics before G0."""

from __future__ import annotations

import argparse
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
    verify_proposal_lock,
)


SOURCES = (
    "systems/66_canonical_counterfactual_residual_state_transformer/configs/train_gate.yaml",
    "systems/66_canonical_counterfactual_residual_state_transformer/model/__init__.py",
    "systems/66_canonical_counterfactual_residual_state_transformer/model/canonical_residual.py",
    "systems/66_canonical_counterfactual_residual_state_transformer/train/__init__.py",
    "systems/66_canonical_counterfactual_residual_state_transformer/train/data_bridge.py",
    "systems/66_canonical_counterfactual_residual_state_transformer/execution/__init__.py",
    "systems/66_canonical_counterfactual_residual_state_transformer/execution/locking.py",
    "systems/66_canonical_counterfactual_residual_state_transformer/execution/freeze_g0_lock.py",
    "systems/66_canonical_counterfactual_residual_state_transformer/execution/run_g0.py",
    "systems/66_canonical_counterfactual_residual_state_transformer/tests/test_model.py",
    "systems/65_counterfactual_residual_state_transformer/model/counterfactual_residual.py",
    "systems/64_end_to_end_lm_representation_probe/model/adaptive_joint_ranker.py",
    "systems/64_end_to_end_lm_representation_probe/train/data.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/train_gate.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    _, proposal_hash = verify_proposal_lock(config)
    target = REPO_ROOT / config["paths"]["g0_lock"]
    value = {
        "candidate_id": "c66",
        "created_at": timestamp(),
        "decision": "authorize_one_canonical_G0",
        "proposal_lock_sha256": proposal_hash,
        "source_sha256": {
            source: sha256_file(REPO_ROOT / source) for source in SOURCES
        },
        "outcome_boundary": {
            "train_labels_opened": False,
            "validation_labels_opened": False,
            "fresh_dev_test_qrels_opened": False,
        },
    }
    atomic_json(target, value)
    print(target.relative_to(REPO_ROOT))
    print(sha256_file(target))


if __name__ == "__main__":
    main()
