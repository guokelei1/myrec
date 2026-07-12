"""Freeze C64 mechanics before label-free G0."""

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
    "systems/64_end_to_end_lm_representation_probe/configs/kuai_probe.yaml",
    "systems/64_end_to_end_lm_representation_probe/model/__init__.py",
    "systems/64_end_to_end_lm_representation_probe/model/adaptive_joint_ranker.py",
    "systems/64_end_to_end_lm_representation_probe/train/__init__.py",
    "systems/64_end_to_end_lm_representation_probe/train/data.py",
    "systems/64_end_to_end_lm_representation_probe/execution/__init__.py",
    "systems/64_end_to_end_lm_representation_probe/execution/locking.py",
    "systems/64_end_to_end_lm_representation_probe/execution/freeze_g0_lock.py",
    "systems/64_end_to_end_lm_representation_probe/execution/run_g0.py",
    "systems/64_end_to_end_lm_representation_probe/tests/test_model.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/kuai_probe.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    _, proposal_hash = verify_proposal_lock(config)
    target = REPO_ROOT / config["paths"]["g0_lock"]
    value = {
        "candidate_id": "c64",
        "created_at": timestamp(),
        "decision": "authorize_one_registered_label_free_G0",
        "proposal_lock_sha256": proposal_hash,
        "source_sha256": {
            source: sha256_file(REPO_ROOT / source) for source in SOURCES
        },
        "outcome_boundary": {
            "fit_labels_opened": False,
            "fresh_features_scores_labels_opened": False,
            "dev_test_qrels_opened": False,
        },
    }
    atomic_json(target, value)
    print(target.relative_to(REPO_ROOT))
    print(sha256_file(target))


if __name__ == "__main__":
    main()
