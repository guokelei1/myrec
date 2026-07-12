"""Freeze C62 implementation before observing the synthetic G0 outcome."""

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
    "systems/62_write_once_preference_memory_transformer/configs/train_gate.yaml",
    "systems/62_write_once_preference_memory_transformer/model/__init__.py",
    "systems/62_write_once_preference_memory_transformer/model/write_once_memory.py",
    "systems/62_write_once_preference_memory_transformer/execution/__init__.py",
    "systems/62_write_once_preference_memory_transformer/execution/locking.py",
    "systems/62_write_once_preference_memory_transformer/execution/freeze_g0_lock.py",
    "systems/62_write_once_preference_memory_transformer/execution/run_g0.py",
    "systems/62_write_once_preference_memory_transformer/tests/test_model.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=SYSTEM_ROOT / "configs/train_gate.yaml",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    _, proposal_hash = verify_proposal_lock(config)
    target = REPO_ROOT / config["paths"]["g0_lock"]
    value = {
        "candidate_id": "c62",
        "created_at": timestamp(),
        "decision": "authorize_exactly_three_registered_synthetic_seeds",
        "proposal_lock_sha256": proposal_hash,
        "source_sha256": {
            path: sha256_file(REPO_ROOT / path) for path in SOURCES
        },
        "outcome_boundary": {
            "repository_records_opened": False,
            "exposed_fit_labels_opened": False,
            "fresh_features_scores_labels_opened": False,
            "dev_test_qrels_opened": False,
        },
    }
    atomic_json(target, value)
    print(target.relative_to(REPO_ROOT))
    print(sha256_file(target))


if __name__ == "__main__":
    main()
