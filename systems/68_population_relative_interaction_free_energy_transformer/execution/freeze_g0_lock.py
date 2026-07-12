"""Freeze C68 implementation before its sole data-free G0."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from execution.locking import (  # noqa: E402
    atomic_json,
    load_config,
    sha256_file,
    timestamp,
    verify_proposal_lock,
)


SOURCES = (
    "systems/68_population_relative_interaction_free_energy_transformer/configs/g0.yaml",
    "systems/68_population_relative_interaction_free_energy_transformer/model/__init__.py",
    "systems/68_population_relative_interaction_free_energy_transformer/model/population_relative_free_energy.py",
    "systems/68_population_relative_interaction_free_energy_transformer/execution/__init__.py",
    "systems/68_population_relative_interaction_free_energy_transformer/execution/locking.py",
    "systems/68_population_relative_interaction_free_energy_transformer/execution/synthetic.py",
    "systems/68_population_relative_interaction_free_energy_transformer/execution/freeze_proposal_lock.py",
    "systems/68_population_relative_interaction_free_energy_transformer/execution/freeze_g0_lock.py",
    "systems/68_population_relative_interaction_free_energy_transformer/execution/run_g0.py",
    "systems/68_population_relative_interaction_free_energy_transformer/execution/aggregate_g0.py",
    "systems/68_population_relative_interaction_free_energy_transformer/tests/__init__.py",
    "systems/68_population_relative_interaction_free_energy_transformer/tests/test_model.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/g0.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    _, proposal_hash = verify_proposal_lock(config)
    target = REPO_ROOT / config["paths"]["g0_lock"]
    value = {
        "candidate_id": "c68",
        "created_at": timestamp(),
        "decision": "authorize_three_registered_data_free_G0_seeds",
        "proposal_lock_sha256": proposal_hash,
        "source_sha256": {
            relative: sha256_file(REPO_ROOT / relative) for relative in SOURCES
        },
        "outcome_boundary": {
            "repository_data_opened": False,
            "labels_opened": False,
            "dev_test_qrels_opened": False,
        },
    }
    atomic_json(target, value)
    print(target.relative_to(REPO_ROOT))
    print(sha256_file(target))


if __name__ == "__main__":
    main()
