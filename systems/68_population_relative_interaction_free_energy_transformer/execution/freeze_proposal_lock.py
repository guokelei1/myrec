"""Freeze C68's design before any outcome-bearing run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from execution.locking import atomic_json, load_config, sha256_file, timestamp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/g0.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    target = REPO_ROOT / config["paths"]["proposal_lock"]
    design = {
        "config": SYSTEM_ROOT / "configs/g0.yaml",
        "nearest_neighbors": REPO_ROOT / config["paths"]["nearest_neighbors"],
        "preimplementation_review": REPO_ROOT
        / config["paths"]["preimplementation_review"],
        "proposal": REPO_ROOT / config["paths"]["proposal"],
        "readme": SYSTEM_ROOT / "README.md",
    }
    value = {
        "candidate_id": "c68",
        "created_at": timestamp(),
        "decision": "freeze_population_relative_interaction_free_energy_G0",
        "design_sha256": {name: sha256_file(path) for name, path in design.items()},
        "outcome_boundary": {
            "repository_data_authorized": False,
            "labels_authorized": False,
            "dev_test_qrels_authorized": False,
        },
    }
    atomic_json(target, value)
    print(target.relative_to(REPO_ROOT))
    print(sha256_file(target))


if __name__ == "__main__":
    main()
