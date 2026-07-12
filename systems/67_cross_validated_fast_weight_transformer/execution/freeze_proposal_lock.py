"""Freeze C67's design before any outcome-bearing GPU run."""

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
)


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
        "candidate_id": "c67",
        "created_at": timestamp(),
        "decision": "freeze_data_free_cross_validated_fast_weight_falsifier",
        "design_sha256": {
            name: sha256_file(source) for name, source in design.items()
        },
        "predecessor_sha256": {
            "c66_report": sha256_file(REPO_ROOT / config["paths"]["c66_report"])
        },
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
