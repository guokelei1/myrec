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
    proposal_sources,
    sha256_file,
    timestamp,
    verify_registered_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/signal_gate.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    verify_registered_inputs(config)
    target = REPO_ROOT / config["paths"]["proposal_lock"]
    value = {
        "candidate_id": "c71",
        "created_at": timestamp(),
        "decision": "freeze_fresh_unpacked_logged_choice_signal_gate",
        "design_sha256": {
            name: sha256_file(path) for name, path in proposal_sources(config).items()
        },
        "outcome_boundary": {
            "target_labels_before_A0": False,
            "source_episode_labels": False,
            "dev_test_qrels": False,
            "c70_architecture_authorized": False,
        },
    }
    atomic_json(target, value)
    print(target.relative_to(REPO_ROOT))
    print(sha256_file(target))


if __name__ == "__main__":
    main()
