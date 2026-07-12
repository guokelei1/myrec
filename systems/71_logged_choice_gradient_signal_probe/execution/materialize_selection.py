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
    verify_registered_inputs,
)
from execution.selection import materialize_selection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/signal_gate.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    _, proposal_hash = verify_proposal_lock(config)
    verify_registered_inputs(config)
    value = materialize_selection(config, repo_root=REPO_ROOT)
    value["created_at"] = timestamp()
    value["proposal_lock_sha256"] = proposal_hash
    target = REPO_ROOT / config["paths"]["selection"]
    atomic_json(target, value)
    print(target.relative_to(REPO_ROOT))
    print(sha256_file(target))
    print(value["status"])


if __name__ == "__main__":
    main()
