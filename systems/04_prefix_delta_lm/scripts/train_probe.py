#!/usr/bin/env python
"""Train the paired C04 probe or one preregistered train/internal control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CANDIDATE_ROOT.parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from cpdlr.io import load_yaml
from cpdlr.model import PrefixDeltaRanker
from cpdlr.train import train_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CANDIDATE_ROOT / "configs/probe.yaml"))
    parser.add_argument("--mode", choices=sorted(PrefixDeltaRanker.VALID_MODES), required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_yaml(args.config)
    run_id = args.run_id or config["run_ids"][args.mode]
    result = train_probe(config, args.config, args.mode, run_id, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
