#!/usr/bin/env python
"""Write the C04 proposal lock before any GPU model outcome."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CANDIDATE_ROOT.parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from cpdlr.freeze import freeze_proposal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CANDIDATE_ROOT / "configs/probe.yaml"))
    args = parser.parse_args()
    result = freeze_proposal(args.config, CANDIDATE_ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
