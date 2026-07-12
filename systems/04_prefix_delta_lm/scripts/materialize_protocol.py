#!/usr/bin/env python
"""Freeze C04's label-free repeat/non-repeat/no-history request subsets."""

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
from cpdlr.protocol import materialize_structural_subsets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CANDIDATE_ROOT / "configs/probe.yaml"))
    parser.add_argument("--output-dir", default=str(CANDIDATE_ROOT / "protocol"))
    args = parser.parse_args()
    result = materialize_structural_subsets(load_yaml(args.config), args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
