#!/usr/bin/env python
"""Audit the one-call C04 screening after shared evaluation/comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CANDIDATE_ROOT.parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from cpdlr.audit import audit_screening
from cpdlr.io import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CANDIDATE_ROOT / "configs/probe.yaml"))
    parser.add_argument(
        "--nonrepeat-comparison",
        default=str(CANDIDATE_ROOT / "notes/screen_nonrepeat_vs_d2p.json"),
    )
    parser.add_argument(
        "--repeat-comparison",
        default=str(CANDIDATE_ROOT / "notes/screen_repeat_vs_item.json"),
    )
    parser.add_argument(
        "--deterministic-a",
        default="artifacts/c04_prefix_delta_lm/determinism_a/scores.jsonl",
    )
    parser.add_argument(
        "--deterministic-b",
        default="artifacts/c04_prefix_delta_lm/determinism_b/scores.jsonl",
    )
    parser.add_argument(
        "--output", default=str(CANDIDATE_ROOT / "notes/screening_audit.json")
    )
    args = parser.parse_args()
    result = audit_screening(
        load_yaml(args.config),
        args.output,
        args.nonrepeat_comparison,
        args.repeat_comparison,
        args.deterministic_a,
        args.deterministic_b,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
