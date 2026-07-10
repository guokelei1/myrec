#!/usr/bin/env python
"""Evaluate frozen features on the train-only internal validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.supervised_diagnostics import evaluate_internal_reference_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/analysis/supervised_motivation_diagnostics.yaml",
    )
    parser.add_argument(
        "--output",
        default="artifacts/analysis/supervised_motivation_diagnostics/internal_reference_metrics.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    result = evaluate_internal_reference_features(config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
