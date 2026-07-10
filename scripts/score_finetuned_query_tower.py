#!/usr/bin/env python
"""Score label-free dev with the frozen D2t/D2p controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.finetuned_query_tower import score_query_tower


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--config",
        default="configs/analysis/finetuned_nonpersonalized_control.yaml",
    )
    parser.add_argument(
        "--final-config",
        default="configs/analysis/finetuned_nonpersonalized_control_final.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with Path(args.final_config).open("r", encoding="utf-8") as handle:
        final_config = yaml.safe_load(handle)
    result = score_query_tower(
        config,
        args.config,
        final_config,
        args.final_config,
        args.seed,
        args.checkpoint,
        args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
