#!/usr/bin/env python
"""Train one supervised query/history diagnostic using train labels only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.supervised_diagnostics import train_diagnostic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/analysis/supervised_motivation_diagnostics.yaml",
    )
    parser.add_argument("--variant", choices=["d1q", "d1m", "d1a"], required=True)
    parser.add_argument("--mode", choices=["calibrate", "final"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    result = train_diagnostic(
        config=config,
        config_path=args.config,
        variant=args.variant,
        seed=args.seed,
        mode=args.mode,
        device=args.device,
        base_checkpoint=args.base_checkpoint,
        epochs=args.epochs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
