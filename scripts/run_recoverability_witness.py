#!/usr/bin/env python
"""Score one dev condition with the train-only recoverability witness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.recoverability_witness import (
    write_recoverability_witness_scores,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("base", "full"), required=True)
    parser.add_argument(
        "--history-condition",
        choices=("base", "true", "null", "wrong"),
        required=True,
    )
    parser.add_argument("--history-assignments")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--config")
    args = parser.parse_args()
    result = write_recoverability_witness_scores(
        args.standardized_dir,
        args.model_dir,
        args.run_id,
        mode=args.mode,
        history_condition=args.history_condition,
        history_assignments_path=args.history_assignments,
        runs_dir=args.runs_dir,
        config_path=args.config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
