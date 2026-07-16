#!/usr/bin/env python
"""Score one true/null/wrong condition with a fixed HSTU/SASRec checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "baselines" / "hstu"))

from myrec.baselines.sequence_ranker_training import write_sequence_ranker_scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--history-assignments", required=True)
    parser.add_argument("--history-condition", choices=("true", "null", "wrong"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", default="dev", choices=("dev", "confirmation"))
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--method-id")
    args = parser.parse_args()
    result = write_sequence_ranker_scores(
        args.standardized_dir,
        args.feature_store,
        args.checkpoint_dir,
        args.history_assignments,
        args.run_id,
        history_condition=args.history_condition,
        split=args.split,
        runs_dir=args.runs_dir,
        device=args.device,
        batch_size=args.batch_size,
        method_id=args.method_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
