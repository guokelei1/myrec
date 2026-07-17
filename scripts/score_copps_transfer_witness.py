#!/usr/bin/env python
"""Score one full/null/wrong assignment with the fixed W0 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.copps_transfer_witness import (  # noqa: E402
    BATCH_REQUESTS,
    write_copps_transfer_witness_scores,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--history-assignments", required=True)
    parser.add_argument(
        "--history-condition",
        choices=("full", "true", "null", "wrong"),
        required=True,
        help="full is normalized to the shared evaluator's true condition",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=("dev", "confirmation"), default="dev")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-requests", type=int, default=BATCH_REQUESTS)
    parser.add_argument(
        "--config",
        required=True,
        help="Frozen W0 config; production scoring refuses an implicit recipe.",
    )
    args = parser.parse_args()
    result = write_copps_transfer_witness_scores(
        args.standardized_dir,
        args.feature_store,
        args.checkpoint_dir,
        args.history_assignments,
        args.run_id,
        history_condition=args.history_condition,
        split=args.split,
        runs_dir=args.runs_dir,
        device=args.device,
        batch_requests=args.batch_requests,
        config_path=args.config,
        command=sys.argv,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
