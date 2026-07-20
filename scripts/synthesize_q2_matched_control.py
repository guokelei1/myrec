#!/usr/bin/env python3
"""Run the preregistered joined per-request Q2 matched-control DID synthesis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.mechanism.matched_control_synthesis import (  # noqa: E402
    ANALYSIS_RUN_ID,
    BALANCED_PAIR_RUN_ID,
    ORIGINAL_PAIR_RUN_ID,
    REGISTRATION_PATH,
    TOP_LEVEL_RUN_ID,
    synthesize_q2_matched_control_did,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-run-id", default=ANALYSIS_RUN_ID)
    parser.add_argument("--top-level-run-id", default=TOP_LEVEL_RUN_ID)
    parser.add_argument("--original-pair-run-id", default=ORIGINAL_PAIR_RUN_ID)
    parser.add_argument("--balanced-pair-run-id", default=BALANCED_PAIR_RUN_ID)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    parser.add_argument("--registration", default=str(REGISTRATION_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = synthesize_q2_matched_control_did(
        analysis_run_id=args.analysis_run_id,
        top_level_run_id=args.top_level_run_id,
        original_pair_run_id=args.original_pair_run_id,
        balanced_pair_run_id=args.balanced_pair_run_id,
        runs_dir=args.runs_dir,
        dev_eval_log_path=args.dev_eval_log,
        registration_path=args.registration,
        command=sys.argv,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
