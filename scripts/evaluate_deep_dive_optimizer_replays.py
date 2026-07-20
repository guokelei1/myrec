#!/usr/bin/env python3
"""Aggregate exact train-only Q2/Q3 step-501 replays."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.optimizer_replay_evaluator import evaluate_optimizer_replays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q2-bundle", required=True)
    parser.add_argument("--q3-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    args = parser.parse_args()
    result = evaluate_optimizer_replays(
        args.q2_bundle, args.q3_bundle, args.output_dir, args.analysis_run_id
    )
    print(json.dumps({"analysis_type": result["analysis_type"], "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
