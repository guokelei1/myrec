#!/usr/bin/env python3
"""Evaluate four M2 representation conditions after bundle integrity checks."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.representation_evaluator import evaluate_m2_representations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument(
        "--bundle",
        action="append",
        nargs=2,
        metavar=("CONDITION", "PATH"),
        required=True,
    )
    parser.add_argument("--probe-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    bundles = {condition: path for condition, path in args.bundle}
    if len(bundles) != len(args.bundle):
        parser.error("duplicate --bundle condition")
    result = evaluate_m2_representations(
        args.standardized_dir,
        bundles,
        args.probe_model_dir,
        args.output_dir,
        args.analysis_run_id,
        dev_eval_log_path=args.dev_eval_log,
        command=sys.argv,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
