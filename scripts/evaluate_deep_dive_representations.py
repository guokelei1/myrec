#!/usr/bin/env python3
"""Evaluate D1 full/null all-layer representations after integrity gates."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.deep_dive_representation_evaluator import (
    evaluate_deep_dive_representations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--full-bundle", required=True)
    parser.add_argument("--null-bundle", required=True)
    parser.add_argument("--probe-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    result = evaluate_deep_dive_representations(
        args.standardized_dir,
        args.full_bundle,
        args.null_bundle,
        args.probe_model_dir,
        args.output_dir,
        args.analysis_run_id,
        dev_eval_log_path=args.dev_eval_log,
        command=sys.argv,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("analysis_run_id", "method_id", "status")
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

