#!/usr/bin/env python3
"""Evaluate one qrels-blind N19 Q3 LoRA branch bundle."""

from __future__ import annotations

import argparse
import sys

from myrec.mechanism.q3_lora_branch_evaluator import evaluate_q3_lora_branch_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    args = parser.parse_args()
    result = evaluate_q3_lora_branch_bundle(
        args.standardized_dir,
        args.bundle,
        args.output_dir,
        args.analysis_run_id,
        command=sys.argv,
    )
    print(result["analysis_type"], result["status"])


if __name__ == "__main__":
    main()
