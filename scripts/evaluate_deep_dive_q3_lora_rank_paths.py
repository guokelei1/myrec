#!/usr/bin/env python3
"""Evaluate the registered N10 Q3 LoRA rank-path bundle."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.q3_lora_rank_evaluator import evaluate_q3_lora_rank_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--qrels-split-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    args = parser.parse_args()
    result = evaluate_q3_lora_rank_bundle(
        args.standardized_dir,
        args.qrels_split_dir,
        args.bundle,
        args.output_dir,
        args.analysis_run_id,
        command=sys.argv,
    )
    print(json.dumps({"analysis_run_id": result["analysis_run_id"], "status": result["status"], "family_rows": len(result["family_rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()

